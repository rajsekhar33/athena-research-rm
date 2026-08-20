"""
Basic data operations for Athenak simulation data.
"""
import numpy as np
from ..core.base import xp, asnumpy, cupy_enabled
from ..utils.batch_processing import determine_blocks_per_batch
from ..core.utils import clear_backend_memory, get_distributed_block_bounds, reduce_array_mpi, maybe_trim_gpu_memory_pool

# MPI support
try:
    from ..backends.mpi_utils import MPIManager
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def calc_data(ad, varl=['mass', 'eint', 'ekin'], varsuf='', operation='sum', scale='', 
            weights='vol', redo=False, simultaneous_blocks=None, use_mpi=False, debug=False):
    """
    Calculate aggregated statistics across mesh-block data with memory optimization.
    
    This unified function handles sum, min, max and average operations across mesh blocks in
    a memory-efficient way using batched processing.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : list of str, optional
        Names of the variables to process. Defaults to ['mass', 'eint', 'ekin'].
    varsuf : str, optional
        Suffix appended to variable names in results. Defaults to ''.
    operation : {'sum', 'min', 'max', 'average'}, optional
        Statistical operation to perform on the data. Defaults to 'sum'.
    scale : {'', 'log10', 'ln'}, optional
        Data scaling to apply before calculation. Choose from '' (no scaling), 
        'log10', or 'ln'. Defaults to ''.
    weights : str, optional
        Name of weighting variable (used for 'sum' and 'average' operations). Defaults to 'vol'.
    redo : bool, optional
        If True, forces recalculation even if results exist. Defaults to False.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, use MPI to distribute computation across ranks. Defaults to False.
    debug : bool, optional
        If True, print debug messages during execution. Defaults to False.
    
    Returns
    -------
    float
        The calculated value (sum, min, max, or average) for the last variable in varl.
        All results are also stored in the appropriate dictionary.
    
    Notes
    -----
    - For 'sum' and 'average' operations, values are weighted by the specified weights parameter
    - For 'min' and 'max' operations, weights are ignored
    - NaN and infinite values are handled by replacing them with zeros
    - Memory usage is optimized through batch processing for large datasets
    """
    # Initialize MPI if requested
    mpi_manager = None
    if use_mpi and MPI_AVAILABLE:
        mpi_manager = MPIManager()
        rank = mpi_manager.rank
        size = mpi_manager.size
        if debug:
            print(f"[DEBUG rank {rank}] calc_data({operation}): MPIManager initialized", flush=True)
    else:
        rank = 0
        size = 1
    
    # Validate operation parameter
    if operation not in ['sum', 'min', 'max', 'average']:
        raise ValueError(f"Invalid operation '{operation}'. Must be one of: 'sum', 'min', 'max', 'average'")
        
    # Set up target dictionary based on operation
    if operation == 'sum':
        target_dict = ad.sum
        # Initial value for accumulation
        initial_value = 0.0
    elif operation == 'min':
        target_dict = ad.min
        # Initial value for min comparison
        initial_value = xp.finfo(xp.float64).max
    elif operation == 'max':
        target_dict = ad.max
        # Initial value for max comparison
        initial_value = xp.finfo(xp.float64).min
    elif operation == 'average':
        # Create avg dictionary if it doesn't exist
        if not hasattr(ad, 'avg'):
            ad.avg = {}
        target_dict = ad.avg
        # No initial value needed for average - we'll calculate total and weight separately
        initial_value = (0.0, 0.0)  # (weighted_sum, total_weight)
    
    # Use _determine_blocks_per_batch to optimize batch size
    no_vars = len(varl)
    n_weights = 1 if weights is not None else 0  # Only one weight if specified
    n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
    
    if simultaneous_blocks is None:
        # Determine optimal batch size using the helper function
        simultaneous_blocks = determine_blocks_per_batch(
            n_mbs=ad.get_local_mb_count() if (use_mpi and mpi_manager is not None) else ad.n_mbs,
            no_vars=no_vars,
            n_weights=n_weights,
            n_points_per_block=n_points_per_block
        )
    
    # Use locally owned meshblocks (data is already distributed at load time)
    if use_mpi and mpi_manager is not None:
        mb_start_global, mb_end_global, total_mbs = get_distributed_block_bounds(ad)
    else:
        mb_start_global = 0
        mb_end_global = ad.n_mbs
        total_mbs = ad.n_mbs
    
    # Process blocks in optimized batches
    num_batches = (total_mbs + simultaneous_blocks - 1) // simultaneous_blocks
    
    for var in varl:
        varname = var + varsuf
        if redo or varname not in target_dict:
            result_value = initial_value
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * simultaneous_blocks
                batch_end = min(batch_start + simultaneous_blocks, total_mbs)
                
                # Convert to global meshblock indices
                batch_start_global = mb_start_global + batch_start
                batch_end_global = mb_start_global + batch_end
                
                if batch_end <= batch_start:
                    continue
                
                # Load and transform data for this batch
                if scale == 'log10':
                    data = xp.log10(ad.data(var, batch_start_global, batch_end_global))
                elif scale == 'ln':
                    data = xp.log(ad.data(var, batch_start_global, batch_end_global))
                else:
                    data = ad.data(var, batch_start_global, batch_end_global)
                
                # Handle NaN and infinite values
                xp.nan_to_num(data, copy=False, posinf=0.0, neginf=0.0)
                
                # Perform the requested operation
                if operation == 'sum':
                    weinorm = ad.data(weights, batch_start_global, batch_end_global)
                    batch_result = xp.sum(data * weinorm)
                    result_value += batch_result
                elif operation == 'min':
                    batch_result = xp.min(data)
                    result_value = min(result_value, batch_result)
                elif operation == 'max':
                    batch_result = xp.max(data)
                    result_value = max(result_value, batch_result)
                elif operation == 'average':
                    weinorm = ad.data(weights, batch_start_global, batch_end_global)
                    # Calculate weighted sum and total weights for this batch
                    weighted_sum = xp.sum(data * weinorm)
                    total_weight = xp.sum(weinorm)
                    # Add to the running total
                    result_value = (result_value[0] + weighted_sum, result_value[1] + total_weight)
                
                # Free memory if using GPU
                if cupy_enabled:
                    maybe_trim_gpu_memory_pool()
            
            # MPI aggregation
            if mpi_manager is not None:
                if operation == 'sum':
                    result_value = reduce_array_mpi(mpi_manager, result_value, op='sum')
                elif operation == 'min':
                    result_value = reduce_array_mpi(mpi_manager, result_value, op='min')
                elif operation == 'max':
                    result_value = reduce_array_mpi(mpi_manager, result_value, op='max')
                elif operation == 'average':
                    result_value = (
                        reduce_array_mpi(mpi_manager, result_value[0], op='sum'),
                        reduce_array_mpi(mpi_manager, result_value[1], op='sum')
                    )
            
            # Store the result
            if operation == 'average':
                # Calculate the weighted average from accumulated values
                if result_value[1] > 0:  # Avoid division by zero
                    target_dict[varname] = result_value[0] / result_value[1]
                else:
                    target_dict[varname] = 0.0
            else:
                target_dict[varname] = result_value
    
    # Return the target dictionary
    return target_dict

def calc_sum(ad, varl=['mass', 'eint', 'ekin'], varsuf='', scale='', weights='vol',
                redo=False, simultaneous_blocks=None, use_mpi=False):
    """
    Calculate the total sum of specified variables across mesh-block data and store the results.

    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : (list of str, optional)
        Names of the variables for which to compute the sum. Defaults to ['mass', 'eint', 'ekin'].
    varsuf : (str, optional)
        A suffix appended to the variable name in the results. Defaults to ''.
    scale : (str, optional)
        The data scaling to apply. Choose from '' (no scaling), 'log10', or 'ln'. Defaults to ''.
    weights : (str, optional)
        The name of the weighting variable. Defaults to 'vol'.
    redo : (bool, optional)
        Whether to redo the calculation even if a prior result exists. Defaults to False.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, use MPI to distribute computation across ranks. Defaults to False.

    Returns:
        float: The sum value for the last variable processed.
    """
    return calc_data(ad,varl=varl, varsuf=varsuf, operation='sum', 
                            scale=scale, weights=weights, redo=redo, simultaneous_blocks=simultaneous_blocks, use_mpi=use_mpi)

def calc_min(ad, varl=['mass', 'eint', 'ekin'], varsuf='', redo=False, simultaneous_blocks=None, use_mpi=False, debug=False):
    """
    Calculate the global minimum value for one or more variables across all mesh blocks.

    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : list of str, optional
        The names of the variables to compute minimum values for. Default variables include
        'mass', 'eint', and 'ekin'.
    varsuf : str, optional
        A string suffix appended to each variable name, allowing computation of specialized
        or derived variables.
    redo : bool, optional
        If True, forces the function to recompute all minimum values even if cached results
        are already available. Defaults to False.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, use MPI to distribute computation across ranks. Defaults to False.
    debug : bool, optional
        If True, print debug messages during execution. Defaults to False.

    Returns
    -------
    float
        The minimum value computed for the last variable in varl.
    """
    return calc_data(ad,varl=varl, varsuf=varsuf, operation='min', 
                            scale='', weights=None, redo=redo, simultaneous_blocks=simultaneous_blocks, use_mpi=use_mpi, debug=debug)

def calc_max(ad, varl=['mass', 'eint', 'ekin'], varsuf='', redo=False, simultaneous_blocks=None, use_mpi=False, debug=False):
    """
    Calculate the maximum values for the specified variables in the mesh block data.

    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : list of str, optional
        List of variable names to process. Defaults to ['mass', 'eint', 'ekin'].
    varsuf : str, optional
        Suffix to append to variable names if needed. Defaults to an empty string.
    redo : bool, optional
        Whether to recalculate values even if they are already present. Defaults to False.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, use MPI to distribute computation across ranks. Defaults to False.
    debug : bool, optional
        If True, print debug messages during execution. Defaults to False.
    Returns
    -------
    float
        The maximum value among the specified variables in the mesh block data.
    """
    return calc_data(ad,varl=varl, varsuf=varsuf, operation='max', 
                            scale='', weights=None, redo=redo, simultaneous_blocks=simultaneous_blocks, use_mpi=use_mpi, debug=debug)
    
def calc_avg(ad, varl=['dens', 'temp', 'ekin'], varsuf='', scale='', weights='vol', 
               redo=False, simultaneous_blocks=None, use_mpi=False, debug=False):
    """
    Calculate the weighted average of variables across mesh blocks.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : list of str, optional
        Names of the variables to calculate averages for. Defaults to ['dens', 'temp', 'ekin'].
    varsuf : str, optional
        Suffix appended to variable names in results. Defaults to ''.
    scale : {'', 'log10', 'ln'}, optional
        Data scaling to apply before calculation. Default is ''.
    weights : str, optional
        Name of weighting variable. Defaults to 'vol'.
    redo : bool, optional
        If True, forces recalculation even if results exist. Defaults to False.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, use MPI to distribute computation across ranks. Defaults to False.
    debug : bool, optional
        If True, print debug messages during execution. Defaults to False.
    
    Returns
    -------
    float
        The weighted average for the last variable in varl.
        All results are also stored in ad.avg dictionary.
    """
    return calc_data(ad, varl=varl, varsuf=varsuf, operation='average',
                        scale=scale, weights=weights, redo=redo, 
                        simultaneous_blocks=simultaneous_blocks, use_mpi=use_mpi, debug=debug)


def calc_filtered_extrema(ad, varl=["dens"], scales="", weights="vol",
                          vert_range=None, simultaneous_blocks=None,
                          use_mpi=False, debug=False):
    """
    Calculate per-variable minima and maxima over the effective support of a
    filtered distribution.

    This helper applies the same basic support rules used by weighted/sub-region
    PDF calculations: optional coordinate filtering, optional non-zero-weight
    filtering, and finite-value filtering after any requested transform.
    """
    mpi_manager = None
    if use_mpi and MPI_AVAILABLE:
        mpi_manager = MPIManager()
        rank = mpi_manager.rank
    else:
        rank = 0

    if isinstance(scales, str):
        scale_list = [scales] * len(varl)
    else:
        scale_list = list(scales)
        if len(scale_list) != len(varl):
            raise ValueError("scales must be a string or match len(varl)")

    if simultaneous_blocks is None:
        n_weights = 1 if weights is not None else 0
        if vert_range is not None:
            n_weights += 1
        simultaneous_blocks = determine_blocks_per_batch(
            n_mbs=ad.n_mbs if not (use_mpi and mpi_manager is not None) else ad.get_local_mb_count(),
            no_vars=len(varl),
            n_weights=n_weights,
            n_points_per_block=ad.nx1 * ad.nx2 * ad.nx3,
        )

    if use_mpi and mpi_manager is not None:
        mb_start_global = ad.local_mb_start
        total_mbs = ad.get_local_mb_count()
    else:
        mb_start_global = 0
        total_mbs = ad.n_mbs

    simultaneous_blocks = min(simultaneous_blocks, total_mbs) if total_mbs > 0 else 0
    num_batches = (total_mbs + simultaneous_blocks - 1) // simultaneous_blocks if simultaneous_blocks > 0 else 0

    local_mins = [float("inf")] * len(varl)
    local_maxs = [float("-inf")] * len(varl)
    local_counts = [0] * len(varl)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * simultaneous_blocks
        batch_end = min(batch_start + simultaneous_blocks, total_mbs)
        if batch_end <= batch_start:
            continue

        batch_start_global = mb_start_global + batch_start
        batch_end_global = mb_start_global + batch_end

        weight_data = ad.data(weights, batch_start_global, batch_end_global) if weights is not None else None
        z_data = ad.data("z", batch_start_global, batch_end_global) if vert_range is not None else None

        base_mask = None
        if vert_range is not None:
            base_mask = xp.logical_and(z_data > vert_range[0], z_data < vert_range[1])
        if weight_data is not None:
            weight_mask = weight_data != 0
            base_mask = weight_mask if base_mask is None else xp.logical_and(base_mask, weight_mask)

        for ivar, (var, scale) in enumerate(zip(varl, scale_list)):
            if scale == "log10":
                data = xp.log10(ad.data(var, batch_start_global, batch_end_global))
            elif scale == "ln":
                data = xp.log(ad.data(var, batch_start_global, batch_end_global))
            else:
                data = ad.data(var, batch_start_global, batch_end_global)

            valid_mask = xp.isfinite(data)
            if base_mask is not None:
                valid_mask = xp.logical_and(valid_mask, base_mask)

            if cupy_enabled:
                valid_count = int(xp.sum(valid_mask).get())
            else:
                valid_count = int(xp.sum(valid_mask))

            if valid_count == 0:
                continue

            data_valid = data[valid_mask]
            batch_min = float(asnumpy(xp.min(data_valid)))
            batch_max = float(asnumpy(xp.max(data_valid)))
            local_mins[ivar] = min(local_mins[ivar], batch_min)
            local_maxs[ivar] = max(local_maxs[ivar], batch_max)
            local_counts[ivar] += valid_count

        if cupy_enabled:
            maybe_trim_gpu_memory_pool()

    if mpi_manager is not None:
        mins_np = np.asarray(local_mins, dtype=np.float64)
        maxs_np = np.asarray(local_maxs, dtype=np.float64)
        counts_np = np.asarray(local_counts, dtype=np.int64)

        mins_total = mpi_manager.allreduce(mins_np, op="min")
        maxs_total = mpi_manager.allreduce(maxs_np, op="max")
        counts_total = mpi_manager.allreduce(counts_np, op="sum")

        local_mins = np.asarray(mins_total, dtype=np.float64).tolist()
        local_maxs = np.asarray(maxs_total, dtype=np.float64).tolist()
        local_counts = np.asarray(counts_total, dtype=np.int64).tolist()

    mins = {}
    maxs = {}
    for ivar, var in enumerate(varl):
        if local_counts[ivar] == 0:
            raise ValueError(f"No valid data points found for filtered extrema of {var}")
        mins[var] = local_mins[ivar]
        maxs[var] = local_maxs[ivar]

    if debug and rank == 0:
        for var in varl:
            print(f"[DEBUG calc_filtered_extrema] {var}: min={mins[var]}, max={maxs[var]}")

    return mins, maxs