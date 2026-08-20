"""
Profile calculation for Athenak simulation data.
"""
import numpy as np
from ..core.base import cupy_enabled, xp, asnumpy
from ..core.utils import clear_backend_memory, get_distributed_block_bounds, reduce_array_mpi
from ..utils.batch_processing import determine_blocks_per_batch
from ..core.utils import maybe_trim_gpu_memory_pool
from .basic_operations import calc_min, calc_max

# Try to import MPI utilities
try:
    from ..backends.mpi_utils import MPIManager
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

def set_profile(ad, varl=['dens','temp','pres','mdot'], varsuf='',
                bins=None, weights='vol', redo=False, vert_range=None, simultaneous_blocks=None, use_mpi=False, debug=False,
                axis='z', log_scale=None):
    """
    Compute 1D profiles along a Cartesian axis ('x','y','z'). Optionally
    computes profiles of log(variable) for specified variables.

    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : list of str, optional
        A list of variables to process. Defaults to ['dens','temp','pres','mdot'].
    varsuf : str, optional
        Optional suffix appended to each variable name. Defaults to ''.
    bins : int, optional
        Number of bins for histogramming. If None, defaults to mesh resolution
        along the chosen axis (ad.Nx1/Nx2/Nx3 or ad.nx1/nx2/nx3).
    weights : str, optional
        Name of the field used as the weighting function. Defaults to 'vol'.
    redo : bool, optional
        If True, force recalculation of profiles even if they already exist. Defaults to False.
    vert_range : tuple of float, optional
        Tuple specifying the coordinate range (min, max) along the chosen axis.
        If None, the domain extent for that axis is used.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, distributes computation across MPI ranks. Defaults to False.
    debug : bool, optional
        If True, print debug information.
    axis : {'x','y','z'}, optional
        Axis along which to compute the profile (default: 'z').
    log_scale : list or set of str, optional
        Names of variables to compute profiles of log(var) instead of var. Defaults to None.

    Returns
    -------
    Returns the updated ad.vert dictionary with entries per variable. Each entry
    contains at least 'coord' (bin centers), 'profile' (means), and 'sigma' (std).
    For backward compatibility, if axis == 'z' 'z' and 'sigma_z' keys are also set.
    """
    # Normalize log_scale to a set for faster lookups
    if log_scale is None:
        log_scale = set()
    elif isinstance(log_scale, (list, tuple)):
        log_scale = set(log_scale)
    
    # Initialize MPI if requested
    mpi_manager = None
    if use_mpi and MPI_AVAILABLE:
        mpi_manager = MPIManager()
        rank = mpi_manager.rank
        size = mpi_manager.size
    else:
        rank = 0
        size = 1
    
    # Identify variables that need processing
    var_to_do = []
    varl_to_do = []
    # Normalize axis and choose defaults based on axis
    axis = axis.lower()
    if axis not in ('x', 'y', 'z'):
        raise ValueError("axis must be one of 'x','y','z'")

    if bins is None:
        if axis == 'x':
            bins = getattr(ad, 'Nx1', getattr(ad, 'nx1', None))
        elif axis == 'y':
            bins = getattr(ad, 'Nx2', getattr(ad, 'nx2', None))
        else:
            bins = getattr(ad, 'Nx3', getattr(ad, 'nx3', None))
        if bins is None:
            bins = 100

    if vert_range is None:
        if axis == 'x':
            vert_range = (ad.x1min, ad.x1max)
        elif axis == 'y':
            vert_range = (ad.x2min, ad.x2max)
        else:
            vert_range = (ad.x3min, ad.x3max)
    
    for var in varl:
        varname = var + varsuf
        if redo or varname not in ad.vert.keys():
            varl_to_do.append(varname)
            var_to_do.append(var)
    
    # If nothing to do, return early
    if not var_to_do:
        return ad.vert
    
    # Initialize accumulators - one array per variable
    hist_values = [xp.zeros(bins, dtype=xp.float64) for _ in range(len(varl_to_do))]
    hist_sq_values = [xp.zeros(bins, dtype=xp.float64) for _ in range(len(varl_to_do))]
    weight_values = xp.zeros(bins, dtype=xp.float64)
    
    # Compute bin centers once (independent of data, same for all ranks)
    coord_bin_edges = xp.linspace(vert_range[0], vert_range[1], bins + 1)
    coord_bin_centers = 0.5 * (coord_bin_edges[1:] + coord_bin_edges[:-1])
    
    # Use pre-distributed meshblock data from load time
    if mpi_manager is not None:
        mb_start_global, mb_end_global, n_mbs_local = get_distributed_block_bounds(ad)
    else:
        mb_start_global = 0
        mb_end_global = ad.n_mbs
        n_mbs_local = ad.n_mbs
    
    if debug:
        if rank == 0:
            print(f"[DEBUG set_profile] coord_bin_edges shape: {coord_bin_edges.shape}, range: [{asnumpy(coord_bin_edges[0])}, {asnumpy(coord_bin_edges[-1])}]")
            print(f"[DEBUG set_profile] vert_range: {vert_range}, bins: {bins}")
        print(f"[DEBUG set_profile] Rank {rank}: Processing meshblocks {mb_start_global} to {mb_end_global} (local count: {n_mbs_local})")
    
    # Determine optimal batch size using determine_blocks_per_batch helper
    if simultaneous_blocks is None:
        # Calculate parameters for the helper function
        no_vars = len(var_to_do)
        n_weights = 1  # We're using one weight variable
        n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
        
        # Use helper function to determine optimal batch size
        simultaneous_blocks = determine_blocks_per_batch(
            n_mbs=n_mbs_local,
            no_vars=no_vars,
            n_weights=n_weights,
            n_points_per_block=n_points_per_block
        )
    
    # Process blocks in optimized batches
    num_batches = (n_mbs_local + simultaneous_blocks - 1) // simultaneous_blocks
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * simultaneous_blocks
        batch_end = min(batch_start + simultaneous_blocks, n_mbs_local)
        
        if batch_end <= batch_start:
            continue
        
        # Convert to global meshblock indices
        batch_start_global = mb_start_global + batch_start
        batch_end_global = mb_start_global + batch_end
            
        # Load weights and coordinate values for the entire batch at once
        weinorm = ad.data(weights, batch_start_global, batch_end_global)
        coord_name = {'x': 'x', 'y': 'y', 'z': 'z'}[axis]
        coord_loc = ad.data(coord_name, batch_start_global, batch_end_global)

        if debug and batch_idx == 0:
            print(f"[DEBUG set_profile] Rank {rank} Batch 0: coord_loc shape={coord_loc.shape}, range=[{xp.min(coord_loc)}, {xp.max(coord_loc)}]")
            print(f"[DEBUG set_profile] Rank {rank} Batch 0: weinorm shape={weinorm.shape}, sum={xp.sum(weinorm)}")
        
        # The weight histogram does not depend on the profiled variable.
        weight_vals, _ = xp.histogram(coord_loc, bins=coord_bin_edges, weights=weinorm, density=False)
        weight_values += weight_vals

        # Process each variable for the current batch
        for ivar, var in enumerate(var_to_do):
            # Load variable data for this batch
            data = ad.data(var, batch_start_global, batch_end_global)
            
            # Apply log transform if requested
            if var in log_scale:
                data = xp.log10(data)
            
            # Calculate histograms for this variable and batch using pre-computed bin edges
            hist_vals, _ = xp.histogram(coord_loc, bins=coord_bin_edges, weights=data * weinorm, density=False)
            hist_sq_vals, _ = xp.histogram(coord_loc, bins=coord_bin_edges, weights=data**2 * weinorm, density=False)
            
            if debug and rank == 0 and batch_idx == 0 and ivar == 0:
                print(f"[DEBUG set_profile] Var {var}: hist_vals sum={asnumpy(hist_vals.sum())}, weight_vals sum={asnumpy(weight_vals.sum())}")
            
            # Accumulate results - add the values to our running totals
            hist_values[ivar] += hist_vals
            hist_sq_values[ivar] += hist_sq_vals
        
        # Free memory after each batch if using GPU
        if cupy_enabled:
            maybe_trim_gpu_memory_pool()
    
    # Gather results from all MPI ranks
    if mpi_manager is not None:
        weight_np = asnumpy(weight_values)
        weight_total = mpi_manager.allreduce(weight_np, op='sum')
        weight_values = xp.asarray(weight_total)

        for ivar in range(len(varl_to_do)):
            if debug:
                print(f"[DEBUG set_profile] Rank {rank} ivar {ivar} BEFORE MPI: hist_sum={asnumpy(hist_values[ivar]).sum():.6f}, weight_sum={asnumpy(weight_values).sum():.6f}")

            hist_total = reduce_array_mpi(mpi_manager, hist_values[ivar], op='sum')
            hist_sq_total = reduce_array_mpi(mpi_manager, hist_sq_values[ivar], op='sum')

            if debug:
                print(f"[DEBUG set_profile] Rank {rank} ivar {ivar} AFTER MPI: hist_sum={asnumpy(hist_total).sum():.6f}, weight_sum={asnumpy(weight_values).sum():.6f}")

            hist_values[ivar] = hist_total
            hist_sq_values[ivar] = hist_sq_total
    
    # Compute final statistics for each variable
    for ivar, varname in enumerate(varl_to_do):
        # Calculate mean and standard deviation
        # Note: In MPI mode, some bins may have no data from any rank
        # We compute for all bins but mask will handle zeros

        if debug:
            valid_bins = weight_values > 0
            print(f"[DEBUG set_profile] ivar {ivar} ({varname}): weight_values sum={xp.sum(weight_values):.6f}, valid_bins count={xp.sum(valid_bins)}")

        # Compute mean: where weight > 0, divide; elsewhere leave as 0
        mean_coord = xp.where(weight_values > 0,
                              hist_values[ivar] / weight_values,
                              0.0)

        # Compute variance and std dev
        ms_coord = xp.where(weight_values > 0,
                            hist_sq_values[ivar] / weight_values,
                            0.0)
        variance = xp.maximum(0, ms_coord - mean_coord**2)
        sigma_coord = xp.sqrt(variance)

        # Store results in ad.vert dictionary. Provide generic 'coord' key and
        # backward-compatible 'z' keys when axis == 'z'.
        entry = {
            'coord': asnumpy(coord_bin_centers),
            'profile': asnumpy(mean_coord),
            'sigma': asnumpy(sigma_coord),
            'axis': axis
        }
        if axis == 'z':
            entry['z'] = entry['coord']
            entry['sigma_z'] = entry['sigma']

        # Store under axis-suffixed key to avoid overwriting profiles from other axes
        store_key = f"{varname}_{axis}"
        ad.vert[store_key] = entry
        # Keep legacy key for z-axis to preserve backward compatibility
        if axis == 'z':
            ad.vert[varname] = entry
    
    return ad.vert

def set_vertical(ad, varl=['dens','temp','pres','mdot'], varsuf='',
                bins=None, weights='vol', redo=False, vert_range=None, simultaneous_blocks=None, use_mpi=False, debug=False,
                axis='z', log_scale=None):
    """
    Backward-compatible wrapper for set_profile. Computes 1D profiles along z-axis.
    
    This function is maintained for backward compatibility. New code should use
    set_profile() directly.
    
    Parameters
    ----------
    All parameters are passed directly to set_profile. See set_profile docstring for details.
    log_scale : list or set of str, optional
        Names of variables to compute profiles of log(var) instead of var. Defaults to None.
    
    Returns
    -------
    Returns the updated ad.vert dictionary. See set_profile for details.
    """
    return set_profile(ad, varl=varl, varsuf=varsuf, bins=bins, weights=weights, redo=redo,
                      vert_range=vert_range, simultaneous_blocks=simultaneous_blocks, use_mpi=use_mpi,
                      debug=debug, axis=axis, log_scale=log_scale)

def set_radial(ad, varl=['dens','temp','pres','mdot'], varsuf='',
                bins=100, weights='vol', redo=False, radial_range=None, 
                flx_flag=False, rad_scale='log', simultaneous_blocks=None, use_mpi=False, log_scale=None):
    """
    Compute radial distributions of specified variables with optimized memory usage.
    
    This method aggregates data from mesh blocks to produce radial profiles
    of the specified variables. It computes histograms for each variable over a
    range of radial bins, applies weighting, and calculates the
    mean and standard deviation, while efficiently managing memory. Optionally
    computes profiles of log(variable) for specified variables.
    
    Parameters
    ----------
    varl : list of str, optional
        List of variable names to process. Default is ['dens','temp','pres','mdot'].
    varsuf : str, optional
        Suffix appended to each variable name when storing results in `ad.rad`.
    bins : int, optional
        Number of radial bins in the histogram. Default is 100.
    weights : str, optional
        Name of the quantity used as weights in the histogram. Default is 'vol'.
    redo : bool, optional
        If True, recalculates distributions even if they already exist. Default is False.
    radial_range : tuple of float or None, optional
        Custom radial range for the histogram (r_min, r_max). If None, it is inferred.
    flx_flag : bool, optional
        If True, treats all variables as fluxes and scales results by 4πr². Default is False.
    rad_scale : {'log', 'linear'}, optional
        Sets the scale for radial coordinates: 'log' for log-scale bins (default)
        and 'linear' for regular bins.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, distributes computation across MPI ranks. Defaults to False.
    log_scale : list or set of str, optional
        Names of variables to compute profiles of log(var) instead of var. Defaults to None.
    
    Returns
    -------
    Returns the updated `ad.rad` dictionary containing the radial distributions
    for the specified variables and others already in the dictionary.
    Notes
    -----
    - The method calculates histograms for each variable, accumulating weighted sums
    and squared sums to derive means and variances.
    - For flux variables (e.g., mass flux, momentum flux), the histogram results are
    multiplied by 4πr².
    - Results are stored in the `ad.rad` dictionary keyed by variable name + suffix.
    - Memory usage is optimized through batch processing and efficient array handling.
    - MPI support distributes meshblocks across ranks and aggregates results
    """
    # Normalize log_scale to a set for faster lookups
    if log_scale is None:
        log_scale = set()
    elif isinstance(log_scale, (list, tuple)):
        log_scale = set(log_scale)
    # Initialize MPI if requested
    mpi_manager = None
    if use_mpi and MPI_AVAILABLE:
        mpi_manager = MPIManager()
        rank = mpi_manager.rank
        size = mpi_manager.size
    else:
        rank = 0
        size = 1
    
    # Identify variables that need processing
    var_to_do = []
    varl_to_do = []
    
    # Determine radial range if not provided
    if radial_range is None:
        if rad_scale == 'log': 
            mins = calc_min(ad, varl=['dx'], use_mpi=use_mpi)
            maxs = calc_max(ad, varl=['r'], use_mpi=use_mpi)
            min_radial_range = xp.log10(mins['dx'])
            max_radial_range = xp.log10(maxs['r'])
        else:
            mins = calc_min(ad, varl=['r'], use_mpi=use_mpi)
            maxs = calc_max(ad, varl=['r'], use_mpi=use_mpi)
            min_radial_range = mins['r']
            max_radial_range = maxs['r']
        radial_range = (min_radial_range, max_radial_range)
    
    # Find which variables need processing
    for var in varl:
        varname = var + varsuf
        if redo or varname not in ad.rad.keys():
            varl_to_do.append(varname)
            var_to_do.append(var)
    
    # If nothing to do, return early
    if not var_to_do:
        return ad.rad
    
    # Initialize accumulators for histograms - using lists of arrays instead of multi-dimensional arrays
    hist_r = [xp.zeros(bins) for _ in range(len(varl_to_do))]
    hist_sq_r = [xp.zeros(bins) for _ in range(len(varl_to_do))]
    weight_sum_r = xp.zeros(bins)
    
    # Compute bin edges and centers once (independent of data, same for all ranks)
    if rad_scale == 'log':
        bin_edges = xp.linspace(radial_range[0], radial_range[1], bins + 1)
        rad_r = 10 ** ((bin_edges[:-1] + bin_edges[1:]) / 2)
        dr_r = 10 ** bin_edges[1:] - 10 ** bin_edges[:-1]
    else:
        bin_edges = xp.linspace(radial_range[0], radial_range[1], bins + 1)
        rad_r = (bin_edges[:-1] + bin_edges[1:]) / 2
        dr_r = xp.diff(bin_edges)
    
    # Use pre-distributed meshblock data from load time
    if mpi_manager is not None:
        # Data already distributed at load time - use local meshblock range
        mb_start_global = ad.local_mb_start
        mb_end_global = ad.local_mb_end
        n_mbs_local = ad.get_local_mb_count()
    else:
        mb_start_global = 0
        mb_end_global = ad.n_mbs
        n_mbs_local = ad.n_mbs
    
    # Determine optimal batch size using determine_blocks_per_batch helper
    if simultaneous_blocks is None:
        # Calculate parameters for the helper function
        no_vars = len(var_to_do)
        n_weights = 1  # We're using one weight variable
        n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
        
        # Use helper function to determine optimal batch size
        simultaneous_blocks = determine_blocks_per_batch(
            n_mbs=n_mbs_local,
            no_vars=no_vars,
            n_weights=n_weights,
            n_points_per_block=n_points_per_block
        )
    
    # Process blocks in optimized batches
    num_batches = (n_mbs_local + simultaneous_blocks - 1) // simultaneous_blocks
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * simultaneous_blocks
        batch_end = min(batch_start + simultaneous_blocks, n_mbs_local)
        
        if batch_end <= batch_start:
            continue
        
        # Convert to global meshblock indices
        batch_start_global = mb_start_global + batch_start
        batch_end_global = mb_start_global + batch_end
            
        # Load weights and radial coordinates for the entire batch at once
        weinorm = ad.data(weights, batch_start_global, batch_end_global)
        
        # Transform radial coordinate based on scale
        if rad_scale == 'log':
            var_r = xp.log10(ad.data('r', batch_start_global, batch_end_global))
            xp.nan_to_num(var_r, copy=False, posinf=0.0, neginf=-2.0)
        else:
            var_r = ad.data('r', batch_start_global, batch_end_global)
        
        # Compute weight histogram for this batch
        weight_vals, _ = xp.histogram(var_r, bins=bin_edges, weights=weinorm, density=False)
        
        # Accumulate weights
        weight_sum_r += weight_vals
        
        # Process each variable for the current batch
        for ivar, var in enumerate(var_to_do):
            # Load variable data for this batch
            data = ad.data(var, batch_start_global, batch_end_global)
            
            # Apply log transform if requested
            if var in log_scale:
                data = xp.log10(data)
            
            # Handle NaN or inf values
            xp.nan_to_num(data, copy=False, posinf=0.0, neginf=0.0)
            
            # Calculate histograms for this variable and batch using pre-computed bin edges
            hist_vals, _ = xp.histogram(var_r, bins=bin_edges, weights=data * weinorm, density=False)
            hist_sq_vals, _ = xp.histogram(var_r, bins=bin_edges, weights=data**2 * weinorm, density=False)
            
            # Accumulate results
            hist_r[ivar] += hist_vals
            hist_sq_r[ivar] += hist_sq_vals
        
        # Free memory after each batch if using GPU
        if cupy_enabled:
            maybe_trim_gpu_memory_pool()
    
    # Gather results from all MPI ranks
    if mpi_manager is not None:
        # Convert to numpy for MPI communication
        weight_sum_np = asnumpy(weight_sum_r)
        
        # Sum weight histogram across all ranks
        weight_sum_total = mpi_manager.allreduce(weight_sum_np, op='sum')
        weight_sum_r = xp.asarray(weight_sum_total)
        
        # Aggregate variable histograms
        for ivar in range(len(varl_to_do)):
            hist_np = asnumpy(hist_r[ivar])
            hist_sq_np = asnumpy(hist_sq_r[ivar])
            
            hist_total = mpi_manager.allreduce(hist_np, op='sum')
            hist_sq_total = mpi_manager.allreduce(hist_sq_np, op='sum')
            
            hist_r[ivar] = xp.asarray(hist_total)
            hist_sq_r[ivar] = xp.asarray(hist_sq_total)
    
    # Compute final statistics for each variable
    for ivar, var in enumerate(var_to_do):
        varname = varl_to_do[ivar]
        
        # Auto-detect if this is a flux variable (unless explicitly set)
        is_flux = flx_flag
        if not flx_flag:
            flux_vars = ['mflxr', 'mflxrin', 'mflxrout', 'momflxr', 'momflxrin', 
                        'momflxrout', 'ekflxr', 'ekflxrin', 'ekflxrout', 
                        'eflxtot', 'eflxtotin', 'eflxtotout']
            is_flux = any(flux_var in var for flux_var in flux_vars)
        
        # Filter out bins with no data
        valid_bins = weight_sum_r > 0
        
        # Calculate mean and standard deviation for valid bins
        mean_r = xp.zeros_like(weight_sum_r)
        ms_r = xp.zeros_like(weight_sum_r)
        
        if xp.any(valid_bins):
            mean_r[valid_bins] = hist_r[ivar][valid_bins] / weight_sum_r[valid_bins]
            ms_r[valid_bins] = hist_sq_r[ivar][valid_bins] / weight_sum_r[valid_bins]
            
            # Apply flux correction if needed
            if is_flux:
                mean_r[valid_bins] *= 4.0 * xp.pi * rad_r[valid_bins]**2
                ms_r[valid_bins] *= 4.0 * xp.pi * rad_r[valid_bins]**2
            
            # Calculate standard deviation, with numerical stability safeguards
            variance = xp.maximum(0, ms_r - mean_r**2)
            sigma_r = xp.sqrt(variance)
        else:
            # Set defaults for empty bins
            sigma_r = xp.zeros_like(weight_sum_r)
        
        # Store results
        ad.rad[varname] = {
            'dr': asnumpy(dr_r),
            'r': asnumpy(rad_r),
            'norm': asnumpy(weight_sum_r),
            'profile': asnumpy(mean_r),
            'sigma_r': asnumpy(sigma_r)
        }
    
    return ad.rad
