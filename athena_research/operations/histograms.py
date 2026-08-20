"""
Histogram and distribution calculation for Athenak simulation data.
"""
from ..core.base import xp, asnumpy, cupy_enabled
from ..core.utils import clear_backend_memory, get_distributed_block_bounds, reduce_array_mpi
from ..utils.batch_processing import determine_blocks_per_batch
from ..core.utils import maybe_trim_gpu_memory_pool
from .basic_operations import calc_filtered_extrema

# Try to import MPI utilities
try:
    from ..backends.mpi_utils import MPIManager
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

def set_dist(ad, varl=['dens', 'temp', 'pres'], varsuf='', scale='log10', 
             bins=1000, weights='vol', redo=False, pdf_range=None, 
             vert_range=None, simultaneous_blocks=None, use_mpi=False, debug=False):
    """
    Generate distributions and corresponding statistics for the specified variables.

    This method computes histograms of the data for each variable in ``varl`` and
    derives various statistical measures such as mean, RMS, standard deviation,
    skewness, and kurtosis. The results are stored in the ``ad.dist`` dictionary
    for subsequent analysis or visualization.

    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : list of str, optional
        List of variable names to process. Defaults to ['dens', 'temp', 'pres'].
    varsuf : str, optional
        A suffix appended to each variable name when storing results in ``ad.dist``.
    scale : {'log10', 'ln', 'linear'}, optional
        Determines the scaling applied to the data prior to histogram computation.
        If 'log10', the data is transformed using log base 10; if 'ln', natural log
        is applied; otherwise no transformation is applied. Defaults to 'log10'.
    bins : int, optional
        Number of bins for the histogram. Defaults to 1000.
    weights : str or array-like, optional
        Specifies the weighting to be used for each data point in the histogram.
        By default, uses the value of 'vol' from ``ad.data`` if weights is a string,
        or accepts an external weights array for custom weighting.
    redo : bool, optional
        If True, forces recalculation of the distribution even if the variable
        already exists in ``ad.dist``. Defaults to False.
    pdf_range : tuple of float, optional
        The lower and upper range of the bins. If None, the range is determined
        automatically from the input data.
    vert_range : tuple or None, optional
        Vertical range (min, max) to filter data by z-coordinate. If None, uses all data.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, distributes computation across MPI ranks. Defaults to False.

    Returns
    -------
    Returns the updated dictionary of distributions for the specified variables.
    
    Notes
    -----
    Statistics calculated for each histogram include:
    - mean
    - RMS
    - standard deviation (sigma)
    - skewness
    - kurtosis

    These quantities can be used to characterize the distribution of the data
    across the specified variables.
    """
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
    for var in varl:
        varname = var + varsuf
        if redo or varname not in ad.dist.keys():
            varl_to_do.append(varname)
            var_to_do.append(var)
    
    # If nothing to do, return early
    if not var_to_do:
        return
    
    # Determine PDF ranges from the same filtered support used by the histogram.
    if pdf_range is None:
        set_pdf_range_bool = True
        pdf_ranges = []

        mins, maxs = calc_filtered_extrema(
            ad,
            var_to_do,
            scales=scale,
            weights=weights,
            vert_range=vert_range,
            simultaneous_blocks=simultaneous_blocks,
            use_mpi=use_mpi,
            debug=debug,
        )

        for var in var_to_do:
            pdf_ranges.append((mins[var], maxs[var]))
    else:
        set_pdf_range_bool = False
        pdf_ranges = [pdf_range] * len(var_to_do)

    
    # Initialize histogram accumulators
    pdf = xp.zeros((len(varl_to_do), bins))
    
    # Pre-compute bin edges for each variable (independent of data, same for all ranks)
    bin_width = []
    binned_data = []
    bin_edges_list = []
    
    for ivar, var in enumerate(var_to_do):
        current_range = pdf_ranges[ivar]
        
        # Create bin edges
        bin_edges = xp.linspace(current_range[0], current_range[1], bins + 1)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        widths = xp.diff(bin_edges)
        
        bin_edges_list.append(bin_edges)
        binned_data.append(bin_centers)
        bin_width.append(widths)
    
    # Convert lists to arrays for indexing
    binned_data = xp.array(binned_data)
    bin_width = xp.array(bin_width)
    
    if debug and rank == 0:
        print(f"[DEBUG set_dist] Computed {len(bin_edges_list)} bin edge arrays")
        for ivar, var in enumerate(var_to_do):
            print(f"[DEBUG set_dist] Var {var}: range={pdf_ranges[ivar]}, bins={bins}")
    
    # Use pre-distributed meshblock data from load time
    if mpi_manager is not None:
        mb_start_global, mb_end_global, n_mbs_local = get_distributed_block_bounds(ad)
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
    simultaneous_blocks = min(simultaneous_blocks, n_mbs_local)
    
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
            
        # Load weights and z coordinates for the entire batch at once
        weinorm = ad.data(weights, batch_start_global, batch_end_global)
        z_data = ad.data('z', batch_start_global, batch_end_global)
        
        # Apply vertical filtering if needed
        bool_z = xp.ones(z_data.shape, dtype=bool)
        if vert_range is not None:
            bool_z = xp.logical_and(z_data > vert_range[0], z_data < vert_range[1])
        
        # Pre-compute flattened weights for this batch (same for all variables)
        weights_combined = weinorm * bool_z
        weights_flat = weights_combined.ravel()
        
        # Skip batch if no valid data points (prevents zero-size array error)
        if cupy_enabled:
            n_valid = int(xp.sum(weights_flat > 0).get())
        else:
            n_valid = int(xp.sum(weights_flat > 0))
        
        if n_valid == 0:
            if debug and rank == 0:
                print(f"[DEBUG set_dist] Skipping empty batch {batch_idx}")
            continue
        
        # Process each variable for the current batch
        for ivar, var in enumerate(var_to_do):
            # Load and transform data for this variable
            if scale == 'log10':
                data = xp.log10(ad.data(var, batch_start_global, batch_end_global))
            elif scale == 'ln':
                data = xp.log(ad.data(var, batch_start_global, batch_end_global))
            else:
                data = ad.data(var, batch_start_global, batch_end_global)
            
            # Handle numerical issues
            xp.nan_to_num(data, copy=False, posinf=0.0, neginf=0.0)
            
            # Flatten data array
            data_flat = data.ravel()
            
            # Verify shapes match
            if data_flat.shape != weights_flat.shape:
                raise ValueError(f"Shape mismatch for variable {var}: data_flat.shape={data_flat.shape}, "
                               f"weights_flat.shape={weights_flat.shape}, data.shape={data.shape}, "
                               f"weinorm.shape={weinorm.shape}, bool_z.shape={bool_z.shape}")
            
            # Compute histogram for this batch using pre-computed bin edges
            hist_vals, _ = xp.histogram(data_flat, bins=bin_edges_list[ivar], weights=weights_flat, density=False)
            
            # Accumulate results
            pdf[ivar] += hist_vals
        
        # Free memory after each batch if using GPU
        if cupy_enabled:
            maybe_trim_gpu_memory_pool()
    
    # Gather results from all MPI ranks
    if mpi_manager is not None:
        pdf = reduce_array_mpi(mpi_manager, pdf, op='sum')
    
    # Compute final statistics for each variable
    for ivar in range(len(varl_to_do)):
        var = var_to_do[ivar]
        varname = varl_to_do[ivar]
        
        # Normalize the PDF
        normalized_pdf = pdf[ivar] / xp.sum(pdf[ivar] * bin_width[ivar])
        
        # Calculate statistical moments
        mean = xp.sum(binned_data[ivar] * normalized_pdf * bin_width[ivar]) / xp.sum(normalized_pdf * bin_width[ivar])
        rms = xp.sqrt(xp.sum(binned_data[ivar]**2 * normalized_pdf * bin_width[ivar]) / xp.sum(normalized_pdf * bin_width[ivar]))
        sigma = xp.sqrt(rms**2 - mean**2)
        skew = xp.sum((binned_data[ivar] - mean)**3 * normalized_pdf * bin_width[ivar]) / xp.sum(normalized_pdf * bin_width[ivar])
        kurtosis = xp.sum((binned_data[ivar] - mean)**4 * normalized_pdf * bin_width[ivar]) / xp.sum(normalized_pdf * bin_width[ivar])
        
        # Store results
        ad.dist[varname] = {
            'dat': asnumpy(normalized_pdf),
            'loc': asnumpy(binned_data[ivar]),
            'mean': asnumpy(mean),
            'rms': asnumpy(rms),
            'sigma': asnumpy(sigma),
            'skew': asnumpy(skew),
            'kurtosis': asnumpy(kurtosis)
        }
    
    return ad.dist

def set_dist2d(ad, varl2d=[['dens','temp'],['dens','pres']], scales=['log10','log10'], 
                weightnorm=True, varsuf='', bins=200, weights='vol', redo=False, 
                pdf_range=None, vert_range=None, simultaneous_blocks=None, use_mpi=False, debug=False):
    """
    Generates 2D probability distributions for pairs of variables from meshblock data.

    This method calculates two-dimensional histograms (PDFs) for each pair of variables 
    specified in varl2d with optimized memory usage. It applies scaling transformations
    and can filter data by vertical range if needed.

    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl2d : list of list of str, optional
        A list of variable pairs (e.g. [["dens", "temp"], ["dens", "pres"]]) 
        for which 2D distributions will be computed.
    scales : list of str, optional
        Scale transformations to apply to each variable ("log10", "ln", "linear").
        Default is ["log10", "log10"].
    weightnorm : bool, optional
        Whether to normalize the PDF by the total weighted sum (True) or by the total 
        bin area only (False). Default is True.
    varsuf : str, optional
        Suffix appended to 2D distribution names in dist2d. Default is "".
    bins : int or tuple/list of ints, optional
        Number of bins along each dimension for the 2D histogram. If an integer, 
        same number of bins is used for both dimensions. If a tuple/list, it should 
        contain (nbins_dim1, nbins_dim2). Default is 200.
    weights : str, optional
        The type of weighting to apply (e.g., "vol" for volume weighting). Default is "vol".
    redo : bool, optional
        If True, forces recalculation even if distributions exist. Default is False.
    pdf_range : tuple or None, optional
        A tuple of two ranges (min, max) for each dimension. If None, determined automatically.
    vert_range : tuple or None, optional
        Vertical range (min, max) to filter data by z-coordinate. If None, uses all data.
    simultaneous_blocks : int, optional
        Number of blocks to process simultaneously. If None, determined automatically.
    use_mpi : bool, optional
        If True, distributes computation across MPI ranks. Defaults to False.
    debug : bool, optional
        If True, print debug information during computation. Defaults to False.

    Returns
    -------
    The updated dist2d dictionary including the 2D distributions for the specified variable pairs.
    """
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
    varname_to_do = []
    for varl in varl2d:
        varname = varl[0] + "_" + varl[1] + varsuf
        if redo or varname not in ad.dist2d.keys():
            varname_to_do.append(varname)
            var_to_do.append(varl)
    
    # If nothing to do, return early
    if not var_to_do:
        return
    
    # Handle bins parameter to ensure proper format
    if isinstance(bins, (list, tuple)):
        nbins_x, nbins_y = bins
    else:
        nbins_x = nbins_y = bins
    
    # Determine PDF ranges from the same filtered support used by the histogram.
    if pdf_range is None:
        set_pdf_range_bool = True
        pdf_ranges = []

        vars_x = list(dict.fromkeys(varl[0] for varl in var_to_do))
        vars_y = list(dict.fromkeys(varl[1] for varl in var_to_do))

        mins_x, maxs_x = calc_filtered_extrema(
            ad,
            vars_x,
            scales=scales[0],
            weights=weights,
            vert_range=vert_range,
            simultaneous_blocks=simultaneous_blocks,
            use_mpi=use_mpi,
            debug=debug,
        )
        mins_y, maxs_y = calc_filtered_extrema(
            ad,
            vars_y,
            scales=scales[1],
            weights=weights,
            vert_range=vert_range,
            simultaneous_blocks=simultaneous_blocks,
            use_mpi=use_mpi,
            debug=debug,
        )

        for varl in var_to_do:
            pdf_ranges.append([
                (mins_x[varl[0]], maxs_x[varl[0]]),
                (mins_y[varl[1]], maxs_y[varl[1]]),
            ])
    else:
        set_pdf_range_bool = False
        pdf_ranges = [pdf_range] * len(var_to_do)


    # Use pre-distributed meshblock data from load time
    if mpi_manager is not None:
        mb_start_global, mb_end_global, n_mbs_local = get_distributed_block_bounds(ad)
    else:
        mb_start_global = 0
        mb_end_global = ad.n_mbs
        n_mbs_local = ad.n_mbs

    # Initialize accumulators for histograms
    # histogram2d with bins=[edges_x, edges_y] returns shape (nbins_x, nbins_y)
    pdf2d = xp.zeros((len(varname_to_do), nbins_x, nbins_y))
    
    # Pre-compute bin edges for each variable pair (independent of data, same for all ranks)
    binned_data1 = []
    binned_data2 = []
    bin_area = []
    
    for ivar2d, varl in enumerate(var_to_do):
        current_ranges = pdf_ranges[ivar2d] if set_pdf_range_bool else pdf_range
        
        # Create bin edges
        bin_edges_1 = xp.linspace(current_ranges[0][0], current_ranges[0][1], nbins_x + 1)
        bin_edges_2 = xp.linspace(current_ranges[1][0], current_ranges[1][1], nbins_y + 1)
        
        binned_data1.append(bin_edges_1)
        binned_data2.append(bin_edges_2)
        
        # Compute bin area
        bin_area.append(xp.outer(xp.diff(bin_edges_1), xp.diff(bin_edges_2)))
    
    # Convert lists to arrays for consistency
    binned_data1 = xp.array(binned_data1)
    binned_data2 = xp.array(binned_data2)
    bin_area = xp.array(bin_area)
    
    # Determine optimal batch size using determine_blocks_per_batch helper
    if simultaneous_blocks is None:
        # Calculate parameters for the helper function
        no_vars = 2 * len(var_to_do)  # Each 2D hist requires 2 variables
        n_weights = 1  # We're using one weight variable
        n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
        
        # Use helper function to determine optimal batch size
        simultaneous_blocks = determine_blocks_per_batch(
            n_mbs=n_mbs_local,
            no_vars=no_vars,
            n_weights=n_weights,
            n_points_per_block=n_points_per_block
        )
    simultaneous_blocks = min(simultaneous_blocks, n_mbs_local)
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
            
        # Load weights and z coordinates for the entire batch at once
        weinorm = ad.data(weights, batch_start_global, batch_end_global)
        z_data = ad.data('z', batch_start_global, batch_end_global)
        
        # Apply vertical filtering if needed
        bool_z = xp.ones(z_data.shape, dtype=bool)
        if vert_range is not None:
            bool_z = xp.logical_and(z_data > vert_range[0], z_data < vert_range[1])
        
        # Process each 2D variable pair for the current batch
        for ivar2d, varl in enumerate(var_to_do):
            # Store the transformed data for both variables
            dats = [None, None]
            
            # Load and transform both variables
            for i, var in enumerate(varl):
                # Apply the appropriate transformation
                if scales[i] == 'log10':
                    dats[i] = xp.log10(ad.data(var, batch_start_global, batch_end_global))
                elif scales[i] == 'ln':
                    dats[i] = xp.log(ad.data(var, batch_start_global, batch_end_global))
                else:
                    dats[i] = ad.data(var, batch_start_global, batch_end_global)
                
                # Handle numerical issues
                xp.nan_to_num(dats[i], copy=False, posinf=0.0, neginf=0.0)
            
            # Prepare weights for histogram
            weight_flat = weinorm.ravel()
            bool_flat = bool_z.ravel()
            
            # Ensure shapes match before multiplication
            if weight_flat.shape != bool_flat.shape:
                raise ValueError(f"Shape mismatch: weinorm.ravel()={weight_flat.shape}, bool_z.ravel()={bool_flat.shape}")
            
            # Check data shapes match weights
            dats0_flat = dats[0].ravel()
            dats1_flat = dats[1].ravel()
            if dats0_flat.shape != weight_flat.shape:
                raise ValueError(f"Shape mismatch: dats[0].ravel()={dats0_flat.shape}, weights={weight_flat.shape}, "
                               f"dats[0].shape={dats[0].shape}, weinorm.shape={weinorm.shape}")
            if dats1_flat.shape != weight_flat.shape:
                raise ValueError(f"Shape mismatch: dats[1].ravel()={dats1_flat.shape}, weights={weight_flat.shape}, "
                               f"dats[1].shape={dats[1].shape}, weinorm.shape={weinorm.shape}")
            
            # Combine weights and boolean mask
            weights_combined = weight_flat * bool_flat
            
            # Skip if no valid data points (prevents zero-size array error in CuPy)
            if cupy_enabled:
                n_valid = int(xp.sum(weights_combined > 0).get())
            else:
                n_valid = int(xp.sum(weights_combined > 0))
            
            if n_valid == 0:
                if debug and rank == 0:
                    print(f"[DEBUG set_dist2d] Skipping empty batch for variable pair {varl}")
                continue
            
            # Compute 2D histogram for this batch and variable pair using pre-computed bin edges
            if debug and rank == 0:
                print(f"[DEBUG set_dist2d] About to call histogram2d:")
                print(f"[DEBUG]   dats0_flat.shape={dats0_flat.shape}")
                print(f"[DEBUG]   dats1_flat.shape={dats1_flat.shape}")
                print(f"[DEBUG]   bins[0] (binned_data1[{ivar2d}]).shape={binned_data1[ivar2d].shape}")
                print(f"[DEBUG]   bins[1] (binned_data2[{ivar2d}]).shape={binned_data2[ivar2d].shape}")
                print(f"[DEBUG]   Expected output shape: ({len(binned_data1[ivar2d])-1}, {len(binned_data2[ivar2d])-1})")
            
            hist_result, _, _ = xp.histogram2d(
                dats0_flat,
                dats1_flat,
                bins=[binned_data1[ivar2d], binned_data2[ivar2d]],
                weights=weights_combined,
                density=False
            )
            
            if debug and rank == 0:
                print(f"[DEBUG]   hist_result.shape={hist_result.shape}")
                print(f"[DEBUG]   This will be accumulated into pdf2d[{ivar2d}] with shape={pdf2d[ivar2d].shape}")
            
            # Accumulate results
            pdf2d[ivar2d] += hist_result
        
        # Free memory after each batch if using GPU
        if cupy_enabled:
            maybe_trim_gpu_memory_pool()
    
    # Gather results from all MPI ranks
    if mpi_manager is not None:
        # Convert to numpy for MPI communication
        pdf2d = reduce_array_mpi(mpi_manager, pdf2d, op='sum')
    
    # Normalize and store the final results
    for ivar2d, varname in enumerate(varname_to_do):
        # Normalize the PDF based on the specified normalization method
        if weightnorm:
            norm = xp.sum(pdf2d[ivar2d] * bin_area[ivar2d])
            pdf2d_norm = pdf2d[ivar2d] / norm
        else:
            norm = xp.sum(bin_area[ivar2d])
            pdf2d_norm = pdf2d[ivar2d] / norm
        
        # Store the results in the class dictionary
        ad.dist2d[varname] = {
            'dat': asnumpy(pdf2d_norm),
            'norm': asnumpy(norm),
            'loc1': asnumpy(binned_data1[ivar2d]),
            'loc2': asnumpy(binned_data2[ivar2d])
        }
    
    return ad.dist2d
