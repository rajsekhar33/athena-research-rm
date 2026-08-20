"""
Spectral analysis for Athenak simulation data.
"""
import numpy as np
from ..core.base import xp, asnumpy
from ..utils.batch_processing import determine_blocks_per_batch
from scipy.signal.windows import hamming

# Try to import MPI utilities
try:
    from ..backends.mpi_utils import MPIManager
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

def fft_comp(data, nindex=1):
    """
    Compute FFT of a field.
    
    Parameters
    ----------
    data : array-like
        Input data for FFT
    nindex : int, optional
        Power index
        
    Returns
    -------
    array-like
        FFT power spectrum
    """
    nz, ny, nx = data.shape

    ru = xp.fft.fftn(data ** nindex)
    ru = ru / xp.sqrt(nx * ny * nz)

    return xp.abs(ru) ** 2

def get_spectrum(ad, var, strat_flag=False, skip=0.0, nbins=256, log_bin_flag=False, debug=False):
    """
    Compute the energy spectrum from the given variable data using Fourier transforms.
    Parameters
    ----------
    var : str or array-like
        The input variable name or data array to be processed.
    strat_flag : bool, optional
        If True, indicates usage of a restricted vertical domain (with a Hamming window),
        otherwise uses the full domain. Defaults to False.
    skip : float, optional
        The amount of data (in the vertical dimension) to skip at each boundary when
        strat_flag is True. Defaults to 0.0.
    nbins : int, optional
        Number of bins for the histogram that determines the spectrum. Defaults to 256.
    log_bin_flag : bool, optional
        If True, uses a logarithmic scale for the histogram bins; otherwise uses a
        linear scale. Defaults to False.
    debug : bool, optional
        If True, prints diagnostic information. Defaults to False.
    Returns
    -------
    k_ : array-like
        The representative wavenumbers for each bin.
    kbins : array-like
        The bin edges for k.
    nk : array-like
        The count of data points in each wavenumber bin.
    E_spectrum : array-like
        The total spectral power in each bin.
    E_spectral_density : array-like
        The spectral power density (power per count) in each bin.
    E_spectrum_norm : array-like
        The power spectrum normalized by 4πk².
    """
    if(strat_flag==False):
        nz, ny, nx = dims = xp.asarray([ad.Nx3,ad.Nx2,ad.Nx1])
        L = xp.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min)])
    else:
        # set z direction stuff sizes same as y direction stuff
        nz, ny, nx = dims = xp.asarray([int(xp.abs(ad.Nx3*(ad.x3max-ad.x3min-skip*2)/(ad.x3max-ad.x3min))),ad.Nx2,ad.Nx1])
        L = xp.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min-skip*2)])
    kx = xp.fft.fftfreq(nx) * nx / L[0]
    ky = xp.fft.fftfreq(ny) * ny / L[1]
    kz = xp.fft.fftfreq(nz) * nz / L[2]
    kmax = xp.min(dims / L)
    if(log_bin_flag):
        kmin = 0.25*xp.min(1.0 / L)
        kbins=xp.logspace(xp.log10(kmin), xp.log10(kmax), nbins+1)
        k_ = xp.sqrt(kbins[:-1] * kbins[1:])
    else:
        kmin = 0.
        kbins = xp.linspace(kmin, kmax+kmin, nbins)
        k_ = 0.5*(kbins[:-1] + kbins[1:])
    kz3d, ky3d, kx3d = xp.meshgrid(kz, ky, kx, indexing="ij")
    k = xp.sqrt(kx3d ** 2 + ky3d ** 2 + kz3d ** 2)

    if(strat_flag==False):
        real_data = ad.get_refined_data(var)
    else:
        real_data = ad.get_refined_data(var, xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])
        n_hamming=real_data.shape[0]
        hamming_filter=xp.asarray(hamming(n_hamming))/xp.sqrt(xp.sum(hamming(n_hamming))/n_hamming)
        real_data = real_data*hamming_filter[:,xp.newaxis,xp.newaxis]

    if debug:
        # DEBUG: Check input data
        import numpy as np
        print(f"[DEBUG get_spectrum] var={var}, shape={real_data.shape}")
        print(f"[DEBUG get_spectrum] data: min={np.min(real_data):.6e}, max={np.max(real_data):.6e}, mean={np.mean(real_data):.6e}")
        print(f"[DEBUG get_spectrum] data: nonzero={np.count_nonzero(real_data)}, total={real_data.size}")

    Kk=fft_comp(real_data)
    
    if debug:
        # DEBUG: Check FFT result
        import numpy as np
        Kk_np = Kk if isinstance(Kk, np.ndarray) else asnumpy(Kk)
        print(f"[DEBUG get_spectrum] FFT power: min={np.min(Kk_np):.6e}, max={np.max(Kk_np):.6e}, sum={np.sum(Kk_np):.6e}")
    
    '''
    whichbin = xp.digitize(k.flat, kbins)
    ncount = xp.bincount(whichbin)
    E_spectrum = xp.zeros(len(ncount) - 1)
    for n in range(1, len(ncount)):
        E_spectrum[n - 1] = xp.sum(Kk.flat[whichbin == n])
    #'''

    E_spectrum = xp.histogram(k,bins=kbins,weights=Kk,density=False)[0]
    nk = xp.histogram(k,bins=kbins,density=False)[0]
    
    if debug:
        # DEBUG: Check histogram results
        import numpy as np
        E_spectrum_np = E_spectrum if isinstance(E_spectrum, np.ndarray) else asnumpy(E_spectrum)
        nk_np = nk if isinstance(nk, np.ndarray) else asnumpy(nk)
        print(f"[DEBUG get_spectrum] E_spectrum: min={np.min(E_spectrum_np):.6e}, max={np.max(E_spectrum_np):.6e}, sum={np.sum(E_spectrum_np):.6e}")
        print(f"[DEBUG get_spectrum] nk: min={np.min(nk_np)}, max={np.max(nk_np)}, sum={np.sum(nk_np)}, nonzero bins={np.count_nonzero(nk_np)}")
    
    # Suppress divide-by-zero warning (handled by nan_to_num)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        E_spectral_density = xp.nan_to_num(E_spectrum/nk)
    E_spectrum_norm = 4.0*xp.pi*k_*k_*E_spectral_density
    return k_, kbins, nk, E_spectrum, E_spectral_density, E_spectrum_norm

def get_spectrum_mb(ad, var, strat_flag=False, skip=0.0, nbins=256, 
                log_bin_flag=False, weight_name='ones', ndiv=None, debug=False):
    """
    Perform a 3D Fast Fourier Transform (FFT) on the specified variable with slab decomposition 
    and weighted transforms, then compute the power spectrum.
    Parameters
    ----------
    var : str
        The name of the variable to transform.
    strat_flag : bool, optional
        Flag indicating whether to apply a skipping region (True) or to use the entire domain (False).
    skip : float, optional
        The size of the region to skip at both ends of the z-axis, used only if strat_flag is True.
    nbins : int, optional
        Number of bins to use for the radial wavenumber histogram.
    log_bin_flag : bool, optional
        If True, uses logarithmically spaced bins; otherwise uses linear spacing.
    weight_name : str, optional
        The name of the weighting variable. If not 'ones', the power spectrum is multiplied by the 
        transformed weighting field.
    ndiv : int
        Number of slabs or divisions along the z-direction for FFT slab decomposition.
    Returns
    -------
    k_ : array-like
        The central wavenumber values for each bin.
    kbins : array-like
        The edges of the wavenumber bins.
    nk : array-like
        The integer count of Fourier modes in each wavenumber bin.
    E_spectrum : array-like
        The unnormalized power spectrum across k-bins.
    E_spectral_density : array-like
        The power spectrum density, or the power spectrum per mode in each bin.
    E_spectrum_norm : array-like
        The power spectrum normalized by a spherical factor (4πk²).
    Notes
    -----
    1. Processes the domain slab-by-slab (sequentially, on a single process/GPU)
       to bound peak memory usage for large datasets.
    2. If strat_flag is True, a Hamming window is applied in the z-direction to reduce edge effects.
    3. The final normalization (1/√(nx*ny*nz)) is performed after all transforms are computed.
    """
    # Clear GPU memory before starting
    if hasattr(xp, 'cuda') and hasattr(xp, 'get_default_memory_pool'):
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
        
    if(strat_flag==False):
        nz, ny, nx = dims = np.array([ad.Nx3,ad.Nx2,ad.Nx1])
        L = np.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min)])
    else:
        # set z direction stuff sizes same as y direction stuff
        nz, ny, nx = dims = np.asarray([int(np.abs(ad.Nx3*(ad.x3max-ad.x3min-skip*2)/(ad.x3max-ad.x3min))),ad.Nx2,ad.Nx1])
        L = np.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min-skip*2)])
    
    kx = np.fft.fftfreq(nx) * nx / L[0]
    ky = np.fft.fftfreq(ny) * ny / L[1]
    kz = np.fft.fftfreq(nz) * nz / L[2]
    kmax = np.min(dims / L)

    if log_bin_flag:
        kmin = 0.25 * np.min(1.0 / L)
        kbins = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    else:
        kmin = 0.0
        kbins = np.linspace(kmin, kmax + kmin, nbins)

    kz3d, ky3d, kx3d = np.meshgrid(kz, ky, kx, indexing="ij")
    k = np.sqrt(kx3d ** 2 + ky3d ** 2 + kz3d ** 2)
    kbins = xp.array(kbins)
    
    if(strat_flag==False):
        real_data = ad.get_refined_data_mb(var)
        if(weight_name != 'ones'):
            weight = ad.get_refined_data_mb(var=weight_name)
    else:
        real_data = ad.get_refined_data_mb(var, xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])
        n_hamming=real_data.shape[0]
        hamming_filter=np.asarray(hamming(n_hamming))/np.sqrt(np.sum(hamming(n_hamming))/n_hamming)
        real_data = real_data*hamming_filter[:,np.newaxis,np.newaxis]
        if(weight_name != 'ones'):
            weight = ad.get_refined_data_mb(var=weight_name, xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])
    
    if debug:
        print(f"[DEBUG get_spectrum_mb] var={var}, shape={real_data.shape}")
        print(f"[DEBUG get_spectrum_mb] min={np.min(real_data):.6e}, max={np.max(real_data):.6e}, mean={np.mean(real_data):.6e}")
        print(f"[DEBUG get_spectrum_mb] nonzero={np.count_nonzero(real_data)}/{real_data.size}")
    
    if(ndiv is None):
        ntot_cells = ad.Nx1 * ad.Nx2 * ad.Nx3
        if(ntot_cells > 512**3):
            ndiv = int(max( 4, ntot_cells // 512**3))
        else:
            ndiv = 4 # default value
    ndiv_ = ndiv
    if(nz % ndiv != 0):
        ndiv_ = ndiv+1
    local_nz = nz // ndiv
    
    spectrum_mb = xp.zeros(nbins-1)
    spectrum_bins_mb = xp.zeros(nbins-1)
    
    # Step 1: 2D FFTs along x and y for each z-slab
    partial_results = []
    weight_partial_results = []
    for slab_id in range(ndiv_):
        start_z = slab_id * local_nz
        end_z = min(start_z + local_nz, nz)
        slab = xp.array(real_data[start_z:end_z])
        # Perform 2D FFTs for entire slab at once
        local_fft_2d = xp.fft.fft2(slab, axes=(1,2))
        partial_results.append(asnumpy(local_fft_2d))
        if(weight_name != 'ones'):
            weight_slab = xp.array(weight[start_z:end_z])
            local_fft_2d_weight = xp.fft.fft2(weight_slab, axes=(1,2))
            weight_partial_results.append(asnumpy(local_fft_2d_weight))
    
    # Step 2: Combine results
    data_combined =np.concatenate(partial_results, axis=0)
    if(weight_name != 'ones'):
        weight_combined = np.concatenate(weight_partial_results, axis=0)
    
    # Step 3: Transpose for z-direction FFT
    data_transpose = np.transpose(data_combined, (1, 2, 0))  # ny, nx, nz
    if(weight_name != 'ones'):
        weight_transpose = np.transpose(weight_combined, (1, 2, 0))  # ny, nx, nz
    
    k_transpose = np.transpose(k, (1, 2, 0)) # ny, nx, nz

    # Step 4 : FFT along the z-direction
    local_ny = ny // ndiv

    for slab_id in range(ndiv_):
        start_y = slab_id * local_ny
        end_y = min(start_y + local_ny, ny)  # Ensure we don't exceed ny
        slab = xp.array(data_transpose[start_y:end_y])
        k_slab = xp.array(k_transpose[start_y:end_y])
        # Perform 1D FFTs for entire slab at once
        ru = xp.fft.fft(slab, axis=2)
        
        ru = ru / xp.sqrt(nx * ny * nz)
        Kk = xp.abs(ru) ** 2
        if(weight_name != 'ones'):
            weight_slab = xp.array(weight_transpose[start_y:end_y])
            ru_weight = xp.fft.fft(weight_slab, axis=2)
            ru_weight = ru_weight / xp.sqrt(nx * ny * nz)
            Kk_weight = xp.abs(ru_weight) ** 2
            Kk = Kk * Kk_weight
        
        # Now calculate the spectrum
        spectrum = xp.histogram(k_slab,bins=kbins,weights=Kk, density=False)
        spectrum_bins = xp.histogram(k_slab,bins=kbins, density=False)
        if log_bin_flag:
            k_ = xp.sqrt(spectrum[1][1:]*spectrum[1][:-1])
        else:
            k_ = (spectrum[1][1:]+spectrum[1][:-1])*0.5
        spectrum_mb += spectrum[0]
        spectrum_bins_mb += spectrum_bins[0]
        
    E_spectrum = spectrum_mb
    # Suppress divide-by-zero warning (handled by nan_to_num below)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        E_spectral_density = spectrum_mb/spectrum_bins_mb
    E_spectral_density = xp.nan_to_num(E_spectral_density)
    E_spectrum_norm = 4.0*xp.pi*k_*k_*E_spectral_density
    # Set dtype of nk to int
    nk = xp.array(spectrum_bins_mb, dtype=int)

    # Free GPU memory (only if using CuPy)
    if hasattr(xp, '__name__') and xp.__name__ == 'cupy':
        if hasattr(xp, 'get_default_memory_pool'):
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
    # Return results
    return k_, kbins, nk, E_spectrum, E_spectral_density, E_spectrum_norm

def get_spectrum_mb_mpi(ad, var, mpi_manager, strat_flag=False, skip=0.0, nbins=256, 
                        log_bin_flag=False, debug=False, force_slab_loading=False):
    """
    Compute power spectrum using MPI with slab decomposition (similar to get_spectrum_mb).
    
    This implementation divides the domain along z-axis among MPI ranks, performs
    2D FFTs (x,y) on each rank, gathers and redistributes data, performs 1D FFT (z),
    then bins the power spectrum.
    
    Parameters
    ----------
    ad : AthenaData
        Athena data object
    var : str
        Variable name to compute spectrum for
    mpi_manager : MPIManager
        MPI manager for communication
    strat_flag : bool, optional
        Whether to use stratified domain with skip
    skip : float, optional
        Skip distance for stratified case
    nbins : int, optional
        Number of spectral bins
    log_bin_flag : bool, optional
        Whether to use logarithmic binning
    debug : bool, optional
        Enable debug output
        
    Returns
    -------
    k_ : array
        Wavenumber bin centers
    kbins : array
        Wavenumber bin edges
    nk : array
        Mode counts per bin
    E_spectrum : array
        Total power spectrum
    E_spectral_density : array
        Power spectral density
    E_spectrum_norm : array
        Normalized power spectrum
    """
    rank = mpi_manager.rank
    size = mpi_manager.size
    comm = mpi_manager.comm
    
    # Get global domain parameters
    if strat_flag:
        nz_global = int(np.abs(ad.Nx3*(ad.x3max-ad.x3min-skip*2)/(ad.x3max-ad.x3min)))
        ny_global, nx_global = ad.Nx2, ad.Nx1
        L = np.array([(ad.x1max-ad.x1min), (ad.x2max-ad.x2min), (ad.x3max-ad.x3min-skip*2)])
    else:
        nz_global, ny_global, nx_global = ad.Nx3, ad.Nx2, ad.Nx1
        L = np.array([(ad.x1max-ad.x1min), (ad.x2max-ad.x2min), (ad.x3max-ad.x3min)])
    
    dims = np.array([nz_global, ny_global, nx_global])
    
    # Setup k-space (same as single-node version)
    kx = np.fft.fftfreq(nx_global) * nx_global / L[0]
    ky = np.fft.fftfreq(ny_global) * ny_global / L[1]
    kz = np.fft.fftfreq(nz_global) * nz_global / L[2]
    kmax = np.min(dims / L)
    
    if log_bin_flag:
        kmin = 0.25 * np.min(1.0 / L)
        kbins = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
        k_ = np.sqrt(kbins[:-1] * kbins[1:])
    else:
        kmin = 0.0
        kbins = np.linspace(kmin, kmax + kmin, nbins)
        k_ = 0.5 * (kbins[:-1] + kbins[1:])
    
    # Divide z-dimension among ranks
    nz_per_rank = nz_global // size
    z_start_idx = rank * nz_per_rank
    if rank == size - 1:
        z_end_idx = nz_global
    else:
        z_end_idx = (rank + 1) * nz_per_rank
    
    local_nz = z_end_idx - z_start_idx
    
    # Calculate physical bounds for this rank's z-slice
    dz = (ad.x3max - ad.x3min) / nz_global
    z_min_rank = ad.x3min + z_start_idx * dz
    z_max_rank = ad.x3min + z_end_idx * dz
    
    if debug:
        print(f"[DEBUG rank {rank}] z_slice=[{z_start_idx}:{z_end_idx}], shape will be ({local_nz}, {ny_global}, {nx_global})")
        print(f"[DEBUG rank {rank}] physical z=[{z_min_rank:.4f}, {z_max_rank:.4f}]")
    
    # Strategy: Try loading full data first (much faster if it fits in memory)
    # Fall back to slab loading if OOM or if forced by user
    use_slab_loading = force_slab_loading
    full_data = None
    
    if not use_slab_loading:
        try:
            if debug and rank == 0:
                print(f"[DEBUG rank {rank}] Attempting to load full domain data...")
            
            # Try loading full domain on all ranks
            if strat_flag:
                z_min_with_skip = ad.x3min + skip
                z_max_with_skip = ad.x3max - skip
                full_data = ad.get_refined_data_mb(var, xyz=[ad.x1min, ad.x1max, ad.x2min, ad.x2max,
                                                              z_min_with_skip, z_max_with_skip])
                n_hamming = full_data.shape[0]
                hamming_filter = np.asarray(hamming(n_hamming)) / np.sqrt(np.sum(hamming(n_hamming))/n_hamming)
                full_data = full_data * hamming_filter[:, np.newaxis, np.newaxis]
            else:
                # use_mpi_gather=True: each rank needs the FULL domain data for manual slicing
                # MPI will gather contributions from all ranks to create complete domain
                full_data = ad.get_refined_data_mb(var, use_mpi_gather=True)  # Load full domain
            
            if debug and rank == 0:
                print(f"[DEBUG rank {rank}] Successfully loaded full data, shape: {full_data.shape}")
                
        except (MemoryError, RuntimeError) as e:
            if debug and rank == 0:
                print(f"[DEBUG rank {rank}] Full data loading failed ({e}), falling back to slab loading")
            use_slab_loading = True
            full_data = None
    
    # Process data based on loading strategy
    if full_data is not None:
        if debug and rank == 0:
            print(f"[DEBUG rank {rank}] Using full data path with manual slicing")
        
        local_data = full_data[z_start_idx:z_end_idx, :, :]
        
    else:
        # No slab-loading fallback is implemented for when full-domain
        # loading fails (MemoryError/RuntimeError) -- fail loudly instead.
        raise RuntimeError(
            f"[rank {rank}] Full-domain data loading failed for '{var}' and the "
            "MPI slab-loading fallback in get_spectrum_mb_mpi is not implemented. "
            "Reduce the domain size per rank or increase available memory."
        )
    
    
    if debug:
        print(f"[DEBUG rank {rank}] Loaded data shape: {local_data.shape}")
        print(f"[DEBUG rank {rank}] Data: min={np.min(local_data):.6e}, max={np.max(local_data):.6e}, mean={np.mean(local_data):.6e}")
        local_sum_sq = np.sum(local_data**2)
        print(f"[DEBUG rank {rank}] Sum of squares: {local_sum_sq:.6e}")
        # Get total across all ranks
        total_sum_sq = comm.allreduce(local_sum_sq, op=mpi_manager.MPI.SUM)
        if rank == 0:
            print(f"[DEBUG rank {rank}] Total sum of squares across all ranks: {total_sum_sq:.6e}")
    
    # Convert to GPU if available
    local_data_device = xp.array(local_data)
    
    # Step 1: Perform 2D FFT along x and y axes
    if debug:
        print(f"[DEBUG rank {rank}] Performing 2D FFT on axes (1,2)...")
    
    fft_2d = xp.fft.fft2(local_data_device, axes=(1, 2))
    
    if debug:
        fft_2d_np = asnumpy(fft_2d)
        print(f"[DEBUG rank {rank}] After 2D FFT: shape={fft_2d.shape}, max_abs={np.max(np.abs(fft_2d_np)):.6e}")
    
    # Convert back to numpy for MPI communication
    fft_2d_np = asnumpy(fft_2d)
    
    # Step 2: Gather all data to all ranks (Allgather)
    if debug:
        print(f"[DEBUG rank {rank}] Gathering 2D FFT results from all ranks...")
    
    # Prepare receive buffer - need to know sizes from each rank
    recvcounts = comm.allgather(local_nz * ny_global * nx_global)
    
    # Flatten for communication
    send_buf = fft_2d_np.flatten()
    
    # Calculate displacements
    displs = np.zeros(size, dtype=int)
    for i in range(1, size):
        displs[i] = displs[i-1] + recvcounts[i-1]
    
    total_size = sum(recvcounts)
    recv_buf = np.empty(total_size, dtype=np.complex128)
    
    comm.Allgatherv(send_buf, [recv_buf, recvcounts, displs, mpi_manager.MPI.DOUBLE_COMPLEX])
    
    if debug:
        print(f"[DEBUG rank {rank}] Gathered data, total size: {total_size}")
    
    # Reshape back to (nz_global, ny_global, nx_global)
    data_combined = recv_buf.reshape((nz_global, ny_global, nx_global))
    
    if debug:
        print(f"[DEBUG rank {rank}] Combined data shape: {data_combined.shape}")
    
    # Step 3: Transpose for z-direction FFT
    # Transpose from (z, y, x) to (y, x, z)
    data_transpose = np.transpose(data_combined, (1, 2, 0))
    
    if debug:
        print(f"[DEBUG rank {rank}] After transpose: {data_transpose.shape}")
    
    # Step 4: Redistribute along y-axis for z-direction FFT
    # Each rank gets a y-slab
    ny_per_rank = ny_global // size
    y_start_idx = rank * ny_per_rank
    if rank == size - 1:
        y_end_idx = ny_global
    else:
        y_end_idx = (rank + 1) * ny_per_rank
    
    local_ny = y_end_idx - y_start_idx
    
    # Extract this rank's y-slab
    local_slab = data_transpose[y_start_idx:y_end_idx, :, :]
    
    if debug:
        print(f"[DEBUG rank {rank}] Local y-slab: [{y_start_idx}:{y_end_idx}], shape={local_slab.shape}")
    
    # Convert to device
    local_slab_device = xp.array(local_slab)
    
    # Step 5: Perform 1D FFT along z-direction (axis=2)
    if debug:
        print(f"[DEBUG rank {rank}] Performing 1D FFT along z (axis=2)...")
    
    fft_3d_local = xp.fft.fft(local_slab_device, axis=2)
    
    # Normalize
    fft_3d_local /= xp.sqrt(nx_global * ny_global * nz_global)
    
    # Compute local power spectrum
    local_Kk = xp.abs(fft_3d_local) ** 2
    
    if debug:
        local_Kk_np = asnumpy(local_Kk)
        local_sum_Kk = np.sum(local_Kk_np)
        print(f"[DEBUG rank {rank}] Local power: min={np.min(local_Kk_np):.6e}, max={np.max(local_Kk_np):.6e}, sum={local_sum_Kk:.6e}")
        # Check Parseval's theorem: sum(|FFT|^2) should equal sum(|data|^2)
        total_sum_Kk = comm.allreduce(local_sum_Kk, op=mpi_manager.MPI.SUM)
        if rank == 0:
            print(f"[DEBUG rank {rank}] Total power across all ranks: {total_sum_Kk:.6e}")
    
    # Step 6: Build k-space grid for local portion
    # After redistribution, each rank has shape (local_ny, nx_global, nz_global)
    # which corresponds to (y, x, z) dimensions in physical space
    # In Fourier space, this corresponds to (ky_local, kx_full, kz_full)
    ky_local = ky[y_start_idx:y_end_idx]
    kx_full = kx
    kz_full = kz
    
    if debug:
        print(f"[DEBUG rank {rank}] Building k-space grid:")
        print(f"[DEBUG rank {rank}]   ky_local: shape={ky_local.shape}, range=[{np.min(ky_local):.3f}, {np.max(ky_local):.3f}]")
        print(f"[DEBUG rank {rank}]   kx_full: shape={kx_full.shape}, range=[{np.min(kx_full):.3f}, {np.max(kx_full):.3f}]")
        print(f"[DEBUG rank {rank}]   kz_full: shape={kz_full.shape}, range=[{np.min(kz_full):.3f}, {np.max(kz_full):.3f}]")
    
    # Create 3D grid matching data shape (local_ny, nx_global, nz_global)
    # Use meshgrid with proper indexing
    ky3d_local, kx3d_local, kz3d_local = np.meshgrid(ky_local, kx_full, kz_full, indexing="ij")
    k_local = np.sqrt(kx3d_local ** 2 + ky3d_local ** 2 + kz3d_local ** 2)
    
    if debug:
        print(f"[DEBUG rank {rank}] k_local: shape={k_local.shape}")
        print(f"[DEBUG rank {rank}] k_local: min={np.min(k_local):.6f}, max={np.max(k_local):.6f}, mean={np.mean(k_local):.6f}")
        print(f"[DEBUG rank {rank}] kbins: min={kbins[0]:.6f}, max={kbins[-1]:.6f}")
        print(f"[DEBUG rank {rank}] k values in range: {np.sum((k_local >= kbins[0]) & (k_local <= kbins[-1]))} / {k_local.size}")
    
    # Convert to device
    k_local_device = xp.array(k_local)
    kbins_device = xp.array(kbins)
    
    # Step 7: Bin local power spectrum
    if debug:
        print(f"[DEBUG rank {rank}] Binning local power spectrum...")
    
    local_E_spectrum = xp.histogram(k_local_device, bins=kbins_device, weights=local_Kk, density=False)[0]
    local_nk = xp.histogram(k_local_device, bins=kbins_device, density=False)[0]
    
    # Convert back to numpy for MPI reduction
    local_E_spectrum_np = asnumpy(local_E_spectrum)
    local_nk_np = asnumpy(local_nk).astype(int)
    
    if debug:
        print(f"[DEBUG rank {rank}] local_E_spectrum: sum={np.sum(local_E_spectrum_np):.6e}, nonzero={np.count_nonzero(local_E_spectrum_np)}")
        print(f"[DEBUG rank {rank}] local_nk: sum={np.sum(local_nk_np)}, nonzero={np.count_nonzero(local_nk_np)}")
    
    # Step 8: Reduce across all ranks
    E_spectrum = mpi_manager.reduce(local_E_spectrum_np, op='sum', root=0)
    nk = mpi_manager.reduce(local_nk_np, op='sum', root=0)
    
    # Broadcast to all ranks
    E_spectrum = mpi_manager.broadcast(E_spectrum if rank == 0 else None, root=0)
    nk = mpi_manager.broadcast(nk if rank == 0 else None, root=0)
    
    if debug and rank == 0:
        print(f"[DEBUG rank {rank}] FINAL E_spectrum: sum={np.sum(E_spectrum):.6e}, nonzero={np.count_nonzero(E_spectrum)}")
        print(f"[DEBUG rank {rank}] FINAL nk: sum={np.sum(nk)}, nonzero={np.count_nonzero(nk)}")
    
    # Compute spectral density
    with np.errstate(divide='ignore', invalid='ignore'):
        E_spectral_density = np.nan_to_num(E_spectrum / nk)
    
    E_spectrum_norm = 4.0 * np.pi * k_ * k_ * E_spectral_density
    
    return k_, kbins, nk.astype(int), E_spectrum, E_spectral_density, E_spectrum_norm


def get_spectrum_helmholtz(ad, var='vel', strat_flag=False, skip=0.0, nbins=200, log_bin_flag=False):
    """
    Performs a 3D FFT with Helmholtz decomposition on the specified vector field. The function computes both the compressible 
    and solenoidal parts of the velocity field, optionally applying a Hamming filter in the z-direction for stratified data.
    Parameters
    ----------
    var : str, optional
        The base variable name to be used for the velocity components (default is 'vel').
    strat_flag : bool, optional
        If True, applies a Hamming filter in the z-direction for stratified data and adjusts dimensions accordingly 
        (default is False).
    skip : float, optional
        The portion of the domain to skip at both ends in the z-direction if strat_flag is True (default is 0.0).
    nbins : int, optional
        The number of bins for the histogram of wavenumbers (default is 200).
    log_bin_flag : bool, optional
        If True, uses logarithmically spaced bins in wavenumber space; otherwise uses linearly spaced bins (default is False).
    Returns
    -------
    k_ : ndarray
        The midpoints of the wavenumber bins.
    kbins : ndarray
        The edges of the wavenumber bins.
    nk : ndarray
        The number of wavenumber modes in each bin.
    E_spectrum_comp : ndarray
        The histogram values of the compressible component energy.
    E_spectrum_sol : ndarray
        The histogram values of the solenoidal component energy.
    E_spectral_density_comp : ndarray
        The compressible spectral density values.
    E_spectral_density_sol : ndarray
        The solenoidal spectral density values.
    
    """
    if(strat_flag==False):
        nz, ny, nx = dims = xp.array([ad.Nx3,ad.Nx2,ad.Nx1])
        L = xp.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min)])
    else:
        # set z direction stuff sizes same as y direction stuff
        nz, ny, nx = dims = xp.asarray([int(xp.abs(ad.Nx3*(ad.x3max-ad.x3min-skip*2)/(ad.x3max-ad.x3min))),ad.Nx2,ad.Nx1])
        L = xp.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min-skip*2)])
    
    kx = xp.fft.fftfreq(nx) * nx / L[0]
    ky = xp.fft.fftfreq(ny) * ny / L[1]
    kz = xp.fft.fftfreq(nz) * nz / L[2]
    kmax = xp.min(dims / L)

    if log_bin_flag:
        kmin = 0.25 * xp.min(1.0 / L)
        kbins = xp.logspace(xp.log10(kmin), xp.log10(kmax), nbins + 1)
        k_ = xp.sqrt(kbins[:-1] * kbins[1:])
    else:
        kmin = 0.0
        kbins = xp.linspace(kmin, kmax + kmin, nbins)
        k_ = 0.5 * (kbins[:-1] + kbins[1:])

    kz3d, ky3d, kx3d = xp.meshgrid(kz, ky, kx, indexing="ij")
    k = xp.sqrt(kx3d ** 2 + ky3d ** 2 + kz3d ** 2)
    kbins = xp.array(kbins)
    varl=[var+'x', var+'y', var+'z']

    if(strat_flag==False):
        real_data_x = ad.get_refined_data(varl[0])
        real_data_y = ad.get_refined_data(varl[1])
        real_data_z = ad.get_refined_data(varl[2])
    else:
        real_data_x = ad.get_refined_data(varl[0], xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])
        n_hamming=real_data_x.shape[0]
        hamming_filter=xp.asarray(hamming(n_hamming))/xp.sqrt(xp.sum(hamming(n_hamming))/n_hamming)
        real_data_x = real_data_x*hamming_filter[:,xp.newaxis,xp.newaxis]
        real_data_y = ad.get_refined_data(varl[1], xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])*hamming_filter[:,xp.newaxis,xp.newaxis]
        real_data_z = ad.get_refined_data(varl[2], xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])*hamming_filter[:,xp.newaxis,xp.newaxis]
    
    data_x_k = xp.fft.fftn(real_data_x)
    data_x_k = data_x_k/xp.sqrt(nx*ny*nz)

    data_y_k = xp.fft.fftn(real_data_y)
    data_y_k = data_y_k/xp.sqrt(nx*ny*nz)

    data_z_k = xp.fft.fftn(real_data_z)
    data_z_k = data_z_k/xp.sqrt(nx*ny*nz)

    kdotv = (kx3d*data_x_k + ky3d*data_y_k + kz3d*data_z_k)

    data_x_k_comp = kdotv*kx3d/(k**2+1e-16)
    data_y_k_comp = kdotv*ky3d/(k**2+1e-16)
    data_z_k_comp = kdotv*kz3d/(k**2+1e-16)
    
    data_x_k_sol = data_x_k - data_x_k_comp
    data_y_k_sol = data_y_k - data_y_k_comp
    data_z_k_sol = data_z_k - data_z_k_comp

    Kk_comp = xp.abs(data_x_k_comp)**2 + xp.abs(data_y_k_comp)**2 + xp.abs(data_z_k_comp)**2
    Kk_sol  = xp.abs(data_x_k_sol)**2  + xp.abs(data_y_k_sol)**2  + xp.abs(data_z_k_sol) **2

    E_spectrum_comp = xp.histogram(k,bins=kbins,weights=Kk_comp,density=False)[0]
    E_spectrum_sol  = xp.histogram(k,bins=kbins,weights=Kk_sol,density=False) [0]
    nk = xp.histogram(k,bins=kbins,density=False)[0]
    E_spectral_density_comp = xp.nan_to_num(E_spectrum_comp/nk)
    E_spectral_density_sol  = xp.nan_to_num(E_spectrum_sol /nk)

    return k_, kbins, nk, E_spectrum_comp, E_spectrum_sol, E_spectral_density_comp, E_spectral_density_sol

def get_spectrum_helmholtz_mb(ad, var, strat_flag=False, skip=0.0, nbins=200, 
                         log_bin_flag=False, ndiv=None):
    """
    Perform a Helmholtz decomposition-based spectral analysis on the specified 3D field data with slab 
    decomposition for optimized memory management.
    Parameters
    ----------
    var : str
        Prefix of the velocity component variables (e.g., 'v' for 'vx', 'vy', 'vz').
    strat_flag : bool, optional
        If False (default), perform the decomposition directly on the full domain without vertical
        skipping. If True, apply a vertical skipping region and use a Hamming filter along the
        z-direction before performing the decomposition.
    skip : float, optional
        Number of grid units to skip at both ends in the z-direction when strat_flag=True.
        Defaults to 0.0.
    nbins : int, optional
        Number of bins for histogram-based spectral analysis. Defaults to 200.
    log_bin_flag : bool, optional
        If True, use logarithmic spacing for bins. Otherwise use linear spacing.
    ndiv : int
        Number of slabs or divisions along the z-direction for FFT slab decomposition.
    Returns
    -------
    k_ : ndarray
        Representative wavenumbers for each bin (either the geometric mean for log bins or midpoints
        for linear bins).
    kbins : ndarray
        The array of bin edges used to group wavenumbers.
    nk : ndarray
        The number of data points (integer counts) in each wavenumber bin.
    E_spectrum_comp : ndarray
        The energy spectrum of the compressive (longitudinal) component in each bin.
    E_spectrum_sol : ndarray
        The energy spectrum of the solenoidal (transverse) component in each bin.
    E_spectral_density_comp : ndarray
        The energy spectral density of the compressive component in each bin (energy per bin count).
    E_spectral_density_sol : ndarray
        The energy spectral density of the solenoidal component in each bin (energy per bin count).
    """
    if(strat_flag==False):
        nz, ny, nx = dims = np.array([ad.Nx3,ad.Nx2,ad.Nx1])
        L = np.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min)])
    else:
        # set z direction stuff sizes same as y direction stuff
        nz, ny, nx = dims = np.asarray([int(np.abs(ad.Nx3*(ad.x3max-ad.x3min-skip*2)/(ad.x3max-ad.x3min))),ad.Nx2,ad.Nx1])
        L = np.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min-skip*2)])
    
    kx = np.fft.fftfreq(nx) * nx / L[0]
    ky = np.fft.fftfreq(ny) * ny / L[1]
    kz = np.fft.fftfreq(nz) * nz / L[2]
    kmax = np.min(dims / L)
    
    if log_bin_flag:
        kmin = 0.25 * np.min(1.0 / L)
        kbins = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
        k_ = np.sqrt(kbins[:-1] * kbins[1:])
    else:
        kmin = 0.0
        kbins = np.linspace(kmin, kmax + kmin, nbins)
        k_ = 0.5 * (kbins[:-1] + kbins[1:])

    kz3d, ky3d, kx3d = np.meshgrid(kz, ky, kx, indexing="ij")
    k = np.sqrt(kx3d ** 2 + ky3d ** 2 + kz3d ** 2)
    kbins = xp.array(kbins)
    varl=[var+'x', var+'y', var+'z']

    if(strat_flag==False):
        real_data_x = ad.get_refined_data_mb(varl[0])
        real_data_y = ad.get_refined_data_mb(varl[1])
        real_data_z = ad.get_refined_data_mb(varl[2])
    else:
        real_data_x = ad.get_refined_data_mb(varl[0], xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])
        n_hamming=real_data_x.shape[0]
        hamming_filter=np.asarray(hamming(n_hamming))/np.sqrt(np.sum(hamming(n_hamming))/n_hamming)
        real_data_x = real_data_x*hamming_filter[:,np.newaxis,np.newaxis]
        real_data_y = ad.get_refined_data_mb(varl[1], xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])*hamming_filter[:,np.newaxis,np.newaxis]
        real_data_z = ad.get_refined_data_mb(varl[2], xyz=[ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min+skip,ad.x3max-skip])*hamming_filter[:,np.newaxis,np.newaxis]
    if(ndiv is None):
        ntot_cells = ad.Nx1 * ad.Nx2 * ad.Nx3
        if(ntot_cells > 512**3):
            ndiv = int(max( 4, ntot_cells * 3 // 512**3))
        else:
            ndiv = 4 # default value
    ndiv_ = ndiv
    if(nz % ndiv != 0):
        ndiv_ = ndiv+1
    local_nz = nz // ndiv
    
    comp_spectrum_mb = xp.zeros(nbins-1)
    sol_spectrum_mb = xp.zeros(nbins-1)
    spectrum_bins_mb = xp.zeros(nbins-1)
    
    # Step 1: 2D FFTs along x and y for each z-slab
    partial_results_x = []
    partial_results_y = []
    partial_results_z = []

    for slab_id in range(ndiv_):
        start_z = slab_id * local_nz
        end_z = min(start_z + local_nz, nz)
        
        # Extract slabs - keeping original axis order [z, y, x]
        slab_x = xp.array(real_data_x[start_z:end_z, :, :])  # Shape: (slab_z, y, x)
        slab_y = xp.array(real_data_y[start_z:end_z, :, :])
        slab_z = xp.array(real_data_z[start_z:end_z, :, :])
        
        # Perform 2D FFTs along y and x axes (axes=(1,2))
        local_fft_2d_x = xp.fft.fft2(slab_x, axes=(1,2))
        local_fft_2d_y = xp.fft.fft2(slab_y, axes=(1,2))
        local_fft_2d_z = xp.fft.fft2(slab_z, axes=(1,2))
        
        partial_results_x.append(asnumpy(local_fft_2d_x))
        partial_results_y.append(asnumpy(local_fft_2d_y))
        partial_results_z.append(asnumpy(local_fft_2d_z))

    # Step 2: Combine results along z-axis (axis=0)
    data_x_combined = np.concatenate(partial_results_x, axis=0)
    data_y_combined = np.concatenate(partial_results_y, axis=0)
    data_z_combined = np.concatenate(partial_results_z, axis=0)

    # Verify we preserved all data
    assert data_x_combined.shape == real_data_x.shape, f"Shape mismatch: {data_x_combined.shape} vs {real_data_x.shape}"

    # Step 3: Transpose for z-direction FFT - keeping original physics
    data_x_transpose = np.transpose(data_x_combined, (1, 2, 0))  # (y, x, z)
    data_y_transpose = np.transpose(data_y_combined, (1, 2, 0))
    data_z_transpose = np.transpose(data_z_combined, (1, 2, 0))

    k_transpose = np.transpose(k, (1, 2, 0))
    kx3d_transpose = np.transpose(kx3d, (1, 2, 0))
    ky3d_transpose = np.transpose(ky3d, (1, 2, 0))
    kz3d_transpose = np.transpose(kz3d, (1, 2, 0))

    # Step 4: FFT along z-direction with proper boundary handling
    local_ny = ny // ndiv

    for slab_id in range(ndiv_):
        start_y = slab_id * local_ny
        end_y = min(start_y + local_ny, ny)  # Ensure we don't exceed ny
        
        slab_x = xp.array(data_x_transpose[start_y:end_y, :, :])
        slab_y = xp.array(data_y_transpose[start_y:end_y, :, :])
        slab_z = xp.array(data_z_transpose[start_y:end_y, :, :])
        k_slab = xp.array(k_transpose[start_y:end_y, :, :])
        kx3d_slab = xp.array(kx3d_transpose[start_y:end_y, :, :])
        ky3d_slab = xp.array(ky3d_transpose[start_y:end_y, :, :])
        kz3d_slab = xp.array(kz3d_transpose[start_y:end_y, :, :])
        
        # Perform 1D FFTs along z-direction (axis=2)
        data_x_k_slab = xp.fft.fft(slab_x, axis=2) / xp.sqrt(nx * ny * nz)
        data_y_k_slab = xp.fft.fft(slab_y, axis=2) / xp.sqrt(nx * ny * nz)
        data_z_k_slab = xp.fft.fft(slab_z, axis=2) / xp.sqrt(nx * ny * nz)
        
        # All arrays now have the same shape, proceed with Helmholtz decomposition
        kdotv_slab = (kx3d_slab*data_x_k_slab + ky3d_slab*data_y_k_slab + kz3d_slab*data_z_k_slab)
        
        data_x_k_comp_slab = kdotv_slab*kx3d_slab/(k_slab**2+1e-16)  
        data_y_k_comp_slab = kdotv_slab*ky3d_slab/(k_slab**2+1e-16)
        data_z_k_comp_slab = kdotv_slab*kz3d_slab/(k_slab**2+1e-16)
        
        data_x_k_sol_slab = data_x_k_slab - data_x_k_comp_slab
        data_y_k_sol_slab = data_y_k_slab - data_y_k_comp_slab
        data_z_k_sol_slab = data_z_k_slab - data_z_k_comp_slab
        
        Kk_comp_slab = xp.abs(data_x_k_comp_slab)**2 + xp.abs(data_y_k_comp_slab)**2 + xp.abs(data_z_k_comp_slab)**2
        Kk_sol_slab  = xp.abs(data_x_k_sol_slab)**2  + xp.abs(data_y_k_sol_slab)**2  + xp.abs(data_z_k_sol_slab) **2
        
        spectrum_comp = xp.histogram(k_slab,bins=kbins,weights=Kk_comp_slab,density=False)[0]
        spectrum_sol = xp.histogram(k_slab,bins=kbins,weights=Kk_sol_slab,density=False)[0]
        spectrum_bins = xp.histogram(k_slab,bins=kbins,density=False)[0]
        
        comp_spectrum_mb += spectrum_comp
        sol_spectrum_mb += spectrum_sol
        spectrum_bins_mb += spectrum_bins
        
    E_spectrum_comp = comp_spectrum_mb
    E_spectrum_sol = sol_spectrum_mb
    # Suppress divide-by-zero warning (handled by nan_to_num)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        E_spectral_density_comp = xp.nan_to_num(E_spectrum_comp/spectrum_bins_mb)
        E_spectral_density_sol = xp.nan_to_num(E_spectrum_sol/spectrum_bins_mb)
    # Set dtype of nk to int
    nk = xp.array(spectrum_bins_mb, dtype=int)

    return k_, kbins, nk, E_spectrum_comp, E_spectrum_sol, E_spectral_density_comp, E_spectral_density_sol

def get_spectrum_helmholtz_mb_mpi(ad, var, mpi_manager, strat_flag=False, skip=0.0, nbins=200,
                                   log_bin_flag=False, debug=False, force_slab_loading=False):
    """
    Compute Helmholtz decomposed power spectrum using MPI with slab decomposition.
    
    This implementation divides the domain along z-axis among MPI ranks, performs
    2D FFTs (x,y) on each rank, gathers and redistributes data, performs 1D FFT (z),
    then performs Helmholtz decomposition and bins the spectra.
    
    Parameters
    ----------
    ad : AthenaData
        Athena data object
    var : str
        Variable prefix (e.g., 'vel' for velocity components)
    mpi_manager : MPIManager
        MPI manager for communication
    strat_flag : bool, optional
        Whether to use stratified domain with skip
    skip : float, optional
        Skip distance for stratified case
    nbins : int, optional
        Number of spectral bins
    log_bin_flag : bool, optional
        Whether to use logarithmic binning
    debug : bool, optional
        Enable debug output
        
    Returns
    -------
    k_ : array
        Wavenumber bin centers
    kbins : array
        Wavenumber bin edges
    nk : array
        Mode counts per bin
    E_spectrum_comp : array
        Compressive component spectrum
    E_spectrum_sol : array
        Solenoidal component spectrum
    E_spectral_density_comp : array
        Compressive spectral density
    E_spectral_density_sol : array
        Solenoidal spectral density
    """
    rank = mpi_manager.rank
    size = mpi_manager.size
    comm = mpi_manager.comm
    
    # Get global domain parameters
    if strat_flag:
        nz_global = int(np.abs(ad.Nx3*(ad.x3max-ad.x3min-skip*2)/(ad.x3max-ad.x3min)))
        ny_global, nx_global = ad.Nx2, ad.Nx1
        L = np.array([(ad.x1max-ad.x1min), (ad.x2max-ad.x2min), (ad.x3max-ad.x3min-skip*2)])
    else:
        nz_global, ny_global, nx_global = ad.Nx3, ad.Nx2, ad.Nx1
        L = np.array([(ad.x1max-ad.x1min), (ad.x2max-ad.x2min), (ad.x3max-ad.x3min)])
    
    dims = np.array([nz_global, ny_global, nx_global])
    varl = [var+'x', var+'y', var+'z']
    
    # Setup k-space
    kx = np.fft.fftfreq(nx_global) * nx_global / L[0]
    ky = np.fft.fftfreq(ny_global) * ny_global / L[1]
    kz = np.fft.fftfreq(nz_global) * nz_global / L[2]
    kmax = np.min(dims / L)
    
    if log_bin_flag:
        kmin = 0.25 * np.min(1.0 / L)
        kbins = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
        k_ = np.sqrt(kbins[:-1] * kbins[1:])
    else:
        kmin = 0.0
        kbins = np.linspace(kmin, kmax + kmin, nbins)
        k_ = 0.5 * (kbins[:-1] + kbins[1:])
    
    # Create k-space grids
    kz3d, ky3d, kx3d = np.meshgrid(kz, ky, kx, indexing="ij")
    k = np.sqrt(kx3d ** 2 + ky3d ** 2 + kz3d ** 2)
    
    # Divide z-dimension among ranks
    nz_per_rank = nz_global // size
    z_start_idx = rank * nz_per_rank
    if rank == size - 1:
        z_end_idx = nz_global
    else:
        z_end_idx = (rank + 1) * nz_per_rank
    
    local_nz = z_end_idx - z_start_idx
    
    if debug:
        print(f"[DEBUG rank {rank}] z_slice=[{z_start_idx}:{z_end_idx}], shape=({local_nz},{ny_global},{nx_global})")
    
    # Load full data with MPI gather (faster than slab loading)
    if strat_flag:
        z_min_skip = ad.x3min + skip
        z_max_skip = ad.x3max - skip
        full_data_x = ad.get_refined_data_mb(varl[0], xyz=[ad.x1min, ad.x1max, ad.x2min, ad.x2max,
                                                           z_min_skip, z_max_skip])
        n_hamming = full_data_x.shape[0]
        hamming_filter = np.asarray(hamming(n_hamming)) / np.sqrt(np.sum(hamming(n_hamming))/n_hamming)
        full_data_x = full_data_x * hamming_filter[:, np.newaxis, np.newaxis]
        full_data_y = ad.get_refined_data_mb(varl[1], xyz=[ad.x1min, ad.x1max, ad.x2min, ad.x2max,
                                                           z_min_skip, z_max_skip])
        full_data_y = full_data_y * hamming_filter[:, np.newaxis, np.newaxis]
        full_data_z = ad.get_refined_data_mb(varl[2], xyz=[ad.x1min, ad.x1max, ad.x2min, ad.x2max,
                                                           z_min_skip, z_max_skip])
        full_data_z = full_data_z * hamming_filter[:, np.newaxis, np.newaxis]
    else:
        full_data_x = ad.get_refined_data_mb(varl[0], use_mpi_gather=True)
        full_data_y = ad.get_refined_data_mb(varl[1], use_mpi_gather=True)
        full_data_z = ad.get_refined_data_mb(varl[2], use_mpi_gather=True)
    
    # Each rank slices its portion
    local_data_x = full_data_x[z_start_idx:z_end_idx, :, :]
    local_data_y = full_data_y[z_start_idx:z_end_idx, :, :]
    local_data_z = full_data_z[z_start_idx:z_end_idx, :, :]
    
    # Step 1: Perform 2D FFTs (x,y) locally on each rank
    local_fft_2d_x = xp.fft.fft2(xp.array(local_data_x), axes=(1,2))
    local_fft_2d_y = xp.fft.fft2(xp.array(local_data_y), axes=(1,2))
    local_fft_2d_z = xp.fft.fft2(xp.array(local_data_z), axes=(1,2))
    
    # Convert to numpy for MPI communication
    local_fft_2d_x_np = asnumpy(local_fft_2d_x)
    local_fft_2d_y_np = asnumpy(local_fft_2d_y)
    local_fft_2d_z_np = asnumpy(local_fft_2d_z)
    
    # Step 2: Gather all ranks' data (distributed along z)
    if rank == 0:
        recvbuf_x = np.zeros((nz_global, ny_global, nx_global), dtype=np.complex128)
        recvbuf_y = np.zeros((nz_global, ny_global, nx_global), dtype=np.complex128)
        recvbuf_z = np.zeros((nz_global, ny_global, nx_global), dtype=np.complex128)
    else:
        recvbuf_x = None
        recvbuf_y = None
        recvbuf_z = None
    
    # Create list of receive sizes and displacements
    sendcounts = np.array([local_nz * ny_global * nx_global])
    recvcounts = comm.gather(sendcounts[0], root=0)
    
    if rank == 0:
        displacements = [0]
        for i in range(1, size):
            displacements.append(displacements[-1] + recvcounts[i-1])
    else:
        displacements = None
    
    # Gatherv from all ranks
    comm.Gatherv(local_fft_2d_x_np.ravel(), [recvbuf_x, recvcounts, displacements, mpi_manager.MPI.DOUBLE_COMPLEX], root=0)
    comm.Gatherv(local_fft_2d_y_np.ravel(), [recvbuf_y, recvcounts, displacements, mpi_manager.MPI.DOUBLE_COMPLEX], root=0)
    comm.Gatherv(local_fft_2d_z_np.ravel(), [recvbuf_z, recvcounts, displacements, mpi_manager.MPI.DOUBLE_COMPLEX], root=0)
    
    # Step 3: Rank 0 performs 1D FFT along z and Helmholtz decomposition
    if rank == 0:
        # Transpose for z-FFT
        data_x_transpose = np.transpose(recvbuf_x, (1, 2, 0))  # (y, x, z)
        data_y_transpose = np.transpose(recvbuf_y, (1, 2, 0))
        data_z_transpose = np.transpose(recvbuf_z, (1, 2, 0))
        
        k_transpose = np.transpose(k, (1, 2, 0))
        kx3d_transpose = np.transpose(kx3d, (1, 2, 0))
        ky3d_transpose = np.transpose(ky3d, (1, 2, 0))
        kz3d_transpose = np.transpose(kz3d, (1, 2, 0))
        
        # Perform 1D FFT along z
        data_x_k = xp.fft.fft(xp.array(data_x_transpose), axis=2) / xp.sqrt(nx_global * ny_global * nz_global)
        data_y_k = xp.fft.fft(xp.array(data_y_transpose), axis=2) / xp.sqrt(nx_global * ny_global * nz_global)
        data_z_k = xp.fft.fft(xp.array(data_z_transpose), axis=2) / xp.sqrt(nx_global * ny_global * nz_global)
        
        k_xp = xp.array(k_transpose)
        kx3d_xp = xp.array(kx3d_transpose)
        ky3d_xp = xp.array(ky3d_transpose)
        kz3d_xp = xp.array(kz3d_transpose)
        
        # Helmholtz decomposition
        kdotv = (kx3d_xp*data_x_k + ky3d_xp*data_y_k + kz3d_xp*data_z_k)
        
        data_x_k_comp = kdotv*kx3d_xp/(k_xp**2+1e-16)
        data_y_k_comp = kdotv*ky3d_xp/(k_xp**2+1e-16)
        data_z_k_comp = kdotv*kz3d_xp/(k_xp**2+1e-16)
        
        data_x_k_sol = data_x_k - data_x_k_comp
        data_y_k_sol = data_y_k - data_y_k_comp
        data_z_k_sol = data_z_k - data_z_k_comp
        
        Kk_comp = xp.abs(data_x_k_comp)**2 + xp.abs(data_y_k_comp)**2 + xp.abs(data_z_k_comp)**2
        Kk_sol = xp.abs(data_x_k_sol)**2 + xp.abs(data_y_k_sol)**2 + xp.abs(data_z_k_sol)**2
        
        kbins_xp = xp.array(kbins)
        E_spectrum_comp = xp.histogram(k_xp, bins=kbins_xp, weights=Kk_comp, density=False)[0]
        E_spectrum_sol = xp.histogram(k_xp, bins=kbins_xp, weights=Kk_sol, density=False)[0]
        nk = xp.histogram(k_xp, bins=kbins_xp, density=False)[0]
        
        E_spectral_density_comp = E_spectrum_comp / xp.maximum(nk, 1)
        E_spectral_density_sol = E_spectrum_sol / xp.maximum(nk, 1)
        nk = xp.array(nk, dtype=int)
    else:
        E_spectrum_comp = None
        E_spectrum_sol = None
        E_spectral_density_comp = None
        E_spectral_density_sol = None
        nk = None
    
    # Broadcast results to all ranks
    E_spectrum_comp = comm.bcast(E_spectrum_comp, root=0)
    E_spectrum_sol = comm.bcast(E_spectrum_sol, root=0)
    E_spectral_density_comp = comm.bcast(E_spectral_density_comp, root=0)
    E_spectral_density_sol = comm.bcast(E_spectral_density_sol, root=0)
    nk = comm.bcast(nk, root=0)
    
    return k_, kbins, nk, E_spectrum_comp, E_spectrum_sol, E_spectral_density_comp, E_spectral_density_sol

def set_spectrum(ad, varl, redo=False, auto_select=True, use_mpi=False, verbose=True, **kwargs):
    """
    Calculate and store power spectra for given variables with optional MPI parallelization.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : list
        List of variable names to calculate spectra for
    redo : bool, optional
        Whether to recalculate existing spectra
    auto_select : bool, optional
        Whether to automatically select method based on memory
    use_mpi : bool, optional
        If True, distributes computation across MPI ranks. Defaults to False.
    verbose : bool, optional
        If True, prints diagnostic messages. Defaults to True.
    **kwargs : dict
        Additional arguments for spectrum calculation
        
    Returns
    -------
    dict
        Dictionary of calculated spectra
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
    
    # Initialize spectra dictionary if it doesn't exist
    if not hasattr(ad, 'spectra'):
        ad.spectra = {}
    
    # Clear memory before starting (only if using CuPy)
    if hasattr(xp, '__name__') and xp.__name__ == 'cupy':
        if hasattr(xp, 'get_default_memory_pool'):
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
        
    # Determine whether to use the memory-efficient method
    use_mb_method = True  # Default to memory-efficient method
    
    if auto_select:
        # For smaller datasets or when not using MPI, can use standard method
        if (ad.Nx1 * ad.Nx2 * ad.Nx3) <= 512**3 and (mpi_manager is None or size == 1):
            use_mb_method = False
        if rank == 0 and verbose:
            print(f"Memory-based selection: Using {'memory-efficient' if use_mb_method else 'standard'} method for spectrum calculation")
    
    # Process each variable
    for var in varl:
        if (redo or var not in ad.spectra):
            try:
                if rank == 0 and verbose:
                    print(f"Calculating spectrum for {var}...")
                
                # Use MPI-specific implementation when MPI is enabled
                if mpi_manager is not None:
                    k, kbins, nk, spectrum, E_spectral_density, E_spectrum_norm = get_spectrum_mb_mpi(
                        ad, var, mpi_manager, debug=verbose, **kwargs
                    )
                else:
                    # Choose appropriate method based on memory requirements for single-node
                    if use_mb_method:
                        k, kbins, nk, spectrum, E_spectral_density, E_spectrum_norm = get_spectrum_mb(ad, var=var, debug=verbose, **kwargs)
                    else:
                        k, kbins, nk, spectrum, E_spectral_density, E_spectrum_norm = get_spectrum(ad, var=var, debug=verbose, **kwargs)
                
                # Store results
                ad.spectra[var] = {
                    'k': asnumpy(k),
                    'kbins': asnumpy(kbins),
                    'nk': asnumpy(nk),
                    'spectrum': asnumpy(spectrum),
                    'spect_dens': asnumpy(E_spectral_density),
                    'spect_norm': asnumpy(E_spectrum_norm)
                }
                if rank == 0 and verbose:
                    print(f"Spectrum calculation for {var} successful")
            except Exception as e:
                print(f"Error calculating spectrum for {var}: {e}")
                # Try the other method as a fallback
                try:
                    print(f"Trying alternate method for {var}...")
                    if use_mb_method:
                        k, kbins, nk, spectrum, E_spectral_density, E_spectrum_norm = get_spectrum(ad, var=var, **kwargs)
                    else:
                        k, kbins, nk, spectrum, E_spectral_density, E_spectrum_norm = get_spectrum_mb(ad, var=var, **kwargs)
                    
                    # Store results
                    ad.spectra[var] = {
                        'k': asnumpy(k),
                        'kbins': asnumpy(kbins),
                        'nk': asnumpy(nk),
                        'spectrum': asnumpy(spectrum),
                        'spect_dens': asnumpy(E_spectral_density),
                        'spect_norm': asnumpy(E_spectrum_norm)
                    }
                    print(f"Alternate method for {var} successful")
                except Exception as e2:
                    print(f"Both methods failed for {var}: {e2}")
                    raise RuntimeError(f"Both methods failed for {var}: {e2}") from e2
            
            # Clear GPU memory after processing each variable (only if using CuPy)
            if hasattr(xp, '__name__') and xp.__name__ == 'cupy':
                if hasattr(xp, 'get_default_memory_pool'):
                    xp.get_default_memory_pool().free_all_blocks()
                    xp.get_default_pinned_memory_pool().free_all_blocks()
                    if verbose:
                        print(f"GPU memory cleared after processing {var}")
    
    return ad.spectra

def set_spectrum_helmholtz(ad, var, redo=False, auto_select=True, use_mpi=False, verbose=True, **kwargs):
    """
    Calculate and store Helmholtz-decomposed spectra with adaptive memory management.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    var : str
        Vector field prefix
    redo : bool, optional
        Force recalculation
    auto_select : bool, optional
        Whether to automatically select method based on memory
    use_mpi : bool, optional
        If True, distributes computation across MPI ranks. Defaults to False.
    verbose : bool, optional
        If True, prints diagnostic messages. Defaults to True.
    **kwargs : dict
        Additional arguments
        
    Returns
    -------
    dict
        Updated spectra dictionary
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
    
    # Initialize spectra dictionary if it doesn't exist
    if not hasattr(ad, 'spectra'):
        ad.spectra = {}
    
    # Clear memory before starting
    if hasattr(xp, 'cuda') and hasattr(xp, 'get_default_memory_pool'):
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
        
    # Determine whether to use the memory-efficient method
    use_mb_method = True  # Default to memory-efficient method
    
    if auto_select and (ad.Nx1 * ad.Nx2 * ad.Nx3) <= 512**3 and (mpi_manager is None or size == 1):
        use_mb_method = False
    if rank == 0 and verbose:
        print(f"Memory-based selection: Using {'memory-efficient' if use_mb_method else 'standard'} method for Helmholtz spectrum calculation")
    
    if (redo or var not in ad.spectra.keys()):
        try:
            if rank == 0 and verbose:
                print(f"Calculating Helmholtz spectrum for {var}...")
            
            # Use MPI-specific implementation when MPI is enabled
            if mpi_manager is not None:
                k, kbins, nk, spectrum_comp, spectrum_sol, E_spectral_density_comp, E_spectral_density_sol = get_spectrum_helmholtz_mb_mpi(
                    ad, var, mpi_manager, debug=verbose, **kwargs
                )
            else:
                # Choose appropriate method based on memory requirements for single-node
                if use_mb_method:
                    k, kbins, nk, spectrum_comp, spectrum_sol, E_spectral_density_comp, E_spectral_density_sol = get_spectrum_helmholtz_mb(ad, var=var, **kwargs)
                else:
                    k, kbins, nk, spectrum_comp, spectrum_sol, E_spectral_density_comp, E_spectral_density_sol = get_spectrum_helmholtz(ad, var=var, **kwargs)
            
            # Store results
            ad.spectra[var] = {
                'k': asnumpy(k),
                'kbins': asnumpy(kbins),
                'nk': asnumpy(nk),
                'spectrum_sol': asnumpy(spectrum_sol),
                'spectrum_comp': asnumpy(spectrum_comp),
                'spect_dens_sol': asnumpy(E_spectral_density_sol),
                'spect_dens_comp': asnumpy(E_spectral_density_comp)
            }
            if rank == 0 and verbose:
                print(f"Helmholtz spectrum calculation for {var} successful")
        except Exception as e:
            print(f"Error calculating Helmholtz spectrum for {var}: {e}")
            # Try the other method as a fallback
            try:
                if rank == 0:
                    print(f"Trying alternate method for {var} Helmholtz spectrum...")
                if use_mb_method:
                    k, kbins, nk, spectrum_comp, spectrum_sol, E_spectral_density_comp, E_spectral_density_sol = get_spectrum_helmholtz(ad, var=var, **kwargs)
                else:
                    k, kbins, nk, spectrum_comp, spectrum_sol, E_spectral_density_comp, E_spectral_density_sol = get_spectrum_helmholtz_mb(ad, var=var, **kwargs)
                
                # Store results
                ad.spectra[var] = {
                    'k': asnumpy(k),
                    'kbins': asnumpy(kbins),
                    'nk': asnumpy(nk),
                    'spectrum_sol': asnumpy(spectrum_sol),
                    'spectrum_comp': asnumpy(spectrum_comp),
                    'spect_dens_sol': asnumpy(E_spectral_density_sol),
                    'spect_dens_comp': asnumpy(E_spectral_density_comp)
                }
                print(f"Alternate method for {var} Helmholtz spectrum successful")
            except Exception as e2:
                print(f"Both methods failed for {var} Helmholtz spectrum: {e2}")
                raise RuntimeError(f"Both methods failed for {var} Helmholtz spectrum: {e2}") from e2
        
        # Clear GPU memory after processing
        if hasattr(xp, 'cuda') and hasattr(xp, 'get_default_memory_pool'):
            # Use CuPy's memory pool management functions
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
            print(f"GPU memory cleared after processing {var}")
    
    return ad.spectra
