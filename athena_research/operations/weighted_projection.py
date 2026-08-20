"""
Utilities for creating weighted projections of Athena++ simulation data.

This module provides functions to compute density-weighted, volume-weighted, 
and other weighted projections along any specified Cartesian axis.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from ..operations.histograms import set_dist2d
from scipy.signal import windows


def create_weighted_projection(ad, variable, weight_type, axis='x', smooth_sigma=None, 
                              apply_window=False, window_type='hamming', redo=True, debug=False, use_mpi=False):
    """
    Create weighted projection of variable along specified axis using set_dist2d.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    variable : str
        Name of variable to project
    weight_type : str
        Type of weighting to use (e.g. 'vol', 'mass', 'ones', etc.)
    axis : str
        Axis along which to project ('x', 'y', or 'z')
    smooth_sigma : float or None
        Standard deviation for Gaussian kernel to smooth the result.
        If None, no smoothing is applied.
    apply_window : bool
        Whether to apply a window function along the projection axis
        to reduce edge effects
    window_type : str
        Type of window function to apply ('hamming', 'hanning', 'blackman', 'kaiser')
    redo : bool
        If False and no filters are requested, try to recover existing projection 
        from ad.dist2d before computing. If True, always recompute.
    debug : bool
        If True, print debug information about projection recovery and computation.
        Default is False.
    use_mpi : bool
        If True, use MPI for distributed computation. Default is False since
        projections typically require all data on one node.
        
    Returns
    -------
    numpy.ndarray
        The projected data
        
    Notes
    -----
    This function computes the weighted projection of a variable by:
    1. Creating a projection of (variable * weight_type)
    2. Creating a projection of the weight itself
    3. Dividing the first projection by the second to get weighted average
    
    Several common weight types are:
    - 'vol': Volume weighting (good default for density-like quantities)
    - 'mass': Mass weighting (good for temperature, velocity, etc.)
    - 'ones': Simple summation (no weighting)
    
    Examples
    --------
    # Create volume-weighted density projection along z-axis
    dens_proj = create_weighted_projection(ad, 'dens', 'vol', axis='z')
    
    # Create mass-weighted temperature projection along x-axis with smoothing
    temp_proj = create_weighted_projection(ad, 'temp', 'mass', axis='x', smooth_sigma=1.0)
    """
    # Determine the coordinate variables based on projection axis
    if axis == 'x':
        coords = ['y', 'z']
        Nx, Ny = ad.Nx2, ad.Nx3
        axis_size = ad.Nx1
    elif axis == 'y':
        coords = ['x', 'z']
        Nx, Ny = ad.Nx1, ad.Nx3
        axis_size = ad.Nx2
    elif axis == 'z':
        coords = ['x', 'y']
        Nx, Ny = ad.Nx1, ad.Nx2
        axis_size = ad.Nx3
    else:
        raise ValueError(f"Invalid projection axis: {axis}. Must be 'x', 'y', or 'z'.")
    
    # Check if we can recover existing projection when redo=False and no filters
    if not redo and smooth_sigma is None and not apply_window:
        # Initialize dist2d storage if it doesn't exist
        if not hasattr(ad, 'dist2d'):
            ad.dist2d = {}
        
        # Check if projections already exist with variable and weight type info
        weighted_key = f"{coords[0]}_{coords[1]}_{variable}_{weight_type}_weighted"
        norm_key = f"{coords[0]}_{coords[1]}_{variable}_{weight_type}_norm"
        
        # Debug: print what keys exist and what we're looking for
        if debug:
            print(f"Looking for keys: {weighted_key}, {norm_key}")
            print(f"Available keys in ad.dist2d: {list(ad.dist2d.keys())}")
        
        if weighted_key in ad.dist2d and norm_key in ad.dist2d:
            if debug:
                print(f"Found existing projections for {variable} with {weight_type} weighting")
            # Recovery existing projection
            weighted_data = ad.dist2d[weighted_key]['dat']
            norm_data = ad.dist2d[norm_key]['dat']
            
            # Avoid division by zero
            mask = norm_data > 0
            result = np.zeros_like(weighted_data)
            result[mask] = weighted_data[mask] / norm_data[mask]
            
            return result
        else:
            if debug:
                print(f"No existing projections found, will compute new ones")
    
    # Generate window function if requested
    window_data = None
    if apply_window:
        if window_type == 'hamming':
            window = windows.hamming(axis_size)
        elif window_type == 'hanning':
            window = windows.hann(axis_size)
        elif window_type == 'blackman':
            window = windows.blackman(axis_size)
        elif window_type == 'kaiser':
            window = windows.kaiser(axis_size, beta=14.0)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        
        # Normalize window
        window = window / np.sqrt(np.sum(window) / axis_size)
        window_data = window
    
    # Create the weighted variable function
    weighted_var_name = f"{variable}_{weight_type}_{axis}proj"
    
    # Define the weighted variable function
    def weighted_var(self, mbl=None, mbh=None):
        var_data = self.data(variable, mbl, mbh)
        weight_data = self.data(weight_type, mbl, mbh)
        result = var_data * weight_data
        
        # Apply window function if provided
        if window_data is not None:
            if axis == 'x':
                result = result * window_data[np.newaxis, np.newaxis, :]
            elif axis == 'y':
                result = result * window_data[np.newaxis, :, np.newaxis]
            elif axis == 'z':
                result = result * window_data[:, np.newaxis, np.newaxis]
        
        return result
    
    # Define the weight function with window if needed
    def weight_func(self, mbl=None, mbh=None):
        weight_data = self.data(weight_type, mbl, mbh)
        
        # Apply window function if provided
        if window_data is not None:
            if axis == 'x':
                weight_data = weight_data * window_data[np.newaxis, np.newaxis, :]
            elif axis == 'y':
                weight_data = weight_data * window_data[np.newaxis, :, np.newaxis]
            elif axis == 'z':
                weight_data = weight_data * window_data[:, np.newaxis, np.newaxis]
        
        return weight_data
    
    # Add the custom functions to ad
    ad.add_data_func(weighted_var_name, weighted_var)
    weight_func_name = f"{weight_type}_{axis}_window"
    if apply_window:
        ad.add_data_func(weight_func_name, weight_func)
    
    # Reset dist2d storage if it doesn't exist
    if not hasattr(ad, 'dist2d'):
        ad.dist2d = {}
    
    if debug:
        print(f"[DEBUG weighted_proj] axis={axis}, coords={coords}, Nx={Nx}, Ny={Ny}")
        print(f"[DEBUG weighted_proj] Calling set_dist2d with bins=[{Nx}, {Ny}]")
    
    # First compute: variable * weight
    # Note: set_dist2d already has use_mpi support, so we can pass it through if needed
    set_dist2d(
        ad,
        varl2d=[[coords[0], coords[1]]],
        weights=weighted_var_name,
        bins=[Nx, Ny],
        scales=['lin', 'lin'],
        redo=redo,
        weightnorm=False,
        varsuf=f"_{variable}_{weight_type}_weighted",
        use_mpi=use_mpi,
        debug=debug
    )
    
    # Then compute: weight (for normalization)
    weight_to_use = weight_func_name if apply_window else weight_type
    set_dist2d(
        ad, 
        varl2d=[[coords[0], coords[1]]], 
        weights=weight_to_use,
        bins=[Nx, Ny],
        scales=['lin', 'lin'],
        redo=redo,
        weightnorm=False,
        varsuf=f"_{variable}_{weight_type}_norm",
        use_mpi=use_mpi,
        debug=debug
    )
    
    # Get the weighted data and normalization
    weighted_key = f"{coords[0]}_{coords[1]}_{variable}_{weight_type}_weighted"
    norm_key = f"{coords[0]}_{coords[1]}_{variable}_{weight_type}_norm"
    
    if debug:
        print(f"[DEBUG] Looking for keys: weighted_key='{weighted_key}', norm_key='{norm_key}'")
        print(f"[DEBUG] Available keys in ad.dist2d: {list(ad.dist2d.keys())}")
    
    if weighted_key in ad.dist2d and norm_key in ad.dist2d:
        if debug:
            print(f"[DEBUG] Retrieving data from ad.dist2d...")
        weighted_data = ad.dist2d[weighted_key]['dat'].copy()
        if debug:
            print(f"[DEBUG] weighted_data retrieved: shape={weighted_data.shape}, dtype={weighted_data.dtype}")
        
        norm_data = ad.dist2d[norm_key]['dat'].copy()
        if debug:
            print(f"[DEBUG] norm_data retrieved: shape={norm_data.shape}, dtype={norm_data.dtype}")
            print(f"[DEBUG] Expected output shape: ({Nx}, {Ny}) based on axis={axis}, coords={coords}")
            print(f"[DEBUG] Grid dimensions: Nx1={ad.Nx1}, Nx2={ad.Nx2}, Nx3={ad.Nx3}")
        
        # Check if shapes match before transpose
        if weighted_data.shape != norm_data.shape:
            raise ValueError(f"Shape mismatch: weighted_data.shape={weighted_data.shape}, norm_data.shape={norm_data.shape}")
        
        if debug:
            print(f"[DEBUG] Both arrays have same shape: {weighted_data.shape}")
        
        # histogram2d returns (ny, nx) for bins=[nx, ny], so we need to transpose
        # Expected: histogram2d with bins=[256, 512] returns shape (512, 256)
        # We want: (256, 512) for projection
        if debug:
            print(f"[DEBUG] Checking if transpose needed...")
            print(f"[DEBUG]   Target shape (Ny, Nx) = ({Ny}, {Nx})")
            print(f"[DEBUG]   Target shape (Nx, Ny) = ({Nx}, {Ny})")
            print(f"[DEBUG]   Actual shape = {weighted_data.shape}")
        
        if weighted_data.shape == (Ny, Nx):
            if debug:
                print(f"[DEBUG] Shape is (Ny={Ny}, Nx={Nx}), transposing to (Nx={Nx}, Ny={Ny})")
            weighted_data = weighted_data.T
            norm_data = norm_data.T
            if debug:
                print(f"[DEBUG] After transpose: weighted={weighted_data.shape}, norm={norm_data.shape}")
        elif weighted_data.shape == (Nx, Ny):
            if debug:
                print(f"[DEBUG] Already correct shape (Nx={Nx}, Ny={Ny}), no transpose needed")
        else:
            raise ValueError(f"Unexpected shape: {weighted_data.shape}, expected ({Ny}, {Nx}) or ({Nx}, {Ny})")
        
        # Verify shapes match after transpose
        if weighted_data.shape != norm_data.shape:
            raise ValueError(f"Shape mismatch after transpose: weighted={weighted_data.shape}, norm={norm_data.shape}")
        
        if debug:
            print(f"[DEBUG] After transpose check: both arrays shape={weighted_data.shape}")
        
        # Ensure correct final shape
        if weighted_data.shape != (Nx, Ny):
            raise ValueError(f"Final shape mismatch: got {weighted_data.shape}, expected ({Nx}, {Ny})")
        
        if debug:
            print(f"[DEBUG] Final shape verified: {weighted_data.shape} == ({Nx}, {Ny})")
        
        # Avoid division by zero
        if debug:
            print(f"[DEBUG] Creating result array with np.zeros_like(weighted_data)...")
        result = np.zeros_like(weighted_data)
        if debug:
            print(f"[DEBUG] result.shape={result.shape}, result.dtype={result.dtype}")
        
        if debug:
            print(f"[DEBUG] Creating mask with norm_data > 0...")
        mask = norm_data > 0
        if debug:
            print(f"[DEBUG] mask.shape={mask.shape}, mask.dtype={mask.dtype}, mask.sum()={mask.sum()}")
        
        if debug:
            print(f"[DEBUG] About to perform: result[mask] = weighted_data[mask] / norm_data[mask]")
            print(f"[DEBUG]   result[mask].shape would be: {result[mask].shape}")
            print(f"[DEBUG]   weighted_data[mask].shape: {weighted_data[mask].shape}")
            print(f"[DEBUG]   norm_data[mask].shape: {norm_data[mask].shape}")
        
        try:
            result[mask] = weighted_data[mask] / norm_data[mask]
            if debug:
                print(f"[DEBUG] Division successful! result computed.")
        except Exception as e:
            print(f"[ERROR] Division failed!")
            print(f"[ERROR]   result.shape={result.shape}")
            print(f"[ERROR]   mask.shape={mask.shape}")
            print(f"[ERROR]   weighted_data.shape={weighted_data.shape}")
            print(f"[ERROR]   norm_data.shape={norm_data.shape}")
            print(f"[ERROR]   result[mask].shape={result[mask].shape}")
            print(f"[ERROR]   weighted_data[mask].shape={weighted_data[mask].shape}")
            print(f"[ERROR]   norm_data[mask].shape={norm_data[mask].shape}")
            raise
        
        # Apply smoothing if requested
        if smooth_sigma is not None and smooth_sigma > 0:
            result = gaussian_filter(result, sigma=smooth_sigma)
        
        return result
    else:
        raise KeyError(f"Could not find projections in ad.dist2d. Keys: {list(ad.dist2d.keys())}")


def create_projection_stack(ad, variable, weight_type, axes=None, smooth_sigma=None, redo=True, debug=False):
    """
    Create projections along multiple axes in a single call.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object
    variable : str
        Name of variable to project
    weight_type : str
        Type of weighting to use
    axes : list or None
        List of axes along which to project. If None, projects along all axes.
    smooth_sigma : float or None
        Smoothing parameter passed to create_weighted_projection
    redo : bool
        If False, try to recover existing projections before computing
    debug : bool
        If True, print debug information during projection computation.
        Default is False.
        
    Returns
    -------
    dict
        Dictionary mapping axis names to projection arrays
    """
    if axes is None:
        axes = ['x', 'y', 'z']
    
    projections = {}
    for axis in axes:
        projections[axis] = create_weighted_projection(
            ad, variable, weight_type, axis=axis, smooth_sigma=smooth_sigma, redo=redo, debug=debug
        )
    
    return projections
