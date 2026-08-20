"""
Gradient, divergence, and curl data operations for Athenak simulation data.
"""
from ..core.base import xp, asnumpy, cupy_enabled
from ..core.utils import axis_index
from ..utils.batch_processing import determine_blocks_per_batch
from ..utils.meshblock_utils import find_neighbour_blocks

# Try to import MPI utilities
try:
    from ..backends.mpi_utils import MPIManager
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

def gradient(ad, f, axis=None, edge_order=1, auto_select=True, handle_boundaries=True, simultaneous_blocks=None, **kwargs):
    """
    Compute the gradient of a field variable, automatically choosing between whole-domain
    and memory-efficient meshblock methods based on memory availability.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    f : str or array-like
        Field variable name or array data
    axis : str or None, optional
        Direction for the gradient ('x', 'y', 'z'). If None, returns all components.
    edge_order : int, optional
        Gradient accuracy at edges. Default is 1.
    auto_select : bool, optional
        If True (default), automatically selects between whole-domain and meshblock methods
        based on memory availability. If False, uses the memory-safe meshblock method.
    handle_boundaries : bool, optional
        If True (default), use neighbor blocks to calculate gradients at block boundaries
        when using the meshblock method.
    simultaneous_blocks : int, optional
        If provided, directly specifies the number of meshblocks to process simultaneously,
        bypassing the auto_select mechanism. Forces use of the meshblock method.
    **kwargs : dict, optional
        Additional arguments passed to ad.data
        
    Returns
    -------
    array-like
        Gradient of the field. If axis is None, returns a list of gradient components
        [grad_x, grad_y, grad_z].
    """
    # If input is not a string variable name, use the original method
    if not isinstance(f, str):
        return _gradient_original(ad, f, axis, edge_order, **kwargs)
    
    # When auto_select is False or simultaneous_blocks is provided, use the memory-safe method
    if not auto_select or simultaneous_blocks is not None:
        # Use the memory-safe meshblock method
        # In MPI mode, use this rank's global meshblock range
        if hasattr(ad, 'has_full_data') and not ad.has_full_data:
            mb_start = ad.local_mb_start  # Global index of first block this rank owns
            mb_end = ad.local_mb_end      # Global index one past last block this rank owns
        else:
            mb_start = 0
            mb_end = ad.n_mbs
        
        if axis is None:
            # Calculate all components
            grad_x = _gradient_mb(ad, f, mb_start, mb_end, axis='x', edge_order=edge_order, 
                                handle_boundaries=handle_boundaries)
            grad_y = _gradient_mb(ad, f, mb_start, mb_end, axis='y', edge_order=edge_order,
                                handle_boundaries=handle_boundaries)
            grad_z = _gradient_mb(ad, f, mb_start, mb_end, axis='z', edge_order=edge_order,
                                handle_boundaries=handle_boundaries)
            return [grad_x, grad_y, grad_z]
        else:
            # Calculate single component
            return _gradient_mb(ad, f, mb_start, mb_end, axis=axis, edge_order=edge_order,
                            handle_boundaries=handle_boundaries)
    
    # simultaneous_blocks is guaranteed None here (the branch above already
    # returned when it wasn't), so always fall through to the memory-based check.
    n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3

    # For gradient, we need space for the input array and output array
    # (and potentially 3x output arrays if axis is None)
    n_output_arrays = 3 if axis is None else 1

    # Call determine_blocks_per_batch to assess how many blocks we can process at once
    blocks_per_batch = determine_blocks_per_batch(
        n_mbs=ad.n_mbs,
        no_vars=1 + n_output_arrays,  # Input array + output array(s)
        n_weights=0,  # No weight arrays needed
        n_points_per_block=n_points_per_block
    )

    # If we can process all blocks at once, use the original method
    # Otherwise, use the memory-efficient meshblock method
    use_mb_method = blocks_per_batch < (ad.n_mbs * 2)
    
    if use_mb_method:
        # Memory-efficient meshblock method
        # In MPI mode, use this rank's global meshblock range
        if hasattr(ad, 'has_full_data') and not ad.has_full_data:
            mb_start = ad.local_mb_start  # Global index of first block this rank owns
            mb_end = ad.local_mb_end      # Global index one past last block this rank owns
        else:
            mb_start = 0
            mb_end = ad.n_mbs
        
        if axis is None:
            # Calculate all components
            grad_x = _gradient_mb(ad, f, mb_start, mb_end, axis='x', edge_order=edge_order, 
                               handle_boundaries=handle_boundaries)
            grad_y = _gradient_mb(ad, f, mb_start, mb_end, axis='y', edge_order=edge_order,
                               handle_boundaries=handle_boundaries)
            grad_z = _gradient_mb(ad, f, mb_start, mb_end, axis='z', edge_order=edge_order,
                               handle_boundaries=handle_boundaries)
            return [grad_x, grad_y, grad_z]
        else:
            # Calculate single component
            return _gradient_mb(ad, f, mb_start, mb_end, axis=axis, edge_order=edge_order,
                            handle_boundaries=handle_boundaries)
    else:
        # Original whole-domain method
        return _gradient_original(ad, f, axis, edge_order, **kwargs)

def divergence(ad, fx, fy, fz, auto_select=True, edge_order=1, handle_boundaries=True, simultaneous_blocks=None, **kwargs):
    """
    Compute the divergence of a vector field, automatically choosing between whole-domain
    and memory-efficient meshblock methods based on memory availability.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    fx, fy, fz : str or array-like
        Vector field components (either variable names or data arrays)
    auto_select : bool, optional
        If True (default), automatically selects between whole-domain and meshblock methods
        based on memory availability. If False, uses the memory-safe meshblock method.
    edge_order : int, optional
        Order of accuracy for edge point calculations. Default is 1.
    handle_boundaries : bool, optional
        If True (default), use neighbor blocks to calculate gradients at block boundaries
        when using the meshblock method.
    simultaneous_blocks : int, optional
        If provided, directly specifies the number of meshblocks to process simultaneously,
        bypassing the auto_select mechanism. Forces use of the meshblock method.
    **kwargs : dict, optional
        Additional arguments passed to ad.data
        
    Returns
    -------
    array-like
        Divergence of the vector field
    """
    # Check if all inputs are strings for variable names
    all_strings = all(isinstance(f, str) for f in [fx, fy, fz])
    
    # If not all inputs are strings, use the original method
    if not all_strings:
        return _divergence_original(ad, fx, fy, fz, **kwargs)
    
    # Check if components follow naming convention (e.g., 'vel' -> 'velx', 'vely', 'velz')
    use_vector_method = False
    vector_prefix = None
    
    if fx.endswith('x') and fy.endswith('y') and fz.endswith('z'):
        # Extract common prefix
        prefixes = [fx[:-1], fy[:-1], fz[:-1]]
        if len(set(prefixes)) == 1:
            # All components have the same prefix
            vector_prefix = prefixes[0]
            use_vector_method = True
    
    # When auto_select is False or simultaneous_blocks is provided, use the memory-safe method
    if not auto_select or simultaneous_blocks is not None:
        # Always use the memory-safe meshblock method
        # In MPI mode, use this rank's global meshblock range
        if hasattr(ad, 'has_full_data') and not ad.has_full_data:
            mb_start = ad.local_mb_start  # Global index of first block this rank owns
            mb_end = ad.local_mb_end      # Global index one past last block this rank owns
        else:
            mb_start = 0
            mb_end = ad.n_mbs
        
        if use_vector_method:
            # Use the optimized vector field divergence implementation
            return _divergence_mb(ad, vector_prefix, mb_start, mb_end, 
                               edge_order=edge_order, handle_boundaries=handle_boundaries)
        else:
            # Different variable names - calculate divergence using gradient_mb components
            grad_x = _gradient_mb(ad, fx, mb_start, mb_end, axis='x', edge_order=edge_order,
                               handle_boundaries=handle_boundaries)
            grad_y = _gradient_mb(ad, fy, mb_start, mb_end, axis='y', edge_order=edge_order,
                               handle_boundaries=handle_boundaries)
            grad_z = _gradient_mb(ad, fz, mb_start, mb_end, axis='z', edge_order=edge_order,
                               handle_boundaries=handle_boundaries)
            return grad_x + grad_y + grad_z
    
    # simultaneous_blocks is guaranteed None here (the branch above already
    # returned when it wasn't), so always fall through to the memory-based check.
    n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3

    # For divergence, we need space for 3 input arrays and at least 1 output array
    # If we're doing gradient-based divergence, we need 3 input and 3 gradient output arrays
    n_vars = 4 if use_vector_method else 6

    # Call determine_blocks_per_batch to assess how many blocks we can process at once
    blocks_per_batch = determine_blocks_per_batch(
        n_mbs=ad.n_mbs,
        no_vars=n_vars,
        n_weights=0,  # No weight arrays needed
        n_points_per_block=n_points_per_block
    )

    # If we can process all blocks at once, use the original method
    # Otherwise, use the memory-efficient meshblock method
    use_mb_method = blocks_per_batch < (ad.n_mbs * 2)
    
    if use_mb_method:
        # Memory-efficient meshblock method
        if use_vector_method:
            # Use the optimized vector field divergence implementation
            # In MPI mode, use this rank's global meshblock range
            if hasattr(ad, 'has_full_data') and not ad.has_full_data:
                mb_start = ad.local_mb_start  # Global index of first block this rank owns
                mb_end = ad.local_mb_end      # Global index one past last block this rank owns
            else:
                mb_start = 0
                mb_end = ad.n_mbs
            return _divergence_mb(ad, vector_prefix, mb_start, mb_end, 
                              edge_order=edge_order, handle_boundaries=handle_boundaries)
        else:
            # Different variable names - calculate divergence using gradient_mb components
            grad_x = _gradient_mb(ad, fx, mb_start, mb_end, axis='x', edge_order=edge_order,
                              handle_boundaries=handle_boundaries)
            grad_y = _gradient_mb(ad, fy, mb_start, mb_end, axis='y', edge_order=edge_order,
                              handle_boundaries=handle_boundaries)
            grad_z = _gradient_mb(ad, fz, mb_start, mb_end, axis='z', edge_order=edge_order,
                              handle_boundaries=handle_boundaries)
            return grad_x + grad_y + grad_z
    else:
        # Original whole-domain method
        return _divergence_original(ad, fx, fy, fz, **kwargs)
    
def curl(ad, fx, fy, fz, auto_select=True, edge_order=1, handle_boundaries=True, simultaneous_blocks=None, **kwargs):
    """
    Compute the curl of a vector field, automatically choosing between whole-domain
    and memory-efficient meshblock methods based on memory availability.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    fx, fy, fz : str or array-like
        Vector field components (either variable names or data arrays)
    auto_select : bool, optional
        If True (default), automatically selects between whole-domain and meshblock methods
        based on memory availability. If False, uses the memory-safe meshblock method.
    edge_order : int, optional
        Order of accuracy for edge point calculations. Default is 1.
    handle_boundaries : bool, optional
        If True (default), use neighbor blocks to calculate gradients at block boundaries
        when using the meshblock method.
    simultaneous_blocks : int, optional
        If provided, directly specifies the number of meshblocks to process simultaneously,
        bypassing the auto_select mechanism. Forces use of the meshblock method.
    **kwargs : dict, optional
        Additional arguments passed to ad.data
        
    Returns
    -------
    list of array-like
        Curl components [curl_x, curl_y, curl_z] as a list of arrays
    """
    # Check if all inputs are strings for variable names
    all_strings = all(isinstance(f, str) for f in [fx, fy, fz])
    
    # If not all inputs are strings, use the original method
    if not all_strings:
        return _curl_original(ad, fx, fy, fz, **kwargs)
    
    # Check if components follow naming convention (e.g., 'vel' -> 'velx', 'vely', 'velz')
    use_vector_method = False
    vector_prefix = None
    
    if fx.endswith('x') and fy.endswith('y') and fz.endswith('z'):
        # Extract common prefix
        prefixes = [fx[:-1], fy[:-1], fz[:-1]]
        if len(set(prefixes)) == 1:
            # All components have the same prefix
            vector_prefix = prefixes[0]
            use_vector_method = True
    
    # When auto_select is False or simultaneous_blocks is provided, always use the memory-safe method
    if not auto_select or simultaneous_blocks is not None:
        # Always use the memory-safe meshblock method using gradient_mb
        return _curl_using_gradient_mb(ad, fx, fy, fz, 
                                     edge_order=edge_order, 
                                     handle_boundaries=handle_boundaries)
    
    # simultaneous_blocks is guaranteed None here (the branch above already
    # returned when it wasn't), so always fall through to the memory-based check.
    n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3

    # For curl, we need space for 3 input arrays and 6 gradient output arrays (6 partial derivatives)
    n_vars = 9  # 3 input + 6 gradient arrays

    # Call determine_blocks_per_batch to assess how many blocks we can process at once
    blocks_per_batch = determine_blocks_per_batch(
        n_mbs=ad.n_mbs,
        no_vars=n_vars,
        n_weights=0,  # No weight arrays needed
        n_points_per_block=n_points_per_block
    )

    # If we can process all blocks at once, use the original method
    # Otherwise, use the memory-efficient meshblock method
    use_mb_method = blocks_per_batch < (ad.n_mbs * 4)
    
    if use_mb_method:
        # Memory-efficient meshblock method
        return _curl_using_gradient_mb(ad, fx, fy, fz, 
                                     edge_order=edge_order, 
                                     handle_boundaries=handle_boundaries)
    else:
        # Original whole-domain method
        return _curl_original(ad, fx, fy, fz, **kwargs)

def _gradient_original(ad, f, axis=None, edge_order=1, handle_boundaries=True, **kwargs):
    """
    Original gradient implementation working on the whole domain at once.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    f : array-like or str
        Field to compute the gradient of
    axis : str or None, optional
        Direction for the gradient ('x', 'y', 'z'). If None, returns all components.
    edge_order : int, optional
        Order of accuracy for edge points. Default is 1.
    handle_boundaries : bool, optional
        Whether to handle boundaries between mesh blocks. Default is True.
    **kwargs : dict, optional
        Additional arguments passed to ad.data
        
    Returns
    -------
    array-like or list of array-like
        Gradient of the field
    """
    # Convert variable name to data array if needed
    if isinstance(f, str):
        f = ad.data(f, **kwargs)
    
    # Convert axis string to numeric index if provided
    if axis is not None:
        if axis == 'x':
            axis_idx = 0
        elif axis == 'y':
            axis_idx = 1
        elif axis == 'z':
            axis_idx = 2
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', 'z', or None.")
    else:
        axis_idx = None
    
    # Get the mesh block cell spacing
    # Note: We keep the order consistent with Athena++ ordering (x,y,z)
    dx_mb = ad.data('dx_mb')
    dy_mb = ad.data('dy_mb')
    dz_mb = ad.data('dz_mb')
    
    # Check if uniform grid is requested (rarely used, but included for completeness)
    if kwargs.get('dtype') == 'uniform':
        # For uniform grid, use the same spacing for all blocks
        # Assuming cell_length returns [dx, dy, dz]
        dx = [dx_mb[0], dy_mb[0], dz_mb[0]]
        
        if axis_idx is None:
            # Calculate gradient in all directions at once
            return xp.asarray(xp.gradient(f, dx[2], dx[1], dx[0], edge_order=edge_order))
        else:
            # Calculate gradient in specified direction only
            # Map from axis index to appropriate spacing
            if axis_idx == 0:  # x direction
                spacing = dx[0]
            elif axis_idx == 1:  # y direction
                spacing = dx[1]
            else:  # z direction
                spacing = dx[2]
            return xp.gradient(f, spacing, axis=axis_idx, edge_order=edge_order)
    
    # For non-uniform grid (the usual case), handle each mesh block separately
    if axis_idx is None:
        # Calculate gradients in all directions
        # Initialize arrays for gradient components
        grad_x = xp.zeros_like(f)
        grad_y = xp.zeros_like(f)
        grad_z = xp.zeros_like(f)
        
        # Process each mesh block
        for i in range(ad.n_mbs):
            if i < f.shape[0]:  # Make sure block index is valid
                # Get spacing for this block
                block_dx = dx_mb[i]
                block_dy = dy_mb[i]
                block_dz = dz_mb[i]
                block_data = f[i]
                
                # Calculate gradient components for this block
                # Note: xp.gradient returns [grad_z, grad_y, grad_x] due to axis order
                grad_components = xp.gradient(block_data, block_dz, block_dy, block_dx, edge_order=edge_order)
                grad_z[i] = grad_components[0]
                grad_y[i] = grad_components[1]
                grad_x[i] = grad_components[2]
        
        # Handle block boundaries if requested
        if handle_boundaries:
            for i in range(ad.n_mbs):
                if i >= f.shape[0]:  # Skip if block index is invalid
                    continue
                
                # X boundaries
                x_plus_neighbors = find_neighbour_blocks(ad, i, 'x+')
                x_minus_neighbors = find_neighbour_blocks(ad, i, 'x-')
                
                # Right boundary (x+)
                if x_plus_neighbors:
                    neighbor_idx = x_plus_neighbors[0]  # Use first neighbor
                    if neighbor_idx < f.shape[0]:  # Ensure valid index
                        grad_x[i, :, :, -1] = (f[neighbor_idx, :, :, 0] - f[i, :, :, -2]) / (2 * dx_mb[i])
                
                # Left boundary (x-)
                if x_minus_neighbors:
                    neighbor_idx = x_minus_neighbors[0]  # Use first neighbor
                    if neighbor_idx < f.shape[0]:  # Ensure valid index
                        grad_x[i, :, :, 0] = (f[i, :, :, 1] - f[neighbor_idx, :, :, -1]) / (2 * dx_mb[i])
                
                # Y boundaries
                y_plus_neighbors = find_neighbour_blocks(ad, i, 'y+')
                y_minus_neighbors = find_neighbour_blocks(ad, i, 'y-')
                
                # Top boundary (y+)
                if y_plus_neighbors:
                    neighbor_idx = y_plus_neighbors[0]  # Use first neighbor
                    if neighbor_idx < f.shape[0]:  # Ensure valid index
                        grad_y[i, :, -1, :] = (f[neighbor_idx, :, 0, :] - f[i, :, -2, :]) / (2 * dy_mb[i])
                
                # Bottom boundary (y-)
                if y_minus_neighbors:
                    neighbor_idx = y_minus_neighbors[0]  # Use first neighbor
                    if neighbor_idx < f.shape[0]:  # Ensure valid index
                        grad_y[i, :, 0, :] = (f[i, :, 1, :] - f[neighbor_idx, :, -1, :]) / (2 * dy_mb[i])
                
                # Z boundaries
                z_plus_neighbors = find_neighbour_blocks(ad, i, 'z+')
                z_minus_neighbors = find_neighbour_blocks(ad, i, 'z-')
                
                # Front boundary (z+)
                if z_plus_neighbors:
                    neighbor_idx = z_plus_neighbors[0]  # Use first neighbor
                    if neighbor_idx < f.shape[0]:  # Ensure valid index
                        grad_z[i, -1, :, :] = (f[neighbor_idx, 0, :, :] - f[i, -2, :, :]) / (2 * dz_mb[i])
                
                # Back boundary (z-)
                if z_minus_neighbors:
                    neighbor_idx = z_minus_neighbors[0]  # Use first neighbor
                    if neighbor_idx < f.shape[0]:  # Ensure valid index
                        grad_z[i, 0, :, :] = (f[i, 1, :, :] - f[neighbor_idx, -1, :, :]) / (2 * dz_mb[i])
        
        # Return the gradient components
        return [grad_x, grad_y, grad_z]
    else:
        # Calculate gradient in specified direction only
        result = xp.zeros_like(f)
        
        # Process each mesh block
        for i in range(ad.n_mbs):
            if i < f.shape[0]:  # Skip if block index is invalid
                # Get block data and appropriate spacing
                block_data = f[i]
                
                if axis_idx == 0:  # x direction
                    spacing = dx_mb[i]
                    result[i] = xp.gradient(block_data, spacing, axis=2, edge_order=edge_order)
                    
                    # Handle block boundaries if requested
                    if handle_boundaries:
                        # Right boundary (x+)
                        x_plus_neighbors = find_neighbour_blocks(ad, i, 'x+')
                        if x_plus_neighbors:
                            neighbor_idx = x_plus_neighbors[0]  # Use first neighbor
                            if neighbor_idx < f.shape[0]:  # Ensure valid index
                                result[i, :, :, -1] = (f[neighbor_idx, :, :, 0] - f[i, :, :, -2]) / (2 * spacing)
                        
                        # Left boundary (x-)
                        x_minus_neighbors = find_neighbour_blocks(ad, i, 'x-')
                        if x_minus_neighbors:
                            neighbor_idx = x_minus_neighbors[0]  # Use first neighbor
                            if neighbor_idx < f.shape[0]:  # Ensure valid index
                                result[i, :, :, 0] = (f[i, :, :, 1] - f[neighbor_idx, :, :, -1]) / (2 * spacing)
                
                elif axis_idx == 1:  # y direction
                    spacing = dy_mb[i]
                    result[i] = xp.gradient(block_data, spacing, axis=1, edge_order=edge_order)
                    
                    # Handle block boundaries if requested
                    if handle_boundaries:
                        # Top boundary (y+)
                        y_plus_neighbors = find_neighbour_blocks(ad, i, 'y+')
                        if y_plus_neighbors:
                            neighbor_idx = y_plus_neighbors[0]  # Use first neighbor
                            if neighbor_idx < f.shape[0]:  # Ensure valid index
                                result[i, :, -1, :] = (f[neighbor_idx, :, 0, :] - f[i, :, -2, :]) / (2 * spacing)
                        
                        # Bottom boundary (y-)
                        y_minus_neighbors = find_neighbour_blocks(ad, i, 'y-')
                        if y_minus_neighbors:
                            neighbor_idx = y_minus_neighbors[0]  # Use first neighbor
                            if neighbor_idx < f.shape[0]:  # Ensure valid index
                                result[i, :, 0, :] = (f[i, :, 1, :] - f[neighbor_idx, :, -1, :]) / (2 * spacing)
                
                else:  # z direction (axis_idx == 2)
                    spacing = dz_mb[i]
                    result[i] = xp.gradient(block_data, spacing, axis=0, edge_order=edge_order)
                    
                    # Handle block boundaries if requested
                    if handle_boundaries:
                        # Front boundary (z+)
                        z_plus_neighbors = find_neighbour_blocks(ad, i, 'z+')
                        if z_plus_neighbors:
                            neighbor_idx = z_plus_neighbors[0]  # Use first neighbor
                            if neighbor_idx < f.shape[0]:  # Ensure valid index
                                result[i, -1, :, :] = (f[neighbor_idx, 0, :, :] - f[i, -2, :, :]) / (2 * spacing)
                        
                        # Back boundary (z-)
                        z_minus_neighbors = find_neighbour_blocks(ad, i, 'z-')
                        if z_minus_neighbors:
                            neighbor_idx = z_minus_neighbors[0]  # Use first neighbor
                            if neighbor_idx < f.shape[0]:  # Ensure valid index
                                result[i, 0, :, :] = (f[i, 1, :, :] - f[neighbor_idx, -1, :, :]) / (2 * spacing)
        
        return result

def _divergence_original(ad, fx, fy, fz, **kwargs):
    """Original divergence implementation working on the whole domain at once."""
    # Convert variable names to data arrays if needed
    if isinstance(fx, str):
        fx = ad.data(fx, **kwargs)
    if isinstance(fy, str):
        fy = ad.data(fy, **kwargs)
    if isinstance(fz, str):
        fz = ad.data(fz, **kwargs)
        
    return _gradient_original(ad, fx, 'x', **kwargs) + _gradient_original(ad, fy, 'y', **kwargs) + _gradient_original(ad, fz, 'z', **kwargs)

def _gradient_mb(ad, var, mbl, mbh, axis=None, edge_order=1, handle_boundaries=True):
    """
    Internal function: Calculate gradients of a field variable across mesh blocks.
    
    Uses xp.gradient for efficiency with optional handling of block boundaries.
    Supports MPI-distributed data by determining block ownership and handling
    boundaries accordingly.
    
    Parameters
    ----------
    var : str
        Field variable name
    mbl, mbh : int
        Range of mesh blocks to process (mbl <= mb < mbh)
    axis : str or None, optional
        Direction for the gradient ('x', 'y', 'z'). If None, returns all components.
    edge_order : int, optional
        Order of accuracy for edge point calculations (passed to xp.gradient)
    handle_boundaries : bool, optional
        If True (default), use neighbor blocks to calculate gradients at block boundaries.
        In MPI mode, only handles boundaries with neighbors on the same rank.
    
    Returns
    -------
    ndarray or list of ndarray
        Gradient field(s) with same shape as input data. If axis is None, returns a list
        of gradient components [grad_x, grad_y, grad_z].
    """
    # If axis is None, compute all components and return as a list
    if axis is None:
        grad_x = _gradient_mb(ad, var, mbl, mbh, 'x', edge_order, handle_boundaries)
        grad_y = _gradient_mb(ad, var, mbl, mbh, 'y', edge_order, handle_boundaries)
        grad_z = _gradient_mb(ad, var, mbl, mbh, 'z', edge_order, handle_boundaries)
        return [grad_x, grad_y, grad_z]
    
    # Check if we're in MPI mode
    is_mpi_distributed = hasattr(ad, 'has_full_data') and not ad.has_full_data
    
    # Get data for the variable
    data = ad.data(var, mbl, mbh)
    
    # Determine which axis we're working with
    if axis == 'x':
        dir_idx = 2
        dx = ad.data('dx_mb')
    elif axis == 'y':
        dir_idx = 1
        dx = ad.data('dy_mb')
    else:  # axis == 'z'
        dir_idx = 0
        dx = ad.data('dz_mb')
    
    # Initialize result array
    result = xp.zeros_like(data)
    
    # Process each block
    for block_offset in range(mbh - mbl):
        block_idx = mbl + block_offset
        block_data = data[block_offset]
        
        # Convert global block index to local index for dx/dy/dz array access
        # dx/dy/dz arrays are indexed by local meshblock indices
        local_idx = block_offset
        
        # Calculate gradient for this block using xp.gradient
        if axis == 'x':
            grad = xp.gradient(block_data, dx[local_idx], axis=2, edge_order=edge_order)
        elif axis == 'y':
            grad = xp.gradient(block_data, dx[local_idx], axis=1, edge_order=edge_order)
        else:  # axis == 'z'
            grad = xp.gradient(block_data, dx[local_idx], axis=0, edge_order=edge_order)
        
        # Handle boundary points if requested
        if handle_boundaries:
            if axis == 'x':
                # Right boundary (x+)
                plus_neighbors = find_neighbour_blocks(ad, block_idx, 'x+')
                if plus_neighbors:
                    # Use first neighbor if multiple exist at different refinement levels
                    neighbor_idx = plus_neighbors[0]
                    # Check if this neighbor is within our processing range
                    if mbl <= neighbor_idx < mbh:
                        neighbor_offset = neighbor_idx - mbl
                        neighbor_data = data[neighbor_offset]
                        # Use central difference with neighbor's first point
                        grad[:, :, -1] = (neighbor_data[:, :, 0] - block_data[:, :, -2]) / (2 * dx[local_idx])
                    elif ad.owns_meshblock(neighbor_idx):
                        # Need to fetch data for neighbor outside our range but owned by this rank
                        neighbor_data = ad.data(var, neighbor_idx, neighbor_idx+1)
                        if len(neighbor_data) > 0:
                            grad[:, :, -1] = (neighbor_data[0][:, :, 0] - block_data[:, :, -2]) / (2 * dx[local_idx])
                
                # Left boundary (x-)
                minus_neighbors = find_neighbour_blocks(ad, block_idx, 'x-')
                if minus_neighbors:
                    # Use first neighbor if multiple exist at different refinement levels
                    neighbor_idx = minus_neighbors[0]
                    # Check if this neighbor is within our processing range
                    if mbl <= neighbor_idx < mbh:
                        neighbor_offset = neighbor_idx - mbl
                        neighbor_data = data[neighbor_offset]
                        # Use central difference with neighbor's last point
                        grad[:, :, 0] = (block_data[:, :, 1] - neighbor_data[:, :, -1]) / (2 * dx[local_idx])
                    elif ad.owns_meshblock(neighbor_idx):
                        # Need to fetch data for neighbor outside our range but owned by this rank
                        neighbor_data = ad.data(var, neighbor_idx, neighbor_idx+1)
                        if len(neighbor_data) > 0:
                            grad[:, :, 0] = (block_data[:, :, 1] - neighbor_data[0][:, :, -1]) / (2 * dx[local_idx])
            
            elif axis == 'y':
                # Top boundary (y+)
                plus_neighbors = find_neighbour_blocks(ad, block_idx, 'y+')
                if plus_neighbors:
                    neighbor_idx = plus_neighbors[0]
                    if mbl <= neighbor_idx < mbh:
                        neighbor_offset = neighbor_idx - mbl
                        neighbor_data = data[neighbor_offset]
                        grad[:, -1, :] = (neighbor_data[:, 0, :] - block_data[:, -2, :]) / (2 * dx[local_idx])
                    elif ad.owns_meshblock(neighbor_idx):
                        neighbor_data = ad.data(var, neighbor_idx, neighbor_idx+1)
                        if len(neighbor_data) > 0:
                            grad[:, -1, :] = (neighbor_data[0][:, 0, :] - block_data[:, -2, :]) / (2 * dx[local_idx])
                
                # Bottom boundary (y-)
                minus_neighbors = find_neighbour_blocks(ad, block_idx, 'y-')
                if minus_neighbors:
                    neighbor_idx = minus_neighbors[0]
                    if mbl <= neighbor_idx < mbh:
                        neighbor_offset = neighbor_idx - mbl
                        neighbor_data = data[neighbor_offset]
                        grad[:, 0, :] = (block_data[:, 1, :] - neighbor_data[:, -1, :]) / (2 * dx[local_idx])
                    elif ad.owns_meshblock(neighbor_idx):
                        neighbor_data = ad.data(var, neighbor_idx, neighbor_idx+1)
                        if len(neighbor_data) > 0:
                            grad[:, 0, :] = (block_data[:, 1, :] - neighbor_data[0][:, -1, :]) / (2 * dx[local_idx])
            
            else:  # axis == 'z'
                # Front boundary (z+)
                plus_neighbors = find_neighbour_blocks(ad, block_idx, 'z+')
                if plus_neighbors:
                    neighbor_idx = plus_neighbors[0]
                    if mbl <= neighbor_idx < mbh:
                        neighbor_offset = neighbor_idx - mbl
                        neighbor_data = data[neighbor_offset]
                        grad[-1, :, :] = (neighbor_data[0, :, :] - block_data[-2, :, :]) / (2 * dx[local_idx])
                    elif ad.owns_meshblock(neighbor_idx):
                        neighbor_data = ad.data(var, neighbor_idx, neighbor_idx+1)
                        if len(neighbor_data) > 0:
                            grad[-1, :, :] = (neighbor_data[0][0, :, :] - block_data[-2, :, :]) / (2 * dx[local_idx])
                
                # Back boundary (z-)
                minus_neighbors = find_neighbour_blocks(ad, block_idx, 'z-')
                if minus_neighbors:
                    neighbor_idx = minus_neighbors[0]
                    if mbl <= neighbor_idx < mbh:
                        neighbor_offset = neighbor_idx - mbl
                        neighbor_data = data[neighbor_offset]
                        grad[0, :, :] = (block_data[1, :, :] - neighbor_data[-1, :, :]) / (2 * dx[local_idx])
                    elif ad.owns_meshblock(neighbor_idx):
                        neighbor_data = ad.data(var, neighbor_idx, neighbor_idx+1)
                        if len(neighbor_data) > 0:
                            grad[0, :, :] = (block_data[1, :, :] - neighbor_data[0][-1, :, :]) / (2 * dx[local_idx])
        
        # Store gradient for this block
        result[block_offset] = grad
    
    return result
    
def _divergence_mb(ad, var, mbl, mbh, edge_order=1, handle_boundaries=True):
    """
    Internal function: Calculate divergence of a vector field across mesh blocks.
    
    Uses _gradient_mb to calculate the individual partial derivatives.
    """
    # Construct the vector component variable names
    fx = var + 'x'
    fy = var + 'y'
    fz = var + 'z'
    
    # Calculate gradients of each component along corresponding direction
    grad_x = _gradient_mb(ad, fx, mbl, mbh, axis='x', edge_order=edge_order, handle_boundaries=handle_boundaries)
    grad_y = _gradient_mb(ad, fy, mbl, mbh, axis='y', edge_order=edge_order, handle_boundaries=handle_boundaries)
    grad_z = _gradient_mb(ad, fz, mbl, mbh, axis='z', edge_order=edge_order, handle_boundaries=handle_boundaries)
    
    # Sum the components to get divergence
    return grad_x + grad_y + grad_z

def _curl_original(ad, fx, fy, fz, **kwargs):
    """Original curl implementation working on the whole domain at once."""
    # Convert variable names to data arrays if needed
    if isinstance(fx, str):
        fx = ad.data(fx, **kwargs)
    if isinstance(fy, str):
        fy = ad.data(fy, **kwargs)
    if isinstance(fz, str):
        fz = ad.data(fz, **kwargs)
    
    # Calculate derivatives
    dfz_dy = _gradient_original(ad, fz, 'y', **kwargs)
    dfy_dz = _gradient_original(ad, fy, 'z', **kwargs)
    
    dfx_dz = _gradient_original(ad, fx, 'z', **kwargs)
    dfz_dx = _gradient_original(ad, fz, 'x', **kwargs)
    
    dfy_dx = _gradient_original(ad, fy, 'x', **kwargs)
    dfx_dy = _gradient_original(ad, fx, 'y', **kwargs)
    
    # Calculate curl components
    curl_x = dfz_dy - dfy_dz
    curl_y = dfx_dz - dfz_dx
    curl_z = dfy_dx - dfx_dy
    
    return [curl_x, curl_y, curl_z]

def _curl_using_gradient_mb(ad, fx, fy, fz, edge_order=1, handle_boundaries=True):
    """Calculate curl using gradient_mb for memory efficiency."""
    # In MPI mode, use this rank's global meshblock range
    if hasattr(ad, 'has_full_data') and not ad.has_full_data:
        mb_start = ad.local_mb_start  # Global index of first block this rank owns
        mb_end = ad.local_mb_end      # Global index one past last block this rank owns
    else:
        mb_start = 0
        mb_end = ad.n_mbs
    
    # Calculate derivatives
    dfz_dy = _gradient_mb(ad, fz, mb_start, mb_end, axis='y', edge_order=edge_order, handle_boundaries=handle_boundaries)
    dfy_dz = _gradient_mb(ad, fy, mb_start, mb_end, axis='z', edge_order=edge_order, handle_boundaries=handle_boundaries)
    
    dfx_dz = _gradient_mb(ad, fx, mb_start, mb_end, axis='z', edge_order=edge_order, handle_boundaries=handle_boundaries)
    dfz_dx = _gradient_mb(ad, fz, mb_start, mb_end, axis='x', edge_order=edge_order, handle_boundaries=handle_boundaries)
    
    dfy_dx = _gradient_mb(ad, fy, mb_start, mb_end, axis='x', edge_order=edge_order, handle_boundaries=handle_boundaries)
    dfx_dy = _gradient_mb(ad, fx, mb_start, mb_end, axis='y', edge_order=edge_order, handle_boundaries=handle_boundaries)
    
    # Calculate curl components
    curl_x = dfz_dy - dfy_dz
    curl_y = dfx_dz - dfz_dx
    curl_z = dfy_dx - dfx_dy
    
    return [curl_x, curl_y, curl_z]