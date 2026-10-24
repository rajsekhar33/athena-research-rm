"""
Gradient, divergence, and curl data operations for Athenak simulation data.
"""
import numpy as np
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
        return _gradient_original(ad, f, axis, edge_order, handle_boundaries=handle_boundaries, **kwargs)

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

_AXIS_SPACING = {'x': 'dx_mb', 'y': 'dy_mb', 'z': 'dz_mb'}
_AXIS_GEOMETRY = {'x': (0, 'nx1'), 'y': (2, 'nx2'), 'z': (4, 'nx3')}


def _global_cell_width(ad, ax):
    """Cell width of every global meshblock along ``ax``, indexed by global
    block id. Taken from mb_geometry, which every rank holds in full, rather
    than ad.data('dx_mb'), whose no-range result is rank-local under MPI."""
    col, n_attr = _AXIS_GEOMETRY[ax]
    geom = np.asarray(ad.mb_geometry, dtype=np.float64)
    return (geom[:, col + 1] - geom[:, col]) / float(getattr(ad, n_attr))


def _first_global_block(ad, mbl=None):
    """
    Global meshblock index of the first block in ``ad.data(var, mbl, mbh)``.

    Under MPI the returned array is the intersection of the requested range
    with this rank's blocks, and an unbounded request returns only the local
    blocks, so local block ``b`` is global block ``offset + b``.
    """
    distributed = hasattr(ad, 'has_full_data') and not ad.has_full_data
    if mbl is None:
        return ad.local_mb_start if distributed else 0
    return max(mbl, ad.local_mb_start) if distributed else mbl


def _owns_block(ad, mb):
    if hasattr(ad, 'owns_meshblock'):
        return ad.owns_meshblock(mb)
    return getattr(ad, 'has_full_data', True)


def _face_neighbour(ad, f, g0, gidx, direction, var, data_kwargs):
    """
    Data of the same-level neighbour of global block ``gidx`` across a face,
    or None when there is none usable. Neighbours at a different refinement
    level have a different cell size and are not one cell away, so the caller
    keeps the one-sided in-block stencil there.
    """
    neighbours = find_neighbour_blocks(ad, gidx, direction)
    if not neighbours:
        return None
    nbr = neighbours[0]
    if ad.mb_logical[nbr][3] != ad.mb_logical[gidx][3]:
        return None
    if g0 <= nbr < g0 + f.shape[0]:
        return f[nbr - g0]
    if var is None or not _owns_block(ad, nbr):
        return None
    nbr_data = ad.data(var, nbr, nbr + 1, **data_kwargs)
    return nbr_data[0] if len(nbr_data) > 0 else None


def _block_gradient(ad, f, g0, axis, spacing, edge_order=1, handle_boundaries=True,
                    var=None, data_kwargs=None):
    """
    Per-block derivative of ``f`` (shape (n_blocks, nz, ny, nx)) along ``axis``.

    ``f[b]`` is global meshblock ``g0 + b`` and ``spacing`` is the global
    per-block cell width along ``axis``. Block faces use a centred difference
    with the same-level neighbour when it is in ``f`` or, given ``var``,
    loadable on this rank; otherwise the one-sided in-block stencil is kept.
    A block that is one cell thick along ``axis`` (2D data) has no in-block
    stencil: its derivative is the centred difference across both
    neighbours, or zero without them.
    """
    data_kwargs = data_kwargs or {}
    arr_axis = 2 - 'xyz'.index(axis)
    n_ax = f.shape[1 + arr_axis]

    def face(idx):
        return tuple(idx if a == arr_axis else slice(None) for a in range(3))

    result = xp.zeros_like(f)
    for b in range(f.shape[0]):
        gidx = g0 + b
        block = f[b]
        h = float(spacing[gidx])
        if n_ax >= 2:
            grad = xp.gradient(block, h, axis=arr_axis, edge_order=min(edge_order, n_ax - 1))
        else:
            grad = xp.zeros_like(block)

        if handle_boundaries:
            plus = _face_neighbour(ad, f, g0, gidx, axis + '+', var, data_kwargs)
            minus = _face_neighbour(ad, f, g0, gidx, axis + '-', var, data_kwargs)
            if n_ax >= 2:
                if plus is not None:
                    grad[face(-1)] = (plus[face(0)] - block[face(-2)]) / (2 * h)
                if minus is not None:
                    grad[face(0)] = (block[face(1)] - minus[face(-1)]) / (2 * h)
            elif plus is not None and minus is not None:
                grad[face(0)] = (plus[face(0)] - minus[face(-1)]) / (2 * h)

        result[b] = grad
    return result


def _gradient_original(ad, f, axis=None, edge_order=1, handle_boundaries=True,
                       mbl=None, mbh=None, var_name=None, **kwargs):
    """
    Original gradient implementation working on the whole domain (or the
    meshblock range ``mbl:mbh``) at once.

    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    f : array-like or str
        Field to compute the gradient of. An array must be what
        ``ad.data(var, mbl, mbh)`` returns for the same ``mbl``/``mbh``.
    axis : str or None, optional
        Direction for the gradient ('x', 'y', 'z'). If None, returns all components.
    edge_order : int, optional
        Order of accuracy for edge points. Default is 1.
    handle_boundaries : bool, optional
        Whether to handle boundaries between mesh blocks. Default is True.
    mbl, mbh : int, optional
        Global meshblock range of ``f``. None means the full (rank-local) domain.
    var_name : str, optional
        Variable name of an array ``f``; lets block faces at the edge of the
        range use neighbour data outside it.
    **kwargs : dict, optional
        Additional arguments passed to ad.data. ``dtype='uniform'`` uses the
        first block's cell size for every block.

    Returns
    -------
    array-like or list of array-like
        Gradient of the field
    """
    uniform = kwargs.pop('dtype', None) == 'uniform'
    if isinstance(f, str):
        var_name = f
        f = ad.data(f, mbl, mbh, **kwargs)
    f = xp.asarray(f)

    if axis is not None and axis not in _AXIS_SPACING:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', 'z', or None.")

    g0 = _first_global_block(ad, mbl)

    def component(ax):
        spacing = _global_cell_width(ad, ax)
        if uniform:
            spacing = spacing[:1].repeat(len(spacing))
        return _block_gradient(ad, f, g0, ax, spacing, edge_order, handle_boundaries,
                               var=var_name, data_kwargs=kwargs)

    if axis is None:
        return [component('x'), component('y'), component('z')]
    return component(axis)

def _divergence_original(ad, fx, fy, fz, mbl=None, mbh=None, **kwargs):
    """Original divergence implementation working on the whole domain at once."""
    return (_gradient_original(ad, fx, 'x', mbl=mbl, mbh=mbh, **dict(kwargs))
            + _gradient_original(ad, fy, 'y', mbl=mbl, mbh=mbh, **dict(kwargs))
            + _gradient_original(ad, fz, 'z', mbl=mbl, mbh=mbh, **dict(kwargs)))

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
        Faces whose neighbour is at a different refinement level keep the
        one-sided in-block difference.

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
    if axis not in _AXIS_SPACING:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', 'z', or None.")

    data = xp.asarray(ad.data(var, mbl, mbh))
    # The unbounded cell-width request is global (indexed by global block id).
    spacing = _global_cell_width(ad, axis)
    return _block_gradient(ad, data, _first_global_block(ad, mbl), axis, spacing,
                           edge_order, handle_boundaries, var=var)

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

def _curl_original(ad, fx, fy, fz, mbl=None, mbh=None, **kwargs):
    """Original curl implementation working on the whole domain at once."""
    data_kwargs = {k: v for k, v in kwargs.items() if k != 'dtype'}
    names = {}
    # Load each component once; keep its name so faces at the edge of an
    # mbl:mbh batch can still use neighbour blocks outside the batch.
    if isinstance(fx, str):
        names['x'], fx = fx, ad.data(fx, mbl, mbh, **data_kwargs)
    if isinstance(fy, str):
        names['y'], fy = fy, ad.data(fy, mbl, mbh, **data_kwargs)
    if isinstance(fz, str):
        names['z'], fz = fz, ad.data(fz, mbl, mbh, **data_kwargs)

    def grad(f, comp, axis):
        return _gradient_original(ad, f, axis, mbl=mbl, mbh=mbh,
                                  var_name=names.get(comp), **dict(kwargs))

    # Calculate derivatives
    dfz_dy = grad(fz, 'z', 'y')
    dfy_dz = grad(fy, 'y', 'z')
    
    dfx_dz = grad(fx, 'x', 'z')
    dfz_dx = grad(fz, 'z', 'x')
    
    dfy_dx = grad(fy, 'y', 'x')
    dfx_dy = grad(fx, 'x', 'y')
    
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