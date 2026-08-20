"""
Core utility functions for Athenak data analysis.

This module provides common utility functions used across the athena_analysis package,
including array handling, data access functions, and coordinate transformations.
"""
import numpy as np
from .base import asnumpy, xp, cupy_enabled

def pinned_array(array):
    """Create a pinned memory array for faster GPU transfers."""
    if cupy_enabled:
        mem = xp.cuda.alloc_pinned_memory(array.nbytes)
        src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
        src[...] = array
        return src
    return array


def maybe_trim_gpu_memory_pool(min_free_fraction=0.15):
    """Trim cached CuPy allocations only when device free memory gets low."""
    if not cupy_enabled:
        return False

    try:
        free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
    except Exception:
        return False

    if total_bytes and (free_bytes / total_bytes) >= min_free_fraction:
        return False

    xp.cuda.Stream.null.synchronize()
    xp.get_default_memory_pool().free_all_blocks()
    return True

def xyz_bool(x, y, z, xyz=None):
    """Create a boolean mask for a 3D region."""
    if xyz is None:
        return xp.ones(x.shape, dtype=bool)
    
    mask = xp.ones(x.shape, dtype=bool)
    if len(xyz) >= 2:
        mask = mask & (x >= xyz[0]) & (x <= xyz[1])
    if len(xyz) >= 4:
        mask = mask & (y >= xyz[2]) & (y <= xyz[3])
    if len(xyz) >= 6:
        mask = mask & (z >= xyz[4]) & (z <= xyz[5])
    return mask


def clear_backend_memory():
    """Release CuPy memory pools when running on GPU."""
    if not cupy_enabled:
        return
    try:
        xp.cuda.Stream.null.synchronize()
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def get_distributed_block_bounds(ad):
    """Return the global meshblock span and local count for the current rank."""
    if getattr(ad, 'has_full_data', True):
        return 0, ad.n_mbs, ad.n_mbs
    return ad.local_mb_start, ad.local_mb_end, ad.get_local_mb_count()


def reduce_array_mpi(mpi_manager, data, op='sum', array_module=xp):
    """Reduce an array or scalar to rank 0 and broadcast the combined result."""
    if mpi_manager is None:
        return data
    total = mpi_manager.allreduce(asnumpy(data), op=op)
    return array_module.asarray(total)


def load(filename):
    """
    Load AthenaData from file with optional MPI distribution.
    
    If MPI is available and being used, each rank will only load its assigned
    meshblocks to save memory (critical for GPU runs).
    
    Parameters
    ----------
    filename : str
        Path to Athena++ data file
        
    Returns
    -------
    AthenaData
        Loaded data object
    """
    from .athena_data import AthenaData
    
    # Check if MPI is available and in use
    try:
        from ..backends.mpi_utils import MPIManager
        mpi = MPIManager(verbose=False)
        use_mpi = (mpi.size > 1)
    except (ImportError, Exception):
        use_mpi = False
        mpi = None
    
    ad = AthenaData()
    
    # Enable distributed loading if MPI is active
    if use_mpi:
        ad.set_mpi_distribution(mpi.rank, mpi.size)
    
    ad.load(filename, config=True)
    return ad

def axis_index(axis):
    """
    Convert axis string to numerical index.
    
    Parameters
    ----------
    axis : str
        Axis designation ('x', 'y', or 'z')
        
    Returns
    -------
    int
        Numerical index (2 for 'x', 1 for 'y', 0 for 'z')
        
    Raises
    ------
    ValueError
        If the axis is not supported
    """
    if isinstance(axis, str):
        if axis == 'z': 
            return 0
        if axis == 'y': 
            return 1
        if axis == 'x': 
            return 2
    raise ValueError(f"axis '{axis}' not supported")

def cell_length(self, level=0, xyz=None):
    """Get cell lengths (dx, dy, dz) for a given refinement level."""
    return cell_info(self, level, xyz)[3:]

def cell_info(self, level=0, xyz=None):
    """
    Calculate cell information based on a given refinement level and domain limits.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        xf, yf, zf : Arrays of face coordinates in each dimension
        dx, dy, dz : Cell widths in each dimension
    """
    if xyz is None:
        xyz = [self.x1min, self.x1max, self.x2min, self.x2max, self.x3min, self.x3max]
    
    # level is physical level
    nx1_fac = 2**level * self.Nx1 / (self.x1max - self.x1min)
    nx2_fac = 2**level * self.Nx2 / (self.x2max - self.x2min)
    nx3_fac = 2**level * self.Nx3 / (self.x3max - self.x3min)
    
    i_min = int((xyz[0] - self.x1min) * nx1_fac)
    i_max = int(np.ceil((xyz[1] - self.x1min) * nx1_fac))
    j_min = int((xyz[2] - self.x2min) * nx2_fac)
    j_max = int(np.ceil((xyz[3] - self.x2min) * nx2_fac))
    k_min = int((xyz[4] - self.x3min) * nx3_fac)
    k_max = int(np.ceil((xyz[5] - self.x3min) * nx3_fac))
    
    dx = (xyz[1] - xyz[0]) / (i_max - i_min)
    dy = (xyz[3] - xyz[2]) / (j_max - j_min)
    dz = (xyz[5] - xyz[4]) / (k_max - k_min)
    
    xf = xp.linspace(xyz[0], xyz[1], i_max - i_min + 1)
    yf = xp.linspace(xyz[2], xyz[3], j_max - j_min + 1)
    zf = xp.linspace(xyz[4], xyz[5], k_max - k_min + 1)
    
    return xf, yf, zf, dx, dy, dz


def cell_faces(self, level=0, xyz=None):
    """
    Get the face coordinates of cells for a given refinement level and domain.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        xf, yf, zf : Arrays of face coordinates in each dimension
    """
    return cell_info(self, level, xyz)[:3]


def cell_faces_mb(self, level=0, xyz=None):
    """
    Get the face coordinates of cells for a given refinement level and domain using numpy arrays.
    
    This version ensures the returned arrays are numpy arrays, not cupy arrays, regardless 
    of whether cupy is enabled. Useful for operations that need to be performed on the CPU.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        xf, yf, zf : Numpy arrays of face coordinates in each dimension
    """
    if xyz is None:
        xyz = [self.x1min, self.x1max, self.x2min, self.x2max, self.x3min, self.x3max]
    
    # level is physical level
    nx1_fac = 2**level * self.Nx1 / (self.x1max - self.x1min)
    nx2_fac = 2**level * self.Nx2 / (self.x2max - self.x2min)
    nx3_fac = 2**level * self.Nx3 / (self.x3max - self.x3min)
    
    i_min = int((xyz[0] - self.x1min) * nx1_fac)
    i_max = int((xyz[1] - self.x1min) * nx1_fac)
    j_min = int((xyz[2] - self.x2min) * nx2_fac)
    j_max = int((xyz[3] - self.x2min) * nx2_fac)
    k_min = int((xyz[4] - self.x3min) * nx3_fac)
    k_max = int((xyz[5] - self.x3min) * nx3_fac)
    
    xf = np.linspace(xyz[0], xyz[1], i_max - i_min + 1)
    yf = np.linspace(xyz[2], xyz[3], j_max - j_min + 1)
    zf = np.linspace(xyz[4], xyz[5], k_max - k_min + 1)
    
    return xf, yf, zf


def cell_centers(self, level=0, xyz=None):
    """
    Get the center coordinates of cells for a given refinement level and domain.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        xc, yc, zc : Arrays of center coordinates in each dimension
    """
    xf, yf, zf = cell_faces(self, level, xyz)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])
    zc = 0.5 * (zf[:-1] + zf[1:])
    return xc, yc, zc


def cell_centers_mb(self, level=0, xyz=None):
    """
    Get the center coordinates of cells for a given refinement level and domain using numpy arrays.
    
    This version ensures the returned arrays are numpy arrays, not cupy arrays.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        xc, yc, zc : Numpy arrays of center coordinates in each dimension
    """
    xf, yf, zf = cell_faces_mb(self, level, xyz)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])
    zc = 0.5 * (zf[:-1] + zf[1:])
    return xc, yc, zc


def xyz_uniform(self, level=0, xyz=None):
    """
    Create 3D coordinate arrays for uniform mesh at specified refinement level.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        X, Y, Z : 3D arrays of coordinates
    """
    xc, yc, zc = cell_centers(self, level, xyz)
    ZYX = xp.meshgrid(zc, yc, xc, indexing='ij')
    return ZYX[2], ZYX[1], ZYX[0]


def coord_uniform(self, level=0, xyz=None):
    """
    Create 3D coordinate arrays for uniform mesh at specified refinement level.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        X, Y, Z : 3D arrays of coordinates with updated axes ordering
    """
    xc, yc, zc = cell_centers(self, level, xyz)
    ZYX = xp.meshgrid(zc, yc, xc, indexing='ij')
    return ZYX[2].swapaxes(0, 1), ZYX[1].swapaxes(0, 1), ZYX[0].swapaxes(0, 1)


def coord_uniform_mb(self, level=0, xyz=None):
    """
    Create 3D coordinate arrays for uniform mesh at specified refinement level using numpy arrays.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        X, Y, Z : Numpy 3D arrays of coordinates
    """
    xc, yc, zc = cell_centers_mb(self, level, xyz)
    ZYX = np.meshgrid(zc, yc, xc, indexing='ij')
    return ZYX[2].swapaxes(0, 1), ZYX[1].swapaxes(0, 1), ZYX[0].swapaxes(0, 1)


def _coord_uniform(self, var, level=0, xyz=None, **kwargs):
    """
    Get uniform mesh coordinates for a specific coordinate variable.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    var : str
        Coordinate variable name ('x', 'y', 'z', 'dx', 'dy', 'dz')
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    xp.ndarray
        Array of the requested coordinate variable
        
    Raises
    ------
    ValueError
        If the variable name is not supported
    """
    xc, yc, zc = cell_centers(self, level, xyz)
    dx, dy, dz = cell_info(self, level, xyz)[3:]
    
    if var == 'x':
        return xp.meshgrid(zc, yc, xc, indexing='ij')[2]
    if var == 'y':
        return xp.meshgrid(zc, yc, xc, indexing='ij')[1]
    if var == 'z':
        return xp.meshgrid(zc, yc, xc, indexing='ij')[0]
    if var == 'dx':
        return xp.full((zc.size, yc.size, xc.size), dx)
    if var == 'dy':
        return xp.full((zc.size, yc.size, xc.size), dy)
    if var == 'dz':
        return xp.full((zc.size, yc.size, xc.size), dz)
        
    raise ValueError(f"var '{var}' not supported")


def get_slice_coord(self, zoom=0, level=0, xyz=None, axis=0):
    """
    Get coordinate data for a slice through the data volume.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    zoom : int, optional
        Zoom factor applied to domain bounds, default=0
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, calculates based on zoom factor
    axis : int or str, optional
        Axis along which to take the slice (0='z', 1='y', 2='x'), default=0
        
    Returns
    -------
    tuple
        x, y, z : 2D coordinate arrays for the slice
        xyz : Domain limits used
    """
    if xyz is None:
        xyz = [
            self.x1min / 2**zoom, self.x1max / 2**zoom,
            self.x2min / 2**zoom, self.x2max / 2**zoom,
            self.x3min / 2**level / self.Nx3, self.x3max / 2**level / self.Nx3
        ]
        
    x, y, z = xyz_uniform(self, level, xyz)
    
    if isinstance(axis, str):
        axis = axis_index(axis)
        
    return (
        xp.average(x, axis=axis),
        xp.average(y, axis=axis),
        xp.average(z, axis=axis),
        xyz
    )


def get_slice(self, var='dens', zoom=0, level=0, xyz=None, axis=0):
    """
    Get a 2D slice of a 3D variable along a specified axis.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    var : str, optional
        Variable name to slice, default='dens'
    zoom : int, optional
        Zoom factor applied to domain bounds, default=0
    level : int, optional
        Refinement level to use, default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, calculates based on zoom factor
    axis : int or str, optional
        Axis along which to take the slice (0='z', 1='y', 2='x'), default=0
        
    Returns
    -------
    tuple
        data : 2D array of the sliced variable
        xyz : Domain limits used
    """
    if xyz is None:
        xyz = [
            self.x1min / 2**zoom, self.x1max / 2**zoom,
            self.x2min / 2**zoom, self.x2max / 2**zoom,
            self.x3min / 2**level / self.Nx3, self.x3max / 2**level / self.Nx3
        ]
    
    if isinstance(axis, str):
        axis = axis_index(axis)
        
    data = self.get_refined_data(var, level=level, xyz=xyz)
    return xp.average(data, axis=axis), xyz

def get_refined_coord(self, level=0, xyz=None):
    """
    Get coordinates of refined grid cells at a specified refinement level.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    level : int, optional
        Refinement level (0 means original resolution), default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    tuple
        x, y, z : 3D coordinate arrays for cell centers
        dx, dy, dz : Cell spacing in each direction
    """
    if xyz is None:
        xyz = [self.x1min, self.x1max, self.x2min, self.x2max, self.x3min, self.x3max]
    # level is physical level
    nx1_fac = 2**level*self.Nx1/(self.x1max-self.x1min)
    nx2_fac = 2**level*self.Nx2/(self.x2max-self.x2min)
    nx3_fac = 2**level*self.Nx3/(self.x3max-self.x3min)
    i_min = int((xyz[0]-self.x1min)*nx1_fac)
    i_max = int((xyz[1]-self.x1min)*nx1_fac)
    j_min = int((xyz[2]-self.x2min)*nx2_fac)
    j_max = int((xyz[3]-self.x2min)*nx2_fac)
    k_min = int((xyz[4]-self.x3min)*nx3_fac)
    k_max = int((xyz[5]-self.x3min)*nx3_fac)
    
    x = np.linspace(xyz[0], xyz[1], i_max-i_min)
    y = np.linspace(xyz[2], xyz[3], j_max-j_min)
    z = np.linspace(xyz[4], xyz[5], k_max-k_min)
    dx = (xyz[1]-xyz[0])/(i_max-i_min)
    dy = (xyz[3]-xyz[2])/(j_max-j_min)
    dz = (xyz[5]-xyz[4])/(k_max-k_min)
    ZYX = np.meshgrid(z, y, x)
    return ZYX[2].swapaxes(0, 1), ZYX[1].swapaxes(0, 1), ZYX[0].swapaxes(0, 1), dx, dy, dz


def get_refined_data(self, var, level=0, xyz=None):
    """
    Get variable data at specified refinement level for a region of interest.
    
    This function retrieves data for any variable on a uniformly refined grid,
    handling both direct data access and interpolation between refinement levels.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    var : str
        Variable name to retrieve
    level : int, optional
        Refinement level (0 means original resolution), default=0
    xyz : list or None, optional
        Domain limits [x1min, x1max, x2min, x2max, x3min, x3max]
        If None, uses the full simulation domain
        
    Returns
    -------
    xp.ndarray
        Data array on a uniform grid at the requested refinement level
    """
    if xyz is None:
        xyz = [self.x1min, self.x1max, self.x2min, self.x2max, self.x3min, self.x3max]
    
    # physical level is the actual grid resolution level
    physical_level = level
    nx1_fac = 2**level*self.Nx1/(self.x1max-self.x1min)
    nx2_fac = 2**level*self.Nx2/(self.x2max-self.x2min)
    nx3_fac = 2**level*self.Nx3/(self.x3max-self.x3min)
    i_min = int((xyz[0]-self.x1min)*nx1_fac)
    i_max = int((xyz[1]-self.x1min)*nx1_fac)
    j_min = int((xyz[2]-self.x2min)*nx2_fac)
    j_max = int((xyz[3]-self.x2min)*nx2_fac)
    k_min = int((xyz[4]-self.x3min)*nx3_fac)
    k_max = int((xyz[5]-self.x3min)*nx3_fac)
    data = xp.zeros((k_max-k_min, j_max-j_min, i_max-i_min))
    
    # Use get_data function from data_functions module
    from .data_functions import get_data
    raw = get_data(self, var)
    
    # In MPI mode, iterate over ALL meshblocks but only process ones this rank owns
    # In non-MPI mode, iterate over local meshblocks
    total_blocks = len(self.mb_logical)
    
    for nmb in range(total_blocks):
        # Check if this rank owns this meshblock
        if nmb < self.local_mb_start or nmb >= self.local_mb_end:
            continue  # Skip blocks not owned by this rank
        
        # Local index for accessing this rank's data arrays
        local_idx = nmb - self.local_mb_start
        
        block_level = self.mb_logical[nmb,-1]
        block_loc = self.mb_logical[nmb,:3]
        block_data = raw[local_idx]
        
        # Prolongate coarse data and copy same-level data
        if block_level <= physical_level:
            s = int(2**(physical_level - block_level))
            # Calculate destination indices, without selection
            il_d = block_loc[0] * self.nx1 * s if self.Nx1 > 1 else 0
            jl_d = block_loc[1] * self.nx2 * s if self.Nx2 > 1 else 0
            kl_d = block_loc[2] * self.nx3 * s if self.Nx3 > 1 else 0
            iu_d = il_d + self.nx1 * s if self.Nx1 > 1 else 1
            ju_d = jl_d + self.nx2 * s if self.Nx2 > 1 else 1
            ku_d = kl_d + self.nx3 * s if self.Nx3 > 1 else 1
            
            # Calculate (prolongated) source indices, with selection
            il_s = max(il_d, i_min) - il_d
            jl_s = max(jl_d, j_min) - jl_d
            kl_s = max(kl_d, k_min) - kl_d
            iu_s = min(iu_d, i_max) - il_d
            ju_s = min(ju_d, j_max) - jl_d
            ku_s = min(ku_d, k_max) - kl_d
            
            if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                continue
                
            # Account for selection in destination indices
            il_d = max(il_d, i_min) - i_min
            jl_d = max(jl_d, j_min) - j_min
            kl_d = max(kl_d, k_min) - k_min
            iu_d = min(iu_d, i_max) - i_min
            ju_d = min(ju_d, j_max) - j_min
            ku_d = min(ku_d, k_max) - k_min
            
            if s > 1:
                if self.Nx1 > 1:
                    block_data = xp.repeat(block_data, s, axis=2)
                if self.Nx2 > 1:
                    block_data = xp.repeat(block_data, s, axis=1)
                if self.Nx3 > 1:
                    block_data = xp.repeat(block_data, s, axis=0)
            
            data[kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_data[kl_s:ku_s, jl_s:ju_s, il_s:iu_s]
            
        # Restrict fine data, volume average
        else:
            # Calculate scale
            s = int(2 ** (block_level - physical_level))
            
            # Calculate destination indices, without selection
            il_d = int(block_loc[0] * self.nx1 / s) if self.Nx1 > 1 else 0
            jl_d = int(block_loc[1] * self.nx2 / s) if self.Nx2 > 1 else 0
            kl_d = int(block_loc[2] * self.nx3 / s) if self.Nx3 > 1 else 0
            iu_d = int(il_d + self.nx1 / s) if self.Nx1 > 1 else 1
            ju_d = int(jl_d + self.nx2 / s) if self.Nx2 > 1 else 1
            ku_d = int(kl_d + self.nx3 / s) if self.Nx3 > 1 else 1
            
            # Calculate (restricted) source indices, with selection
            il_s = max(il_d, i_min) - il_d
            jl_s = max(jl_d, j_min) - jl_d
            kl_s = max(kl_d, k_min) - kl_d
            iu_s = min(iu_d, i_max) - il_d
            ju_s = min(ju_d, j_max) - jl_d
            ku_s = min(ku_d, k_max) - kl_d
            
            if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                continue
                
            # Account for selection in destination indices
            il_d = max(il_d, i_min) - i_min
            jl_d = max(jl_d, j_min) - j_min
            kl_d = max(kl_d, k_min) - k_min
            iu_d = min(iu_d, i_max) - i_min
            ju_d = min(ju_d, j_max) - j_min
            ku_d = min(ku_d, k_max) - k_min
            
            # Account for restriction in source indices
            num_extended_dims = 0
            if self.Nx1 > 1:
                il_s *= s
                iu_s *= s
                num_extended_dims += 1
            if self.Nx2 > 1:
                jl_s *= s
                ju_s *= s
                num_extended_dims += 1
            if self.Nx3 > 1:
                kl_s *= s
                ku_s *= s
                num_extended_dims += 1
            
            # Calculate fine-level offsets
            io_vals = range(s) if self.Nx1 > 1 else (0,)
            jo_vals = range(s) if self.Nx2 > 1 else (0,)
            ko_vals = range(s) if self.Nx3 > 1 else (0,)

            # Assign values
            for ko in ko_vals:
                for jo in jo_vals:
                    for io in io_vals:
                        data[kl_d:ku_d, jl_d:ju_d, il_d:iu_d] += block_data[
                                                                kl_s+ko:ku_s:s,
                                                                jl_s+jo:ju_s:s,
                                                                il_s+io:iu_s:s] / (s**num_extended_dims)
    return data


def get_refined_data_mb(self, var, level=0, xyz=None, use_mpi_gather=True):
    """
    Reconstructs variable data from multi-block simulation at a specified refinement level.
    
    This method extracts and properly combines data from multiple mesh blocks to create a 
    uniform grid representation of the specified variable at the requested refinement level.
    It handles both coarse data (which is prolongated/upsampled) and fine data (which is
    restricted/downsampled) to achieve the target resolution.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance
    var : str
        Name of the variable to reconstruct. Can be a raw variable from simulation data
        or a derived variable.
    level : int, optional
        Target refinement level. Default is 0 (base grid level).
        Higher levels give finer resolution with cells smaller by a factor of 2^level.
    xyz : list, optional
        Spatial limits of the region to extract, specified as 
        [x1min, x1max, x2min, x2max, x3min, x3max].
        If empty, uses the full domain bounds.
    use_mpi_gather : bool, optional
        If True (default), use MPI allreduce to gather data from all ranks when in MPI mode.
        Set to False when different ranks intentionally request different spatial regions
        (e.g., for slab decomposition in MPI FFT).
        
    Returns
    -------
    numpy.ndarray
        A 3D array containing the requested variable data at the specified refinement 
        level within the specified region. Array dimensions are [nz, ny, nx].
        
    Notes
    -----
    - For coarse blocks (block_level <= target level), data is prolongated by repeating values.
    - For fine blocks (block_level > target level), data is restricted by averaging.
    - Data from overlapping blocks is combined consistently, with finer blocks taking precedence.
    - The algorithm handles arbitrary mesh refinement patterns with varying levels.
    - For large domains with high refinement levels, this method may consume significant memory.
    
    Examples
    --------
    # Get density at base level for full domain
    >>> density = dataset.get_refined_data_mb('dens')
    
    # Get temperature at refinement level 2 for central region
    >>> temp = dataset.get_refined_data_mb('temp', level=2, 
    ...                             xyz=[0.25, 0.75, 0.25, 0.75, 0.25, 0.75])
    """
    if xyz is None:
        xyz = [self.x1min, self.x1max, self.x2min, self.x2max, self.x3min, self.x3max]
    
    # physical level is the actual grid resolution level
    physical_level = level
    nx1_fac = 2**level*self.Nx1/(self.x1max-self.x1min)
    nx2_fac = 2**level*self.Nx2/(self.x2max-self.x2min)
    nx3_fac = 2**level*self.Nx3/(self.x3max-self.x3min)
    i_min = int((xyz[0]-self.x1min)*nx1_fac)
    i_max = int((xyz[1]-self.x1min)*nx1_fac)
    j_min = int((xyz[2]-self.x2min)*nx2_fac)
    j_max = int((xyz[3]-self.x2min)*nx2_fac)
    k_min = int((xyz[4]-self.x3min)*nx3_fac)
    k_max = int((xyz[5]-self.x3min)*nx3_fac)
    data = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min))
    
    # Iterate over ALL meshblocks, but only process ones this rank owns
    # This allows MPI ranks to contribute their data and combine via allreduce
    mbl = self.local_mb_start
    mbh = self.local_mb_end
    total_blocks = len(self.mb_logical)
    
    for nmb in range(total_blocks):
        # Skip meshblocks not owned by this rank
        if nmb < mbl or nmb >= mbh:
            continue
        
        local_idx = nmb - mbl  # Local index for data arrays
            
        block_level = self.mb_logical[nmb,-1]
        block_loc = self.mb_logical[nmb,:3]
        
        # Get block data based on variable type
        if var in self.coord.keys():
            block_data = self.coord[var][local_idx]
        elif var in self.data_raw.keys():
            block_data = self.data_raw[var][local_idx]
        else:
            block_data = asnumpy(self.data(var, nmb, nmb+1)[0])
        
        # Prolongate coarse data and copy same-level data
        if block_level <= physical_level:
            s = int(2**(physical_level - block_level))
            # Calculate destination indices, without selection
            il_d = block_loc[0] * self.nx1 * s if self.Nx1 > 1 else 0
            jl_d = block_loc[1] * self.nx2 * s if self.Nx2 > 1 else 0
            kl_d = block_loc[2] * self.nx3 * s if self.Nx3 > 1 else 0
            iu_d = il_d + self.nx1 * s if self.Nx1 > 1 else 1
            ju_d = jl_d + self.nx2 * s if self.Nx2 > 1 else 1
            ku_d = kl_d + self.nx3 * s if self.Nx3 > 1 else 1
            
            # Calculate (prolongated) source indices, with selection
            il_s = max(il_d, i_min) - il_d
            jl_s = max(jl_d, j_min) - jl_d
            kl_s = max(kl_d, k_min) - kl_d
            iu_s = min(iu_d, i_max) - il_d
            ju_s = min(ju_d, j_max) - jl_d
            ku_s = min(ku_d, k_max) - kl_d
            
            if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                continue
                
            # Account for selection in destination indices
            il_d = max(il_d, i_min) - i_min
            jl_d = max(jl_d, j_min) - j_min
            kl_d = max(kl_d, k_min) - k_min
            iu_d = min(iu_d, i_max) - i_min
            ju_d = min(ju_d, j_max) - j_min
            ku_d = min(ku_d, k_max) - k_min
            
            if s > 1:
                if self.Nx1 > 1:
                    block_data = np.repeat(block_data, s, axis=2)
                if self.Nx2 > 1:
                    block_data = np.repeat(block_data, s, axis=1)
                if self.Nx3 > 1:
                    block_data = np.repeat(block_data, s, axis=0)
            
            src = block_data[kl_s:ku_s, jl_s:ju_s, il_s:iu_s]
            data[kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = src.get() if hasattr(src, 'get') else src
            
        # Restrict fine data, volume average
        else:
            # Calculate scale
            s = int(2 ** (block_level - physical_level))
            
            # Calculate destination indices, without selection
            il_d = int(block_loc[0] * self.nx1 / s) if self.Nx1 > 1 else 0
            jl_d = int(block_loc[1] * self.nx2 / s) if self.Nx2 > 1 else 0
            kl_d = int(block_loc[2] * self.nx3 / s) if self.Nx3 > 1 else 0
            iu_d = int(il_d + self.nx1 / s) if self.Nx1 > 1 else 1
            ju_d = int(jl_d + self.nx2 / s) if self.Nx2 > 1 else 1
            ku_d = int(kl_d + self.nx3 / s) if self.Nx3 > 1 else 1
            
            # Calculate (restricted) source indices, with selection
            il_s = max(il_d, i_min) - il_d
            jl_s = max(jl_d, j_min) - jl_d
            kl_s = max(kl_d, k_min) - kl_d
            iu_s = min(iu_d, i_max) - il_d
            ju_s = min(ju_d, j_max) - jl_d
            ku_s = min(ku_d, k_max) - kl_d
            
            if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                continue
                
            # Account for selection in destination indices
            il_d = max(il_d, i_min) - i_min
            jl_d = max(jl_d, j_min) - j_min
            kl_d = max(kl_d, k_min) - k_min
            iu_d = min(iu_d, i_max) - i_min
            ju_d = min(ju_d, j_max) - j_min
            ku_d = min(ku_d, k_max) - k_min
            
            # Account for restriction in source indices
            num_extended_dims = 0
            if self.Nx1 > 1:
                il_s *= s
                iu_s *= s
                num_extended_dims += 1
            if self.Nx2 > 1:
                jl_s *= s
                ju_s *= s
                num_extended_dims += 1
            if self.Nx3 > 1:
                kl_s *= s
                ku_s *= s
                num_extended_dims += 1
            
            # Calculate fine-level offsets
            io_vals = range(s) if self.Nx1 > 1 else (0,)
            jo_vals = range(s) if self.Nx2 > 1 else (0,)
            ko_vals = range(s) if self.Nx3 > 1 else (0,)

            # Assign values
            for ko in ko_vals:
                for jo in jo_vals:
                    for io in io_vals:
                        data[kl_d:ku_d, jl_d:ju_d, il_d:iu_d] += block_data[
                                                                kl_s+ko:ku_s:s,
                                                                jl_s+jo:ju_s:s,
                                                                il_s+io:iu_s:s] / (s**num_extended_dims)
    
    # If using MPI, combine contributions from all ranks
    # Each rank filled in data ONLY for meshblocks it owns (rest are zeros)
    # Use allreduce with SUM to combine contributions from all ranks
    # This correctly reconstructs any spatial region, not just full domain
    # UNLESS use_mpi_gather=False (for spectrum slab decomposition)
    if use_mpi_gather:
        try:
            from ..backends.mpi_utils import MPIManager
            mpi_manager = MPIManager()
            if mpi_manager.is_initialized and mpi_manager.size > 1:
                # Use MPI allreduce to sum contributions from all ranks
                # Non-overlapping contributions will sum correctly
                data = mpi_manager.allreduce(data, op='sum')
        except (ImportError, Exception):
            pass  # No MPI available or not initialized, use local data as-is
    
    return data

def _attach_helper_methods():
    """
    Attach coordinate and slice utility functions to the AthenaData class.
    
    This function should be called at the end of the utils.py module to ensure
    that all the utility functions are properly attached to AthenaData when
    the module is imported.
    """
    from .athena_data import AthenaData
    
    # Attach coordinate utility functions
    AthenaData.cell_faces = cell_faces
    AthenaData.cell_faces_mb = cell_faces_mb
    AthenaData.cell_centers = cell_centers
    AthenaData.cell_centers_mb = cell_centers_mb
    AthenaData.coord_uniform = coord_uniform
    AthenaData.coord_uniform_mb = coord_uniform_mb
    AthenaData.xyz_uniform = xyz_uniform
    AthenaData.cell_length = cell_length
    AthenaData.cell_info = cell_info
    AthenaData.axis_index = axis_index
    
    # Attach slice utility functions
    AthenaData.get_slice = get_slice
    AthenaData.get_slice_coord = get_slice_coord
    
    # Attach refined grid functions
    AthenaData.get_refined_coord = get_refined_coord
    AthenaData.get_refined_data = get_refined_data
    AthenaData.get_refined_data_mb = get_refined_data_mb

# Call the function at the end of the module to attach methods when imported
_attach_helper_methods()
