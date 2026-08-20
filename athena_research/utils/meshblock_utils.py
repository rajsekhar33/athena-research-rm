"""
Utilities for handling meshblocks in Athenak simulations.

This module provides helper functions for working with meshblocks,
including determining which meshblocks to process, locating data
within meshblocks, and calculating geometric properties.
"""

from ..core import xp, cupy_enabled, asnumpy, np

def get_block_coords(mb_geometry, mb_id, nx1, nx2, nx3):
    """
    Generate coordinates for a meshblock.
    
    Parameters
    ----------
    mb_geometry : array
        Array containing geometry information for all meshblocks
    mb_id : int
        Meshblock ID
    nx1 : int
        Number of cells in x1 direction
    nx2 : int
        Number of cells in x2 direction
    nx3 : int
        Number of cells in x3 direction
        
    Returns
    -------
    tuple
        x, y, z coordinate arrays for the meshblock
    """
    x1min, x1max, x2min, x2max, x3min, x3max = mb_geometry[mb_id]

    # Create coordinate arrays (cell centers)
    dx1 = (x1max - x1min) / nx1
    dx2 = (x2max - x2min) / nx2
    dx3 = (x3max - x3min) / nx3

    x1 = xp.linspace(x1min + dx1/2, x1max - dx1/2, nx1)
    x2 = xp.linspace(x2min + dx2/2, x2max - dx2/2, nx2)
    x3 = xp.linspace(x3min + dx3/2, x3max - dx3/2, nx3)

    x, y, z = xp.meshgrid(x1, x2, x3, indexing='ij')
    
    return x, y, z

def get_block_batches(n_mbs, blocks_per_batch):
    """
    Create batches of meshblock indices for processing.
    
    Parameters
    ----------
    n_mbs : int
        Total number of meshblocks
    blocks_per_batch : int
        Number of blocks per batch
        
    Returns
    -------
    list
        List of batches, where each batch is a list of meshblock indices
    """
    batches = []
    for i in range(0, n_mbs, blocks_per_batch):
        batch = list(range(i, min(i + blocks_per_batch, n_mbs)))
        batches.append(batch)
    
    return batches

def get_block_volume(mb_geometry, mb_id):
    """
    Calculate the volume of a meshblock.
    
    Parameters
    ----------
    mb_geometry : array
        Array containing geometry information for all meshblocks
    mb_id : int
        Meshblock ID
        
    Returns
    -------
    float
        Volume of the meshblock
    """
    x1min, x1max, x2min, x2max, x3min, x3max = mb_geometry[mb_id]
    return (x1max - x1min) * (x2max - x2min) * (x3max - x3min)

def is_block_outside_xyz(mb_geometry, block_idx, xyz):
    """
    Check if a block is completely outside the specified xyz limits.
    
    Parameters
    ----------
    mb_geometry : array
        Array containing geometry information for all meshblocks
    block_idx : int
        Index of the block to check
    xyz : list of float
        Spatial limits [x1min, x1max, x2min, x2max, x3min, x3max]
        
    Returns
    -------
    bool
        True if block is completely outside limits, False otherwise
    """
    if xyz is None:
        return False
        
    x1min, x1max, x2min, x2max, x3min, x3max = mb_geometry[block_idx]

    # Check if block is completely outside specified limits
    if (x1max < xyz[0] or x1min > xyz[1] or  # Block outside x limits
        x2max < xyz[2] or x2min > xyz[3] or  # Block outside y limits
        x3max < xyz[4] or x3min > xyz[5]):   # Block outside z limits
        return True
    
    return False

def find_neighbour_blocks(ad, block_idx, direction):
    """
    Find the neighboring mesh block index in the specified direction using logical coordinates.
    
    This function uses the mb_logical array, which stores block locations and AMR levels,
    to efficiently identify neighboring blocks along the specified axis.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    block_idx : int
        Index of the source mesh block for which to find neighbors
    direction : str
        Direction to search for neighbors. Must be one of:
        'x+', 'x-', 'y+', 'y-', 'z+', 'z-'
        where '+' indicates the positive direction and '-' the negative direction
        
    Returns
    -------
    list
        List of mesh block indices that neighbor the source block in the specified
        direction. May contain multiple blocks if neighbors have different refinement levels.
    """
    # Validate inputs
    if block_idx < 0 or block_idx >= ad.n_mbs:
        raise ValueError(f"Block index {block_idx} is out of range [0, {ad.n_mbs-1}]")
    
    valid_directions = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
    if direction not in valid_directions:
        raise ValueError(f"Direction must be one of {valid_directions}, got {direction}")
    
    source_x, source_y, source_z, source_level = ad.mb_logical[block_idx]

    periodic_x = 'periodic' in ad._header['mesh']['ix1_bc']
    periodic_y = 'periodic' in ad._header['mesh']['ix2_bc']
    periodic_z = 'periodic' in ad._header['mesh']['ix3_bc']

    neighbors = []
    
    # Determine target logical coordinates based on direction
    target_x, target_y, target_z = source_x, source_y, source_z
    
    if direction == 'x+':
        target_x = source_x + 1
    elif direction == 'x-':
        target_x = source_x - 1
    elif direction == 'y+':
        target_y = source_y + 1
    elif direction == 'y-':
        target_y = source_y - 1
    elif direction == 'z+':
        target_z = source_z + 1
    elif direction == 'z-':
        target_z = source_z - 1
    
    # Handle periodic boundaries
    # Get logical domain size (based on root blocks at level 0)
    nx_root = ad.Nx1 // ad.nx1
    ny_root = ad.Nx2 // ad.nx2
    nz_root = ad.Nx3 // ad.nx3
    
    # Apply periodicity if needed
    if periodic_x and (target_x < 0 or target_x >= nx_root * (2**source_level)):
        if target_x < 0:
            target_x = nx_root * (2**source_level) - 1  # Wrap to other end
        else:
            target_x = 0  # Wrap to beginning
    
    if periodic_y and (target_y < 0 or target_y >= ny_root * (2**source_level)):
        if target_y < 0:
            target_y = ny_root * (2**source_level) - 1
        else:
            target_y = 0
    
    if periodic_z and (target_z < 0 or target_z >= nz_root * (2**source_level)):
        if target_z < 0:
            target_z = nz_root * (2**source_level) - 1
        else:
            target_z = 0
    
    # If we hit a boundary and it's not periodic, return an empty list
    if ((target_x < 0 or target_x >= nx_root * (2**source_level)) and not periodic_x) or \
    ((target_y < 0 or target_y >= ny_root * (2**source_level)) and not periodic_y) or \
    ((target_z < 0 or target_z >= nz_root * (2**source_level)) and not periodic_z):
        return []
    
    # Find potential neighbor blocks at same refinement level
    for i in range(ad.n_mbs):
        if i == block_idx:
            continue  # Skip the source block
            
        # Get logical coordinates of potential neighbor block
        block_x, block_y, block_z, block_level = ad.mb_logical[i]
        
        # Check if this is a neighbor at the same refinement level
        if (block_level == source_level and
            block_x == target_x and
            block_y == target_y and
            block_z == target_z):
            neighbors.append(i)
    
    # If we found same-level neighbors, return them
    if neighbors:
        return neighbors
    
    # If no same-level neighbors, check for neighbors at different refinement levels
    # Check for coarser neighbors (1 level down)
    if source_level > 0:  # Only check if source isn't at the base level
        # Get position within parent block (0 or 1 in each dimension)
        pos_in_parent_x = source_x % 2
        pos_in_parent_y = source_y % 2
        pos_in_parent_z = source_z % 2
        
        # Calculate parent block coordinates
        parent_x = source_x // 2
        parent_y = source_y // 2
        parent_z = source_z // 2
        
        # Determine if we're at the edge of a parent block
        at_edge = False
        parent_target_x, parent_target_y, parent_target_z = parent_x, parent_y, parent_z
        
        if direction == 'x+' and pos_in_parent_x == 1:
            at_edge = True
            parent_target_x = parent_x + 1
        elif direction == 'x-' and pos_in_parent_x == 0:
            at_edge = True
            parent_target_x = parent_x - 1
        elif direction == 'y+' and pos_in_parent_y == 1:
            at_edge = True
            parent_target_y = parent_y + 1
        elif direction == 'y-' and pos_in_parent_y == 0:
            at_edge = True
            parent_target_y = parent_y - 1
        elif direction == 'z+' and pos_in_parent_z == 1:
            at_edge = True
            parent_target_z = parent_z + 1
        elif direction == 'z-' and pos_in_parent_z == 0:
            at_edge = True
            parent_target_z = parent_z - 1
        
        # Handle periodic boundaries for parent coordinates
        if at_edge:
            nx_parent_root = nx_root * (2**(source_level-1))
            ny_parent_root = ny_root * (2**(source_level-1))
            nz_parent_root = nz_root * (2**(source_level-1))
            
            if periodic_x and (parent_target_x < 0 or parent_target_x >= nx_parent_root):
                if parent_target_x < 0:
                    parent_target_x = nx_parent_root - 1
                else:
                    parent_target_x = 0
            
            if periodic_y and (parent_target_y < 0 or parent_target_y >= ny_parent_root):
                if parent_target_y < 0:
                    parent_target_y = ny_parent_root - 1
                else:
                    parent_target_y = 0
            
            if periodic_z and (parent_target_z < 0 or parent_target_z >= nz_parent_root):
                if parent_target_z < 0:
                    parent_target_z = nz_parent_root - 1
                else:
                    parent_target_z = 0
            
            # If valid parent block coordinates, look for blocks at coarser level
            if not ((parent_target_x < 0 or parent_target_x >= nx_parent_root) and not periodic_x) and \
            not ((parent_target_y < 0 or parent_target_y >= ny_parent_root) and not periodic_y) and \
            not ((parent_target_z < 0 or parent_target_z >= nz_parent_root) and not periodic_z):
                
                for i in range(ad.n_mbs):
                    block_x, block_y, block_z, block_level = ad.mb_logical[i]
                    
                    if (block_level == source_level - 1 and
                        block_x == parent_target_x and
                        block_y == parent_target_y and
                        block_z == parent_target_z):
                        neighbors.append(i)
    
    # Check for finer neighbors (1 level up)
    # Need to look for up to 4 blocks that could be neighbors
    for i in range(ad.n_mbs):
        block_x, block_y, block_z, block_level = ad.mb_logical[i]
        
        if block_level != source_level + 1:
            continue  # Skip blocks not at the next finer level
        
        # Calculate this block's parent coordinates
        child_parent_x = block_x // 2
        child_parent_y = block_y // 2
        child_parent_z = block_z // 2
        
        # Calculate position within parent block
        child_pos_x = block_x % 2
        child_pos_y = block_y % 2
        child_pos_z = block_z % 2
        
        # Check if the potential child is on the correct side of the source block
        if direction == 'x+' and child_parent_x == source_x and child_pos_x == 0:
            if child_parent_y == source_y and child_parent_z == source_z:
                neighbors.append(i)
        elif direction == 'x-' and child_parent_x == source_x - 1 and child_pos_x == 1:
            if child_parent_y == source_y and child_parent_z == source_z:
                neighbors.append(i)
        elif direction == 'y+' and child_parent_y == source_y and child_pos_y == 0:
            if child_parent_x == source_x and child_parent_z == source_z:
                neighbors.append(i)
        elif direction == 'y-' and child_parent_y == source_y - 1 and child_pos_y == 1:
            if child_parent_x == source_x and child_parent_z == source_z:
                neighbors.append(i)
        elif direction == 'z+' and child_parent_z == source_z and child_pos_z == 0:
            if child_parent_x == source_x and child_parent_y == source_y:
                neighbors.append(i)
        elif direction == 'z-' and child_parent_z == source_z - 1 and child_pos_z == 1:
            if child_parent_x == source_x and child_parent_y == source_y:
                neighbors.append(i)
    
    return neighbors
