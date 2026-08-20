"""
Batch processing utilities for memory-efficient data handling.
"""
import numpy as np

from ..core.base import xp, cupy_enabled

def determine_blocks_per_batch(n_mbs, no_vars, n_weights, n_points_per_block):
    """
    Determine optimal number of blocks to process at once based on memory.
    
    Parameters
    ----------
    n_mbs : int
        Total number of mesh blocks
    no_vars : int
        Number of variables to process
    n_weights : int
        Number of weight schemes
    n_points_per_block : int
        Points per block
    Returns
    -------
    int
        Blocks to process in each batch
    """
    if cupy_enabled:
        free_memory = xp.cuda.Device().mem_info[0]  # Free memory in bytes
        total_memory = xp.cuda.Device().mem_info[1]  # Total memory in bytes
        
        # Use only up to 40% of total memory or 50% of free memory (whichever is less)
        target_memory = min(0.4 * total_memory, 0.5 * free_memory)
        
        # Estimate memory needed per block
        bytes_per_float64 = 8
        memory_per_block = n_points_per_block * bytes_per_float64  # Base data size
        
        # Each block requires memory for:
        # - 3D coordinates (x, y, z) = 3 * memory_per_block
        # - Variable data for each variable = no_vars * memory_per_block  
        # - Weight data for each weight = n_weights * memory_per_block
        memory_per_block *= (3 + no_vars + n_weights)
        
        # Calculate blocks per batch, maintaining a safety margin
        blocks_per_batch = max(2, int(target_memory / memory_per_block))
    else:
        # For CPU processing
        blocks_per_batch = max(2, n_mbs)
    
    return blocks_per_batch

def process_data_in_batches(mb_data, var_names, n_mbs, blocks_per_batch, process_func, **kwargs):
    """
    Process mesh block data in batches to manage memory usage.
    
    Parameters
    ----------
    mb_data : dict
        Dictionary containing all meshblock data
    var_names : list
        List of variable names to process
    n_mbs : int
        Total number of mesh blocks
    blocks_per_batch : int
        Number of blocks to process in each batch
    process_func : callable
        Function to process each batch of data
    **kwargs : dict
        Additional arguments to pass to process_func
        
    Returns
    -------
    dict
        Results from batch processing
    """
    results = {}

    for batch_start in range(0, n_mbs, blocks_per_batch):
        batch_end = min(batch_start + blocks_per_batch, n_mbs)

        batch_data = {var: mb_data[var][batch_start:batch_end] for var in var_names}

        batch_results = process_func(
            batch_data, 
            block_range=(batch_start, batch_end), 
            **kwargs
        )
        
        # Merge results
        if not results:
            results = batch_results
        else:
            for key, value in batch_results.items():
                if isinstance(value, np.ndarray) or (cupy_enabled and isinstance(value, xp.ndarray)):
                    results[key] = xp.concatenate([results[key], value])
                elif isinstance(value, (int, float)):
                    results[key] += value
                else:
                    results[key].extend(value)
    
    return results
