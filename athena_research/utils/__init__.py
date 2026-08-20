"""
Utility functions for Athenak data analysis.

This module provides various utility functions for working with Athenak data,
including memory-efficient batch processing and meshblock handling.
"""

# Import core functionality
from ..core import xp, np, cupy_enabled, asnumpy, pinned_array

# Import utilities submodules
from .batch_processing import (
    determine_blocks_per_batch,
    process_data_in_batches
)

from .meshblock_utils import (
    get_block_coords,
    get_block_batches,
    get_block_volume,
    is_block_outside_xyz,
    find_neighbour_blocks
)

__all__ = [
    # Memory handling
    'determine_blocks_per_batch',
    'process_data_in_batches',

    # Meshblock utilities
    'get_block_coords',
    'get_block_batches',
    'get_block_volume',
    'is_block_outside_xyz',
    'find_neighbour_blocks',

    # Array handling
    'asnumpy',
    'pinned_array',
    
    # Backend configuration
    'xp',
    'np',
    'cupy_enabled'
]