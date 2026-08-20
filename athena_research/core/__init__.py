"""
Core package for Athenak analysis tools.

This package provides the fundamental tools and utilities for analyzing
Athenak simulation data, focusing on efficient data loading, processing,
and analysis of large-scale hydrodynamic and MHD simulations.
"""

# Import essential functionality from submodules
from .utils import load, pinned_array
from .data_functions import derived_var_list
from .athena_data import AthenaData
from .athena_dataset import AthenaDataSet
from .base import asnumpy, xp, cupy_enabled

# Import particle I/O functions
try:
    from .particle_io import (
        particle_column_xy,
        particle_density_grid,
        particle_profile_x,
        read_particle_binary,
        read_particle_binary_header,
        read_particle_binary_positions,
        read_particle_hdf5,
        read_particle_vtk,
        wrap_particle_positions,
    )
    PARTICLE_IO_AVAILABLE = True
    # Aliases for convenience
    read_particle_h5 = read_particle_hdf5
except ImportError:
    PARTICLE_IO_AVAILABLE = False
    particle_column_xy = None
    particle_density_grid = None
    particle_profile_x = None
    read_particle_binary = None
    read_particle_binary_header = None
    read_particle_binary_positions = None
    read_particle_hdf5 = None
    read_particle_h5 = None
    read_particle_vtk = None
    wrap_particle_positions = None

# Configure array backend
import numpy as np

# Expose common constants and configurations
__all__ = [
    # Core data classes
    'AthenaData',
    'AthenaDataSet',
    
    # I/O functions
    'load',
    
    # Particle I/O functions
    'particle_column_xy',
    'particle_density_grid',
    'particle_profile_x',
    'read_particle_binary',
    'read_particle_binary_header',
    'read_particle_binary_positions',
    'read_particle_hdf5',
    'read_particle_h5',
    'read_particle_vtk',
    'wrap_particle_positions',
    'PARTICLE_IO_AVAILABLE',
    
    # Unit handling - Note: Need to add Units class once implemented
    
    # Utility functions and variables
    'asnumpy',
    'pinned_array',
    'xp',
    'cupy_enabled',
    'np',
    'derived_var_list',
]

# Version information
__version__ = '0.1.0'
