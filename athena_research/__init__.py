"""
Athenak data analysis package.

This package provides generic tools for analyzing data from Athenak simulations.
"""
# Core functionality
from .core.athena_data import AthenaData
from .core.athena_dataset import AthenaDataSet
from .core.utils import load, asnumpy, pinned_array, xyz_bool
from .core.data_functions import config_data_functions

# Operations
try:
    from .operations.profiles import set_vertical
except ImportError:
    pass
# Spectra
try:
    from .operations.spectra import (
        set_spectrum,
        set_spectrum_helmholtz,
    )
except ImportError:
    pass
# Structure functions
try:
    from .operations.structure_functions import (
        set_sf,
        set_sf_helmholtz
    )
except ImportError:
    pass

__all__ = [
    # Core classes
    'AthenaData', 
    'AthenaDataSet',
    
    # Core utilities
    'load',
    'asnumpy',
    'pinned_array',
    'config_data_functions',
    'xyz_bool',
    
    # Operations
    'set_vertical',
    
    # Spectra
    'set_spectrum',
    'set_spectrum_helmholtz',
    
    # Structure functions
    'set_sf',
    'set_sf_helmholtz',
]