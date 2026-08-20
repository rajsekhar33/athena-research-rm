"""
Operations module for Athenak data analysis.

This module provides various operations for analyzing Athenak simulation data,
including basic operations, histograms, profiles, spectral analysis, and
structure functions.
"""

# Import operation submodules
try:
    from .basic_operations import *
    from .grad_div_curl import *
    from .histograms import *
    from .profiles import *
    from .weighted_projection import *
    from .spectra import *
    from .structure_functions import *
    from .area_functions import set_area, calc_area_mb_based, calc_areas_all_steps, STEP_SIZES
except ImportError as e:
    import warnings
    warnings.warn(f"Some operations modules could not be imported: {e}")

__all__ = [
    # Basic operations
    'calc_data', 'calc_min', 'calc_max', 'calc_sum', 'calc_avg',

    # Histogram operations
    'set_dist', 'set_dist2d',

    # Profile operations
    'set_profile', 'set_radial', 'set_vertical',

    # Weighted projection operations
    'create_weighted_projection', 'create_projection_stack',

    # Spectral operations
    'set_spectrum', 'set_spectrum_helmholtz',

    # Structure function operations
    'set_sf', 'set_sf_helmholtz', 'set_sf_aniso_mhd',

    # Gradient, divergence, and curl operations
    'gradient', 'divergence', 'curl',

    # Area functions
    'set_area', 'calc_area_mb_based', 'calc_areas_all_steps', 'STEP_SIZES',
]
