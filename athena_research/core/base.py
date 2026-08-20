"""
Base module for shared functionality to avoid circular imports.

This module provides backward-compatible access to the array backend (CPU/GPU).
The actual backend is determined at import time: CuPy if available, otherwise NumPy.

Set environment variable ATHENA_RESEARCH_CPU_ONLY=1 to force CPU mode even if CuPy is available.
"""

import numpy as np
import os

force_cpu = os.environ.get('ATHENA_RESEARCH_CPU_ONLY', '0') == '1'

if force_cpu:
    import numpy as xp
    cupy_enabled = False
    
    def asnumpy(a):
        """Return NumPy array unchanged."""
        return a
else:
    try:
        import cupy as xp
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy is installed, but no CUDA-capable device is available")
        cupy_enabled = True

        def asnumpy(a):
            """Convert CuPy array to NumPy array if necessary."""
            if isinstance(a, xp.ndarray):
                return a.get()
            return a
    except Exception:
        import numpy as xp
        cupy_enabled = False
        
        def asnumpy(a):
            """Return NumPy array unchanged."""
            return a
