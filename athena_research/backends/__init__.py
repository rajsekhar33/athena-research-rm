"""
Backend utilities for distributed computing.

This module provides MPI utilities for distributed computation.
The actual CPU/GPU abstraction is handled by the simple xp module in core/base.py.
"""

from .backend_manager import BackendManager, get_backend
from .mpi_utils import MPIManager, setup_mpi_environment

__all__ = ['BackendManager', 'MPIManager', 'get_backend', 'setup_mpi_environment']
