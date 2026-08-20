"""
Lightweight backend manager for CPU/GPU execution.
"""
import os

import numpy as np

from ..core.base import xp, cupy_enabled, asnumpy


def _cpu_memory_info():
    """Best-effort CPU memory information without extra dependencies."""
    page_size = os.sysconf('SC_PAGE_SIZE')
    total_pages = os.sysconf('SC_PHYS_PAGES')
    avail_pages = os.sysconf('SC_AVPHYS_PAGES')
    total = page_size * total_pages
    free = page_size * avail_pages
    used = total - free
    return {
        'free': free,
        'total': total,
        'used': used,
        'used_by_cupy': 0,
    }


class BackendManager:
    """Minimal backend manager compatible with the example and MPI helpers."""

    def __init__(self, backend_type='auto', device_id=None, enable_mpi=False, multi_gpu=False):
        if backend_type == 'auto':
            backend_type = 'gpu' if cupy_enabled else 'cpu'
        if backend_type == 'gpu' and not cupy_enabled:
            raise RuntimeError("GPU backend requested, but CuPy is not available")

        self.backend_type = backend_type
        self.device_id = device_id
        self.enable_mpi = enable_mpi
        self.multi_gpu = multi_gpu
        self.xp = xp if backend_type == 'gpu' else np

        if self.backend_type == 'gpu' and self.device_id is not None:
            xp.cuda.Device(self.device_id).use()

    def asnumpy(self, array):
        return asnumpy(array)

    def to_device(self, array):
        if self.backend_type == 'gpu':
            return xp.asarray(array)
        return np.asarray(array)

    def synchronize(self):
        if self.backend_type == 'gpu':
            xp.cuda.Stream.null.synchronize()

    def free_memory(self):
        if self.backend_type == 'gpu':
            self.synchronize()
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()

    def get_memory_info(self):
        if self.backend_type == 'gpu':
            free, total = xp.cuda.runtime.memGetInfo()
            return {
                'free': free,
                'total': total,
                'used': total - free,
                'used_by_cupy': xp.get_default_memory_pool().used_bytes(),
            }
        return _cpu_memory_info()


def get_backend(backend_type='auto', device_id=None, enable_mpi=False, multi_gpu=False):
    """Construct a backend manager."""
    return BackendManager(
        backend_type=backend_type,
        device_id=device_id,
        enable_mpi=enable_mpi,
        multi_gpu=multi_gpu,
    )
