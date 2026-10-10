"""Tiny uniform mesh with bounded data access for boundary regressions."""
import numpy as np
from athena_research.core.base import xp


class BoundaryMesh:
    def __init__(self, log_temp, block_shape=(4, 4, 4),
                 periodic=(True, True, False), spacing=(.7, 1.1, 1.3)):
        self.logs = np.log10(np.power(10., np.asarray(log_temp, dtype=float)))
        self.Nx3, self.Nx2, self.Nx1 = self.logs.shape
        self.nx3, self.nx2, self.nx1 = block_shape
        assert all(n % b == 0 for n, b in zip(self.logs.shape, block_shape))
        self.x1min = self.x2min = self.x3min = 0.
        self.x1max, self.x2max, self.x3max = (
            self.Nx1*spacing[0], self.Nx2*spacing[1], self.Nx3*spacing[2])
        self.spacing = spacing
        self.periodic = periodic
        self.has_full_data = True
        self.calls = []
        logical, blocks, geometry = [], [], []
        bz, by, bx = block_shape
        for z, y, x in np.ndindex(self.Nx3//bz, self.Nx2//by, self.Nx1//bx):
            logical.append((x, y, z, 0))
            blocks.append(np.power(10., self.logs[
                z*bz:(z+1)*bz, y*by:(y+1)*by, x*bx:(x+1)*bx]))
            geometry.append((x*bx*spacing[0], (x+1)*bx*spacing[0],
                             y*by*spacing[1], (y+1)*by*spacing[1],
                             z*bz*spacing[2], (z+1)*bz*spacing[2]))
        self.mb_logical = np.asarray(logical)
        self.mb_geometry = np.asarray(geometry)
        self.blocks = np.asarray(blocks)
        self.n_mbs = len(blocks)
        self.fields = {}
        self._header = {'mesh': {
            side+str(axis)+'_bc': 'periodic' if periodic[axis-1] else 'outflow'
            for axis in (1, 2, 3) for side in ('ix', 'ox')}}

    def data(self, var, mbl=None, mbh=None):
        if var in ('dx_mb', 'dy_mb', 'dz_mb'):
            return xp.full(self.n_mbs, self.spacing['xyz'.index(var[1])])
        assert mbl is not None and mbh is not None, 'unbounded field access'
        assert 0 <= mbl < mbh <= self.n_mbs
        self.calls.append((var, mbl, mbh))
        field = self.blocks if var == 'temp' else self.fields[var]
        return xp.asarray(field[mbl:mbh])

    def owns_meshblock(self, index):
        return 0 <= index < self.n_mbs
