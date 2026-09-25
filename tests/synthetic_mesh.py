"""Small analytic mesh with explicit global meshblock access for regressions."""
import numpy as np
from athena_research.core.base import xp


class AnalyticMesh:
    def __init__(self, n=4, rank=0, size=1):
        self.nx1 = self.nx2 = self.nx3 = n
        self.Nx1 = self.Nx2 = self.Nx3 = 2 * n
        self.x1min = self.x2min = self.x3min = 0.0
        self.x1max = self.x2max = self.x3max = 1.0
        self.mb_logical = np.array([(i, j, k, 0) for k in range(2)
                                    for j in range(2) for i in range(2)])
        self.mb_geometry = np.array([(i/2, (i+1)/2, j/2, (j+1)/2, k/2, (k+1)/2)
                                     for i, j, k, _ in self.mb_logical])
        self.n_mbs = 8
        self.has_full_data = size == 1
        q, rem = divmod(self.n_mbs, size)
        self.local_mb_start = rank*q + min(rank, rem)
        self.local_mb_end = self.local_mb_start + q + (rank < rem)
        self._header = {'mesh': {'ix1_bc': 'periodic', 'ox1_bc': 'periodic',
                                'ix2_bc': 'periodic', 'ox2_bc': 'periodic',
                                'ix3_bc': 'outflow', 'ox3_bc': 'outflow'}}
        coords = []
        for i, j, k, _ in self.mb_logical:
            z, y, x = np.meshgrid((k+(np.arange(n)+.5)/n)/2,
                                 (j+(np.arange(n)+.5)/n)/2,
                                 (i+(np.arange(n)+.5)/n)/2, indexing='ij')
            coords.append((x, y, z))
        x, y, z = np.moveaxis(np.array(coords), 1, 0)
        self.fields = {'x': x, 'y': y, 'z': z, 'scalar': z,
                       'vx': np.zeros_like(z), 'vy': np.zeros_like(z),
                       'vz': np.ones_like(z), 'ones': np.ones_like(z),
                       'dens': 1+x+2*y+3*z, 'phase': (z > .3).astype(float),
                       'velx': x, 'vely': y, 'velz': z,
                       'bcc1': np.zeros_like(z), 'bcc2': np.zeros_like(z),
                       'bcc3': np.ones_like(z)}

    def data(self, var, mbl=None, mbh=None, **kwargs):
        assert mbl is not None and mbh is not None, 'unbounded data access'
        assert self.local_mb_start <= mbl <= mbh <= self.local_mb_end
        return xp.asarray(self.fields[var][mbl:mbh])
