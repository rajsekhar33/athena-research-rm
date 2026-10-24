"""Regressions for histogram, profile, average and projection reductions on a
tiny four-meshblock mesh with explicit meshblock-range access."""
import numpy as np
import pytest

from athena_research.core.base import xp, asnumpy
import athena_research.operations.histograms as H
from athena_research.operations.histograms import set_dist, set_dist2d
from athena_research.operations.profiles import set_profile, set_radial
from athena_research.operations.basic_operations import calc_avg, calc_sum
from athena_research.operations.weighted_projection import create_weighted_projection


class StripMesh:
    """Four 4^3 meshblocks along x; domain [0,4]x[0,1]x[0,1], dx=0.25.

    Raw fields are backend arrays and data() returns views of them, like
    AthenaData.data does for cached raw arrays, so in-place writes by a
    reduction are visible to the test."""

    def __init__(self, n_local=None, local_start=0):
        self.n_mbs = 4
        self.nx1 = self.nx2 = self.nx3 = 4
        self.Nx1, self.Nx2, self.Nx3 = 16, 4, 4
        self.x1min, self.x1max = 0.0, 4.0
        self.x2min = self.x3min = 0.0
        self.x2max = self.x3max = 1.0
        self.dist, self.dist2d, self.vert, self.rad = {}, {}, {}, {}
        self.sum, self.min, self.max, self.avg = {}, {}, {}, {}
        self.has_full_data = n_local is None
        self.local_mb_start = local_start
        self.local_mb_end = local_start + (4 if n_local is None else n_local)
        dx = 0.25
        k, j, i = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing='ij')
        x = np.stack([(m*4 + i + 0.5)*dx for m in range(4)])
        y = np.stack([(j + 0.5)*dx]*4)
        z = np.stack([(k + 0.5)*dx]*4)
        f = {'x': x, 'y': y, 'z': z, 'vol': np.full(x.shape, dx**3),
             'ones': np.ones(x.shape), 'velx': x - 2.0,
             'dens': np.where(x < 2, 1.0, 2.0),
             'withnan': np.where(x < 1, np.nan, 3.0),
             'r': np.sqrt(x**2 + y**2 + z**2)}
        self.raw = {key: xp.asarray(val) for key, val in f.items()}
        self.funcs = {}

    def get_local_mb_count(self):
        return self.local_mb_end - self.local_mb_start

    def add_data_func(self, name, func):
        self.funcs[name] = func

    def data(self, var, mbl=None, mbh=None):
        if not isinstance(var, str):
            return var
        assert mbl is not None and mbh is not None, 'unbounded data access'
        if var in self.raw:
            return self.raw[var][mbl:mbh]
        return self.funcs[var](self, mbl, mbh)


def _np(a):
    return np.asarray(asnumpy(a))


# D3: all-negative weight batches must not be skipped.
def test_set_dist_signed_weights_independent_of_batch_size():
    results = []
    for sb in (1, 4):
        ad = StripMesh()
        ad.raw['w'] = (ad.raw['velx'] - 0.5) * ad.raw['vol']
        set_dist(ad, varl=['dens'], scale='', bins=4, weights='w',
                 pdf_range=(0.5, 2.5), simultaneous_blocks=sb, redo=True)
        results.append(_np(ad.dist['dens']['dat']))
    hist, edges = np.histogram(_np(ad.raw['dens']).ravel(), bins=4, range=(0.5, 2.5),
                               weights=_np(ad.raw['w']).ravel())
    expect = hist / np.sum(hist * np.diff(edges))
    np.testing.assert_allclose(results[0], expect)
    np.testing.assert_allclose(results[1], expect)


def test_set_dist2d_signed_weights_independent_of_batch_size():
    results = []
    for sb in (1, 4):
        ad = StripMesh()
        ad.raw['w'] = ad.raw['velx'] * ad.raw['vol']
        set_dist2d(ad, varl2d=[['x', 'y']], scales=['', ''], bins=[16, 4],
                   weights='w', weightnorm=False,
                   pdf_range=[(0.0, 4.0), (0.0, 1.0)], simultaneous_blocks=sb,
                   redo=True)
        results.append(_np(ad.dist2d['x_y']['dat']))
    np.testing.assert_allclose(results[0], results[1])
    assert (results[0][:8] < 0).all()


# D4: non-finite cells get zero weight and cached raw data is not modified.
def test_set_dist_log_of_zero_is_excluded_not_binned_at_one():
    ad = StripMesh()
    x = ad.raw['x']
    ad.raw['q'] = xp.where(x < 1, 0.0, xp.where(x < 2.5, 0.1, 10.0))
    set_dist(ad, varl=['q'], scale='log10', bins=4, weights='vol', redo=True)
    d = ad.dist['q']
    loc, dat = _np(d['loc']), _np(d['dat'])
    width = loc[1] - loc[0]
    # only log10 = -1 (6 of 12 valid cells) and +1 (6 of 12) are populated
    np.testing.assert_allclose(dat * width, [0.5, 0, 0, 0.5], atol=1e-12)
    assert float(d['mean']) == pytest.approx(0.0, abs=1e-12)


def test_set_dist_nan_excluded_and_raw_untouched():
    ad = StripMesh()
    set_dist(ad, varl=['withnan'], scale='', bins=4, weights='vol',
             pdf_range=(0.0, 4.0), redo=True)
    assert np.isnan(_np(ad.raw['withnan'])).sum() == 64
    dat = _np(ad.dist['withnan']['dat'])
    np.testing.assert_allclose(dat, [0, 0, 0, 1.0])


def test_set_dist2d_nan_excluded_and_raw_untouched():
    ad = StripMesh()
    set_dist2d(ad, varl2d=[['withnan', 'dens']], scales=['', ''], bins=4,
               weights='vol', weightnorm=False,
               pdf_range=[(0.0, 4.0), (0.0, 4.0)], redo=True)
    assert np.isnan(_np(ad.raw['withnan'])).sum() == 64
    dat = _np(ad.dist2d['withnan_dens']['dat'])
    # weightnorm=False divides by the total bin area (16); only the finite
    # volume (3 of 4) lands, all at withnan=3
    expect = np.zeros((4, 4))
    expect[3, 1] = 1.0 / 16
    expect[3, 2] = 2.0 / 16
    np.testing.assert_allclose(dat, expect)


def test_calc_avg_excludes_nan_and_leaves_raw_untouched():
    ad = StripMesh()
    calc_avg(ad, varl=['withnan'], weights='vol', redo=True)
    assert np.isnan(_np(ad.raw['withnan'])).sum() == 64
    assert float(_np(ad.avg['withnan'])) == pytest.approx(3.0)


def test_set_radial_excludes_nan_and_leaves_raw_untouched():
    ad = StripMesh()
    set_radial(ad, varl=['withnan'], bins=8, weights='vol', rad_scale='linear',
               radial_range=(0.0, 5.0), redo=True)
    assert np.isnan(_np(ad.raw['withnan'])).sum() == 64
    prof = _np(ad.rad['withnan']['profile'])
    filled = prof != 0
    assert filled.any()
    np.testing.assert_allclose(prof[filled], 3.0)


# D6: the cache must not return a result computed with other weights/scale.
def test_calc_avg_and_sum_cache_respects_weights_and_scale():
    ad = StripMesh()
    calc_avg(ad, varl=['velx'], weights='vol')
    a_vol = float(_np(ad.avg['velx']))
    calc_avg(ad, varl=['velx'], weights='dens')
    a_dens = float(_np(ad.avg['velx']))
    v, d = _np(ad.raw['velx']), _np(ad.raw['dens'])
    assert a_vol == pytest.approx(0.0, abs=1e-12)
    assert a_dens == pytest.approx(np.sum(v * d) / np.sum(d))

    calc_sum(ad, varl=['dens'], weights='vol')
    calc_sum(ad, varl=['dens'], weights='vol', scale='log10')
    expect = np.sum(np.log10(d) * _np(ad.raw['vol']))
    assert float(_np(ad.sum['dens'])) == pytest.approx(expect)


# D7: an x profile after a z profile must be computed, not skipped.
def test_set_profile_axes_do_not_collide():
    ad = StripMesh()
    set_profile(ad, varl=['dens'], axis='z')
    set_profile(ad, varl=['dens'], axis='x')
    assert 'dens_x' in ad.vert and ad.vert['dens_x']['axis'] == 'x'
    np.testing.assert_allclose(_np(ad.vert['dens_x']['profile']),
                               [1.0]*8 + [2.0]*8)
    assert ad.vert['dens']['axis'] == 'z'
    # repeated call is served from the cache
    ad.vert['dens_x']['marker'] = True
    set_profile(ad, varl=['dens'], axis='x')
    assert ad.vert['dens_x'].get('marker') is True


# D8: sigma of a flux variable scales with 4 pi r^2, like its mean.
def test_set_radial_flux_sigma_scales_linearly_with_area():
    ad = StripMesh()
    ad.raw['mflxr'] = ad.raw['velx']
    set_radial(ad, varl=['mflxr'], bins=6, weights='vol', rad_scale='linear',
               radial_range=(0.0, 4.5), redo=True)
    ad.raw['plain'] = ad.raw['velx']
    set_radial(ad, varl=['plain'], bins=6, weights='vol', rad_scale='linear',
               radial_range=(0.0, 4.5), redo=True)
    r = _np(ad.rad['mflxr']['r'])
    area = 4*np.pi*r**2
    np.testing.assert_allclose(_np(ad.rad['mflxr']['profile']),
                               _np(ad.rad['plain']['profile']) * area)
    np.testing.assert_allclose(_np(ad.rad['mflxr']['sigma_r']),
                               _np(ad.rad['plain']['sigma_r']) * area,
                               rtol=1e-8, atol=1e-12)


# D5: numerator/denominator share one pixel grid; window works per block.
def test_projection_of_signed_and_zero_support_fields(monkeypatch):
    monkeypatch.setattr(H, 'determine_blocks_per_batch', lambda **k: 1)
    ad = StripMesh()
    p = create_weighted_projection(ad, 'velx', 'vol', axis='z', redo=True)
    expect = (np.arange(16) + 0.5)*0.25 - 2.0
    np.testing.assert_allclose(p, np.repeat(expect[:, None], 4, axis=1))

    ad = StripMesh()
    ad.raw['s'] = xp.where(ad.raw['x'] < 1, 0.0, 1.0)
    p = create_weighted_projection(ad, 's', 'vol', axis='z', redo=True)
    np.testing.assert_allclose(p[:, 0], [0.0]*4 + [1.0]*12)
    w = ad.dist2d['x_y_s_vol_weighted']['loc1']
    n = ad.dist2d['x_y_s_vol_norm']['loc1']
    np.testing.assert_array_equal(w, n)
    assert w[0] == 0.0 and w[-1] == 4.0


def test_projection_window_on_multiblock_axis_and_cache_key():
    ad = StripMesh()
    plain = create_weighted_projection(ad, 'velx', 'vol', axis='x', redo=True)
    # constant along x within each (y,z) column after windowing too:
    win = create_weighted_projection(ad, 'velx', 'vol', axis='x',
                                     apply_window=True, redo=False)
    assert win.shape == (4, 4)
    # window weights the projection axis symmetrically about the box centre,
    # and velx is antisymmetric about it, so the windowed mean is still 0
    np.testing.assert_allclose(win, 0.0, atol=1e-12)
    np.testing.assert_allclose(plain, 0.0, atol=1e-12)
    # a windowed projection of an asymmetric field differs from the plain one,
    # and redo=False must not hand back the unwindowed cache entry
    ad.raw['q'] = ad.raw['x']**2
    plain = create_weighted_projection(ad, 'q', 'vol', axis='x', redo=True)
    win = create_weighted_projection(ad, 'q', 'vol', axis='x',
                                     apply_window=True, redo=False)
    assert not np.allclose(plain, win)
    again = create_weighted_projection(ad, 'q', 'vol', axis='x', redo=False)
    np.testing.assert_allclose(again, plain)
    smooth = create_weighted_projection(ad, 'q', 'vol', axis='x', redo=False,
                                        smooth_sigma=1.0)
    np.testing.assert_allclose(smooth, plain)  # uniform image stays uniform


# D10: a rank owning no meshblocks must not divide by zero.
class _FakeMPI:
    rank, size = 0, 2

    def __init__(self):
        self.calls = 0

    def allreduce(self, data, op='sum'):
        _FakeMPI.last = self
        self.calls += 1
        return data


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_histograms_on_rank_without_meshblocks(monkeypatch):
    monkeypatch.setattr(H, 'MPI_AVAILABLE', True)
    monkeypatch.setattr(H, 'MPIManager', _FakeMPI, raising=False)
    ad = StripMesh(n_local=0, local_start=4)
    set_dist(ad, varl=['dens'], scale='', bins=4, weights='vol',
             pdf_range=(0.0, 4.0), use_mpi=True, redo=True)
    assert _FakeMPI.last.calls == 1
    set_dist2d(ad, varl2d=[['x', 'y']], scales=['', ''], bins=4, weights='vol',
               pdf_range=[(0.0, 4.0), (0.0, 1.0)], use_mpi=True, redo=True)
    assert _FakeMPI.last.calls == 1


# D11: skewness and kurtosis are standardized.
def test_set_dist_standardized_moments():
    ad = StripMesh()
    set_dist(ad, varl=['x'], scale='', bins=400, weights='vol',
             pdf_range=(0.0, 4.0), redo=True)
    d = ad.dist['x']
    assert float(d['skew']) == pytest.approx(0.0, abs=1e-10)
    # discrete uniform over 16 cell centres: kurtosis = 3/5 (3n^2-7)/(n^2-1)
    n = 16
    assert float(d['kurtosis']) == pytest.approx(0.6*(3*n*n - 7)/(n*n - 1), rel=1e-6)
