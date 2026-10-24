"""Regressions for spectrum binning, windowing, slabbing, Helmholtz split and caching."""
import numpy as np
import pytest
from scipy.signal.windows import hamming

from athena_research.core.base import asnumpy, xp
from athena_research.operations import spectra as S


class SpectrumMesh:
    """Uniform periodic grid exposing the refined-data accessors spectra.py uses.

    ``fields`` hold (nz, ny, nx) arrays; an ``xyz`` request crops z with the
    same int() truncation as a cell-centred refined-grid crop.
    """

    def __init__(self, fields, L=(1.0, 1.0, 1.0)):
        self.fields = {k: np.asarray(v, dtype=float) for k, v in fields.items()}
        shape = next(iter(self.fields.values())).shape
        self.Nx3, self.Nx2, self.Nx1 = shape
        self.x1min = self.x2min = self.x3min = 0.0
        self.x1max, self.x2max, self.x3max = L
        self.spectra = {}

    def _crop(self, var, xyz):
        f = self.fields[var]
        if xyz is None:
            return f.copy()
        fac = self.Nx3 / (self.x3max - self.x3min)
        k0 = int((xyz[4] - self.x3min) * fac)
        k1 = int((xyz[5] - self.x3min) * fac)
        return f[k0:k1].copy()

    def get_refined_data(self, var, xyz=None):
        return xp.asarray(self._crop(var, xyz))

    def get_refined_data_mb(self, var=None, xyz=None, use_mpi_gather=True):
        return self._crop(var, xyz)


def _power(fn, ad, var='u', **kw):
    return float(np.sum(asnumpy(fn(ad, var, **kw)[3])))


SCALAR = [S.get_spectrum, lambda ad, var, **kw: S.get_spectrum_mb(ad, var, ndiv=4, **kw)]
HELM = [S.get_spectrum_helmholtz,
        lambda ad, var, **kw: S.get_spectrum_helmholtz_mb(ad, var, ndiv=4, **kw)]


@pytest.mark.parametrize('fn', SCALAR)
@pytest.mark.parametrize('shape,L', [((32, 64, 64), (2.0, 2.0, 1.0)),
                                     ((64, 32, 32), (1.0, 1.0, 2.0)),
                                     ((16, 16, 14), (0.875, 1.0, 1.0)),
                                     ((1, 16, 16), (1.0, 1.0, 1.0))])
def test_noncubic_and_2d_boxes_bin_every_mode(fn, shape, L):
    f = np.random.default_rng(0).standard_normal(shape)
    ad = SpectrumMesh({'u': f}, L)
    assert _power(fn, ad, nbins=64) == pytest.approx(np.sum(f**2), rel=1e-12)


def test_kmax_uses_matching_axis_order():
    # Non-cubic box: Nx1/Lx = Nx2/Ly = Nx3/Lz = 16 with Lx = 0.875.
    ad = SpectrumMesh({'u': np.zeros((16, 16, 14))}, (0.875, 1.0, 1.0))
    kbins = asnumpy(S.get_spectrum(ad, 'u', nbins=17)[1])
    assert kbins[-1] == pytest.approx(16.0)
    kbins_mb = asnumpy(S.get_spectrum_mb(ad, 'u', nbins=17)[1])
    np.testing.assert_allclose(kbins_mb, kbins)


def test_2d_kmax_ignores_single_cell_axis():
    ad = SpectrumMesh({'u': np.zeros((1, 16, 16))}, (1.0, 1.0, 1.0))
    kbins = asnumpy(S.get_spectrum(ad, 'u', nbins=17)[1])
    assert kbins[-1] == pytest.approx(16.0)


@pytest.mark.parametrize('fn', SCALAR)
def test_hamming_window_preserves_power_of_z_uniform_field(fn):
    plane = np.random.default_rng(1).standard_normal((1, 16, 16))
    f = np.repeat(plane, 32, axis=0)
    ad = SpectrumMesh({'u': f})
    got = _power(fn, ad, nbins=64, strat_flag=True, skip=0.0)
    assert got == pytest.approx(np.sum(f**2), rel=1e-12)


@pytest.mark.parametrize('fn', SCALAR)
def test_strat_crop_uses_loaded_plane_count(fn):
    # 64 rows, skip=0.105: the old formula gives 50 planes, the crop has 51.
    f = np.random.default_rng(2).standard_normal((64, 8, 8))
    ad = SpectrumMesh({'u': f})
    crop = ad._crop('u', S._strat_xyz(ad, 0.105))
    assert crop.shape[0] == 51
    w = hamming(51)
    w = w / np.sqrt(np.mean(w**2))
    expected = np.sum((crop * w[:, None, None])**2)
    assert _power(fn, ad, nbins=32, strat_flag=True, skip=0.105) == pytest.approx(expected, rel=1e-12)


def test_log_binning_memory_efficient_matches_standard():
    f = np.random.default_rng(3).standard_normal((16, 16, 16))
    ad = SpectrumMesh({'u': f})
    std = S.get_spectrum(ad, 'u', nbins=24, log_bin_flag=True)
    mb = S.get_spectrum_mb(ad, 'u', nbins=24, log_bin_flag=True, ndiv=4)
    np.testing.assert_allclose(asnumpy(mb[3]), asnumpy(std[3]), rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(asnumpy(mb[2]), asnumpy(std[2]))


def test_log_binning_helmholtz_memory_efficient_runs():
    rng = np.random.default_rng(4)
    ad = SpectrumMesh({c: rng.standard_normal((8, 8, 8)) for c in ('vx', 'vy', 'vz')})
    std = S.get_spectrum_helmholtz(ad, 'v', nbins=12, log_bin_flag=True)
    mb = S.get_spectrum_helmholtz_mb(ad, 'v', nbins=12, log_bin_flag=True, ndiv=4)
    for i in (3, 4):
        np.testing.assert_allclose(asnumpy(mb[i]), asnumpy(std[i]), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('fn', HELM)
@pytest.mark.parametrize('L', [1.0, 3.0e20])
def test_helmholtz_split_is_independent_of_length_units(fn, L):
    n = 16
    x = (np.arange(n) + 0.5) / n
    z3, _, x3 = np.meshgrid(x, x, x, indexing='ij')
    ad = SpectrumMesh({'vx': np.sin(2*np.pi*x3),          # compressive
                       'vy': np.sin(2*np.pi*z3),          # solenoidal
                       'vz': np.zeros((n, n, n))}, (L, L, L))
    r = fn(ad, 'v', nbins=32)
    assert float(np.sum(asnumpy(r[3]))) / n**3 == pytest.approx(0.5, rel=1e-10)
    assert float(np.sum(asnumpy(r[4]))) / n**3 == pytest.approx(0.5, rel=1e-10)


@pytest.mark.parametrize('shape,ndiv', [((32, 30, 32), 4), ((7, 8, 8), 4), ((3, 8, 8), 4),
                                        ((8, 5, 8), 3)])
def test_slabbing_covers_every_row(shape, ndiv):
    rng = np.random.default_rng(5)
    f = rng.standard_normal(shape)
    ad = SpectrumMesh({'u': f})
    std = S.get_spectrum(ad, 'u', nbins=32)
    mb = S.get_spectrum_mb(ad, 'u', nbins=32, ndiv=ndiv)
    np.testing.assert_allclose(asnumpy(mb[3]), asnumpy(std[3]), rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(asnumpy(mb[2]), asnumpy(std[2]))

    vel = SpectrumMesh({c: rng.standard_normal(shape) for c in ('vx', 'vy', 'vz')})
    hs = S.get_spectrum_helmholtz(vel, 'v', nbins=32)
    hm = S.get_spectrum_helmholtz_mb(vel, 'v', nbins=32, ndiv=ndiv)
    for i in (2, 3, 4):
        np.testing.assert_allclose(asnumpy(hm[i]), asnumpy(hs[i]), rtol=1e-10, atol=1e-12)


def test_set_spectrum_cache_respects_settings():
    ad = SpectrumMesh({'u': np.random.default_rng(6).standard_normal((8, 8, 8))})
    S.set_spectrum(ad, ['u'], nbins=16, verbose=False)
    assert len(ad.spectra['u']['spectrum']) == 15
    S.set_spectrum(ad, ['u'], nbins=32, verbose=False)
    assert len(ad.spectra['u']['spectrum']) == 31
    assert ad.spectra['u']['settings']['nbins'] == 32


def test_scalar_and_helmholtz_entries_do_not_alias():
    rng = np.random.default_rng(7)
    ad = SpectrumMesh({c: rng.standard_normal((8, 8, 8)) for c in ('v', 'vx', 'vy', 'vz')})
    S.set_spectrum_helmholtz(ad, 'v', verbose=False)
    assert 'spectrum_sol' in ad.spectra['v']
    S.set_spectrum(ad, ['v'], verbose=False)
    assert 'spectrum' in ad.spectra['v']
    S.set_spectrum_helmholtz(ad, 'v', verbose=False)
    assert 'spectrum_sol' in ad.spectra['v']


class _FakeMPIManager:
    rank, size = 0, 2


@pytest.mark.parametrize('setter,mpi_fn,args', [
    (S.set_spectrum, 'get_spectrum_mb_mpi', (['u'],)),
    (S.set_spectrum_helmholtz, 'get_spectrum_helmholtz_mb_mpi', ('v',)),
])
def test_mpi_failure_is_raised_not_replaced_by_rank_local_spectrum(monkeypatch, setter, mpi_fn, args):
    rng = np.random.default_rng(8)
    ad = SpectrumMesh({c: rng.standard_normal((8, 8, 8)) for c in ('u', 'vx', 'vy', 'vz')})

    def fail(*a, **k):
        raise RuntimeError('mpi spectrum failed')

    monkeypatch.setattr(S, 'MPI_AVAILABLE', True)
    monkeypatch.setattr(S, 'MPIManager', _FakeMPIManager, raising=False)
    monkeypatch.setattr(S, mpi_fn, fail)
    with pytest.raises(RuntimeError, match='mpi spectrum failed'):
        setter(ad, *args, use_mpi=True, verbose=False)
    assert ad.spectra == {}


def test_parseval_verification_keeps_lowest_nonzero_modes(monkeypatch, capsys):
    from athena_research.operations import verify_parseval_theorem as V

    n = 16
    x = (np.arange(n) + 0.5) / n
    z3, y3, x3 = np.meshgrid(x, x, x, indexing='ij')
    fields = {'u': 3.0 + np.sin(2*np.pi*x3),
              'vx': 1.0 + np.sin(2*np.pi*x3),
              'vy': -2.0 + np.sin(2*np.pi*z3),
              'vz': 0.5 + np.cos(2*np.pi*y3),
              'vol': np.ones((n, n, n))}
    ad = SpectrumMesh(fields)
    ad.avg = {}

    def calc_avg(ad, varl, weights='vol', redo=True):
        for v in varl:
            ad.avg[v] = float(np.mean(ad.fields[v]))

    monkeypatch.setattr(V, 'calc_avg', calc_avg)
    pv = V.ParsevalVerification(verbose=False)
    for method in ('standard', 'memory_efficient'):
        res = pv.verify_parseval_scalar(ad, 'u', method=method)
        assert res['spectrum_sum'] == pytest.approx(0.5, rel=1e-10)
        assert res['passed']
        res = pv.verify_helmholtz_consistency(ad, 'v', method=method)
        assert res['total_helmholtz'] == pytest.approx(1.5, rel=1e-10)
        assert res['total_compressive'] == pytest.approx(0.5, rel=1e-10)
