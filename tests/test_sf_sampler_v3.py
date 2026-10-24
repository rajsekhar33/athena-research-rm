"""Sampler v3: unbiased same-block weights, CPU/GPU draw parity, robust
radial edges and the per-bin increment histogram."""
import numpy as np
import pytest

from athena_research.core.base import xp, asnumpy, cupy_enabled
from athena_research.operations import structure_functions as sfm
from synthetic_mesh import AnalyticMesh


def _block(n, seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, size=(3, n))
    data = rng.normal(size=(2, n))
    return pos, data


def _exact_pair_sums(pos, data, rbins, same_block, pos2=None, data2=None):
    """Exact sum of |delta| over distinct pairs per radial bin (non-periodic)."""
    if same_block:
        i, j = np.triu_indices(pos.shape[1], k=1)
        p1, p2, d1, d2 = pos[:, i], pos[:, j], data[:, i], data[:, j]
    else:
        i, j = np.meshgrid(np.arange(pos.shape[1]), np.arange(pos2.shape[1]), indexing='ij')
        i, j = i.ravel(), j.ravel()
        p1, p2, d1, d2 = pos[:, i], pos2[:, j], data[:, i], data2[:, j]
    r = np.sqrt(np.sum((p2 - p1) ** 2, axis=0))
    b = np.searchsorted(rbins, r, side='right') - 1
    ok = (b >= 0) & (b < len(rbins) - 1)
    return np.array([np.bincount(b[ok], weights=np.abs(d2[v] - d1[v])[ok], minlength=len(rbins) - 1)
                     for v in range(d1.shape[0])])


def _cpu_pair(pos1, data1, pos2, data2, rbins, n_pairs, seed, same, delta_edges=None):
    w1, w2 = np.ones(pos1.shape[1]), np.ones(pos2.shape[1])
    return sfm._compute_sf_cpu_vectorized(
        data1.ravel(), data2.ravel(), data1.shape[0], w1, w2, *pos1, *pos2,
        rbins, 2, 1.0, 1.0, 1.0, False, False, False, n_pairs, seed, same,
        delta_edges=delta_edges)


@pytest.mark.parametrize('n', [2, 3, 5])
def test_same_block_random_weight_is_unbiased(n):
    pos, data = _block(n, n)
    rbins = np.array([0.0, 0.5, 2.0])
    exact = _exact_pair_sums(pos, data, rbins, True)
    n_pairs = 400000
    sf, _, _, _ = _cpu_pair(pos, data, pos, data, rbins, n_pairs, sfm._pair_seed(3, 3), True)
    estimate = sfm._random_pair_weight(n, n, True, n_pairs) * sf[:, 0]
    filled = exact > 0
    np.testing.assert_allclose(estimate[filled], exact[filled], rtol=0.02)


def test_cross_block_random_weight_is_unbiased():
    pos1, data1 = _block(4, 1)
    pos2, data2 = _block(6, 2)
    rbins = np.array([0.0, 0.5, 2.0])
    exact = _exact_pair_sums(pos1, data1, rbins, False, pos2, data2)
    n_pairs = 400000
    sf, _, _, _ = _cpu_pair(pos1, data1, pos2, data2, rbins, n_pairs, sfm._pair_seed(0, 1), False)
    estimate = sfm._random_pair_weight(4, 6, False, n_pairs) * sf[:, 0]
    np.testing.assert_allclose(estimate, exact, rtol=0.02)


def test_pair_seeds_are_unique_beyond_ten_thousand_blocks():
    seeds = {int(sfm._pair_seed(b1, b2)) for b1 in (0, 1, 9999, 10000, 10001) for b2 in (0, 1, 10000, 20001)}
    assert len(seeds) == 20


def test_radial_edges_bin_rmin_and_rmax():
    edges, _ = sfm.radial_bin_edges(0.125, np.sqrt(0.75), 8, True)
    edges = asnumpy(edges)
    r = np.array([0.125, np.sqrt(0.75)])
    bins = np.searchsorted(edges, r, side='right') - 1
    assert list(bins) == [0, 7]
    lin, _ = sfm.radial_bin_edges(0.125, np.sqrt(0.75), 8, False)
    assert np.searchsorted(asnumpy(lin), np.sqrt(0.75), side='right') - 1 == 7


def test_delta_histogram_columns_and_totals():
    edges = np.array([0.1, 1.0, 10.0])
    assert list(sfm._delta_bin_index(np.array([0.0, 0.1, 0.5, 1.0, 10.0, 50.0]), edges)) == [0, 1, 1, 2, 3, 3]
    pos, data = _block(40, 7)
    rbins = np.array([0.0, 0.3, 0.6, 2.0])
    sf, num, _, pdf = _cpu_pair(pos, data, pos, data, rbins, 20000, sfm._pair_seed(0, 0), True,
                                delta_edges=edges)
    assert pdf.shape == (2, 3, 4)
    for v in range(2):
        np.testing.assert_array_equal(pdf[v].sum(axis=1), num)


@pytest.mark.skipif(not cupy_enabled, reason='needs CUDA')
@pytest.mark.parametrize('same', [True, False])
def test_gpu_kernel_draws_match_cpu_twin(same):
    pos1, data1 = _block(37, 11)
    pos2, data2 = (pos1, data1) if same else _block(23, 12)
    rbins = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
    edges = np.array([0.05, 0.5, 2.0])
    n_pairs, seed = 50000, sfm._pair_seed(5, 5 if same else 9)
    sf_c, num_c, w_c, pdf_c = _cpu_pair(pos1, data1, pos2, data2, rbins, n_pairs, seed, same, edges)
    g = lambda a: xp.ascontiguousarray(xp.asarray(a, dtype=xp.float64))
    d1 = g(data1.ravel())
    d2 = d1 if same else g(data2.ravel())
    w1 = g(np.ones(pos1.shape[1]))
    w2 = w1 if same else g(np.ones(pos2.shape[1]))
    c1 = [g(c) for c in pos1]
    c2 = c1 if same else [g(c) for c in pos2]
    nb = len(rbins) - 1
    sf = xp.zeros(2 * 2 * nb)
    num = xp.zeros(nb)
    wsum = xp.zeros(nb)
    pdf = xp.zeros(2 * nb * (len(edges) + 1))
    sfm.structure_function_varlist_kernel(
        ((n_pairs + 255) // 256,), (256,),
        (d1, d2, 2, w1, w2, *c1, *c2, g(rbins), sf, num, wsum, n_pairs,
         pos1.shape[1], pos2.shape[1], 1.0, 1.0, 1.0, nb, 2, False, False, False,
         seed, pdf, g(edges), len(edges)))
    np.testing.assert_array_equal(asnumpy(num), num_c)
    np.testing.assert_allclose(asnumpy(sf).reshape(2, 2, nb), sf_c, rtol=1e-12)
    np.testing.assert_allclose(asnumpy(wsum), w_c, rtol=1e-12)
    np.testing.assert_array_equal(asnumpy(pdf).reshape(2, nb, -1), pdf_c)


def test_get_sf_mb_histogram_option_leaves_moments_unchanged():
    ad = AnalyticMesh(n=4)
    kw = dict(weights='phase', max_order=3, npairs=1e5, nbins=8, nsamples_block_min=10, sparse_data=True)
    plain = sfm.get_sf_mb(ad, ['dens', 'velx'], **kw)
    with_pdf = sfm.get_sf_mb(ad, ['dens', 'velx'], delta_edges=[0.01, 0.1, 1.0, 3.0], **kw)
    for a, b in zip(plain, with_pdf[:4]):
        np.testing.assert_array_equal(asnumpy(a), asnumpy(b))
    pdf = asnumpy(with_pdf[4])
    assert pdf.shape == (2, 8, 5)
    assert np.all(pdf >= 0) and pdf.sum() > 0


def _helmholtz_brute_force(ad, rbins, comp_edges, sol_edges, order=2):
    pos = np.stack([np.asarray(ad.fields[k]).ravel() for k in ('x', 'y', 'z')])
    vel = np.stack([np.asarray(ad.fields[k], dtype=float).ravel() for k in ('velx', 'vely', 'velz')])
    i, j = np.triu_indices(pos.shape[1], k=1)
    d = pos[:, j] - pos[:, i]
    for axis in (0, 1):  # x and y periodic, L = 1
        d[axis] = np.where(np.abs(d[axis]) > 0.5, d[axis] - np.sign(d[axis]), d[axis])
    r = np.sqrt(np.sum(d * d, axis=0))
    dv = vel[:, j] - vel[:, i]
    comp = np.sum(dv * d, axis=0) / r
    sol = np.sqrt(np.maximum(np.sum(dv * dv, axis=0) - comp ** 2, 0.0))
    b = np.searchsorted(rbins, r, side='right') - 1
    ok = (b >= 0) & (b < len(rbins) - 1)
    b, comp, sol = b[ok], comp[ok], sol[ok]
    nb = len(rbins) - 1

    def hist(values, edges):
        cols = np.searchsorted(edges, values, side='right')
        return np.bincount(b * (len(edges) + 1) + cols, minlength=nb * (len(edges) + 1)).reshape(nb, -1)
    return (np.array([np.bincount(b, weights=np.abs(comp) ** p, minlength=nb) for p in range(1, order + 1)]),
            np.array([np.bincount(b, weights=sol ** p, minlength=nb) for p in range(1, order + 1)]),
            hist(comp, comp_edges), hist(sol, sol_edges))


@pytest.mark.skipif(not cupy_enabled, reason='Helmholtz kernels need CUDA')
def test_helmholtz_exhaustive_moments_and_histograms_match_brute_force():
    ad = AnalyticMesh(n=4)
    comp_edges = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
    sol_edges = np.array([0.01, 0.1, 0.5])
    out = sfm.get_sf_helmholtz_mb(ad, 'vel', weights='ones', max_order=2, npairs=1e8, nbins=6,
                                  delta_edges_comp=comp_edges, delta_edges_sol=sol_edges)
    r, sf_comp, sf_sol, wsum, num, pdf_comp, pdf_sol = [asnumpy(x) for x in out]
    rbins = asnumpy(sfm.radial_bin_edges(0.125, np.sqrt(0.75), 6, True)[0])
    want_comp, want_sol, want_hc, want_hs = _helmholtz_brute_force(ad, rbins, comp_edges, sol_edges)
    np.testing.assert_allclose(sf_comp, want_comp, rtol=1e-10, atol=1e-12)
    # sqrt(|dv|^2 - comp^2) cancels for near-parallel increments: looser match.
    np.testing.assert_allclose(sf_sol, want_sol, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(pdf_comp, want_hc)
    np.testing.assert_array_equal(pdf_sol, want_hs)
    np.testing.assert_array_equal(pdf_comp.sum(axis=1), num)


def test_helmholtz_mb_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(sfm, 'cupy_enabled', False)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        sfm.get_sf_helmholtz_mb(AnalyticMesh(n=4), 'vel')


def test_delta_edges_from_dist_covers_support_and_normalises():
    class Dist:
        dist = {'velx': {'loc': np.linspace(-2.95, 2.95, 60), 'dat': np.ones(60) / 6.0}}
    edges, sigma = sfm.delta_edges_from_dist(Dist(), 'velx', n_delta=10)
    assert edges[-1] >= 6.0 and edges[0] > 0 and np.all(np.diff(edges) > 0)
    np.testing.assert_allclose(sigma, 6.0 / np.sqrt(12.0), rtol=1e-3)
    signed, _ = sfm.delta_edges_from_dist(Dist(), 'velx', n_delta=10, signed=True)
    assert signed[0] <= -6.0 and signed[-1] >= 6.0 and np.all(np.diff(signed) > 0)
    norm, s = sfm.delta_edges_from_dist(Dist(), 'velx', normalized_edges=[0.1, 1.0, 3.0])
    np.testing.assert_allclose(norm, np.array([0.1, 1.0, 3.0]) * s)
    with pytest.raises(KeyError):
        sfm.delta_edges_from_dist(Dist(), 'dens')


def _set_sf_kw(**extra):
    return dict(max_order=2, npairs=1e5, nbins=6, nsamples_block_min=10, auto_select=False, **extra)


def test_set_sf_weight_list_keeps_weight_axis_per_variable():
    ad = AnalyticMesh(n=4)
    sfm.set_sf(ad, ['dens', 'velx'], weights=['ones', 'phase'], **_set_sf_kw())
    direct = sfm.get_sf_mb(AnalyticMesh(n=4), ['dens', 'velx'], weights=['ones', 'phase'],
                           max_order=2, npairs=1e5, nbins=6, nsamples_block_min=10)
    assert ad.sf['dens']['sf'].shape == (2, 2, 6)
    np.testing.assert_array_equal(ad.sf['velx']['sf'], asnumpy(direct[1])[:, 1])
    assert ad.sf['dens']['weight_sum_bin'].shape == (2, 6)


def test_set_sf_reuses_only_current_entries_with_same_settings():
    ad = AnalyticMesh(n=4)
    sfm.set_sf(ad, ['dens'], weights='ones', **_set_sf_kw())
    entry = ad.sf['dens']
    assert sfm.sf_entry_is_current(entry)
    sfm.set_sf(ad, ['dens'], weights='ones', **_set_sf_kw())
    assert ad.sf['dens'] is entry
    sfm.set_sf(ad, ['dens'], weights='phase', **_set_sf_kw())
    assert ad.sf['dens'] is not entry
    stale = dict(ad.sf['dens']); stale['sampler'] = 'an_older_sampler'
    ad.sf['dens'] = stale
    sfm.set_sf(ad, ['dens'], weights='phase', **_set_sf_kw())
    assert ad.sf['dens'] is not stale


def test_set_sf_stores_increment_histogram():
    ad = AnalyticMesh(n=4)
    edges = [0.01, 0.1, 1.0, 3.0]
    sfm.set_sf(ad, ['dens', 'velx'], weights='phase', sparse_data=True, delta_edges=edges, **_set_sf_kw())
    e = ad.sf['velx']
    assert e['delta_pdf'].shape == (6, 5)
    np.testing.assert_array_equal(e['delta_edges'], edges)


def test_helmholtz_and_aniso_cannot_overwrite_each_other():
    ad = AnalyticMesh(n=4)
    ad.sf = {'vel_x': {'sf_comp': np.zeros(1), 'sf_sol': np.zeros(1)}}
    with pytest.raises(ValueError, match='helmholtz'):
        sfm.set_sf_aniso_mhd(ad, varsuf='_x')
    ad.sf = {'vel_x': {'sf': np.zeros(1), 'sf_prll': np.zeros(1)}}
    with pytest.raises(ValueError, match='aniso'):
        sfm.set_sf_helmholtz(ad, 'vel', varsuf='_x')


def _scalar_brute_force(ad, var, weight, rbins, edges, order=2):
    pos = np.stack([np.asarray(ad.fields[k]).ravel() for k in ('x', 'y', 'z')])
    q = np.asarray(ad.fields[var], dtype=float).ravel()
    w = np.asarray(ad.fields[weight], dtype=float).ravel()
    i, j = np.triu_indices(q.size, k=1)
    d = np.abs(pos[:, j] - pos[:, i])
    d[:2] = np.minimum(d[:2], 1.0 - d[:2])
    r = np.sqrt(np.sum(d * d, axis=0))
    b = np.searchsorted(rbins, r, side='right') - 1
    ok = (b >= 0) & (b < len(rbins) - 1) & (w[i] != 0) & (w[j] != 0)
    b, i, j = b[ok], i[ok], j[ok]
    diff = np.abs(w[j] * q[j] - w[i] * q[i])
    nb = len(rbins) - 1
    cols = np.searchsorted(edges, diff, side='right')
    return (np.array([np.bincount(b, weights=diff ** p, minlength=nb) for p in range(1, order + 1)]),
            np.bincount(b, weights=w[i] + w[j], minlength=nb),
            np.bincount(b * (len(edges) + 1) + cols, minlength=nb * (len(edges) + 1)).reshape(nb, -1))


@pytest.mark.parametrize('sparse', [False, True])
def test_use_exhaustive_matches_brute_force(sparse):
    ad = AnalyticMesh(n=4)
    edges = np.array([0.05, 0.3, 1.0, 2.0])
    r, sf, wsum, num, pdf = sfm.get_sf_mb(ad, ['dens'], weights='phase', max_order=2, npairs=1e8, nbins=6,
                                          sparse_data=sparse, use_exhaustive=True, delta_edges=edges)
    rbins = asnumpy(sfm.radial_bin_edges(0.125, np.sqrt(0.75), 6, True)[0])
    want_sf, want_w, want_pdf = _scalar_brute_force(ad, 'dens', 'phase', rbins, edges)
    np.testing.assert_allclose(asnumpy(sf)[0], want_sf, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(asnumpy(wsum), want_w, rtol=1e-12)
    np.testing.assert_allclose(asnumpy(pdf)[0], want_pdf, rtol=1e-12)
