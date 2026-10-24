"""Correlation functions against a brute-force sum over all distinct cell pairs."""
import numpy as np
import pytest

from athena_research.operations.correlation_functions import (
    get_cf, set_cf, cf_entry_is_current, CF_ESTIMATOR)
from synthetic_mesh import AnalyticMesh


def brute_force(ad, a, b, weight, edges, subtract_mean=False):
    pos = np.stack([np.asarray(ad.fields[k]).ravel() for k in ('x', 'y', 'z')], axis=1)
    va = np.asarray(ad.fields[a], dtype=float).ravel()
    vb = np.asarray(ad.fields[b], dtype=float).ravel()
    w = np.asarray(ad.fields[weight], dtype=float).ravel()
    if subtract_mean:
        va = va - np.sum(w * va) / np.sum(w)
        vb = vb - np.sum(w * vb) / np.sum(w)
    i, j = np.triu_indices(len(w), k=1)
    d = np.abs(pos[j] - pos[i])
    d[:, :2] = np.minimum(d[:, :2], 1.0 - d[:, :2])  # x, y periodic with L = 1
    bins = np.searchsorted(edges, np.sqrt(np.sum(d * d, axis=1)), side='right') - 1
    keep = (bins >= 0) & (bins < len(edges) - 1) & (w[i] > 0) & (w[j] > 0)
    i, j, bins = i[keep], j[keep], bins[keep]
    pw = w[i] * w[j]
    n = len(edges) - 1
    return (np.bincount(bins, weights=pw * (va[i] * vb[j] + vb[i] * va[j]), minlength=n),
            2 * np.bincount(bins, weights=pw, minlength=n),
            2 * np.bincount(bins, minlength=n))


@pytest.fixture
def ad():
    return AnalyticMesh(n=4)


@pytest.mark.parametrize('a,b,weight,subtract', [
    ('dens', None, 'ones', False),
    ('dens', 'velx', 'ones', False),
    ('dens', 'velx', 'phase', True),
])
def test_exhaustive_matches_brute_force(ad, a, b, weight, subtract):
    # 64 cells per block: every block pair is small enough to enumerate fully.
    out = get_cf(ad, a, b, weights=weight, npairs=1e8, nbins=12, subtract_mean=subtract)
    cf, wsum, count = brute_force(ad, a, a if b is None else b, weight, out['edges'], subtract)
    np.testing.assert_array_equal(out['num_bin_points'], count)
    np.testing.assert_allclose(out['weight_sum_bin'], wsum, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out['cf'][0], cf, rtol=1e-11, atol=1e-11)


def test_random_sampling_is_unbiased(ad):
    big = AnalyticMesh(n=8)
    out = get_cf(big, 'dens', weights='ones', npairs=64 * 64 * 64, nbins=8, nsamples_block_min=1)
    cf, wsum, _ = brute_force(big, 'dens', 'dens', 'ones', out['edges'])
    filled = wsum > 0.01 * wsum.sum()
    np.testing.assert_allclose(out['weight_sum_bin'].sum(), wsum.sum(), rtol=0.02)
    np.testing.assert_allclose((out['cf'][0] / out['weight_sum_bin'])[filled],
                               (cf / wsum)[filled], rtol=0.03)


def test_weight_list_keeps_weight_axis(ad):
    out = get_cf(ad, 'dens', weights=['ones', 'phase'], npairs=1e8, nbins=6, max_order=2)
    assert out['cf'].shape == (2, 2, 6)
    assert out['weight_sum_bin'].shape == (2, 6)


def test_set_cf_reuses_only_matching_current_entries(ad):
    set_cf(ad, 'dens', nbins=6, npairs=1e8)
    entry = ad.cf['dens']
    assert cf_entry_is_current(entry) and entry['metadata']['estimator'] == CF_ESTIMATOR
    assert set_cf(ad, 'dens', nbins=6, npairs=1e8)['dens'] is entry
    assert set_cf(ad, 'dens', nbins=8, npairs=1e8)['dens'] is not entry
    set_cf(ad, 'dens', 'velx', varsuf='_hot', nbins=6, npairs=1e8)
    assert 'dens_velx_hot' in ad.cf


def test_rejects_distributed_data():
    with pytest.raises(ValueError, match='full-data'):
        get_cf(AnalyticMesh(n=4, rank=0, size=2), 'dens')
