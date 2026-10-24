"""Isotropic two-point correlation functions <A(x) B(x+r)> on meshblock data.

Block pairs are visited once each (i <= j). Within a pair, cells are either
enumerated exhaustively or drawn at random with independent counter-hashed
indices, and random draws are re-weighted to the exhaustive sum, so every
block pair contributes in proportion to its number of cell pairs. Both
orientations A(x)B(x+r) and B(x)A(x+r) enter each radial bin, and same-cell
pairs are excluded.

Stored moment p is sum(w1*w2*(A*B)**p); divide by weight_sum_bin for the
weighted expectation. Weights enter once, not to the power p. With
subtract_mean=True, A and B are replaced by their fluctuations about the
weighted domain means before the products are formed.

Serial or full-data ownership only (every rank holding the whole mesh).
"""
import json

import numpy as np

from ..core.base import xp, asnumpy, cupy_enabled
from . import structure_functions as legacy
from .sf_checkpoint import pair_ordinal, pair_work
from ..utils.batch_processing import determine_blocks_per_batch

CF_SAMPLER_VERSION = 'counter_splitmix64_importance_v3'
CF_ESTIMATOR = 'raw_product_pair_weight_symmetrized_importance_v3'
_CHUNK = 65536


def cf_entry_is_current(entry):
    """True for ad.cf entries written by the current estimator."""
    if not isinstance(entry, dict):
        return False
    metadata = entry.get('metadata', {})
    if not isinstance(metadata, dict):
        return False
    tag = metadata.get('estimator', '')
    if hasattr(tag, 'item'):
        tag = tag.item()
    if isinstance(tag, bytes):
        tag = tag.decode()
    return str(tag) == CF_ESTIMATOR


def _pair_indices(indices, i, j, nblocks, points):
    """Independent counter-hashed draws, stable for a given block pair.

    Successive low LCG bits must not be reused for power-of-two meshblock
    sizes: the second index would then be a function of the first.
    """
    salt = np.uint64((pair_ordinal(i, j, nblocks) * 0xD1B54A32D192ED03) % 2**64)

    def mix(value):
        value = (value ^ (value >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> 27)) * np.uint64(0x94D049BB133111EB)
        return value ^ (value >> 31)
    counter = 2 * indices + salt + np.uint64(0x9E3779B97F4A7C15)
    return mix(counter) % points, mix(counter + np.uint64(1)) % points


def _sample_count(i, j, centers, lengths, periodic, rmin, rmax, samples, minimum):
    """Samples per block pair, falling off as (distance/rmax)**-2."""
    d = np.abs(centers[j] - centers[i])
    for axis in range(3):
        if periodic[axis]:
            d[axis] = min(d[axis], abs(lengths[axis] - d[axis]))
    distance = np.linalg.norm(d)
    value = samples / rmax / (rmax / rmin - 1.) * (samples if distance == 0 else (distance / rmax) ** -2.)
    return int(np.clip(value, minimum, samples))


def _correlation_pair(first, second, i, j, weights, variables, means, edges,
                      lengths, periodic, attempts_per_block, points, order, nblocks, exhaustive):
    nbins = len(edges) - 1
    moments = xp.zeros((len(weights), order, nbins), dtype=xp.float64)
    counts = xp.zeros((len(weights), nbins), dtype=xp.float64)
    sums = xp.zeros_like(counts)
    attempts = points * points if exhaustive else attempts_per_block
    # Random attempts draw ordered (p, q) uniformly from points**2. Rescale to
    # the exhaustive sum over both orientations; within one block each
    # unordered pair is drawn in both orders, hence the extra 1/2.
    scale = 1.0 if exhaustive else points * points / attempts * (0.5 if i == j else 1.0)
    a, b = variables
    for iw, weight in enumerate(weights):
        w1, w2 = first['weights'][weight], second['weights'][weight]
        if bool(xp.any(~xp.isfinite(w1) | (w1 < 0))) or bool(xp.any(~xp.isfinite(w2) | (w2 < 0))):
            raise ValueError('correlation weights must be finite and nonnegative')
        ma, mb = means[iw]
        a1, b1 = first['vars'][a] - ma, first['vars'][b] - mb
        a2, b2 = second['vars'][a] - ma, second['vars'][b] - mb
        for start in range(0, attempts, _CHUNK):
            indices = xp.arange(start, min(start + _CHUNK, attempts), dtype=xp.uint64)
            if exhaustive:
                p, q = indices // points, indices % points
            else:
                p, q = _pair_indices(indices, i, j, nblocks, points)
            p, q = p.astype(xp.int64), q.astype(xp.int64)
            valid = (w1[p] > 0) & (w2[q] > 0)
            if i == j:
                valid &= p < q if exhaustive else p != q
            p, q = p[valid], q[valid]
            if not p.size:
                continue
            rsq = xp.zeros(p.size, dtype=xp.float64)
            for axis, key in enumerate(('x', 'y', 'z')):
                d = xp.abs(second['coords'][key][q] - first['coords'][key][p])
                if periodic[axis]:
                    d = xp.minimum(d, xp.abs(lengths[axis] - d))
                rsq += d * d
            bins = xp.searchsorted(edges, xp.sqrt(rsq), side='right') - 1
            good = (bins >= 0) & (bins < nbins)
            bins, p, q = bins[good], p[good], q[good]
            if not bins.size:
                continue
            pair_weight = w1[p] * w2[q]
            ab = a1[p] * b2[q]
            ba = b1[p] * a2[q]
            if not bool(xp.all(xp.isfinite(ab))) or not bool(xp.all(xp.isfinite(ba))):
                raise ValueError('nonfinite included correlation data')
            counts[iw] += 2 * xp.bincount(bins, minlength=nbins)
            sums[iw] += 2 * scale * xp.bincount(bins, weights=pair_weight, minlength=nbins)
            for power in range(1, order + 1):
                moments[iw, power - 1] += scale * xp.bincount(
                    bins, weights=pair_weight * (ab ** power + ba ** power), minlength=nbins)
    return moments, counts, sums


def _weighted_means(ad, variables, weights, batch):
    """Weighted domain means of each variable, accumulated per meshblock range."""
    totals = np.zeros((len(weights), len(variables)))
    norms = np.zeros(len(weights))
    for lo in range(0, int(ad.n_mbs), batch):
        hi = min(int(ad.n_mbs), lo + batch)
        values = [ad.data(v, lo, hi) for v in variables]
        for iw, weight in enumerate(weights):
            w = ad.data(weight, lo, hi)
            norms[iw] += float(xp.sum(w))
            for iv, v in enumerate(values):
                totals[iw, iv] += float(xp.sum(w * v))
    if np.any(norms <= 0):
        raise ValueError('a correlation weight has no positive support')
    return totals / norms[:, None]


def get_cf(ad, field_a, field_b=None, weights='ones', npairs=1e8, nbins=100,
           max_order=1, log_bin_flag=True, subtract_mean=False,
           simultaneous_blocks=None, nsamples_block_min=1000):
    """Two-point correlation of field_a with field_b (autocorrelation if None).

    Returns a dict with 'r' (bin centres), 'edges', 'cf' of shape
    (n_weights, max_order, nbins) holding raw product sums, 'weight_sum_bin'
    and 'num_bin_points' of shape (n_weights, nbins), and 'means' of shape
    (n_weights, 2) (zeros unless subtract_mean). A string weight drops the
    leading weight axis. cf / weight_sum_bin is the weighted correlation.
    """
    if getattr(ad, 'has_full_data', True) is False:
        raise ValueError('get_cf requires serial/full-data ownership of the mesh')
    field_b = field_a if field_b is None else field_b
    scalar_weight = isinstance(weights, str)
    weights = [weights] if scalar_weight else list(weights)
    if not weights or len(set(weights)) != len(weights):
        raise ValueError('unique nonempty weight list required')
    if not np.isfinite(npairs) or npairs <= 0 or nbins < 2 or not 1 <= max_order <= 10:
        raise ValueError('invalid pair sampling/bin/order settings')
    if nsamples_block_min < 1:
        raise ValueError('minimum sample count must be positive')
    n = int(ad.n_mbs)
    points = int(ad.nx1 * ad.nx2 * ad.nx3)
    if n < 1 or points < 1 or int(np.sqrt(npairs) / n) < 1:
        raise ValueError('pair sampling needs at least one sample per block')
    variables = [field_a, field_b]
    if simultaneous_blocks is not None:
        if int(simultaneous_blocks) != simultaneous_blocks or simultaneous_blocks < 1:
            raise ValueError('batch size must be a positive integer')
        batch = min(n, int(simultaneous_blocks))
    else:
        batch = min(n, determine_blocks_per_batch(n, len(variables) * 3, len(weights), points))
    geom = np.asarray(ad.mb_geometry)
    lengths = np.array([ad.x1max - ad.x1min, ad.x2max - ad.x2min, ad.x3max - ad.x3min])
    cells = (geom[0, 1::2] - geom[0, ::2]) / [ad.nx1, ad.nx2, ad.nx3]
    rmin = float(cells[cells > 0].min())
    rmax = float(np.sqrt(.25 * np.sum(lengths ** 2)))
    if not 0 < rmin < rmax:
        raise ValueError('invalid radial geometry')
    edges, radial = legacy.radial_bin_edges(rmin, rmax, nbins, log_bin_flag)
    means = (_weighted_means(ad, variables, weights, batch) if subtract_mean
             else np.zeros((len(weights), 2)))
    centers = .5 * (geom[:, ::2] + geom[:, 1::2])
    periodic = ['periodic' in str(ad._header['mesh'][f'ix{k}_bc']) for k in (1, 2, 3)]
    samples = min(int(np.sqrt(npairs) / n), points)
    moments = xp.zeros((len(weights), max_order, nbins), dtype=xp.float64)
    counts = xp.zeros((len(weights), nbins), dtype=xp.float64)
    weight_sums = xp.zeros_like(counts)
    cache = {}
    resident = None
    try:
        for a, b, i, j, _ in pair_work(n, batch):
            if resident != (a, b):
                first = list(range(a * batch, min(n, (a + 1) * batch)))
                second = list(range(b * batch, min(n, (b + 1) * batch)))
                for key in list(cache):
                    if key not in first:
                        del cache[key]
                legacy._load_batch_data(ad, cache, first, weights, variables)
                legacy._load_batch_data(ad, cache, second, weights, variables)
                resident = (a, b)
            n2 = _sample_count(i, j, centers, lengths, periodic, rmin, rmax, samples, nsamples_block_min)
            exhaustive = points * points < 5 * samples * n2
            local = _correlation_pair(cache[i], cache[j], i, j, weights, variables, means, edges,
                                      lengths, periodic, samples * n2, points, max_order, n, exhaustive)
            moments += local[0]
            counts += local[1]
            weight_sums += local[2]
    finally:
        cache.clear()
    if cupy_enabled:
        xp.cuda.Stream.null.synchronize()
    result = dict(r=asnumpy(radial), edges=asnumpy(edges), cf=asnumpy(moments),
                  weight_sum_bin=asnumpy(weight_sums), num_bin_points=asnumpy(counts).astype(np.int64),
                  means=means)
    if scalar_weight:
        for key in ('cf', 'weight_sum_bin', 'num_bin_points', 'means'):
            result[key] = result[key][0]
    return result


def set_cf(ad, field_a, field_b=None, varsuf='', redo=False, **kwargs):
    """Compute get_cf and store it in ad.cf[key].

    The key is field_a+varsuf for an autocorrelation and
    field_a+'_'+field_b+varsuf otherwise. A cached entry is reused only if
    it carries the current estimator tag and identical settings.
    """
    field_b_name = field_a if field_b is None else field_b
    key = (field_a if field_b_name == field_a else f'{field_a}_{field_b_name}') + varsuf
    settings = dict(field_a=field_a, field_b=field_b_name, **kwargs)
    settings_json = json.dumps(settings, sort_keys=True, default=str)
    if not hasattr(ad, 'cf') or ad.cf is None:
        ad.cf = {}
    entry = ad.cf.get(key)
    if not redo and cf_entry_is_current(entry):
        stored = entry['metadata'].get('settings_json', '')
        if hasattr(stored, 'item'):
            stored = stored.item()
        if isinstance(stored, bytes):
            stored = stored.decode()
        if stored == settings_json:
            return ad.cf
    result = get_cf(ad, field_a, field_b, **kwargs)
    result['metadata'] = dict(estimator=CF_ESTIMATOR, sampler=CF_SAMPLER_VERSION,
                              settings_json=settings_json)
    ad.cf[key] = result
    return ad.cf
