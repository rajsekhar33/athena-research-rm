"""Regressions for block-face gradients: subranges, MPI offsets, refinement, 2D, wrappers."""
from types import SimpleNamespace

import numpy as np
import pytest

from athena_research.core.base import asnumpy, xp
from athena_research.operations import grad_div_curl as G
from boundary_mesh import BoundaryMesh


def _blocked_mesh(global_fields, block_shape=(4, 4, 4), periodic=(True, True, True)):
    """BoundaryMesh whose ``fields`` are the given global (nz, ny, nx) arrays split into blocks."""
    shape = next(iter(global_fields.values())).shape
    mesh = BoundaryMesh(np.zeros(shape), block_shape=block_shape, periodic=periodic)
    bz, by, bx = block_shape
    for name, g in global_fields.items():
        mesh.fields[name] = np.asarray([
            g[z*bz:(z+1)*bz, y*by:(y+1)*by, x*bx:(x+1)*bx]
            for x, y, z, _ in mesh.mb_logical])
    return mesh


def _to_blocks(mesh, g):
    bz, by, bx = mesh.nx3, mesh.nx2, mesh.nx1
    return np.asarray([g[z*bz:(z+1)*bz, y*by:(y+1)*by, x*bx:(x+1)*bx]
                       for x, y, z, _ in mesh.mb_logical])


def _periodic_derivative(mesh, g, axis):
    a = 2 - 'xyz'.index(axis)
    return _to_blocks(mesh, (np.roll(g, -1, a) - np.roll(g, 1, a))
                      / (2 * mesh.spacing['xyz'.index(axis)]))


@pytest.fixture
def periodic_mesh():
    rng = np.random.default_rng(0)
    fields = {n: rng.standard_normal((8, 8, 8)) for n in ('fx', 'fy', 'fz', 'u')}
    return _blocked_mesh(fields), fields


@pytest.mark.parametrize('lo,hi', [(0, 8), (4, 8), (2, 6), (1, 2)])
@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_gradient_original_subrange_uses_global_block_ids(periodic_mesh, lo, hi, axis):
    mesh, g = periodic_mesh
    expected = _periodic_derivative(mesh, g['u'], axis)[lo:hi]
    got = G._gradient_original(mesh, 'u', axis, mbl=lo, mbh=hi)
    np.testing.assert_allclose(asnumpy(got), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('lo,hi', [(0, 8), (4, 8), (2, 6), (1, 2)])
def test_curl_original_batches_match_periodic_curl(periodic_mesh, lo, hi):
    mesh, g = periodic_mesh
    d = lambda f, ax: _periodic_derivative(mesh, g[f], ax)[lo:hi]
    expected = [d('fz', 'y') - d('fy', 'z'), d('fx', 'z') - d('fz', 'x'), d('fy', 'x') - d('fx', 'y')]
    got = G._curl_original(mesh, 'fx', 'fy', 'fz', mbl=lo, mbh=hi)
    for c, e in zip(got, expected):
        np.testing.assert_allclose(asnumpy(c), e, rtol=1e-12, atol=1e-12)


def test_divergence_original_batch_matches_periodic(periodic_mesh):
    mesh, g = periodic_mesh
    expected = sum(_periodic_derivative(mesh, g[f], ax)[2:6] for f, ax in
                   (('fx', 'x'), ('fy', 'y'), ('fz', 'z')))
    got = G._divergence_original(mesh, 'fx', 'fy', 'fz', mbl=2, mbh=6)
    np.testing.assert_allclose(asnumpy(got), expected, rtol=1e-12, atol=1e-12)


def test_gradient_mb_indexes_cell_width_by_global_block():
    # Blocks 4..7 have twice the cell width of blocks 0..3 (per-block dx as
    # with SMR); interior derivative of f = slope * x_cell is the slope.
    mesh = _blocked_mesh({'u': np.zeros((8, 8, 8))})
    dx = np.where(np.arange(mesh.n_mbs) < 4, 1.0, 2.0)
    i = np.arange(mesh.nx1)
    mesh.fields['u'] = np.asarray([np.broadcast_to(3.0 * (i + 0.5) * dx[b], (4, 4, 4))
                                   for b in range(mesh.n_mbs)])
    geom = np.array(mesh.mb_geometry, dtype=float)
    geom[:, 1] = geom[:, 0] + dx * mesh.nx1
    mesh.mb_geometry = geom
    got = G._gradient_mb(mesh, 'u', 4, 8, axis='x', handle_boundaries=False)
    np.testing.assert_allclose(asnumpy(got), 3.0, rtol=1e-12)


def test_gradient_mb_mpi_rank_offsets():
    rng = np.random.default_rng(1)
    g = rng.standard_normal((8, 8, 8))
    full = _blocked_mesh({'u': g})
    rank = _blocked_mesh({'u': g})
    rank.has_full_data = False
    rank.local_mb_start, rank.local_mb_end = 4, 8
    base_data = rank.data

    def local_data(var, mbl=None, mbh=None):
        if var.endswith('_mb'):
            return base_data(var, mbl, mbh)
        assert mbl is not None and 4 <= mbl < mbh <= 8, 'rank-local access only'
        return base_data(var, mbl, mbh)

    rank.data = local_data
    rank.owns_meshblock = lambda m: 4 <= m < 8
    for axis in 'xyz':
        ref = asnumpy(G._gradient_mb(full, 'u', 0, 8, axis=axis))[4:8]
        got = asnumpy(G._gradient_mb(rank, 'u', 4, 8, axis=axis))
        # Faces whose neighbour lives on another rank stay one-sided; every
        # other cell must match the single-process result.
        a = 2 - 'xyz'.index(axis)
        interior = [slice(None)] * 4
        interior[1 + a] = slice(1, -1)
        np.testing.assert_allclose(got[tuple(interior)], ref[tuple(interior)], rtol=1e-12, atol=1e-12)
        if axis in 'xy':  # blocks 4..7 are the z=1 layer: x/y neighbours are on this rank
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def _refined_mesh():
    """Coarse block 0 (level 0, dx=1) with level-1 blocks (dx=0.5) on its x+ side."""
    logical = np.array([(0, 0, 0, 0), (2, 0, 0, 1), (2, 1, 0, 1), (2, 0, 1, 1), (2, 1, 1, 1)])
    geo = np.array([(0, 4, 0, 4, 0, 4), (4, 6, 0, 2, 0, 2), (4, 6, 2, 4, 0, 2),
                    (4, 6, 0, 2, 2, 4), (4, 6, 2, 4, 2, 4)], dtype=float)
    blocks = []
    for g in geo:
        x = g[0] + (np.arange(4) + 0.5) * (g[1] - g[0]) / 4
        blocks.append(np.broadcast_to(x, (4, 4, 4)).copy())
    blocks = np.asarray(blocks)
    widths = {'dx_mb': (geo[:, 1] - geo[:, 0]) / 4, 'dy_mb': (geo[:, 3] - geo[:, 2]) / 4,
              'dz_mb': (geo[:, 5] - geo[:, 4]) / 4}

    def data(var, mbl=None, mbh=None):
        if var in widths:
            return xp.asarray(widths[var])
        return xp.asarray(blocks[mbl:mbh])

    return SimpleNamespace(n_mbs=5, Nx1=8, Nx2=4, Nx3=4, nx1=4, nx2=4, nx3=4,
                           mb_logical=logical, mb_geometry=geo, has_full_data=True,
                           _header={'mesh': {f'ix{i}_bc': 'outflow' for i in (1, 2, 3)}},
                           data=data, owns_meshblock=lambda m: True)


def test_refinement_level_faces_fall_back_to_one_sided():
    mesh = _refined_mesh()
    got = asnumpy(G._gradient_mb(mesh, 'x', 0, 5, axis='x'))
    np.testing.assert_allclose(got, 1.0, rtol=1e-12)
    got = asnumpy(G._gradient_original(mesh, 'x', 'x', mbl=0, mbh=5))
    np.testing.assert_allclose(got, 1.0, rtol=1e-12)


@pytest.mark.parametrize('periodic_z', [True, False])
def test_2d_data_gradients(periodic_z):
    rng = np.random.default_rng(2)
    g = rng.standard_normal((1, 8, 8))
    mesh = _blocked_mesh({'u': g, 'fx': g, 'fy': 2*g, 'fz': 3*g}, block_shape=(1, 4, 4),
                         periodic=(True, True, periodic_z))
    n = mesh.n_mbs
    gz = G._gradient_mb(mesh, 'u', 0, n, axis='z')
    np.testing.assert_array_equal(asnumpy(gz), 0.0)
    gx, gy, gz2 = G._gradient_original(mesh, 'u', mbl=0, mbh=n)
    np.testing.assert_allclose(asnumpy(gx), _periodic_derivative(mesh, g, 'x'), atol=1e-12)
    np.testing.assert_allclose(asnumpy(gy), _periodic_derivative(mesh, g, 'y'), atol=1e-12)
    np.testing.assert_array_equal(asnumpy(gz2), 0.0)
    cx, cy, cz = G._curl_using_gradient_mb(mesh, 'fx', 'fy', 'fz')
    np.testing.assert_allclose(asnumpy(cz), 2*_periodic_derivative(mesh, g, 'x')
                               - _periodic_derivative(mesh, g, 'y'), atol=1e-12)


def test_divergence_mb_with_unmatched_component_names(periodic_mesh, monkeypatch):
    mesh, g = periodic_mesh
    monkeypatch.setattr(G, 'determine_blocks_per_batch', lambda **kw: 1)
    got = G.divergence(mesh, 'fx', 'fy', 'u')
    expected = (_periodic_derivative(mesh, g['fx'], 'x') + _periodic_derivative(mesh, g['fy'], 'y')
                + _periodic_derivative(mesh, g['u'], 'z'))
    np.testing.assert_allclose(asnumpy(got), expected, rtol=1e-12, atol=1e-12)


def test_uniform_dtype_matches_per_block_spacing(periodic_mesh):
    mesh, _ = periodic_mesh
    ref = G._gradient_original(mesh, 'u', mbl=0, mbh=mesh.n_mbs)
    got = G._gradient_original(mesh, 'u', mbl=0, mbh=mesh.n_mbs, dtype='uniform')
    for a, b in zip(got, ref):
        np.testing.assert_allclose(asnumpy(a), asnumpy(b), rtol=1e-12)
