"""Regressions for area_functions: coarse sampling on axes the step does not
tile, case-12 cubes with corners exactly on the iso level, interface z-level
selection, cache metadata, and non-periodic x/y padding."""
import ctypes
import itertools
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from athena_research.core.base import asnumpy, cupy_enabled, xp
from athena_research.operations import area_functions as af
from boundary_mesh import BoundaryMesh
from test_operations_area_boundaries import (  # noqa: F401  (fixtures)
    area_backend, calculate, cube_area, reference,
)


def _z_plane(shape, k_hot):
    logs = np.full(shape, -.3)
    logs[k_hot:] = .3
    return logs


# ── D21: coarse grid on axes the step does not tile ──────────────────────────

@pytest.mark.parametrize('step', [8, 16])
def test_coarse_sampling_non_dividing_and_oversized_steps(step):
    # Nx2=12 is not a multiple of 8; 16 exceeds both Nx1=8 and Nx2=12.
    mesh = BoundaryMesh(_z_plane((16, 12, 8), 7), block_shape=(4, 4, 4))
    _, areas = af.calc_area_global_coarse_sampled_offsets(
        mesh, 1., step_sizes=[step], verbose=False)
    np.testing.assert_allclose(areas, mesh.x1max*mesh.x2max, rtol=1e-12)


def test_coarse_sampling_unchanged_when_step_tiles_axes():
    # Steps that tile the periodic axes, surface away from the shorter N-1
    # closure interval of non-periodic z: identical to uniform spacing.
    import skimage.measure
    rng = np.random.default_rng(3)
    k = np.arange(32)[:, None, None]
    logs = .05*(k - 15.3) + .02*rng.uniform(-1, 1, (1, 16, 16))
    mesh = BoundaryMesh(logs)
    for step in (2, 4):
        coarse, spacing, idx = af._build_coarse_sampled_grid_offset(
            mesh, 1., step, verbose=False, return_indices=True)
        verts, faces, _, _ = skimage.measure.marching_cubes(
            coarse, level=0., spacing=spacing, allow_degenerate=False)
        expected = skimage.measure.mesh_surface_area(verts, faces)
        _, areas = af.calc_area_global_coarse_sampled_offsets(
            mesh, 1., step_sizes=[step], offsets='none', verbose=False)
        np.testing.assert_allclose(areas[0], expected, rtol=1e-6)  # skimage float32


# ── D22: case-12 cubes with corners exactly on the iso level ─────────────────

_CORNER_SETS = [(-1., 0., 1.), (-.3, 0., 2.)]


def _all_cubes():
    for values in _CORNER_SETS:
        for combo in itertools.product(values, repeat=8):
            yield combo


@pytest.fixture(scope='module')
def cpu_kernels(tmp_path_factory):
    if not shutil.which('g++'):
        pytest.skip('requires g++')
    library_path = tmp_path_factory.mktemp('mc_case12')/'mc.so'
    source = af._build_cuda_source().split(af._KERNEL_MAIN, 1)[0]
    source = source.replace('__device__', '')
    source = '#include <cmath>\n#include <cfloat>\n#include <iostream>\n' + source
    source += r'''
extern "C" double phys(const double *v) {
    Cube c(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
    return process_cube_phys(c, 1.0, 1.0, 1.0);
}
extern "C" double header(const double *v) {
    Cube c(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
    return process_cube(c);
}
'''
    subprocess.run(['g++', '-x', 'c++', '-std=c++17', '-O2', '-shared',
                    '-fPIC', '-o', str(library_path), '-'],
                   input=source, text=True, check=True)
    library = ctypes.CDLL(str(library_path))
    out = {}
    for name in ('phys', 'header'):
        fn = getattr(library, name)
        fn.argtypes = [ctypes.POINTER(ctypes.c_double)]
        fn.restype = ctypes.c_double
        out[name] = fn
    return out


@pytest.mark.parametrize('name', ['phys', 'header'])
def test_exact_iso_corners_give_finite_cube_area(cpu_kernels, name):
    fn = cpu_kernels[name]
    bad = []
    for combo in _all_cubes():
        v = np.asarray(combo, dtype=np.float64)
        area = fn(v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if not np.isfinite(area):
            bad.append(combo)
    assert not bad, f'{len(bad)} non-finite cubes, e.g. {bad[:3]}'


def test_strict_crossing_cubes_unchanged(cpu_kernels):
    # Without exact-level corners the strict-sign pass is the only pass, so
    # the shared header and the physical copy must agree exactly.
    rng = np.random.default_rng(7)
    for _ in range(2000):
        v = rng.uniform(-1, 1, 8)
        p = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        assert cpu_kernels['phys'](p) == cpu_kernels['header'](p)


@pytest.mark.skipif(not cupy_enabled, reason='CUDA backend unavailable')
def test_exact_iso_corners_finite_in_cuda_area_and_alignment_kernels():
    area_kernel = af._get_kernel_mb()
    align_kernel = af._get_kernel_align_mb()
    assert area_kernel is not None and align_kernel is not None
    pos = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
           (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
    cubes = list(_all_cubes())
    bad = 0
    ones = xp.ones((2, 2, 2))
    for combo in cubes:
        a = np.empty((2, 2, 2))
        for p, v in zip(pos, combo):
            a[p] = v
        d = xp.asarray(a)
        out = xp.zeros(1)
        area_kernel((1, 1, 1), (1, 1, 1),
                    (d, out, np.int32(1), np.int32(1), np.int32(1), np.int32(1),
                     np.int32(1), np.float64(0.), np.float64(1.), np.float64(1.),
                     np.float64(1.)))
        area_out, align_out = xp.zeros(1), xp.zeros(1)
        align_kernel((1, 1, 1), (1, 1, 1),
                     (d, ones, ones, ones, area_out, align_out,
                      np.int32(1), np.int32(1), np.int32(1), np.int32(1),
                      np.int32(1), np.float64(0.), np.float64(1.),
                      np.float64(1.), np.float64(1.)))
        vals = np.concatenate([asnumpy(out), asnumpy(area_out), asnumpy(align_out)])
        bad += int(not np.all(np.isfinite(vals)))
    assert bad == 0


def _bfield_mesh(logs, **kwargs):
    mesh = BoundaryMesh(logs, **kwargs)
    for name, value in (('bcc1', 1.), ('bcc2', 0.), ('bcc3', 0.)):
        mesh.fields[name] = np.full(mesh.blocks.shape, value)
    return mesh


def test_alignment_refuses_nonfinite_results(monkeypatch):
    assert af._alignment_results_finite([1., 0.], [.5, np.nan])
    assert not af._alignment_results_finite([np.nan], [.5])
    assert not af._alignment_results_finite([1.], [np.nan])
    with pytest.raises(RuntimeError, match='non-finite'):
        af._check_alignment_results_finite([1], [np.nan], [.3])


def test_alignment_ignores_cached_nonfinite_entry():
    mesh = _bfield_mesh(_z_plane((8, 8, 8), 4))
    mesh.align_mc = {
        'step_sizes': [1], 'mean_cos2': [np.nan], 'total_area': [np.nan],
        'iso_value': 1., 'T_peak': 1., 'iso_field': 'temp',
        'vector_fields': ['bcc1', 'bcc2', 'bcc3'], 'log_iso_field': True,
        'n_mb_buffer': 1,
    }
    cos2, area = af.calc_bhat_nhat_alignment_mb_based(
        mesh, 1., step_sizes=[1], verbose=False, use_gpu=cupy_enabled)
    np.testing.assert_allclose(area, mesh.x1max*mesh.x2max, rtol=1e-6)
    np.testing.assert_allclose(cos2, 0., atol=1e-12)


# ── D23: interface z-level selection ─────────────────────────────────────────

def _two_interface_mesh():
    # Cold below k=6, hot above, plus a cold layer k=24..27 far above the
    # mean-profile crossing; its lower face sits exactly on a block face.
    logs = _z_plane((32, 8, 8), 6)
    logs[24:28] = -.3
    return BoundaryMesh(logs)


def test_z_selection_includes_far_crossing_levels():
    mesh = _two_interface_mesh()
    with pytest.warns(UserWarning, match='outside the interface window'):
        z_sel, crossing = af.find_interface_z_levels(mesh, 1., n_buffer=1)
    levels = np.sort(np.unique(mesh.mb_geometry[:, 4]))
    selected = np.searchsorted(levels, z_sel)
    # Levels 5 (owns the k=23|24 cubes) and 6 must both be present.
    assert {1, 5, 6}.issubset(set(selected.tolist()))
    np.testing.assert_array_equal(np.diff(selected), 1)


def test_z_selection_for_generic_field_matches():
    mesh = _two_interface_mesh()
    with pytest.warns(UserWarning):
        a, _ = af.find_interface_z_levels(mesh, 1., n_buffer=1)
    with pytest.warns(UserWarning):
        b, _ = af._find_interface_z_levels_for_field(mesh, 'temp', 1., True, n_buffer=1)
    np.testing.assert_array_equal(a, b)


def test_z_selection_unchanged_without_far_crossings():
    mesh = BoundaryMesh(_z_plane((32, 8, 8), 14))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        z_sel, crossing = af.find_interface_z_levels(mesh, 1., n_buffer=1)
    levels = np.sort(np.unique(mesh.mb_geometry[:, 4]))
    lo, hi = max(0, crossing - 1), min(len(levels) - 1, crossing + 1)
    np.testing.assert_array_equal(z_sel, levels[lo:hi + 1])


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_mb_area_counts_far_interfaces():
    mesh = _two_interface_mesh()
    _, areas = af.calc_area_mb_based(mesh, 1., step_sizes=[1, 2, 4],
                                     use_gpu=cupy_enabled, verbose=False)
    np.testing.assert_allclose(areas, 3*mesh.x1max*mesh.x2max, rtol=1e-6)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slab_area_counts_far_interfaces():
    mesh = _two_interface_mesh()
    _, areas = af.calc_areas_all_steps(mesh, 1., step_sizes=[1, 2, 4],
                                       use_gpu=False, verbose=False)
    np.testing.assert_allclose(areas, 3*mesh.x1max*mesh.x2max, rtol=1e-6)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_alignment_counts_far_interfaces():
    logs = _z_plane((32, 8, 8), 6)
    logs[24:28] = -.3
    mesh = _bfield_mesh(logs)
    _, area = af.calc_bhat_nhat_alignment_mb_based(
        mesh, 1., step_sizes=[1], verbose=False, use_gpu=cupy_enabled)
    np.testing.assert_allclose(area, 3*mesh.x1max*mesh.x2max, rtol=1e-6)


# ── area cache metadata ──────────────────────────────────────────────────────

def test_area_cache_records_buffer_and_reads_legacy_entries():
    meta = {'method': af.AREA_METHOD_MARCHING_CUBES, 'method_version': 1,
            'T_ref': 1., 'n_mb_buffer': 1}
    legacy = {'step_sizes': [1], 'areas': [2.], 'T_peak': 1.}
    assert af._area_cache_metadata_matches(legacy, meta)
    assert af._area_cache_metadata_matches(dict(legacy, n_mb_buffer=1), meta)
    assert not af._area_cache_metadata_matches(dict(legacy, n_mb_buffer=2), meta)
    assert af._area_cache_metadata_matches(
        dict(legacy, n_mb_buffer=np.array(1)), meta)


def test_set_area_stores_buffer(tmp_path):
    mesh = BoundaryMesh(_z_plane((16, 8, 8), 6))
    mesh.time = 0.
    result = af.set_area(mesh, 1., str(tmp_path/'a.h5'), step_sizes=[1],
                         use_gpu=cupy_enabled, verbose=False, n_mb_buffer=2)
    assert result['n_mb_buffer'] == 2
    # A legacy cache (no buffer key, T_peak naming) is reused as is.
    mesh.area_mc = {'step_sizes': [1], 'areas': np.array([123.]), 'T_peak': 1.,
                    'method': 'marching_cubes', 'method_version': 1}
    again = af.set_area(mesh, 1., str(tmp_path/'a.h5'), step_sizes=[1],
                        use_gpu=cupy_enabled, verbose=False)
    assert float(np.asarray(again['areas'])[0]) == 123.


# ── non-periodic x/y padding ─────────────────────────────────────────────────

@pytest.mark.parametrize('periodic', [(False, True, False), (True, False, False),
                                      (False, False, True)])
def test_non_periodic_xy_partition_invariance(area_backend, cube_area, periodic):
    mesh = BoundaryMesh(np.random.default_rng(11).uniform(-.4, .4, (8, 8, 8)),
                        periodic=periodic)
    np.testing.assert_allclose(calculate(mesh), reference(mesh, (1, 2, 4), cube_area),
                               rtol=1e-12)


def test_aug_array_replicates_on_non_periodic_axes():
    slab = np.random.default_rng(2).normal(size=(3, 4, 5))
    for px, py in itertools.product((True, False), repeat=2):
        expected = np.pad(slab, ((0, 0), (0, 0), (0, 1)), mode='wrap' if px else 'edge')
        expected = np.pad(expected, ((0, 0), (0, 1), (0, 0)), mode='wrap' if py else 'edge')
        np.testing.assert_array_equal(af._aug_array(slab, (px, py)), expected)


@pytest.mark.parametrize('periodic', [(True, True, False), (False, True, False),
                                      (False, False, False), (True, True, True)])
def test_alignment_padding_matches_global_reference(periodic):
    rng = np.random.default_rng(5)
    mesh = BoundaryMesh(rng.uniform(-.4, .4, (8, 8, 8)), periodic=periodic)
    neighbors = af._find_mc_neighbors(mesh)
    centers = list(range(mesh.n_mbs))
    runs = af._alignment_batch_fields(
        mesh, af._alignment_required_blocks(centers, neighbors), 'temp',
        ('temp', 'temp', 'temp'), log_iso_field=True)
    s = 2
    padded = asnumpy(af._build_alignment_padded_batch(runs, centers, neighbors, s, 'scalar'))
    glob = mesh.logs
    for axis, per in enumerate(periodic):
        widths = [(0, 0)]*3
        widths[2 - axis] = (0, s)
        glob = np.pad(glob, widths, mode='wrap' if per else 'edge')
    for i, m in enumerate(centers):
        bx, by, bz, _ = mesh.mb_logical[m]
        k0, j0, i0 = bz*4, by*4, bx*4
        np.testing.assert_allclose(
            padded[i], glob[k0:k0+4+s, j0:j0+4+s, i0:i0+4+s], rtol=0, atol=1e-12)
        pad_one = af._build_padded_block(mesh, m, neighbors, s, 4, 4, 4)
        np.testing.assert_allclose(pad_one, padded[i], rtol=0, atol=1e-12)
