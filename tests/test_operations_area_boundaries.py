"""Generic boundary-culling regressions, with CPU geometry and real CUDA.

CPU tests compile the exact bundled process_cube_phys geometry and emulate
the unchanged CUDA tile walk. An independently padded global field is the
reference. All inputs are tiny synthetic meshes, never production snapshots.
"""
import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from athena_research.core.base import cupy_enabled
from athena_research.operations import area_functions as af
from boundary_mesh import BoundaryMesh

_ORIGINAL_Z_SELECTOR = af.find_interface_z_levels


@pytest.fixture(scope='session')
def cube_area(tmp_path_factory):
    if not shutil.which('g++'):
        pytest.skip('CPU geometry regression requires g++')
    library_path = tmp_path_factory.mktemp('cube_geometry')/'geometry.so'
    source = af._build_cuda_source().split(af._KERNEL_MAIN, 1)[0]
    source = source.replace('__device__', '')
    source = '#include <cmath>\n#include <cfloat>\n#include <iostream>\n' + source
    source += r'''
extern "C" double test_cube(const double *v, double dx, double dy, double dz) {
    Cube cube(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
    return process_cube_phys(cube,dx,dy,dz);
}
'''
    subprocess.run(['g++', '-x', 'c++', '-std=c++17', '-O2', '-shared',
                    '-fPIC', '-o', str(library_path), '-'],
                   input=source, text=True, check=True)
    library = ctypes.CDLL(str(library_path))
    cube = library.test_cube
    cube.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.c_double]*3
    cube.restype = ctypes.c_double

    def evaluate(values, spacing):
        vertices = np.asarray(values, dtype=np.float64)
        return cube(vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    *spacing)
    return evaluate


def tiled_area(log_temp, shape, step, iso, spacing, cube_area):
    total = 0.
    for k, j, i in np.ndindex(*(n//step for n in shape)):
        k, j, i = k*step, j*step, i*step
        vertices = np.asarray([
            log_temp[k,j,i], log_temp[k,j,i+step],
            log_temp[k,j+step,i+step], log_temp[k,j+step,i],
            log_temp[k+step,j,i], log_temp[k+step,j,i+step],
            log_temp[k+step,j+step,i+step], log_temp[k+step,j+step,i],
        ]) - iso
        total += cube_area(vertices, tuple(step*d for d in spacing))
    return total


def reference(mesh, steps, cube_area):
    padded = mesh.logs.copy()
    for axis, periodic in enumerate(mesh.periodic):
        widths = [(0, 0)]*3
        widths[2-axis] = (0, max(steps))
        padded = np.pad(padded, widths, mode='wrap' if periodic else 'edge')
    return np.asarray([tiled_area(padded, mesh.logs.shape, s, 0.,
                                  mesh.spacing, cube_area) for s in steps])


@pytest.fixture(params=['cpu_geometry', 'cuda'])
def area_backend(request, monkeypatch, cube_area):
    if request.param == 'cuda':
        if not cupy_enabled:
            pytest.skip('CUDA backend unavailable')
        assert af._get_module() is not None, 'CUDA tests must not fall back'
    else:
        def kernel(grid, block, args):
            data, out, nz, ny, nx, ghost, step, iso, dz, dy, dx = args
            out[0] += tiled_area(data, (nz, ny, nx), step, iso,
                                 (dx, dy, dz), cube_area)
        monkeypatch.setattr(af, 'xp', np)
        monkeypatch.setattr(af, 'cupy_enabled', True)
        monkeypatch.setattr(af, '_get_module', lambda: object())
        monkeypatch.setattr(af, '_get_kernel_mb', lambda: kernel)
    # Isolate boundary orchestration from the existing interface-slab policy.
    # One unpatched end-to-end slab-selection test is provided below.
    monkeypatch.setattr(af, 'find_interface_z_levels',
                        lambda ad, *a, **k: (np.unique(ad.mb_geometry[:,4]), 0))
    return request.param


def calculate(mesh, steps=(1, 2, 4), mpi_manager=None):
    _, values = af.calc_area_mb_based(mesh, 1., step_sizes=steps,
                                     verbose=False, mpi_manager=mpi_manager)
    assert all(high == low+1 for _, low, high in mesh.calls)
    return values


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_uniform_face_boundary_plane(area_backend, cube_area, axis):
    logs = np.full((8, 8, 8), -.3)
    selection = [slice(None)]*3
    selection[axis] = slice(4, None)
    logs[tuple(selection)] = .3
    mesh = BoundaryMesh(logs)
    assert all(np.min(b) == np.max(b) for b in mesh.blocks)
    measured = calculate(mesh)
    expected = reference(mesh, (1, 2, 4), cube_area)
    np.testing.assert_allclose(measured, expected, rtol=1e-12)
    lengths = [mesh.x3max, mesh.x2max, mesh.x1max]
    planar_area = np.prod([v for i,v in enumerate(lengths) if i != axis])
    if axis != 0:  # x/y periodic: interior plane and wrap plane
        planar_area *= 2
    np.testing.assert_allclose(measured, planar_area, rtol=1e-12)


def test_corner_only_crossing(area_backend, cube_area):
    logs = np.full((8, 8, 8), -.3)
    logs[4:,4:,4:] = .3
    mesh = BoundaryMesh(logs)
    np.testing.assert_allclose(calculate(mesh), reference(mesh, (1,2,4), cube_area),
                               rtol=1e-12)


@pytest.mark.parametrize('shape', [(4,4,4), (8,4,4), (4,8,4),
                                  (4,4,8), (8,8,8)])
def test_random_field_partition_invariance(area_backend, cube_area, shape):
    mesh = BoundaryMesh(np.random.default_rng(100).uniform(-.4,.4,(8,8,8)),
                        block_shape=shape)
    np.testing.assert_allclose(calculate(mesh), reference(mesh, (1,2,4), cube_area),
                               rtol=1e-12)


@pytest.mark.parametrize('warm', [0., .3])
def test_exact_level_is_warm(area_backend, cube_area, warm):
    logs = np.full((8,8,8), -.3)
    logs[4:,:,:] = warm
    mesh = BoundaryMesh(logs)
    actual = calculate(mesh)
    assert np.all(np.isfinite(actual)) and np.all(actual > 0)
    np.testing.assert_allclose(actual, reference(mesh, (1,2,4), cube_area),
                               rtol=1e-12)


@pytest.mark.parametrize('level', [-.3, 0., .3])
def test_constant_fields_have_no_surface(area_backend, level):
    np.testing.assert_array_equal(calculate(BoundaryMesh(np.full((8,8,8),level))), 0.)


def test_single_block_periodic_z(area_backend, cube_area):
    mesh = BoundaryMesh(np.random.default_rng(48).uniform(-.4,.4,(4,8,8)),
                        periodic=(True,True,True))
    np.testing.assert_allclose(calculate(mesh), reference(mesh, (1,2,4), cube_area),
                               rtol=1e-12)


def test_interface_slab_selection_keeps_boundary_plane(area_backend, monkeypatch):
    # Restore the real slab-selection helper patched by the backend fixture.
    monkeypatch.setattr(af, 'find_interface_z_levels', _ORIGINAL_Z_SELECTOR)
    logs = np.full((8,8,8), -.3)
    logs[4:,:,:] = .3
    mesh = BoundaryMesh(logs)
    np.testing.assert_allclose(calculate(mesh), mesh.x1max*mesh.x2max, rtol=1e-12)


def test_mpi_area_matches_replicated_serial_reference(area_backend, cube_area):
    from athena_research.backends.mpi_utils import MPIManager
    mpi = MPIManager()
    if mpi.size < 2:
        pytest.skip('requires mpiexec with two or more ranks')
    mesh = BoundaryMesh(np.random.default_rng(18).uniform(-.4,.4,(8,8,8)))
    actual = calculate(mesh, mpi_manager=mpi)
    np.testing.assert_allclose(actual, reference(mesh, (1,2,4), cube_area),
                               rtol=1e-12)
