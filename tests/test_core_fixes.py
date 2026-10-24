"""Regressions for core/, utils/ and backends/ fixes (I/O, MPI-local access,
derived fields). Tiny synthetic data only."""
import os
import struct
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from athena_research.core.athena_data import AthenaData
from athena_research.core.base import asnumpy, cupy_enabled, xp
from athena_research.core import io_utils
from athena_research.core.io_utils import (load_dict_from_hdf5, read_binary,
                                           save_dict_to_hdf5)


def _header(N, nx1, units=True):
    lines = ['<mesh>', f'nx1={N}', f'nx2={N}', f'nx3={N}', 'nghost=2',
             'x1min=-1', 'x1max=1', 'x2min=-1', 'x2max=1', 'x3min=-1', 'x3max=1',
             '<meshblock>', f'nx1={nx1}', f'nx2={N}', f'nx3={N}',
             '<hydro>', 'gamma=1.6666666666666667', 'eos=ideal']
    if units:
        lines += ['<units>', 'length_cgs=3.0857e18', 'mass_cgs=1e30',
                  'time_cgs=3.0857e13', 'mu=0.6']
    return lines


def make_binary(N=8, nb=4, units=True):
    """filedata dict as read_binary returns: nb blocks split along x."""
    nx1 = N // nb
    rng = np.random.default_rng(0)
    mb = {v: rng.random((nb, N, N, nx1)) + 0.5
          for v in ['dens', 'eint', 'velx', 'vely', 'velz']}
    geo = [[-1 + 2*b/nb, -1 + 2*(b+1)/nb, -1, 1, -1, 1] for b in range(nb)]
    return dict(n_mbs=nb, mb_logical=np.array([[b, 0, 0, 0] for b in range(nb)]),
                mb_geometry=np.array(geo), var_names=list(mb), mb_data=mb,
                header=_header(N, nx1, units), time=0.0, cycle=0)


def make_ad(rank=0, size=1, filename='synth.00000.bin', **kw):
    ad = AthenaData()
    if size > 1:
        ad.set_mpi_distribution(rank, size)
    ad.filename = filename
    ad._load_from_binary(make_binary(**kw))
    ad.config()
    return ad


def write_bin(path, nmb=4, nx=(2, 8, 8), N=8, ghost=False):
    """Minimal AthenaK .bin (version 1.1, float32 data) with nmb x-blocks."""
    ng = 2
    vars_ = ['dens', 'eint', 'velx', 'vely', 'velz']
    par = '\n'.join(_header(N, nx[0])) + '\n'
    pre = ("Athena binary output version=1.1\n  size of preheader=5\n  time=1.5e+00\n"
           "  cycle=3\n  size of location=8\n  size of variable=4\n"
           f"  number of variables={len(vars_)}\n  variables:  {'  '.join(vars_)}  \n")
    rng = np.random.default_rng(1)
    truth = []
    g = ng if ghost else 0
    with open(path, 'wb') as f:
        f.write(pre.encode())
        f.write(f"  header offset={len(par)}\n".encode())
        f.write(par.encode())
        for m in range(nmb):
            f.write(struct.pack('6i', ng - g, ng + nx[0] - 1 + g, ng, ng + nx[1] - 1,
                                ng, ng + nx[2] - 1))
            f.write(struct.pack('4i', m, 0, 0, 0))
            f.write(struct.pack('6d', -1 + 2*m/nmb, -1 + 2*(m+1)/nmb, -1, 1, -1, 1))
            d = rng.random((len(vars_), nx[2], nx[1], nx[0] + 2*g)).astype('f4') + 0.5
            truth.append(d)
            f.write(d.tobytes())
    return truth


# --------------------------------------------------------------- C1 / MPI-local
def test_no_range_derived_fields_are_rank_local():
    full = make_ad()
    ad = make_ad(rank=1, size=4)
    assert ad.data('dens').shape == ad.data('x').shape == (1, 8, 8, 2)
    np.testing.assert_allclose(asnumpy(ad.data('x')), asnumpy(full.data('x'))[1:2])
    np.testing.assert_allclose(asnumpy(ad.data('mass')), asnumpy(full.data('mass'))[1:2])
    np.testing.assert_allclose(asnumpy(ad.data('velr')), asnumpy(full.data('velr'))[1:2])


def test_get_refined_data_places_rank_local_blocks():
    full = asnumpy(make_ad().get_refined_data('x'))
    part = asnumpy(make_ad(rank=1, size=4).get_refined_data('x'))
    np.testing.assert_allclose(part[:, :, 2:4], full[:, :, 2:4])
    assert not part[:, :, :2].any() and not part[:, :, 4:].any()


# ---------------------------------------------------------------------- C2
def test_allreduce_min_max_arrays_single_rank():
    from athena_research.backends.mpi_utils import MPIManager
    m = MPIManager()
    a = np.array([3.0, -1.0, 2.0])
    np.testing.assert_array_equal(m.allreduce(a, 'min'), a)
    assert m.allreduce(np.float32(2.5), 'max') == np.float32(2.5)
    with pytest.raises(ValueError, match='unknown reduction op'):
        m.allreduce(a, 'mean')


# ---------------------------------------------------------------------- C3
@pytest.mark.parametrize('value', [np.int32(7), np.uint64(3), np.float16(1.0),
                                   np.complex128(1j), np.str_('abc'), None, {}, True,
                                   np.arange(3, dtype=np.int64)])
def test_hdf5_values_round_trip(tmp_path, value):
    path = str(tmp_path / 'rt.h5data')
    save_dict_to_hdf5({'sf': {'k': value}}, path)
    out = load_dict_from_hdf5(path)['sf']['k']
    if value is None:
        assert out is None
    elif isinstance(value, dict):
        assert out == {}
    elif isinstance(value, np.ndarray):
        assert out.dtype == value.dtype and np.array_equal(out, value)
    else:
        assert out == value
        if isinstance(value, np.generic) and not isinstance(value, np.str_):
            assert out.dtype == value.dtype


def test_hdf5_int_keys_and_slash_keys(tmp_path):
    path = str(tmp_path / 'k.h5data')
    save_dict_to_hdf5({'sf': {5: {'a': np.ones(2)}, 6: np.zeros(1)}}, path)
    out = load_dict_from_hdf5(path)['sf']
    assert set(out) == {'5', '6'} and np.array_equal(out['5']['a'], np.ones(2))
    with pytest.raises(ValueError, match="'/'"):
        save_dict_to_hdf5({'sf': {'a/b': 1.0}}, path)


def test_hdf5_unsupported_value_raises(tmp_path):
    with pytest.raises(TypeError, match='unsupported type set'):
        save_dict_to_hdf5({'sf': {'s': {1, 2}}}, str(tmp_path / 'u.h5data'))


@pytest.mark.skipif(not cupy_enabled, reason='needs CuPy')
def test_hdf5_cupy_arrays_keep_dtype(tmp_path):
    path = str(tmp_path / 'c.h5data')
    big = 2**24 + 1
    save_dict_to_hdf5({'n': xp.asarray([big], dtype=xp.int64),
                       'f': xp.asarray([0.1], dtype=xp.float64)}, path)
    out = load_dict_from_hdf5(path)
    assert out['n'].dtype == np.int64 and out['n'][0] == big
    assert out['f'].dtype == np.float64 and out['f'][0] == 0.1


def test_ad_save_skips_runtime_objects_but_keeps_products(tmp_path):
    ad = make_ad()
    ad.dist['dens'] = {'dat': np.ones(3), 'count': np.int32(5), 'meta': None, 'tag': 'v2'}
    path = str(tmp_path / 'synth.00007.h5data')
    ad.save(path)
    back = AthenaData().load(path)
    assert sorted(back.dist['dens']) == ['count', 'dat', 'meta', 'tag']
    assert back.dist['dens']['meta'] is None and back.dist['dens']['count'] == 5


# ---------------------------------------------------------------------- C4
def test_failed_save_leaves_existing_file_intact(tmp_path, monkeypatch):
    path = str(tmp_path / 'cache.00001.h5data')
    save_dict_to_hdf5({'sum': {'dens': 1.0}}, path)
    real = io_utils._recursively_save_dict_contents_to_group

    def boom(h5file, p, dic, exclude_keys=None):
        h5file['partial'] = 1.0
        raise RuntimeError('disk full')
    monkeypatch.setattr(io_utils, '_recursively_save_dict_contents_to_group', boom)
    with pytest.raises(RuntimeError):
        save_dict_to_hdf5({'sum': {'dens': 2.0}}, path)
    monkeypatch.setattr(io_utils, '_recursively_save_dict_contents_to_group', real)
    assert load_dict_from_hdf5(path) == {'sum': {'dens': 1.0}}
    assert os.listdir(tmp_path) == ['cache.00001.h5data']


def test_checkpoint_pattern_temp_then_replace(tmp_path):
    """Caller-managed temp file + os.replace (checkpoint-style save)."""
    import tempfile
    final = tmp_path / 'snap.00003.h5data'
    ad = make_ad()
    ad.sum['dens'] = 4.0
    fd, temporary = tempfile.mkstemp(prefix=final.stem + '.checkpoint-',
                                     suffix='.h5data', dir=tmp_path)
    os.close(fd)
    ad.save(filename=temporary)
    os.replace(temporary, final)
    assert AthenaData().load(str(final)).sum['dens'] == 4.0
    assert os.listdir(tmp_path) == [final.name]


# ------------------------------------------------------------------ C5 / C6
def test_read_binary_reads_only_requested_blocks(tmp_path):
    path = str(tmp_path / 't.00001.bin')
    truth = write_bin(path)
    fd = read_binary(path, mb_range=(1, 3))
    assert fd['n_mbs'] == 4 and fd['mb_data_start'] == 1 and len(fd['mb_geometry']) == 4
    assert len(fd['mb_data']['dens']) == 2
    np.testing.assert_array_equal(fd['mb_data']['velx'][1], truth[2][2])
    assert fd['mb_data']['dens'][0].dtype == np.float64


def test_distributed_binary_load_keeps_only_local_blocks(tmp_path):
    path = str(tmp_path / 't.00001.bin')
    truth = write_bin(path)
    ad = AthenaData()
    ad.set_mpi_distribution(1, 2)
    ad.load(path)
    assert (ad.local_mb_start, ad.local_mb_end) == (2, 4)
    assert ad.data_raw['dens'].shape == (2, 8, 8, 2)
    np.testing.assert_array_equal(ad.data_raw['dens'][0], truth[2][0])
    assert 'mb_data' not in ad.__dict__ and 'mb_data' not in ad.binary
    serial = AthenaData().load(path)
    assert serial.data_raw['dens'].shape == (4, 8, 8, 2)


def test_ghost_zone_binary_rejected(tmp_path):
    path = str(tmp_path / 'g.00001.bin')
    write_bin(path, ghost=True)
    with pytest.raises(ValueError, match='Ghost-zone and sliced'):
        AthenaData().load(path)


def test_sliced_binary_rejected():
    b = make_binary()
    for v in b['mb_data']:
        b['mb_data'][v] = b['mb_data'][v][:, :1]
    ad = AthenaData()
    ad.filename = 'synth.00000.bin'
    with pytest.raises(ValueError, match='Ghost-zone and sliced'):
        ad._load_from_binary(b)


# ----------------------------------------------------------------- C7 / C8
def test_moved_h5data_reports_its_own_identity_and_units(tmp_path):
    ad = make_ad(filename='/orig/dir/synth.00007.bin')
    ad.save(str(tmp_path / 'synth.00007.h5data'))
    os.makedirs(tmp_path / 'moved')
    moved = str(tmp_path / 'moved' / 'synth.00008.h5data')
    os.replace(tmp_path / 'synth.00007.h5data', moved)
    b = AthenaData().load(moved)
    assert (b.filename, b.num, b.path) == (moved, 8, str(tmp_path / 'moved'))
    assert b.units is not None and b.units.length_cgs == pytest.approx(3.0857e18)
    assert b.units.mu == pytest.approx(0.6)


def test_legacy_h5data_without_units_params_rebuilds_units(tmp_path):
    path = str(tmp_path / 'old.00004.h5data')
    with h5py.File(path, 'w') as f:
        for k, v in {'length_cgs': '2.0', 'mass_cgs': '3.0', 'time_cgs': '4.0', 'mu': '0.5'}.items():
            f[f'_header/units/{k}'] = v
        f['_header/mesh/nx1'] = '8'
        f['gamma'] = 1.4
        f['sum/dens'] = 8.0
        f['filename'] = '/somewhere/else.00001.bin'
    b = AthenaData().load(path)
    assert b.num == 4 and b.sum['dens'] == 8.0
    assert b.units.length_cgs == 2.0 and b.units.gamma == 1.4


def test_snapshot_number_required():
    with pytest.raises(ValueError, match='snapshot number'):
        make_ad(filename='/x/merged.vtk')
    ad = AthenaData(num=12)
    ad.filename = '/x/merged.vtk'
    ad._load_from_binary(make_binary())
    ad.config()
    assert ad.num == 12
    assert make_ad(filename='/x/p.00031.h5data.bak').num == 31


# ---------------------------------------------------------- C9 / C10 / C12
def test_derived_cache_respects_kwargs():
    ad = make_ad()
    ad.cache_data_funcs = True
    temp = asnumpy(ad.data('temp_cgs'))
    t = float(np.median(temp))
    from athena_research.core import data_functions
    # Default threshold (a branch's problem physics may set it), cached.
    assert float(ad.data('Thotbool').mean()) == pytest.approx(float(np.mean(temp >= data_functions.Thot)))
    assert float(ad.data('Thotbool', T_hot=t).mean()) == pytest.approx(0.5)
    assert float(ad.data('Thotbool', T_hot=1e30).mean()) == 0.0
    assert float(ad.data('Thotbool', 0, 4, T_hot=1e30).mean()) == 0.0


def test_vphi_uses_cylindrical_radius():
    ad = make_ad()
    x, y = asnumpy(ad.data('x')), asnumpy(ad.data('y'))
    vx, vy = asnumpy(ad.data('velx')), asnumpy(ad.data('vely'))
    np.testing.assert_allclose(asnumpy(ad.data('vphi')), (x*vy - y*vx)/np.sqrt(x**2 + y**2))


def test_nrho_meshblock_range_matches_full_domain():
    full = asnumpy(make_ad().data('nrho'))
    ad = make_ad()
    np.testing.assert_allclose(asnumpy(ad.data('nrho', 1, 3)), full[1:3])
    with pytest.raises(ValueError, match='does not accept'):
        ad.data('nrho_hot', T_hot=1.0)


# ---------------------------------------------------------------------- C13
def test_units_block_parsing():
    assert make_ad().units.length_cgs == 3.0857e18
    assert make_ad(units=False).units.length_cgs == 1.0
    b = make_binary()
    b['header'] = [l if not l.startswith('mu=') else 'mu=abc' for l in b['header']]
    ad = AthenaData()
    ad.filename = 'synth.00000.bin'
    with pytest.raises(ValueError, match='not a number'):
        ad._load_from_binary(b)
    b['header'] = [l for l in b['header'] if not l.startswith('mu=')]
    with pytest.raises(ValueError, match="no 'mu'"):
        AthenaData()._load_from_binary(b)


# ---------------------------------------------------------------------- C14
def test_finer_neighbours_in_plus_direction():
    from athena_research.utils.meshblock_utils import find_neighbour_blocks
    logi = [[0, 0, 0, 0]] + [[x, y, z, 1] for z in (0, 1) for y in (0, 1) for x in (2, 3)]
    ad = SimpleNamespace(n_mbs=len(logi), mb_logical=np.array(logi), Nx1=16, nx1=8,
                         Nx2=8, nx2=8, Nx3=8, nx3=8,
                         _header={'mesh': {'ix1_bc': 'outflow', 'ix2_bc': 'outflow',
                                           'ix3_bc': 'outflow'}})
    assert sorted(logi[i][0] for i in find_neighbour_blocks(ad, 0, 'x+')) == [2] * 4
    ad._header['mesh']['ix1_bc'] = 'periodic'
    # Periodic wrap: x- of the coarse root block reaches the children with x=3.
    assert sorted(logi[i][0] for i in find_neighbour_blocks(ad, 0, 'x-')) == [3] * 4


# ---------------------------------------------------------------------- C15
def test_read_athenak_particle_vtk(tmp_path):
    from athena_research.core.particle_io import read_particle_vtk
    n = 3
    path = tmp_path / 'p.00001.vtk'
    with open(path, 'wb') as f:
        f.write(b"# vtk DataFile Version 2.0\n# AthenaK particle data at time= 1.5  "
                b"nranks= 1  cycle=7  variables=prtcl_all\nBINARY\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"\nPOINTS {n} float\n".encode())
        f.write(np.arange(9, dtype='>f4').tobytes())
        f.write(f"\n\nPOINT_DATA {n}\n\nSCALARS gid float\nLOOKUP_TABLE default\n".encode())
        f.write(np.array([0, 1, 2], dtype='>f4').tobytes())
        f.write(b"\nSCALARS ptag float\nLOOKUP_TABLE default\n")
        f.write(np.array([10, 11, 12], dtype='>f4').tobytes())
    p = read_particle_vtk(path)
    assert (p['time'], p['ncycle'], p['nparticles']) == (1.5, 7, 3)
    np.testing.assert_array_equal(p['y'], [1, 4, 7])
    np.testing.assert_array_equal(p['tag'], [10, 11, 12])
    assert p['gid'].dtype == np.int64


# ---------------------------------------------------------------------- C16
def test_athinput_discovery_is_restricted(tmp_path, capsys):
    run = tmp_path / 'top' / 'run'
    (run / 'vtk').mkdir(parents=True)
    vtk = str(run / 'vtk' / 'prob.0001.vtk')
    (tmp_path / 'top' / 'athinput.unrelated').write_text('<domain1>\n')
    assert io_utils._discover_athinput(vtk, 'prob') is None
    (run / 'input').mkdir()
    (run / 'input' / 'prob.par').write_text('<domain1>\n')
    found = io_utils._discover_athinput(vtk, 'prob')
    assert found == str(run / 'input' / 'prob.par')
    io_utils._resolve_athinput(None, vtk, 'prob')
    assert str(run / 'input' / 'prob.par') in capsys.readouterr().out
    # Ambiguous wildcards in the run directory are not guessed.
    (run / 'vtk' / 'athinput.a').write_text('')
    (run / 'vtk' / 'athinput.b').write_text('')
    assert io_utils._discover_athinput(str(run / 'vtk' / 'other.0001.vtk'), 'other') is None


# ---------------------------------------------------------------------- C17
def test_get_render_cache_ignores_ranged_requests():
    ad = make_ad()
    assert ad.get('dens', mbl=0, mbh=1).shape[0] == 1
    assert ad.get('dens').shape[0] == 4


def test_x_sol_and_get_coord():
    ad = make_ad()
    np.testing.assert_allclose(asnumpy(ad.data('x_sol')),
                               asnumpy(ad.data('x')) - 8.0 * ad.units.kpc_cgs / 3.0857e18)
    x, y, z = ad.get_coord()
    assert x.shape == (8, 8, 8)
    np.testing.assert_allclose(asnumpy(x), asnumpy(ad.get_refined_data('x')))


def test_cell_info_matches_refined_data_shape():
    ad = make_ad()
    xyz = [-1, 1, -1, 1, -0.3, 0.3]
    x, y, z = ad.xyz_uniform(xyz=xyz)
    assert x.shape == ad.get_refined_data('dens', xyz=xyz).shape


def test_get_block_coords_layout():
    from athena_research.utils.meshblock_utils import get_block_coords
    x, y, z = get_block_coords(np.array([[0, 1, 0, 1, 0, 1]]), 0, 4, 2, 1)
    assert x.shape == (1, 2, 4)
    np.testing.assert_allclose(asnumpy(x)[0, 0], [0.125, 0.375, 0.625, 0.875])


def test_batch_processing_numpy_scalars():
    from athena_research.utils.batch_processing import process_data_in_batches
    data = {'a': np.arange(6.0)}
    out = process_data_in_batches(
        data, ['a'], 6, 2,
        lambda batch, block_range: {'s': np.float32(batch['a'].sum()),
                                    'n': np.int64(len(batch['a'])),
                                    'z': np.array(1.0)})
    assert out['s'] == 15.0 and out['n'] == 6 and float(out['z']) == 3.0


def test_curl_failure_propagates(monkeypatch):
    from athena_research.operations import grad_div_curl
    ad = make_ad()
    ad.is_mhd = True

    def broken(*args, **kwargs):
        raise RuntimeError('curl failed')
    monkeypatch.setattr(grad_div_curl, '_curl_original', broken)
    with pytest.raises(RuntimeError, match='curl failed'):
        ad.data('curx', 0, 4)


def test_vtk_one_cell_axis_keeps_spacing(tmp_path):
    path = tmp_path / 'g.0001.vtk'
    with open(path, 'wb') as f:
        f.write(b'# vtk DataFile Version 3.0\nPRIMITIVE vars at time= 1.0e+00, level= 0, '
                b'domain= 0\nBINARY\nDATASET STRUCTURED_POINTS\n'
                b'DIMENSIONS 5 3 2\nORIGIN 0 0 0\nSPACING 0.5 0.5 0.25\nCELL_DATA 8\n'
                b'SCALARS density float\nLOOKUP_TABLE default\n')
        f.write(np.ones(8, '>f4').tobytes())
    g = io_utils._vtk_set_grid(str(path))
    assert list(g['Nx']) == [4, 2, 1]
    np.testing.assert_allclose(g['re'], [2.0, 1.0, 0.25])


def test_cache_load_without_config_sets_snapshot_number(tmp_path):
    from athena_research import AthenaData
    ad = AthenaData(); ad.time = 0.0; ad.sf = {'x': {'sf': np.zeros(2)}}
    path = tmp_path / 'Run.hydro_w.00058.h5data'
    ad.save(filename=str(path))
    back = AthenaData().load(str(path), config=False)
    assert back.num == 58 and back.path == str(tmp_path)
