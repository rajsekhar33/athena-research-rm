"""Regression coverage for shared changes ported from private problem branches."""
from collections import Counter
import h5py
import numpy as np
import pytest
from athena_research import AthenaData
from athena_research.core.base import asnumpy, cupy_enabled
from athena_research.core.io_utils import load_dict_from_hdf5, save_dict_to_hdf5
from athena_research.core.utils import build_owner_tiled_mpi_pairs
from athena_research.operations import area_functions as af
from athena_research.operations import structure_functions as sf
from synthetic_mesh import AnalyticMesh


@pytest.mark.parametrize('suffix', ['.h5data', '.h5data.tmp'])
def test_analysis_cache_excludes_raw_arrays_and_preserves_mpi(tmp_path, suffix):
    path = str(tmp_path / ('cache.00001' + suffix))
    writer = AthenaData()
    writer.data_raw = {'dens': np.ones((2, 4, 4, 4))}
    writer.mb_data = np.ones((2, 4, 4, 4))
    writer.meshbdata_func = {'closure': lambda: 1}
    writer.sum = {'dens': 17.0}
    writer.mpi_rank, writer.mpi_size = 3, 8
    writer.local_mb_start, writer.local_mb_end = 6, 8
    writer.has_full_data = False
    writer.save(path)
    with h5py.File(path) as f:
        assert not {'data_raw', 'mb_data', 'meshbdata_func'} & set(f)
    reader = AthenaData()
    reader.mpi_rank, reader.mpi_size = 1, 2
    reader.local_mb_start, reader.local_mb_end = 4, 8
    reader.has_full_data = False
    reader.load(path, config=False)
    assert reader.sum['dens'] == 17
    assert (reader.mpi_rank, reader.mpi_size, reader.local_mb_start,
            reader.local_mb_end, reader.has_full_data) == (1, 2, 4, 8, False)
    serial = AthenaData().load(path, config=False)
    assert serial.has_full_data and serial.mpi_size == 1


def test_legacy_raw_cache_groups_are_skipped(tmp_path):
    path = str(tmp_path / 'old.h5data')
    with h5py.File(path, 'w') as f:
        f['mb_data'] = np.ones(8)
        f['data_raw/dens'] = np.ones(8)
        f['sum/dens'] = 8.0
    assert set(load_dict_from_hdf5(path)) == {'sum'}
    assert 'mb_data' in load_dict_from_hdf5(path, exclude_keys=[])


def test_hdf5_string_array_round_trip(tmp_path):
    path = str(tmp_path / 'strings.h5data.tmp')
    save_dict_to_hdf5({'fields': np.array(['velx', 'bcc3'])}, path)
    assert list(load_dict_from_hdf5(path)['fields']) == ['velx', 'bcc3']


@pytest.mark.parametrize('gpu', [False, True])
@pytest.mark.parametrize('vector,expected', [('vz', 1.0), ('vx', 0.0)])
def test_planar_alignment_analytic(gpu, vector, expected):
    if gpu and not cupy_enabled:
        pytest.skip('GPU backend unavailable')
    if gpu:
        assert af._get_module() is not None, 'GPU test must not silently fall back'
    ad = AnalyticMesh(n=8)
    ad.fields['vx'].fill(vector == 'vx')
    ad.fields['vz'].fill(vector == 'vz')
    cos2, area = af.calc_vector_nhat_alignment_mb_based(
        ad, .37, 'scalar', ('vx', 'vy', 'vz'), step_sizes=[1, 2, 4],
        use_gpu=gpu, verbose=False)
    np.testing.assert_allclose(ad.vector_nhat_alignment['mean_cos2'], expected, atol=1e-12)
    np.testing.assert_allclose(ad.vector_nhat_alignment['total_area'], 1.0, rtol=2e-6)
    ad.data = lambda *a, **k: pytest.fail('compatible cache should be reused')
    assert af.calc_vector_nhat_alignment_mb_based(
        ad, .37, 'scalar', ('vx', 'vy', 'vz'), step_sizes=[1, 2, 4],
        use_gpu=gpu, verbose=False) == (cos2, area)


@pytest.mark.parametrize('nranks', [2, 3, 5, 10])
@pytest.mark.parametrize('anisotropic', [False, True])
def test_mpi_processors_visit_every_assigned_pair(monkeypatch, nranks, anisotropic):
    """Exercise real batching with fake exchanges; detect dropped remote pairs."""
    from types import SimpleNamespace
    visited = []
    def load(ad, cache, blocks, *args, **kwargs):
        cache.update({b: {} for b in blocks})
    def record(ad, b1, b2, *args, **kwargs):
        visited.append((b1, b2))
    monkeypatch.setattr(sf, '_load_all_blocks_upfront', load)
    monkeypatch.setattr(sf, '_process_block_pair', record)
    monkeypatch.setattr(sf, '_process_block_pair_aniso_mhd', record)
    monkeypatch.setattr('athena_research.utils.batch_processing.determine_blocks_per_batch', lambda *a, **k: 3)
    monkeypatch.setattr(sf, 'determine_blocks_per_batch', lambda *a, **k: 3)
    for rank in range(nranks):
        ad = AnalyticMesh(rank=rank, size=nranks)
        mpi = SimpleNamespace(rank=rank, size=nranks, allreduce=lambda a, **k: a)
        pairs = build_owner_tiled_mpi_pairs(ad.n_mbs, nranks, rank)
        if anisotropic:
            import inspect
            fn = sf._process_mpi_pairs_aniso_mhd
            # Only the pair scheduler runs: scalar geometry parameters are
            # unused by the mocked numerical accumulator.
            args = {name: 1 for name,p in inspect.signature(fn).parameters.items()
                    if p.default is inspect.Parameter.empty}
            args.update(ad=ad, my_pairs=pairs, block_cache={}, weights_list=['ones'],
                        vel_varlist=['velx','vely','velz'], bcc_varlist=['bcc1','bcc2','bcc3'],
                        mpi_manager=mpi, mb_offset=ad.local_mb_start, xyzlim=None)
            fn(**args)
        else:
            sf._process_mpi_pairs(ad, pairs, {}, ['ones'], ['dens'], None, None, None,
                                 1, .01, 1, 1, 1, None, 4, 2, 8, 64, 8,
                                 True, True, False, mpi_manager=mpi,
                                 mb_offset=ad.local_mb_start)
    assert Counter(visited) == Counter((i,j) for i in range(8) for j in range(i,8))


@pytest.mark.parametrize('field', ['nvelx', 'nbcc1'])
def test_profile_subtraction_is_independent_of_first_requested_block(field):
    from athena_research.core.data_functions import config_data_functions
    from athena_research.operations.profiles import set_vertical
    from athena_research.core.base import xp
    mesh = AnalyticMesh()
    ad = AthenaData()
    ad.__dict__.update({k: v for k,v in mesh.__dict__.items() if k != 'fields'})
    ad.gamma = 5/3
    ad.data_raw = {k: xp.asarray(v) for k,v in mesh.fields.items()}
    ad.data_raw['bcc1'] = ad.data_raw['velx'].copy()
    config_data_functions(ad)
    ad.register_vertical_profile_func(set_vertical)
    actual = asnumpy(ad.data(field, 1, 2))
    np.testing.assert_allclose(actual, mesh.fields['x'][1:2] - .5, atol=1e-12)
    np.testing.assert_allclose(asnumpy(ad.data(field, 0, 8)), mesh.fields['x'] - .5, atol=1e-12)


def test_anisotropic_cpu_is_rejected_instead_of_returning_zero_arrays():
    if cupy_enabled:
        pytest.skip('CPU backend only')
    with pytest.raises(RuntimeError, match='require the CuPy'):
        sf.get_sf_aniso_mhd_mb(AnalyticMesh())


def test_grouped_gpu_alignment_analytic():
    if not cupy_enabled:
        pytest.skip('requires GPU backend')
    assert af._get_module() is not None
    ad = AnalyticMesh(n=4)
    cos2, area = af.calc_vector_nhat_alignment_mb_based(
        ad, .37, 'scalar', ('vx','vy','vz'), step_sizes=[8], verbose=False)
    assert cos2 == pytest.approx(1.0, abs=1e-12)
    assert area == pytest.approx(1.0, rel=2e-6)


@pytest.mark.parametrize('steps', [[], [0], [-1]])
def test_alignment_rejects_invalid_scales(steps):
    with pytest.raises(ValueError, match='positive integers'):
        af.calc_vector_nhat_alignment_mb_based(
            AnalyticMesh(), .37, 'scalar', ('vx','vy','vz'), step_sizes=steps)


def test_alignment_rejects_sharded_inputs_without_neighbor_halos():
    with pytest.raises(ValueError, match='including neighbor halos'):
        af.calc_vector_nhat_alignment_mb_based(
            AnalyticMesh(rank=0, size=2), .37, 'scalar', ('vx','vy','vz'))
