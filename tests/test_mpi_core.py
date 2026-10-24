"""Run with mpiexec -n 2 python -m pytest tests/test_mpi_core.py.

Collective reductions and the distributed refined-grid gather. Skipped in a
single-process run.
"""
import numpy as np
import pytest
from athena_research.backends.mpi_utils import MPIManager
from athena_research.core.base import asnumpy

from test_core_fixes import make_ad


@pytest.fixture
def mpi():
    m = MPIManager()
    if m.size < 2:
        pytest.skip('requires mpiexec with two or more ranks')
    return m


@pytest.mark.parametrize('op,expect', [('min', np.minimum), ('max', np.maximum),
                                       ('sum', np.add)])
def test_array_reductions_do_not_deadlock(mpi, op, expect):
    rows = [np.array([1.0 + r, 5.0 - r, 2.0]) for r in range(mpi.size)]
    want = expect.reduce(np.stack(rows))
    np.testing.assert_array_equal(mpi.allreduce(rows[mpi.rank], op), want)
    got = mpi.reduce(rows[mpi.rank].astype(np.int64), op, root=0)
    if mpi.rank == 0:
        np.testing.assert_array_equal(got, want.astype(np.int64))
    else:
        assert got is None
    flags = np.array([mpi.rank == 0, True])
    np.testing.assert_array_equal(mpi.allreduce(flags, op='min'), [False, True])
    assert mpi.allreduce(np.float64(mpi.rank), 'max') == mpi.size - 1


def test_distributed_refined_data_matches_serial(mpi):
    full = asnumpy(make_ad().get_refined_data('x'))
    part = asnumpy(make_ad(rank=mpi.rank, size=mpi.size).get_refined_data('x'))
    np.testing.assert_allclose(part, full)


def test_mixed_scalar_and_dtype_reductions_agree(mpi):
    # A rank without meshblocks keeps a Python initial value while the others
    # hold NumPy results; all ranks must still enter the same collective.
    v = float('inf') if mpi.rank == 0 else np.float64(3.0)
    assert mpi.allreduce(v, 'min') == 3.0
    n = 0 if mpi.rank == 0 else np.int64(5)
    assert mpi.allreduce(n, 'sum') == 5 * (mpi.size - 1)
    arr = np.ones(3, np.float32 if mpi.rank == 0 else np.float64)
    np.testing.assert_array_equal(mpi.allreduce(arr, 'sum'), np.full(3, float(mpi.size)))
    with pytest.raises(ValueError):
        mpi.allreduce(np.zeros(2 + mpi.rank), 'sum')


def test_masked_increment_scales_distributed_matches_serial(mpi):
    from athena_research.operations.structure_functions import masked_increment_scales
    full = masked_increment_scales(make_ad(), ['dens', 'temp'], 'vol')
    part = masked_increment_scales(make_ad(rank=mpi.rank, size=mpi.size), ['dens', 'temp'], 'vol',
                                   mpi_manager=mpi)
    np.testing.assert_allclose(part, full, rtol=1e-10)
