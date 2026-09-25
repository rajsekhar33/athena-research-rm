"""Run with mpiexec -n 3 python -m pytest tests/test_mpi_shared_port.py.

Uses a tiny distributed analytic mesh and a separately loaded full-domain
serial reference. No production snapshots or caches are read or written.
"""
import numpy as np
import pytest
from athena_research.backends.mpi_utils import MPIManager
from athena_research.core.base import asnumpy, cupy_enabled
from athena_research.operations.structure_functions import get_sf_mb
from synthetic_mesh import AnalyticMesh


@pytest.mark.parametrize('weights', ['ones', 'phase'])
def test_distributed_sf_matches_full_domain_reference(weights):
    mpi = MPIManager()
    if mpi.size < 2:
        pytest.skip('requires mpiexec with two or more ranks')
    kwargs = dict(varlist=['dens'], weights=weights, max_order=2,
                  npairs=2e5, nbins=8, nsamples_block_min=64,
                  simultaneous_blocks=3)
    reference = get_sf_mb(AnalyticMesh(), **kwargs)
    distributed = get_sf_mb(AnalyticMesh(rank=mpi.rank, size=mpi.size),
                            mpi_manager=mpi, **kwargs)
    for actual, expected in zip(distributed, reference):
        np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=2e-12, atol=1e-10)


def test_distributed_anisotropic_sf_matches_full_domain_reference():
    from athena_research.operations.structure_functions import get_sf_aniso_mhd_mb
    mpi = MPIManager()
    if mpi.size < 2:
        pytest.skip('requires mpiexec with two or more ranks')
    if not cupy_enabled:
        with pytest.raises(RuntimeError, match='require the CuPy'):
            get_sf_aniso_mhd_mb(AnalyticMesh())
        return
    kwargs = dict(max_order=2, npairs=2e5, nbins=8, nbins_prll=8, nbins_perp=8,
                  nsamples_block_min=64, simultaneous_blocks=3, use_nearfield_bias=False)
    reference = get_sf_aniso_mhd_mb(AnalyticMesh(), **kwargs)
    distributed = get_sf_aniso_mhd_mb(AnalyticMesh(rank=mpi.rank, size=mpi.size),
                                     mpi_manager=mpi, **kwargs)
    def compare(actual, expected):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                compare(actual[key], expected[key])
        else:
            np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=2e-12, atol=1e-10)
    assert np.sum(asnumpy(reference['counts']['num'])) > 0
    compare(distributed, reference)
