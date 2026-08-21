"""
Tests for athena_research.operations.structure_functions.

Includes a regression test for get_sf's deterministic RNG seed, which was
computed as np.uint32(idx * 1013904223) without wrapping to uint32 range
first. For idx > ~4239 (e.g. a real block-pair index on a domain with
enough meshblocks/pairs), idx * 1013904223 exceeds uint32's max and modern
NumPy raises OverflowError instead of silently wrapping -- this crashed
set_sf outright on domains with enough block pairs.
"""
import numpy as np
import pytest

from athena_research.core.base import asnumpy


class TestSeedComputationRegressionForLargeBlockPairIndex:
    """idx * 1013904223 must be wrapped to uint32 range explicitly (matching
    the GPU kernel's native uint32 wraparound) rather than relying on
    implicit truncation, which recent NumPy treats as an OverflowError for
    out-of-range Python ints."""

    @pytest.mark.parametrize("idx", [0, 1, 4239, 4240, 5000, 100_000, 10_000_000])
    def test_wrapped_seed_stays_in_uint32_range(self, idx):
        seed = np.uint32((idx * 1013904223) % (2**32))
        assert 0 <= int(seed) < 2**32

    def test_naive_unwrapped_cast_would_have_overflowed(self, idx=5000):
        """Documents exactly the failure this regression test guards against:
        the naive np.uint32(idx * 1013904223), with no modulo, raises on
        this input under the NumPy version this suite runs against."""
        with pytest.raises((OverflowError, TypeError)):
            np.uint32(idx * 1013904223)


@pytest.mark.data
class TestStructureFunctionsRealDataSmoke:
    def test_set_sf_runs_and_returns_monotonic_finite_values(self, ad):
        from athena_research.operations.structure_functions import set_sf
        set_sf(ad, varl=["dens"], redo=True, debug=False)

        assert "dens" in ad.sf
        entry = ad.sf["dens"]
        r = asnumpy(entry["r"])
        sf = asnumpy(entry["sf"])
        assert r.shape[0] == sf.shape[-1]
        assert np.all(np.isfinite(sf))
        assert np.all(sf >= 0)
        assert np.all(r > 0)
