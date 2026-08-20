"""
Tests for athena_research.operations.grad_div_curl.

Includes a regression check for the dead-branch cleanup in gradient()/
divergence()/curl(): before the fix, each function had an unreachable
`if simultaneous_blocks is not None: ...` block nested inside the
auto_select path (dead because the function's very first check already
returns whenever simultaneous_blocks is not None). The cleanup only
removed literally-unreachable code, so behavior must be identical; the
real safety net here is that the two independent code paths
(memory-efficient meshblock method vs. whole-domain method) agree.
"""
import numpy as np
import pytest

from athena_research.core.base import asnumpy


@pytest.mark.data
class TestDivergence:
    def test_auto_select_agrees_with_forced_meshblock_method(self, ad_turb):
        from athena_research.operations.grad_div_curl import divergence

        div_auto = divergence(ad_turb, "velx", "vely", "velz", auto_select=True)
        div_mb = divergence(ad_turb, "velx", "vely", "velz", auto_select=False)

        auto_np = np.concatenate([asnumpy(d).ravel() for d in div_auto]) \
            if isinstance(div_auto, list) else asnumpy(div_auto).ravel()
        mb_np = np.concatenate([asnumpy(d).ravel() for d in div_mb]) \
            if isinstance(div_mb, list) else asnumpy(div_mb).ravel()

        np.testing.assert_allclose(auto_np, mb_np, rtol=1e-10, atol=1e-12)
        assert np.all(np.isfinite(auto_np))

    def test_simultaneous_blocks_matches_auto_select(self, ad_turb):
        """simultaneous_blocks explicitly forces the meshblock method,
        bypassing auto_select entirely -- exercises the exact code path
        that had the dead conditional before the cleanup."""
        from athena_research.operations.grad_div_curl import divergence

        div_auto = divergence(ad_turb, "velx", "vely", "velz", auto_select=True)
        div_forced = divergence(
            ad_turb, "velx", "vely", "velz", simultaneous_blocks=1
        )

        auto_np = np.concatenate([asnumpy(d).ravel() for d in div_auto]) \
            if isinstance(div_auto, list) else asnumpy(div_auto).ravel()
        forced_np = np.concatenate([asnumpy(d).ravel() for d in div_forced]) \
            if isinstance(div_forced, list) else asnumpy(div_forced).ravel()

        np.testing.assert_allclose(auto_np, forced_np, rtol=1e-10, atol=1e-12)


@pytest.mark.data
class TestGradient:
    def test_auto_select_agrees_with_forced_meshblock_method(self, ad_turb):
        from athena_research.operations.grad_div_curl import gradient

        grad_auto = gradient(ad_turb, "dens", axis="x", auto_select=True)
        grad_mb = gradient(ad_turb, "dens", axis="x", auto_select=False)

        np.testing.assert_allclose(
            asnumpy(grad_auto), asnumpy(grad_mb), rtol=1e-10, atol=1e-12
        )
        assert np.all(np.isfinite(asnumpy(grad_auto)))


@pytest.mark.data
class TestCurl:
    def test_auto_select_agrees_with_forced_meshblock_method(self, ad_turb):
        from athena_research.operations.grad_div_curl import curl

        curl_auto = curl(ad_turb, "velx", "vely", "velz", auto_select=True)
        curl_mb = curl(ad_turb, "velx", "vely", "velz", auto_select=False)

        for comp_auto, comp_mb in zip(curl_auto, curl_mb):
            np.testing.assert_allclose(
                asnumpy(comp_auto), asnumpy(comp_mb), rtol=1e-10, atol=1e-12
            )
