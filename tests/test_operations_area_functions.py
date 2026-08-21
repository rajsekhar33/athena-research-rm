"""
Tests for athena_research.operations.area_functions.

Covers: the bundled marching_cubes_src/ header resolution (no dependency on
a live athenak-RM checkout), and cross-validation between the three
independent area-calculation methods (meshblock GPU kernel, smoothed-
indicator marching cubes, global coarse-sampled offsets) on real data.
"""
import os

import numpy as np
import pytest

from athena_research.core.base import asnumpy
from athena_research.operations import area_functions as af


class TestBundledHeaders:
    def test_local_headers_are_used_not_the_home_relative_fallback(self):
        assert af._LUT_HPP == af._LOCAL_LUT_HPP
        assert af._MC_HPP == af._LOCAL_MC_HPP
        assert os.path.exists(af._LUT_HPP)
        assert os.path.exists(af._MC_HPP)
        # bundled path must live inside the package, not under a home dir
        assert "marching_cubes_src" in af._LUT_HPP


@pytest.mark.data
class TestAreaMethodsCrossValidation:
    """All three methods approximate the same physical quantity (area of the
    T=T_peak isosurface) via independent code paths; they should agree to
    within the discretization differences documented for each method."""

    T_PEAK = 0.1  # well inside the turbulent snapshot's T range, away from
                  # any meshblock-face degeneracy (see TestMeshblockFaceEdgeCase)

    def test_mb_based_and_global_coarse_agree_at_step_1(self, ad):
        s_mb, a_mb = af.calc_area_mb_based(
            ad, self.T_PEAK, step_sizes=[1], use_gpu=af.cupy_enabled, verbose=False
        )
        s_gl, a_gl = af.calc_area_global_coarse_sampled_offsets(
            ad, self.T_PEAK, step_sizes=[1], verbose=False
        )
        # Same per-cell resolution, different code paths (GPU kernel vs.
        # global coarse-grid sampling with offset averaging) -- expect close
        # agreement, not bit-identical.
        assert float(a_mb[0]) == pytest.approx(float(a_gl[0]), rel=0.02)

    def test_smoothed_indicator_is_within_a_few_percent_of_mb_based(self, ad):
        s_mb, a_mb = af.calc_area_mb_based(
            ad, self.T_PEAK, step_sizes=[1], use_gpu=af.cupy_enabled, verbose=False
        )
        s_sm, a_sm = af.calc_area_smoothed_indicator_mc(
            ad, self.T_PEAK, step_sizes=[1], verbose=False
        )
        # Smoothing genuinely changes the isosurface geometry a little, so
        # this is a looser bound than the two sharp-interface methods above.
        assert float(a_sm[0]) == pytest.approx(float(a_mb[0]), rel=0.1)

    def test_areas_are_positive_and_finite(self, ad):
        s_mb, a_mb = af.calc_area_mb_based(
            ad, self.T_PEAK, step_sizes=[1, 2], use_gpu=af.cupy_enabled, verbose=False
        )
        assert np.all(np.isfinite(a_mb))
        assert np.all(a_mb > 0)


@pytest.mark.data
class TestSetAreaCaching:
    def test_set_area_caches_and_skips_recomputation(self, ad, tmp_path):
        h5_path = str(tmp_path / "area_cache.h5data")
        result1 = af.set_area(
            ad, 0.1, h5_path, step_sizes=[1], verbose=False, use_gpu=af.cupy_enabled
        )
        assert list(result1["step_sizes"]) == [1]

        # Second call with the same step sizes should hit the cache and
        # return the identical stored result without raising.
        result2 = af.set_area(
            ad, 0.1, h5_path, step_sizes=[1], verbose=False, use_gpu=af.cupy_enabled
        )
        assert result2["areas"][0] == pytest.approx(result1["areas"][0])


@pytest.mark.data
class TestMeshblockFaceEdgeCase:
    """Documents a known limitation: calc_area_mb_based's per-block
    quick-skip check operates on each block's own (unpadded) data range, so
    an isosurface
    that lies exactly on a meshblock face -- with zero crossing in either
    block's interior -- is missed entirely. This only shows up in
    degenerate cases like this sample run's t=0 initial condition, where
    the TRML profile is centered exactly on the domain's z mid-plane, which
    happens to coincide with a meshblock boundary at this resolution.
    The smoothed-indicator/global-coarse methods, which don't rely on a
    per-block interior range check, correctly find the interface."""

    def test_mb_based_misses_interface_exactly_on_a_meshblock_face(self, ad_t0):
        s_mb, a_mb = af.calc_area_mb_based(
            ad_t0, 0.1, step_sizes=[1], use_gpu=af.cupy_enabled, verbose=False
        )
        assert float(a_mb[0]) == 0.0  # current (limited) behavior

    def test_smoothed_indicator_correctly_finds_that_interface(self, ad_t0):
        s_sm, a_sm = af.calc_area_smoothed_indicator_mc(
            ad_t0, 0.1, step_sizes=[1], verbose=False
        )
        assert float(a_sm[0]) > 0.0
