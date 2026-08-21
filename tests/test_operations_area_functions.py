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
from athena_research.operations.basic_operations import calc_min, calc_max


class TestBundledHeaders:
    def test_local_headers_are_used_not_the_home_relative_fallback(self):
        assert af._LUT_HPP == af._LOCAL_LUT_HPP
        assert af._MC_HPP == af._LOCAL_MC_HPP
        assert os.path.exists(af._LUT_HPP)
        assert os.path.exists(af._MC_HPP)
        # bundled path must live inside the package, not under a home dir
        assert "marching_cubes_src" in af._LUT_HPP


def _midrange_temp(ad):
    """Isosurface value used by the cross-validation tests below: the
    midpoint of whatever snapshot's actual temperature range, so the tests
    work with any dataset rather than a value calibrated to one particular
    problem generator's initial condition."""
    calc_min(ad, varl=["temp"], redo=True)
    calc_max(ad, varl=["temp"], redo=True)
    return float((asnumpy(ad.min["temp"]) + asnumpy(ad.max["temp"])) / 2)


@pytest.mark.data
class TestAreaMethodsCrossValidation:
    """All three methods approximate the same physical quantity (area of a
    T=const isosurface) via independent code paths; they should agree to
    within the discretization differences documented for each method."""

    def test_mb_based_and_global_coarse_agree_at_step_1(self, ad):
        T_iso = _midrange_temp(ad)
        s_mb, a_mb = af.calc_area_mb_based(
            ad, T_iso, step_sizes=[1], use_gpu=af.cupy_enabled, verbose=False
        )
        s_gl, a_gl = af.calc_area_global_coarse_sampled_offsets(
            ad, T_iso, step_sizes=[1], verbose=False
        )
        # Same per-cell resolution, different code paths (GPU kernel vs.
        # global coarse-grid sampling with offset averaging) -- expect close
        # agreement, not bit-identical.
        assert float(a_mb[0]) == pytest.approx(float(a_gl[0]), rel=0.02)

    def test_smoothed_indicator_is_within_a_few_percent_of_mb_based(self, ad):
        T_iso = _midrange_temp(ad)
        s_mb, a_mb = af.calc_area_mb_based(
            ad, T_iso, step_sizes=[1], use_gpu=af.cupy_enabled, verbose=False
        )
        s_sm, a_sm = af.calc_area_smoothed_indicator_mc(
            ad, T_iso, step_sizes=[1], verbose=False
        )
        # Smoothing genuinely changes the isosurface geometry a little, so
        # this is a looser bound than the two sharp-interface methods above.
        assert float(a_sm[0]) == pytest.approx(float(a_mb[0]), rel=0.1)

    def test_areas_are_positive_and_finite(self, ad):
        T_iso = _midrange_temp(ad)
        s_mb, a_mb = af.calc_area_mb_based(
            ad, T_iso, step_sizes=[1, 2], use_gpu=af.cupy_enabled, verbose=False
        )
        assert np.all(np.isfinite(a_mb))
        assert np.all(a_mb > 0)


@pytest.mark.data
class TestSetAreaCaching:
    def test_set_area_caches_and_skips_recomputation(self, ad, tmp_path):
        T_iso = _midrange_temp(ad)
        h5_path = str(tmp_path / "area_cache.h5data")
        result1 = af.set_area(
            ad, T_iso, h5_path, step_sizes=[1], verbose=False, use_gpu=af.cupy_enabled
        )
        assert list(result1["step_sizes"]) == [1]

        # Second call with the same step sizes should hit the cache and
        # return the identical stored result without raising.
        result2 = af.set_area(
            ad, T_iso, h5_path, step_sizes=[1], verbose=False, use_gpu=af.cupy_enabled
        )
        assert result2["areas"][0] == pytest.approx(result1["areas"][0])


def _planar_interface_T_iso_at_z_midplane(ad, lateral_rtol=0.05):
    """If the domain's z mid-plane coincides with a mesh-cell face, and the
    temperature field is laterally uniform there (a genuinely planar
    interface, not fully 3D structure), return the interface temperature
    value that sits exactly on that face. Otherwise return None so the
    caller can skip: the meshblock-face edge case below can only be
    demonstrated with a snapshot shaped like that -- the *fact* that a
    domain mid-plane can coincide with a meshblock face is a generic
    property of the mesh decomposition, but any specific temperature value
    for where an interface sits is data-specific and must come from the
    loaded snapshot itself, not a hardcoded constant."""
    z = asnumpy(ad.data("z"))
    temp = asnumpy(ad.data("temp"))
    mid = 0.5 * (ad.x3min + ad.x3max)
    z_unique = np.unique(z)
    below = z_unique[z_unique < mid]
    above = z_unique[z_unique > mid]
    if below.size == 0 or above.size == 0:
        return None
    z_below, z_above = below.max(), above.min()
    cell_size = np.median(np.diff(z_unique))
    if not np.isclose(z_above - z_below, cell_size, rtol=0.05):
        return None  # mid-plane doesn't sit exactly on a cell face here
    t_below = temp[np.isclose(z, z_below)]
    t_above = temp[np.isclose(z, z_above)]
    if (t_below.std() > lateral_rtol * abs(t_below.mean()) or
            t_above.std() > lateral_rtol * abs(t_above.mean())):
        return None  # not laterally uniform -- no clean planar interface here
    return float(0.5 * (t_below.mean() + t_above.mean()))


@pytest.mark.data
class TestMeshblockFaceEdgeCase:
    """Documents a known limitation: calc_area_mb_based's per-block
    quick-skip check operates on each block's own (unpadded) data range, so
    an isosurface that lies exactly on a meshblock face -- with zero
    crossing in either block's interior -- is missed entirely. Demonstrating
    this requires a snapshot with a laterally-uniform (planar) interface
    positioned exactly on the domain's z mid-plane, which
    _planar_interface_T_iso_at_z_midplane() detects generically from
    whatever snapshot is loaded; the tests skip if the loaded data isn't
    shaped that way rather than assuming any particular one is. The
    smoothed-indicator/global-coarse methods, which don't rely on a
    per-block interior range check, correctly find the interface."""

    def test_mb_based_misses_interface_exactly_on_a_meshblock_face(self, ad_t0):
        T_iso = _planar_interface_T_iso_at_z_midplane(ad_t0)
        if T_iso is None:
            pytest.skip("loaded snapshot has no laterally-uniform interface "
                        "exactly on the domain z mid-plane; this edge case "
                        "needs a snapshot shaped like that")
        s_mb, a_mb = af.calc_area_mb_based(
            ad_t0, T_iso, step_sizes=[1], use_gpu=af.cupy_enabled, verbose=False
        )
        assert float(a_mb[0]) == 0.0  # current (limited) behavior

    def test_smoothed_indicator_correctly_finds_that_interface(self, ad_t0):
        T_iso = _planar_interface_T_iso_at_z_midplane(ad_t0)
        if T_iso is None:
            pytest.skip("loaded snapshot has no laterally-uniform interface "
                        "exactly on the domain z mid-plane; this edge case "
                        "needs a snapshot shaped like that")
        s_sm, a_sm = af.calc_area_smoothed_indicator_mc(
            ad_t0, T_iso, step_sizes=[1], verbose=False
        )
        assert float(a_sm[0]) > 0.0
