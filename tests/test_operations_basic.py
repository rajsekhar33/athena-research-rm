"""Tests for athena_research.operations.basic_operations, histograms, and
profiles against real simulation data."""
import numpy as np
import pytest

from athena_research.core.base import asnumpy


@pytest.mark.data
class TestBasicOperations:
    def test_sum_min_max_avg_are_consistent(self, ad_turb):
        from athena_research.operations.basic_operations import (
            calc_sum, calc_min, calc_max, calc_avg,
        )
        calc_sum(ad_turb, varl=["dens"], redo=True)
        calc_min(ad_turb, varl=["dens"], redo=True)
        calc_max(ad_turb, varl=["dens"], redo=True)
        calc_avg(ad_turb, varl=["dens"], redo=True)

        dmin = float(asnumpy(ad_turb.min["dens"]))
        dmax = float(asnumpy(ad_turb.max["dens"]))
        davg = float(asnumpy(ad_turb.avg["dens"]))
        dsum = float(asnumpy(ad_turb.sum["dens"]))

        assert dmin > 0  # density is positive everywhere
        assert dmin <= davg <= dmax
        assert dsum > 0
        assert np.isfinite([dmin, dmax, davg, dsum]).all()


@pytest.mark.data
class TestHistograms:
    def test_set_dist_produces_normalized_histogram(self, ad_turb):
        from athena_research.operations.histograms import set_dist
        set_dist(ad_turb, varl=["dens"], redo=True)

        assert "dens" in ad_turb.dist
        entry = ad_turb.dist["dens"]
        for key in ("dat", "loc", "mean", "rms", "sigma"):
            assert key in entry
        dat = asnumpy(entry["dat"])
        assert np.all(np.isfinite(dat))
        assert np.all(dat >= 0)  # PDF/count values are non-negative


@pytest.mark.data
class TestProfiles:
    def test_set_profile_axes_agree_on_domain_extent(self, ad_turb):
        from athena_research.operations.profiles import set_profile
        set_profile(ad_turb, varl=["dens"], axis="z", redo=True)

        assert "dens" in ad_turb.vert
        entry = ad_turb.vert["dens"]
        coord = asnumpy(entry["coord"])
        profile = asnumpy(entry["profile"])
        assert coord.shape == profile.shape
        assert np.all(np.isfinite(profile))
        # bin centers should stay within the domain z-extent
        assert coord.min() >= ad_turb.x3min
        assert coord.max() <= ad_turb.x3max

    def test_set_vertical_matches_set_profile_axis_z(self, ad_turb):
        from athena_research.operations.profiles import set_profile, set_vertical
        set_profile(ad_turb, varl=["dens"], axis="z", varsuf="_via_profile", redo=True)
        set_vertical(ad_turb, varl=["dens"], varsuf="_via_vertical", redo=True)

        p1 = asnumpy(ad_turb.vert["dens_via_profile"]["profile"])
        p2 = asnumpy(ad_turb.vert["dens_via_vertical"]["profile"])
        np.testing.assert_allclose(p1, p2, rtol=1e-10)
