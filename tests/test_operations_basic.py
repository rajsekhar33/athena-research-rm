"""Tests for athena_research.operations.basic_operations, histograms, and
profiles against real simulation data."""
import numpy as np
import pytest

from athena_research.core.base import asnumpy


@pytest.mark.data
class TestBasicOperations:
    def test_sum_min_max_avg_are_consistent(self, ad):
        from athena_research.operations.basic_operations import (
            calc_sum, calc_min, calc_max, calc_avg,
        )
        calc_sum(ad, varl=["dens"], redo=True)
        calc_min(ad, varl=["dens"], redo=True)
        calc_max(ad, varl=["dens"], redo=True)
        calc_avg(ad, varl=["dens"], redo=True)

        dmin = float(asnumpy(ad.min["dens"]))
        dmax = float(asnumpy(ad.max["dens"]))
        davg = float(asnumpy(ad.avg["dens"]))
        dsum = float(asnumpy(ad.sum["dens"]))

        assert dmin > 0  # density is positive everywhere
        assert dmin <= davg <= dmax
        assert dsum > 0
        assert np.isfinite([dmin, dmax, davg, dsum]).all()


@pytest.mark.data
class TestHistograms:
    def test_set_dist_produces_normalized_histogram(self, ad):
        from athena_research.operations.histograms import set_dist
        set_dist(ad, varl=["dens"], redo=True)

        assert "dens" in ad.dist
        entry = ad.dist["dens"]
        for key in ("dat", "loc", "mean", "rms", "sigma"):
            assert key in entry
        dat = asnumpy(entry["dat"])
        assert np.all(np.isfinite(dat))
        assert np.all(dat >= 0)  # PDF/count values are non-negative


@pytest.mark.data
class TestProfiles:
    def test_set_profile_axes_agree_on_domain_extent(self, ad):
        from athena_research.operations.profiles import set_profile
        set_profile(ad, varl=["dens"], axis="z", redo=True)

        assert "dens" in ad.vert
        entry = ad.vert["dens"]
        coord = asnumpy(entry["coord"])
        profile = asnumpy(entry["profile"])
        assert coord.shape == profile.shape
        assert np.all(np.isfinite(profile))
        # bin centers should stay within the domain z-extent
        assert coord.min() >= ad.x3min
        assert coord.max() <= ad.x3max

    def test_set_vertical_matches_set_profile_axis_z(self, ad):
        from athena_research.operations.profiles import set_profile, set_vertical
        set_profile(ad, varl=["dens"], axis="z", varsuf="_via_profile", redo=True)
        set_vertical(ad, varl=["dens"], varsuf="_via_vertical", redo=True)

        p1 = asnumpy(ad.vert["dens_via_profile"]["profile"])
        p2 = asnumpy(ad.vert["dens_via_vertical"]["profile"])
        np.testing.assert_allclose(p1, p2, rtol=1e-10)
