"""
Tests for athena_research.operations.spectra.

Parseval's theorem (sum(E_spectrum) == <f^2>, excluding the DC/k=0 bin
which carries the mean^2 term) is only exact for a field that is genuinely
periodic on the box. A sample snapshot with a non-periodic boundary
condition along one axis will have a real discontinuity when wrapped
periodically by the FFT, so its fields won't satisfy Parseval to tight
tolerance -- that's expected physics, not a bug: a synthetic field that is
exactly periodic (even with real structure along every axis) matches to
full float64 precision, while the real sample data does not, purely
because the field itself isn't periodic.

So: the real correctness check uses a synthetic periodic field injected
onto a real snapshot's geometry (decoupled from the sample data's BC). Real
sample data is covered by a much looser smoke test only.
"""
import numpy as np
import pytest

from athena_research.core.base import xp, asnumpy, cupy_enabled


def _inject_periodic_field(ad, name, x_wavenumber=3, z_wavenumber=7,
                            x_amp=0.3, z_amp=0.4):
    """Overwrite ad.data_raw[name] with a field that is exactly periodic on
    the box (integer wavelengths along x1 and x3), reusing ad's real
    meshblock geometry. Exactly-periodic-on-the-box is what the FFT-based
    spectrum functions implicitly assume, so Parseval must hold to full
    float64 precision for this field regardless of the sample data's
    actual (non-periodic-in-z) boundary conditions."""
    n_mb, nx3, nx2, nx1 = ad.data_raw["dens"].shape
    mb_geo = ad.mb_geometry
    x1_vals = np.sort(np.unique(mb_geo[:, 0]))
    x3_vals = np.sort(np.unique(mb_geo[:, 4]))
    Lx = ad.x1max - ad.x1min
    Lz = ad.x3max - ad.x3min
    dx = Lx / ad.Nx1
    dz = Lz / ad.Nx3

    synth = np.empty((n_mb, nx3, nx2, nx1), dtype=np.float64)
    for m in range(n_mb):
        i1 = int(np.searchsorted(x1_vals, mb_geo[m, 0]))
        i3 = int(np.searchsorted(x3_vals, mb_geo[m, 4]))
        gi = i1 * nx1 + np.arange(nx1)
        gk = i3 * nx3 + np.arange(nx3)
        xg = ad.x1min + (gi + 0.5) * dx
        zg = ad.x3min + (gk + 0.5) * dz
        field = (
            1.0
            + x_amp * np.sin(2 * np.pi * x_wavenumber * xg / Lx)[None, None, :]
            + z_amp * np.cos(2 * np.pi * z_wavenumber * zg / Lz)[:, None, None]
        )
        synth[m] = np.broadcast_to(field, (nx3, nx2, nx1))

    ad.data_raw[name] = xp.asarray(synth) if cupy_enabled else synth
    # analytic variance: 0.5*x_amp^2 + 0.5*z_amp^2 (orthogonal sin/cos modes)
    return 0.5 * x_amp**2 + 0.5 * z_amp**2


@pytest.mark.data
class TestParsevalTheoremSyntheticPeriodicField:
    """The rigorous correctness check: total spectral power must equal the
    real-space variance to float64 precision for a genuinely periodic field."""

    @pytest.mark.parametrize("method", ["standard", "memory_efficient"])
    def test_scalar_field_parseval(self, ad, method):
        from athena_research.operations.spectra import get_spectrum, get_spectrum_mb

        expected_variance = _inject_periodic_field(ad, "synth_scalar")

        if method == "standard":
            _, _, _, E_spectrum, _, _ = get_spectrum(
                ad, "synth_scalar", strat_flag=False, skip=0.0, nbins=ad.Nx1
            )
        else:
            _, _, _, E_spectrum, _, _ = get_spectrum_mb(
                ad, "synth_scalar", strat_flag=False, skip=0.0,
                nbins=ad.Nx1, ndiv=4,
            )

        n_total = ad.Nx1 * ad.Nx2 * ad.Nx3
        # bin 0 is the DC/k=0 mode (mean^2); exclude it to compare against variance
        spectral_variance = float(asnumpy(xp.sum(xp.asarray(E_spectrum)[1:]))) / n_total

        assert spectral_variance == pytest.approx(expected_variance, rel=1e-9)

    def test_parseval_verification_class_passes_on_periodic_field(self, ad):
        """End-to-end check via the actual verify_parseval_theorem.py class,
        exercising its corrected formula (regression test for the dk/DC-bin
        formula bug)."""
        from athena_research.operations.verify_parseval_theorem import ParsevalVerification

        _inject_periodic_field(ad, "synth_scalar")
        pv = ParsevalVerification(verbose=False)
        result = pv.verify_parseval_scalar(ad, "synth_scalar", method="standard")

        assert result["relative_error"] < 1e-8
        assert result["passed"]


@pytest.mark.data
class TestSpectraRealDataSmoke:
    """Loose smoke tests on real (non-periodic-in-z) sample data -- checks the
    functions run and return physically sane output, without asserting
    Parseval (which doesn't hold for this data; see module docstring)."""

    def test_set_spectrum_runs_and_returns_finite_positive_power(self, ad):
        from athena_research.operations.spectra import set_spectrum
        set_spectrum(ad, varl=["dens"], redo=True, verbose=False)

        assert "dens" in ad.spectra
        spect = asnumpy(ad.spectra["dens"]["spectrum"])
        assert np.all(np.isfinite(spect))
        assert np.all(spect >= 0)

    def test_set_spectrum_helmholtz_runs_and_returns_finite_power(self, ad):
        from athena_research.operations.spectra import set_spectrum_helmholtz
        set_spectrum_helmholtz(ad, "vel", redo=True, verbose=False)

        assert "vel" in ad.spectra
        for key in ("spectrum_comp", "spectrum_sol"):
            values = asnumpy(ad.spectra["vel"][key])
            assert np.all(np.isfinite(values))
            assert np.all(values >= 0)
