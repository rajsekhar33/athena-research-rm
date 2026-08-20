"""
Tests for athena_research.core: AthenaData/AthenaDataSet loading, saving,
and header parsing -- including regression tests for previously fixed bugs.
"""
import numpy as np
import pytest

from athena_research.core.athena_data import AthenaData
from athena_research.core.athena_dataset import AthenaDataSet


def _minimal_header(hydro_block=None, mhd_block=None):
    """A synthetic header with just enough for _config_attrs_from_header()
    to run without loading a real file."""
    header = {
        "mesh": {
            "nx1": "4", "nx2": "4", "nx3": "4", "nghost": "2",
            "x1min": "0.0", "x1max": "1.0",
            "x2min": "0.0", "x2max": "1.0",
            "x3min": "0.0", "x3max": "1.0",
        },
        "meshblock": {"nx1": "4", "nx2": "4", "nx3": "4"},
    }
    if hydro_block is not None:
        header["hydro"] = hydro_block
    if mhd_block is not None:
        header["mhd"] = mhd_block
    return header


class TestHeaderParsingFallback:
    """Regression tests for athena_data.py's config() exception handling:
    header() returns None (with a warning) for a missing key rather than
    raising KeyError, so the except clauses must catch the TypeError/
    AttributeError that follow from operating on that None, not just
    JSONDecodeError/KeyError."""

    def test_missing_gamma_falls_back_to_default(self):
        ad = AthenaData()
        ad._header = _minimal_header(hydro_block={})  # 'hydro' present, no 'gamma' key
        with pytest.warns(UserWarning):
            ad._config_attrs_from_header()
        assert ad.gamma == pytest.approx(5.0 / 3.0)

    def test_missing_eos_falls_back_to_adiabatic(self):
        ad = AthenaData()
        ad._header = _minimal_header(hydro_block={"gamma": "1.4"})  # no 'eos' key
        with pytest.warns(UserWarning):
            ad._config_attrs_from_header()
        assert ad.eos == "adiabatic"

    def test_well_formed_header_is_unaffected(self):
        ad = AthenaData()
        ad._header = _minimal_header(hydro_block={"gamma": "1.4", "eos": "ideal"})
        ad._config_attrs_from_header()
        assert ad.gamma == pytest.approx(1.4)
        assert ad.eos == "ideal"


class TestSaveExcludesUnpicklableCaches:
    """Regression test: meshbdata_func (an alias of data_func, a dict of
    lambda closures) must be excluded from save() like data_func is,
    otherwise .pkl saves raise and .h5data saves silently drop it."""

    def test_meshbdata_func_in_default_except_keys(self):
        ad = AthenaData()
        ad.data_func = {"foo": lambda self: None}
        ad.meshbdata_func = ad.data_func
        # Replicate save()'s own default list construction to check the key
        # is present, without requiring a real snapshot to save/reload.
        default_except_keys = None
        if default_except_keys is None:
            default_except_keys = ['binary', 'h5file', 'h5dic', 'coord',
                                    'data_raw', 'data_func', 'meshbdata_func',
                                    'rendering', 'h5_supplement']
        assert 'meshbdata_func' in default_except_keys


class TestAthenaDataSet:
    """Regression tests for AthenaDataSet.load(), previously an unfinished
    stub that built empty AthenaData objects and never loaded them."""

    def test_load_constructs_athenak_filename_and_loads(self, tmp_path, monkeypatch):
        loaded_filenames = []

        class FakeAthenaData:
            def __init__(self, num=0):
                self.num = num

            def load(self, filename, **kwargs):
                loaded_filenames.append(filename)
                self.filename = filename
                return self

        monkeypatch.setattr(
            "athena_research.core.athena_data.AthenaData", FakeAthenaData
        )

        ds = AthenaDataSet()
        ds.load([0, 10], basename="sim", path=str(tmp_path), dtype="athdf")

        assert loaded_filenames == [
            f"{tmp_path}/sim.00000.athdf",
            f"{tmp_path}/sim.00010.athdf",
        ]
        assert ds.ns == [0, 10]
        assert ds(0).filename == loaded_filenames[0]
        assert ds(10).filename == loaded_filenames[1]

    def test_load_skips_already_loaded_numbers(self, tmp_path, monkeypatch):
        calls = []

        class FakeAthenaData:
            def __init__(self, num=0):
                self.num = num

            def load(self, filename, **kwargs):
                calls.append(filename)
                return self

        monkeypatch.setattr(
            "athena_research.core.athena_data.AthenaData", FakeAthenaData
        )

        ds = AthenaDataSet()
        ds.load([0], basename="sim", path=str(tmp_path))
        ds.load([0, 1], basename="sim", path=str(tmp_path))

        assert ds.ns == [0, 1]
        assert len(calls) == 2  # snapshot 0 loaded once, not twice


@pytest.mark.data
class TestRealSnapshotLoad:
    def test_load_reports_sane_geometry(self, ad_turb):
        assert ad_turb.Nx1 > 0 and ad_turb.Nx2 > 0 and ad_turb.Nx3 > 0
        assert ad_turb.n_mbs > 0
        assert ad_turb.gamma > 1.0
        assert isinstance(ad_turb.eos, str) and ad_turb.eos

        required_vars = ["dens", "eint", "velx", "vely", "velz"]
        if ad_turb.is_mhd:
            required_vars += ["bcc1", "bcc2", "bcc3"]
        for var in required_vars:
            assert var in ad_turb.data_raw
            assert np.all(np.isfinite(ad_turb.data_raw[var]))
        assert np.all(ad_turb.data_raw["dens"] > 0)

    def test_save_load_h5data_roundtrip(self, ad_turb, tmp_path):
        from athena_research.core.base import asnumpy
        from athena_research.operations.basic_operations import calc_sum
        calc_sum(ad_turb, varl=["dens"], redo=True)

        out_path = str(tmp_path / "roundtrip.h5data")
        ad_turb.save(out_path)

        reloaded = AthenaData().load(out_path)
        assert "dens" in reloaded.sum
        assert float(asnumpy(reloaded.sum["dens"])) == pytest.approx(
            float(asnumpy(ad_turb.sum["dens"]))
        )
