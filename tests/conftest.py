"""
Shared pytest fixtures for the athena_research test suite.

Tests that need real simulation data are marked with @pytest.mark.data and
are skipped automatically when no sample data is available -- the suite
still runs (and is useful) on a machine without access to sample data.
Point ATHENA_RESEARCH_TEST_DATA at a directory containing grid-data
snapshots to enable them locally. Snapshot discovery is generic: any
basename, any variable-set token, .athdf or .bin -- AthenaData.load()
dispatches purely on file extension, so nothing here assumes a particular
problem name or output naming convention.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# No hardcoded default on purpose -- point this at a local directory of
# grid-data snapshots to enable the @pytest.mark.data tests.
TEST_DATA_DIR = Path(os.environ["ATHENA_RESEARCH_TEST_DATA"]) \
    if "ATHENA_RESEARCH_TEST_DATA" in os.environ else None


def _is_full_volume_athdf(path):
    """True if this .athdf file's per-meshblock size has no degenerate
    (size-1) dimension. A run directory commonly has slice/2D diagnostic
    dumps alongside the main volumetric snapshots (e.g. AthenaK's <output>
    slice_x1/x2/x3 outputs); those set MeshBlockSize to 1 along the sliced
    axis regardless of what the file is named, so checking this structural
    metadata -- not the filename -- is what lets snapshot discovery stay
    generic across arbitrary basenames/variable-set tokens."""
    import h5py
    with h5py.File(path, "r") as h:
        block_size = h.attrs.get("MeshBlockSize")
        if block_size is None:
            return True  # unknown/unexpected format; don't filter it out
        return bool((block_size > 1).all())


def _discover_snapshots(data_dir):
    """Find grid-data snapshot files generically: any basename, any
    variable-set token, .athdf or .bin. Sorting is lexicographic, which
    matches numeric snapshot order since AthenaK zero-pads the cycle number.
    Prefers .athdf if both are present in the directory."""
    athdf = sorted(data_dir.glob("*.athdf"))
    if athdf:
        volume_dumps = [p for p in athdf if _is_full_volume_athdf(p)]
        return volume_dumps if volume_dumps else athdf
    return sorted(data_dir.glob("*.bin"))


@pytest.fixture(scope="session")
def data_dir():
    if TEST_DATA_DIR is None:
        pytest.skip("ATHENA_RESEARCH_TEST_DATA is not set")
    if not TEST_DATA_DIR.is_dir():
        pytest.skip(f"Sample data directory not found: {TEST_DATA_DIR}")
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def _snapshots(data_dir):
    candidates = _discover_snapshots(data_dir)
    if not candidates:
        pytest.skip(f"No *.athdf or *.bin snapshot files found in {data_dir}")
    return candidates


@pytest.fixture(scope="session")
def snapshot_t0_path(_snapshots):
    """Path to the earliest available snapshot. For the default sample run,
    this happens to be a t=0 initial condition with the interface exactly
    on a meshblock face -- see test_operations_area_functions.py for why
    that matters; with a different sample directory this is just "the
    earliest snapshot available"."""
    return _snapshots[0]


@pytest.fixture(scope="session")
def snapshot_turb_path(_snapshots):
    """Path to the latest available (most evolved/developed) snapshot."""
    return _snapshots[-1]


@pytest.fixture(scope="session")
def ad_t0(snapshot_t0_path):
    """Loaded AthenaData for the earliest snapshot. Session-scoped: grid
    snapshots can be large, so this loads once and is shared read-only
    across tests within a run. Tests that mutate ad-level caches (dist, sf,
    spectra, area_mc, ...) should use a fresh copy or a distinct varsuf."""
    from athena_research.core.athena_data import AthenaData
    return AthenaData().load(str(snapshot_t0_path))


@pytest.fixture(scope="session")
def ad_turb(snapshot_turb_path):
    """Loaded AthenaData for the latest/most evolved snapshot."""
    from athena_research.core.athena_data import AthenaData
    return AthenaData().load(str(snapshot_turb_path))
