"""In-process retry after an exception between SF checkpoint saves."""
import types

import numpy as np
import pytest

from athena_research.core.base import cupy_enabled
import athena_research.operations.sf_checkpoint as sfc
from athena_research.operations.sf_checkpoint import checkpointed_sf

from test_sf_checkpoint import ToyData, SETTINGS

pytestmark = pytest.mark.skipif(cupy_enabled, reason='CPU fixture; ToyData serves NumPy arrays')


def _popcount(state):
    return int(np.unpackbits(state['completed_bitmap'], bitorder='little').sum())


@pytest.mark.parametrize('fail_at', [6, 9, 12])
def test_retry_after_exception_resumes_from_last_snapshot(monkeypatch, fail_at):
    reference = ToyData(4)
    assert checkpointed_sf(reference, 'p', ['dens'], checkpoint=lambda ad, label: None,
                           **SETTINGS)

    # A controllable clock: exactly one periodic save (after the first pair),
    # then progress continues unsaved until the simulated failure.
    clock = {'t': 0.0}
    monkeypatch.setattr(sfc, 'time', types.SimpleNamespace(monotonic=lambda: clock['t']))
    ad = ToyData(4)
    calls = {'n': 0}
    saved = []

    def stop():
        calls['n'] += 1
        if calls['n'] == 2:
            clock['t'] = 100.0
        if calls['n'] == fail_at:
            raise RuntimeError('simulated OOM')
        return False

    with pytest.raises(RuntimeError):
        checkpointed_sf(ad, 'p', ['dens'], checkpoint=lambda ad, label: saved.append(label),
                        checkpoint_seconds=50.0, should_stop=stop, **SETTINGS)
    assert len(saved) == 1
    state = ad.sf_checkpoints['p']
    assert state['completed_pairs'] == _popcount(state) == 1
    np.testing.assert_array_equal(state['cursor'], [0, 0, 0, 1])
    assert state['status'] == 'partial'

    monkeypatch.undo()
    assert checkpointed_sf(ad, 'p', ['dens'], checkpoint=lambda ad, label: None, **SETTINGS)
    for key in ('sf', 'weight_sum_bin', 'num_bin_points'):
        np.testing.assert_allclose(ad.sf['dens_p'][key], reference.sf['dens_p'][key])
