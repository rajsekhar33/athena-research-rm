"""SF checkpoint: sampler tag, code-only continuation and the increment histogram."""
import json
import unittest

import numpy as np

from athena_research.core.base import cupy_enabled
from athena_research.operations import sf_checkpoint as ck
from athena_research.operations.structure_functions import SF_SAMPLER_VERSION, sf_entry_is_current
from test_sf_checkpoint import SETTINGS, ToyData

EDGES = [0.01, 0.1, 0.5, 1.0]


def run(ad, **kw):
    return ck.checkpointed_sf(ad, 'p', ['dens', 'velx'], checkpoint=lambda *a: None, **{**SETTINGS, **kw})


@unittest.skipIf(cupy_enabled, 'CPU fixture suite')
class CheckpointV3Tests(unittest.TestCase):
    def test_published_entries_are_tagged_with_histogram(self):
        ad = ToyData()
        self.assertTrue(run(ad, delta_edges=EDGES))
        entry = ad.sf['dens_p']
        self.assertTrue(sf_entry_is_current(entry))
        self.assertEqual(entry['delta_pdf'].shape, (8, len(EDGES) + 1))
        # Importance-weighted pair counts; with 0/1 mask weights each pair adds 2 to weight_sum_bin.
        np.testing.assert_allclose(entry['delta_pdf'].sum(axis=1), 0.5 * entry['weight_sum_bin'], rtol=1e-12)

    def test_interrupted_histogram_resume_equals_uninterrupted(self):
        whole = ToyData(); self.assertTrue(run(whole, delta_edges=EDGES))
        parts = ToyData(); self.assertFalse(run(parts, delta_edges=EDGES, stop_after_pairs=3))
        self.assertTrue(run(parts, delta_edges=EDGES))
        for key in ('sf', 'delta_pdf', 'weight_sum_bin'):
            np.testing.assert_allclose(parts.sf['velx_p'][key], whole.sf['velx_p'][key], rtol=1e-12)

    def test_other_sampler_state_is_discarded(self):
        ad = ToyData(); self.assertFalse(run(ad, stop_after_pairs=2))
        state = ad.sf_checkpoints['p']
        sig = json.loads(state['signature_json']); sig['sampling'] = 'an_older_sampler'
        state['signature_json'] = json.dumps(sig, sort_keys=True)
        state['sf_sums'] = state['sf_sums'] + 1e6
        self.assertTrue(run(ad))
        fresh = ToyData(); self.assertTrue(run(fresh))
        np.testing.assert_allclose(ad.sf['dens_p']['sf'], fresh.sf['dens_p']['sf'], rtol=1e-12)

    def test_code_only_change_continues(self):
        ad = ToyData(); self.assertFalse(run(ad, stop_after_pairs=2))
        state = ad.sf_checkpoints['p']
        sig = json.loads(state['signature_json']); sig['engine_sha256'] = '0' * 64
        state['signature_json'] = json.dumps(sig, sort_keys=True)
        done_before = int(state['completed_pairs'])
        self.assertTrue(run(ad))
        fresh = ToyData(); self.assertTrue(run(fresh))
        self.assertGreater(done_before, 0)
        np.testing.assert_allclose(ad.sf['dens_p']['sf'], fresh.sf['dens_p']['sf'], rtol=1e-12)
        self.assertEqual(json.loads(ad.sf_checkpoints['p']['signature_json'])['sampling'], SF_SAMPLER_VERSION)


if __name__ == '__main__':
    unittest.main()


@unittest.skipIf(cupy_enabled, 'CPU fixture suite')
class CheckpointPerVariableEdgeTests(unittest.TestCase):
    def test_each_key_gets_its_edge_row(self):
        edges = np.array([[0.01, 0.1, 0.5, 1.0], [0.02, 0.2, 0.6, 1.2]])
        ad = ToyData(); self.assertTrue(run(ad, delta_edges=edges))
        np.testing.assert_array_equal(ad.sf['dens_p']['delta_edges'], edges[0])
        np.testing.assert_array_equal(ad.sf['velx_p']['delta_edges'], edges[1])
