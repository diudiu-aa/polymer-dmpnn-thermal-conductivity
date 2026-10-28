import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from polymer_dmpnn.graph import build_trimer_graph
from polymer_dmpnn.paper import PaperPredictor, paper_registry

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'reproduce'))
from fixed_inputs import load_inputs
from paper_tables import load_inputs as load_saved


class PaperModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.records, cls.splits, _ = load_inputs()
        cls.manifest, cls.saved = load_saved(REPO / 'reproduce/data')

    def test_all_saved_checkpoint_identities_and_state_shapes(self):
        registry = paper_registry()['models']
        self.assertEqual(len(registry), 65)
        for key in registry:
            with self.subTest(model=key):
                predictor = PaperPredictor(key)
                self.assertEqual(predictor.model_id, key)

    def test_actual_paper_predictions_and_multitask_scores(self):
        # Compare real saved predictions from every arm, not a newly generated golden.
        for arm, table in [('open', 3), ('periodic', 3), ('periodic_rank', 3),
                           ('balanced_control', 5), ('multitask', 5)]:
            key = f'table{table}_B600_{arm}_32101'
            predictor = PaperPredictor(key)
            group = next(g for g in self.manifest['groups'] if g['source_table'] == table
                         and g['seed'] == 32101 and g['model'] == arm)
            rows = self.saved[group['id']][:16]
            graphs = [build_trimer_graph(self.records[r['record_id']]['smiles'],
                                       periodic_closure=arm != 'open') for r in rows]
            predictions, scores = predictor.predict_graphs(graphs)
            np.testing.assert_allclose(predictions, [float(r['prediction']) for r in rows], atol=1e-5, rtol=0)
            if scores is not None:
                np.testing.assert_allclose(scores, [float(r['class_score']) for r in rows], atol=1e-5, rtol=0)

    def test_training_entry_validates_fixed_inputs_without_private_paths_or_fit(self):
        for entry, flags in [('strict_cluster_compare.py', []),
                             ('train_graph_models.py', ['--experiment', 'table5', '--arm', 'multitask']),
                             ('train_baselines.py', [])]:
            proc = subprocess.run([sys.executable, str(REPO / 'reproduce' / entry), *flags],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(json.loads(proc.stdout)['run_started'])

    def test_invalid_paper_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown paper model'):
            PaperPredictor('nonexistent')

if __name__ == '__main__':
    unittest.main()
