"""Check that fresh inference does not depend on stored prediction availability."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from polymer_dmpnn.paper import PaperPredictor


class FreshRetestWorkerTests(unittest.TestCase):
    def test_missing_old_predictions_does_not_prevent_new_model_outputs(self):
        root = Path(__file__).resolve().parents[1]
        model_id = 'table5_B600_periodic_32101'
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'new-run'
            missing_references = Path(temp) / 'no-reference-predictions'
            missing_references.mkdir()
            proc = subprocess.run([sys.executable, '-I', str(root / 'reproduce/retest_models.py'),
                '--models', model_id, '--reference-data', str(missing_references), '--output', str(output)],
                capture_output=True, text=True, timeout=90, cwd=temp)
            self.assertEqual(proc.returncode, 1)  # Comparison must fail, never fabricate a PASS.
            receipt = json.loads((output / 'fresh_prediction_receipt.json').read_text())
            self.assertEqual(receipt['predictions'], 400)
            self.assertEqual(receipt['forward_batches'], 25)
            rows = list(csv.DictReader((output / (model_id + '.csv')).open()))
            self.assertEqual(len(rows), 400)
            expected = PaperPredictor(model_id).predict_one(rows[0]['smiles'])
            self.assertAlmostEqual(float(rows[0]['prediction']), expected['predicted_kappa_W_mK'], delta=1e-5)
            self.assertFalse((output / 'result.json').exists())
            self.assertEqual(json.loads((output / 'progress.json').read_text())['state'], 'failed')

if __name__ == '__main__':
    unittest.main()
