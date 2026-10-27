import csv
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from polymer_dmpnn.graph import build_trimer_graph
from polymer_dmpnn.predictor import PolymerThermalPredictor


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = PolymerThermalPredictor()

    def test_saved_model_matches_original_implementation(self):
        reference = json.loads((Path(__file__).parent / "golden_predictions.json").read_text())
        for case in reference["cases"]:
            with self.subTest(smiles=case["smiles"]):
                result = self.predictor.predict_one(case["smiles"])
                self.assertTrue(result["success"])
                self.assertAlmostEqual(result["predicted_kappa_W_mK"], case["mean_W_mK"], delta=1e-5)
                for actual, expected in zip(result["per_seed_predictions_W_mK"].values(), case["per_seed_W_mK"]):
                    self.assertAlmostEqual(actual, expected, delta=1e-5)

    def test_invalid_repeat_units_report_errors(self):
        for smiles in ("", "CC", "*CC", "*CC*.[Na+]"):
            with self.subTest(smiles=smiles):
                result = self.predictor.predict_one(smiles)
                self.assertFalse(result["success"])
                self.assertTrue(result["error"])

    def test_graph_dimensions_and_reverse_edges(self):
        graph = build_trimer_graph("*CC*", periodic_closure=True)
        self.assertEqual(graph.atom_features.shape[1], 133)
        self.assertEqual(graph.edge_features.shape[1], 14)
        self.assertTrue(all(graph.reverse_edge[graph.reverse_edge[i]] == i for i in range(len(graph.reverse_edge))))

    def test_other_rdkit_version_warns_instead_of_failing(self):
        with patch("polymer_dmpnn.predictor.rdBase.rdkitVersion", "2025.03.1"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                PolymerThermalPredictor()
        self.assertTrue(any("validated with" in str(item.message) for item in caught))

    def test_batch_csv_preserves_rows_and_reports_invalid_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.csv"
            output = Path(temp) / "result.csv"
            source.write_text("id,smiles\nvalid,*CC*\ninvalid,CC\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "polymer_dmpnn", "--batch", str(source),
                 "--format", "csv", "--output", str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 2)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["input_id"] for row in rows], ["valid", "invalid"])
            self.assertEqual(rows[0]["success"], "True")
            self.assertEqual(rows[1]["success"], "False")
            self.assertTrue(rows[1]["error"])

    def test_missing_csv_column_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "bad.csv"
            source.write_text("structure\n*CC*\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "polymer_dmpnn", "--batch", str(source)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("smiles", proc.stderr)


if __name__ == "__main__":
    unittest.main()
