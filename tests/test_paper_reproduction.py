"""Integration checks for portable saved-prediction reproduction of Tables 1–5."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPRODUCE = Path(__file__).resolve().parents[1] / "reproduce"
SPEC = importlib.util.spec_from_file_location("paper_tables", REPRODUCE / "paper_tables.py")
paper_tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper_tables)


class PaperReproductionTests(unittest.TestCase):
    def test_complete_reproduction_outside_source_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            portable = root / "download" / "reproduce"
            portable.mkdir(parents=True)
            shutil.copy2(REPRODUCE / "paper_tables.py", portable)
            shutil.copytree(REPRODUCE / "data", portable / "data")
            cwd = root / "unrelated-working-directory"
            cwd.mkdir()
            output = root / "tables"
            run = subprocess.run([sys.executable, "-I", str(portable / "paper_tables.py"),
                                  "--verify", "--output", str(output)], cwd=cwd,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            check = json.loads((output / "verification.json").read_text())
            self.assertEqual(check["groups"], 117)
            self.assertEqual(check["prediction_rows"], 40848)
            self.assertEqual(check["verification"]["display_statistics_checked"], 182)
            self.assertEqual(check["verification"]["display_mismatches"], 0)
            self.assertEqual(check["verification"]["display_statistics_by_table"],
                             {"1": 40, "2": 16, "3": 30, "4": 32, "5": 64})
            self.assertTrue(check["manifest_integrity_verified"])
            self.assertTrue(check["input_integrity_verified"])
            with (output / "display_statistics.csv").open(newline="") as stream:
                statistics_rows = list(csv.DictReader(stream))
            self.assertEqual(len(statistics_rows), 182)
            self.assertEqual({int(r["table"]) for r in statistics_rows}, {1, 2, 3, 4, 5})
            self.assertTrue(all(r["matches"] == "True" for r in statistics_rows))
            for number in range(1, 6):
                self.assertTrue((output / f"table_{number}.csv").is_file())
                self.assertTrue((output / f"table_{number}.md").is_file())

    def test_modified_prediction_file_fails_integrity_check(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            shutil.copytree(REPRODUCE / "data", data)
            path = data / "saved_predictions.csv.gz"
            rows = list(csv.DictReader(io.StringIO(gzip.decompress(path.read_bytes()).decode())))
            rows[0]["prediction"] = str(float(rows[0]["prediction"]) + 0.1)
            text = io.StringIO(newline="")
            writer = csv.DictWriter(text, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
            path.write_bytes(gzip.compress(text.getvalue().encode(), mtime=0))
            with self.assertRaisesRegex(ValueError, "prediction input SHA256 mismatch"):
                paper_tables.load_inputs(data)

    def test_changed_numeric_input_fails_table_verification(self):
        manifest, records = paper_tables.load_inputs(REPRODUCE / "data")
        group_id = manifest["groups"][0]["id"]
        records[group_id][0]["prediction"] = str(float(records[group_id][0]["prediction"]) + 0.1)
        metrics, stats = paper_tables.calculate(manifest, records)
        with self.assertRaisesRegex(ValueError, "numeric verification failed"):
            paper_tables.verify(manifest, metrics, stats)

    def test_modified_threshold_manifest_fails_integrity_check(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            shutil.copytree(REPRODUCE / "data", data)
            path = data / "manifest.json"
            manifest = json.loads(path.read_text())
            group = next(g for g in manifest["groups"] if g["train_q95"] is not None)
            group["train_q95"] += 0.01
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                paper_tables.load_inputs(data)

    def test_paired_model_truth_cannot_diverge(self):
        manifest, records = paper_tables.load_inputs(REPRODUCE / "data")
        group_id = manifest["groups"][0]["id"]
        records[group_id][0]["y_true"] = str(float(records[group_id][0]["y_true"]) + 0.01)
        with self.assertRaisesRegex(ValueError, "paired evaluation record/truth identity mismatch"):
            paper_tables.calculate(manifest, records)

    def test_opc_uses_equal_cluster_weight_not_equal_record_weight(self):
        group = {"id": "example", "n": 3, "prediction_dtype": "float64",
                 "cluster_macro": True, "n_clusters": 2}
        rows = [{"record_id": "1", "y_true": "1", "prediction": "1", "cluster_id": "A"},
                {"record_id": "2", "y_true": "2", "prediction": "1.8", "cluster_id": "A"},
                {"record_id": "3", "y_true": "3", "prediction": "2.4", "cluster_id": "B"}]
        metrics = paper_tables.group_metrics(group, rows)
        self.assertAlmostEqual(metrics["macro_mae"], 0.35)
        self.assertAlmostEqual(metrics["mae"], 0.8 / 3)
        self.assertNotAlmostEqual(metrics["macro_mae"], metrics["mae"])


if __name__ == "__main__":
    unittest.main()
