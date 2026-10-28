"""Recalculate manuscript Tables 1–5 from bundled saved record-level results.

This is saved-prediction metric reproduction, not model training. Only Python's
standard library is required. No source-project directory or network is used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import sys


# Updated only when the reviewed reproduction-data manifest changes.
MANIFEST_SHA256 = "bd23f4af30549c8d113a305731295a550eddc3da9bd2afe6d43707b45a63484b"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_inputs(data_dir: Path) -> tuple[dict, dict[str, list[dict]]]:
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    if sha256(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError("manifest SHA256 mismatch")
    manifest = json.loads(manifest_bytes)
    payload = (data_dir / manifest["input"]["file"]).read_bytes()
    if sha256(payload) != manifest["input"]["sha256"]:
        raise ValueError("prediction input SHA256 mismatch")
    records = {g["id"]: [] for g in manifest["groups"]}
    for row in csv.DictReader(io.StringIO(gzip.decompress(payload).decode("utf-8"))):
        if row["group_id"] not in records:
            raise ValueError("unknown prediction group: " + row["group_id"])
        records[row["group_id"]].append(row)
    if sum(map(len, records.values())) != manifest["input"]["record_rows"]:
        raise ValueError("prediction input row count mismatch")
    return manifest, records


def parse_prediction(text: str, dtype: str) -> float:
    value = float(text)
    if dtype == "float32":
        value = struct.unpack("!f", struct.pack("!f", value))[0]
    elif dtype != "float64":
        raise ValueError("unsupported stored prediction dtype: " + dtype)
    return value


def group_metrics(group: dict, rows: list[dict]) -> dict[str, float]:
    if len(rows) != group["n"] or len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("row count or record identity mismatch: " + group["id"])
    y = [float(r["y_true"]) for r in rows]
    p = [parse_prediction(r["prediction"], group["prediction_dtype"]) for r in rows]
    if not y or not all(math.isfinite(v) for v in y + p):
        raise ValueError("nonfinite or empty input: " + group["id"])
    errors = [abs(a - b) for a, b in zip(y, p)]
    ymean = statistics.fmean(y)
    denominator = math.fsum((v - ymean) ** 2 for v in y)
    if denominator <= 0:
        raise ValueError("undefined R2: " + group["id"])
    result = {
        "mae": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean((a - b) ** 2 for a, b in zip(y, p))),
        "r2": 1 - math.fsum((a - b) ** 2 for a, b in zip(y, p)) / denominator,
    }
    if group.get("cluster_macro"):
        clusters: dict[str, list[float]] = {}
        for row, error in zip(rows, errors):
            if not row["cluster_id"]:
                raise ValueError("missing OPC cluster identity")
            clusters.setdefault(row["cluster_id"], []).append(error)
        result["macro_mae"] = statistics.fmean(statistics.fmean(v) for v in clusters.values())
        if len(clusters) != group["n_clusters"]:
            raise ValueError("OPC cluster count mismatch")
    q95 = group.get("train_q95")
    if q95 is not None:
        tail = [e for value, e in zip(y, errors) if value >= q95]
        if not tail:
            raise ValueError("empty predefined tail: " + group["id"])
        result["tail_mae"] = statistics.fmean(tail)
        result["tail_n"] = len(tail)
        # Python's sort is stable, preserving the saved evaluation row order.
        truth_order = sorted(range(len(y)), key=lambda i: -y[i])
        prediction_order = sorted(range(len(p)), key=lambda i: -p[i])
        class_scores = None
        if group.get("classifier_scores"):
            class_scores = [float(r["class_score"]) for r in rows]
            if not all(math.isfinite(v) for v in class_scores):
                raise ValueError("nonfinite classifier scores")
            class_order = sorted(range(len(y)), key=lambda i: -class_scores[i])
        for k in (10, 20, 100):
            if len(y) < k:
                continue
            truth = set(truth_order[:k])
            result[f"precision_at_{k}"] = len(truth & set(prediction_order[:k])) / k
            if class_scores is not None:
                result[f"class_precision_at_{k}"] = len(truth & set(class_order[:k])) / k
    return result


def calculate(manifest: dict, records: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    # Compare record IDs and truth across paired arms/budgets before scoring.
    # The older Table-1 evaluation set changes with the total label budget;
    # the later learning curve intentionally keeps each seed's V records fixed.
    paired_truth = {}
    for group in manifest["groups"]:
        key = (group["evaluation_partition"], group["seed"],
               group["budget"] if group["source_table"] == 1 else None)
        truth = {r["record_id"]: float(r["y_true"]) for r in records[group["id"]]}
        if key in paired_truth and truth != paired_truth[key]:
            raise ValueError("paired evaluation record/truth identity mismatch: " + group["id"])
        paired_truth[key] = truth
    metrics = {g["id"]: group_metrics(g, records[g["id"]]) for g in manifest["groups"]}
    statistics_rows = []
    for table in manifest["tables"]:
        for row_index, row in enumerate(table["rows"], 1):
            for cell in row["cells"]:
                if cell is None:
                    continue
                values = [metrics[g][cell["metric"]] * cell["scale"] for g in cell["groups"]]
                if len(values) != len(set(cell["groups"])) or len(values) < 2:
                    raise ValueError("invalid aggregation group identities")
                for kind, value, expected, decimals in (
                    ("mean", statistics.fmean(values), cell["expected_mean"], cell["mean_decimals"]),
                    ("sample_sd", statistics.stdev(values), cell["expected_sd"], cell["sd_decimals"]),
                ):
                    displayed = f"{value:.{decimals}f}"
                    statistics_rows.append({
                        "table": table["number"], "row": row_index,
                        "labels": " | ".join(row["labels"]), "metric": cell["metric"],
                        "column": cell["column"], "statistic": kind,
                        "n_seeds": len(values), "value": value,
                        "displayed": displayed, "expected": expected,
                        "matches": displayed == expected,
                    })
    return metrics, statistics_rows


def verify(manifest: dict, metrics: dict, statistics_rows: list[dict]) -> dict:
    mismatches = [r for r in statistics_rows if not r["matches"]]
    reference_mismatches = []
    max_errors: dict[str, float] = {}
    for group in manifest["groups"]:
        tolerance = group["reference_tolerance"]
        for key, expected in group["reference_metrics"].items():
            actual = metrics[group["id"]][key]
            error = abs(actual - expected)
            max_errors[str(group["source_table"])] = max(max_errors.get(str(group["source_table"]), 0), error)
            if not math.isclose(actual, expected, rel_tol=tolerance["relative"], abs_tol=tolerance["absolute"]):
                reference_mismatches.append({"group": group["id"], "metric": key,
                                             "computed": actual, "expected": expected})
    if len(statistics_rows) != manifest["expected_display_statistics"]:
        raise ValueError("display statistic count mismatch")
    if mismatches or reference_mismatches:
        raise ValueError("numeric verification failed: " + json.dumps({
            "display_mismatches": mismatches, "seed_metric_mismatches": reference_mismatches}, ensure_ascii=False))
    table_counts = {str(t["number"]): sum(r["table"] == t["number"] for r in statistics_rows)
                    for t in manifest["tables"]}
    return {"display_statistics_checked": len(statistics_rows), "display_mismatches": 0,
            "display_statistics_by_table": table_counts,
            "seed_metric_mismatches": 0, "max_seed_metric_absolute_error_by_source_table": max_errors}


def write_outputs(out: Path, manifest: dict, metrics: dict, stats: list[dict], check: dict | None) -> None:
    out.mkdir(parents=True, exist_ok=False)
    with (out / "display_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader(); writer.writerows(stats)
    with (out / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["group_id", "source_table", "model", "seed", "budget", "metric", "value"])
        for group in manifest["groups"]:
            for metric, value in metrics[group["id"]].items():
                writer.writerow([group["id"], group["source_table"], group["model"], group["seed"], group["budget"], metric, value])
    for table in manifest["tables"]:
        rendered = []
        for row_index, row in enumerate(table["rows"], 1):
            values = list(row["labels"])
            for cell in row["cells"]:
                if cell is None:
                    values.append("—"); continue
                matching = [s for s in stats if s["table"] == table["number"] and s["row"] == row_index and s["column"] == cell["column"]]
                pair = {s["statistic"]: s["displayed"] for s in matching}
                values.append(pair["mean"] + " ± " + pair["sample_sd"] + ("%" if cell["scale"] == 100 else ""))
            rendered.append(values)
        with (out / f"table_{table['number']}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(table["columns"]); writer.writerows(rendered)
        lines = [f"Table {table['number']}. {table['title']}", "", "| " + " | ".join(table["columns"]) + " |",
                 "| " + " | ".join("---" for _ in table["columns"]) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in rendered]
        (out / f"table_{table['number']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"mode": "saved-record-prediction metric reproduction; no training or inference",
               "data_manifest_sha256": MANIFEST_SHA256,
               "prediction_input_sha256": manifest["input"]["sha256"],
               "manifest_integrity_verified": True, "input_integrity_verified": True,
               "groups": len(metrics),
               "prediction_rows": manifest["input"]["record_rows"], "verification": check,
               "scope": manifest["scope"]}
    (out / "verification.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check all 182 displayed statistics and the original seed metrics")
    parser.add_argument("--output", required=True, type=Path, help="New output directory (must not exist)")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data", help="Bundled reproduction data directory")
    args = parser.parse_args(argv)
    try:
        manifest, records = load_inputs(args.data_dir)
        metrics, stats = calculate(manifest, records)
        check = verify(manifest, metrics, stats) if args.verify else None
        write_outputs(args.output, manifest, metrics, stats, check)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print("paper table reproduction failed: " + str(exc), file=sys.stderr)
        return 1
    print(f"Recomputed Tables 1–5: {len(stats)} displayed statistics from {len(metrics)} saved prediction groups; verified={args.verify}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
