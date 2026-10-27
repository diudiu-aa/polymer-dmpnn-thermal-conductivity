"""CLI for single and batch polymer thermal-conductivity predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .predictor import PolymerThermalPredictor


def _read_batch(path: Path, column: str) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or column not in reader.fieldnames:
                raise ValueError(f"CSV must contain a {column!r} column")
            return [{"input_row": i, "input_id": row.get("id", ""), "smiles": row[column]}
                    for i, row in enumerate(reader, start=1)]
    with path.open(encoding="utf-8") as stream:
        return [{"input_row": i, "input_id": "", "smiles": line.strip()}
                for i, line in enumerate(stream, start=1) if line.strip()]


def _write_csv(results: list[dict], stream) -> None:
    fields = ["input_row", "input_id", "smiles", "success", "predicted_kappa_W_mK",
              "seed_std_W_mK", "per_seed_predictions_W_mK", "seed_std_note", "error"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for item in results:
        row = dict(item)
        if "per_seed_predictions_W_mK" in row:
            row["per_seed_predictions_W_mK"] = json.dumps(row["per_seed_predictions_W_mK"], sort_keys=True)
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict amorphous polymer thermal conductivity at 300 K and 1 atm.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", help="Repeat-unit SMILES with two terminal * connection points")
    group.add_argument("--batch", type=Path, help="UTF-8 text file or CSV containing a smiles column")
    parser.add_argument("--column", default="smiles", help="CSV column containing repeat-unit SMILES")
    parser.add_argument("--format", choices=("json", "csv"), default="json", help="Output format")
    parser.add_argument("--output", type=Path, help="Write results to this file instead of stdout")
    args = parser.parse_args(argv)

    try:
        inputs = ([{"input_row": 1, "input_id": "", "smiles": args.smiles}]
                  if args.smiles is not None else _read_batch(args.batch, args.column))
        if not inputs:
            raise ValueError("batch contains no input rows")
        predictor = PolymerThermalPredictor()
        results = []
        for item in inputs:
            result = predictor.predict_one(item["smiles"])
            results.append({"input_row": item["input_row"], "input_id": item["input_id"], **result})
        if args.output:
            with args.output.open("w", encoding="utf-8", newline="") as stream:
                if args.format == "csv":
                    _write_csv(results, stream)
                else:
                    json.dump(results if args.batch else results[0], stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
        elif args.format == "csv":
            _write_csv(results, sys.stdout)
        else:
            json.dump(results if args.batch else results[0], sys.stdout, indent=2, ensure_ascii=False)
            print()
        return 0 if all(item["success"] for item in results) else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
