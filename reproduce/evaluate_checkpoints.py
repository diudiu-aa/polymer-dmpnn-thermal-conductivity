"""Run saved graph weights on published fixed evaluation inputs; no training."""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

from fixed_inputs import load_inputs
from polymer_dmpnn.graph import build_trimer_graph
from polymer_dmpnn.paper import PaperPredictor, paper_registry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', help='Manifest IDs; default: all 65 graph checkpoints')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--atol', type=float, default=1e-5, help='Absolute prediction tolerance W/(m K)')
    args = p.parse_args(argv)
    if not 0 < args.atol <= 1e-5:
        p.error('Tolerance must be positive and no greater than 1e-5')
    records, splits, _ = load_inputs()
    registry = paper_registry()['models']
    selected = args.models or sorted(registry)
    for key in selected:
        if key not in registry:
            p.error(f'Unknown model {key}')
    args.output.mkdir(parents=True, exist_ok=False)
    # This is a standard-library saved-prediction replay, not another model.
    from paper_tables import load_inputs as load_saved
    saved_manifest, saved_records = load_saved(Path(__file__).resolve().parent / "data")
    reference = {}
    for group in saved_manifest['groups']:
        reference[(group['source_table'], group['seed'], group['model'], group['budget'])] = saved_records[group['id']]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cache, summary = {}, []
    for key in selected:
        predictor = PaperPredictor(key)
        spec = predictor.spec
        seed, experiment = str(spec['seed']), spec['experiment']
        ids = splits[seed][experiment]['validation']
        identity = (int(experiment.removeprefix('table')), spec['seed'], spec['arm'], spec['fit_budget'])
        if identity not in reference:
            raise ValueError(f'Missing saved result identity {identity}')
        group = reference[identity]
        by_id = {r['record_id']: r for r in group}
        if set(ids) != set(by_id):
            raise ValueError(f'Evaluation identity mismatch {key}')
        predictions, class_scores = [], []
        for start in range(0, len(ids), 16):
            graphs = []
            for rid in ids[start:start + 16]:
                ck = (records[rid]['smiles'], predictor.periodic)
                if ck not in cache:
                    cache[ck] = build_trimer_graph(ck[0], periodic_closure=ck[1])
                graphs.append(cache[ck])
            values, scores = predictor.predict_graphs(graphs)
            predictions.extend(values.tolist())
            if scores is not None:
                class_scores.extend(scores.tolist())
        delta = max(abs(x - float(by_id[rid]['prediction'])) for rid, x in zip(ids, predictions))
        class_delta = None
        if class_scores:
            class_delta = max(abs(x - float(by_id[rid]['class_score'])) for rid, x in zip(ids, class_scores))
        ok = delta <= args.atol and (class_delta is None or class_delta <= args.atol)
        with (args.output / f'{key}.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['record_id', 'true', 'prediction', 'class_score'])
            for i, rid in enumerate(ids):
                w.writerow([rid, records[rid]['kappa_W_mK'], predictions[i], class_scores[i] if class_scores else ''])
        row = {'model_id': key, 'n': len(ids), 'max_abs_prediction_difference': delta,
               'max_abs_logit_difference': class_delta, 'atol': args.atol, 'pass': ok}
        summary.append(row)
        print(json.dumps(row), flush=True)
        if not ok:
            (args.output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
            raise ValueError(f'Checkpoint output differs from saved result: {key}')
    (args.output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
