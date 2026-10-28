"""Fresh inference of frozen Table-5 models, followed by a separate comparison.

No stored predictions are opened until every requested model has produced a new
prediction file. No fitting, split selection or hyperparameter changes occur.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
# This public checkout is sufficient even when launched as an isolated script.
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CORE_MODELS = tuple(f'table5_B600_{arm}_{seed}' for seed in range(32101, 32106)
                    for arm in ('open', 'periodic', 'multitask'))
ATOL = 1e-5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def fresh_predictions(output, selected, progress):
    """Only structures, fixed roles and weights enter inference, never old scores."""
    import numpy as np
    import torch
    from rdkit import rdBase
    from fixed_inputs import load_inputs
    from polymer_dmpnn.graph import build_trimer_graph
    from polymer_dmpnn.paper import PaperPredictor, paper_registry
    registry = paper_registry()
    if digest(REPOSITORY / 'reproduce/inputs/manifest.json') != registry['reference_input_manifest_sha256']:
        raise ValueError('Fixed input manifest differs from the published checkpoint reference')
    records, splits, manifest = load_inputs()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    total = sum(len(splits[str(registry['models'][key]['seed'])]['table5']['validation']) for key in selected)
    completed, batches, cache = 0, 0, OrderedDict()
    generated = []
    for index, key in enumerate(selected):
        predictor = PaperPredictor(key)
        ids = splits[str(predictor.spec['seed'])]['table5']['validation']
        if len(ids) != 400:
            raise ValueError('The frozen Table-5 evaluation set must contain 400 records')
        rows = []
        for start in range(0, len(ids), 16):
            graphs = []
            for rid in ids[start:start + 16]:
                ck = (records[rid]['smiles'], predictor.periodic)
                if ck not in cache:
                    cache[ck] = build_trimer_graph(ck[0], periodic_closure=ck[1])
                cache.move_to_end(ck)
                graphs.append(cache[ck])
                if len(cache) > 1024:
                    cache.popitem(last=False)
            values, logits = predictor.predict_graphs(graphs)
            if logits is not None and not np.isfinite(logits).all():
                raise ValueError('Nonfinite classification score')
            for j, rid in enumerate(ids[start:start + 16]):
                rows.append({'record_id': rid, 'smiles': records[rid]['smiles'],
                             'y_true': records[rid]['kappa_W_mK'],
                             'prediction': repr(float(values[j])),
                             'class_score': repr(float(logits[j])) if logits is not None else ''})
            completed += len(values)
            batches += 1
            progress(phase='predicting', model_id=key, models_completed=index,
                     predictions_completed=completed, predictions_total=total,
                     model_predictions_completed=len(rows), model_predictions_total=len(ids),
                     forward_batches=batches)
        file = output / f'{key}.csv'
        with file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        generated.append({'model_id': key, 'file': file.name, 'sha256': digest(file),
                          'weight_sha256': predictor.spec['sha256'], 'n': len(rows)})
        progress(models_completed=index + 1)
        print(json.dumps({'event': 'fresh_model_completed', **generated[-1],
                          'models_completed': index + 1, 'forward_batches': batches}), flush=True)
    environment = {'python': sys.version.split()[0], 'torch': torch.__version__,
                   'numpy': np.__version__, 'rdkit': rdBase.rdkitVersion, 'device': 'CPU', 'threads': 1}
    receipt = {'phase': 'fresh inference completed before stored predictions were loaded',
               'models': generated, 'forward_batches': batches, 'predictions': completed,
               'input_manifest_sha256': digest(REPOSITORY / 'reproduce/inputs/manifest.json'),
               'weight_manifest_sha256': digest(REPOSITORY / 'polymer_dmpnn/paper_weights/manifest.json'),
               'runner_sha256': digest(__file__), 'environment': environment,
               'source_revision': manifest['source']['revision']}
    atomic_json(output / 'fresh_prediction_receipt.json', receipt)
    print(json.dumps({'event': 'all_fresh_predictions_saved', 'predictions': completed,
                      'stored_predictions_loaded': False}), flush=True)
    return receipt


def ranking(values):
    return sorted(range(len(values)), key=lambda i: -values[i])


def compare_predictions(output, receipt, reference_dir, progress):
    # This is the first point where original scores are read.
    from paper_tables import load_inputs, group_metrics, parse_prediction
    from polymer_dmpnn.paper import paper_registry
    manifest, references = load_inputs(reference_dir)
    registry = paper_registry()['models']
    groups = {(g['source_table'], g['seed'], g['model'], g['budget']): g for g in manifest['groups']}
    fresh_metrics, stored_metrics, per_group, plot_candidates, ranking_rows = {}, {}, [], [], []
    selected_groups = set()
    max_difference, max_logit_difference = 0.0, 0.0
    progress(phase='comparing', models_completed=len(receipt['models']))
    for item in receipt['models']:
        spec = registry[item['model_id']]
        group = groups[(5, spec['seed'], spec['arm'], spec['fit_budget'])]
        old = references[group['id']]
        with (output / item['file']).open(newline='') as f:
            new = list(csv.DictReader(f))
        if [r['record_id'] for r in old] != [r['record_id'] for r in new]:
            raise ValueError('Ordered evaluation identity differs: ' + item['model_id'])
        if any(float(a['y_true']) != float(b['y_true']) for a, b in zip(old, new)):
            raise ValueError('Reference labels differ: ' + item['model_id'])
        old_values = [parse_prediction(r['prediction'], group['prediction_dtype']) for r in old]
        new_values = [parse_prediction(r['prediction'], group['prediction_dtype']) for r in new]
        differences = [abs(a - b) for a, b in zip(old_values, new_values)]
        old_order, new_order = ranking(old_values), ranking(new_values)
        old_ranks = {j: rank + 1 for rank, j in enumerate(old_order)}
        new_ranks = {j: rank + 1 for rank, j in enumerate(new_order)}
        class_delta = None
        if group.get('classifier_scores'):
            class_delta = max(abs(float(a['class_score']) - float(b['class_score'])) for a, b in zip(old, new))
            max_logit_difference = max(max_logit_difference, class_delta)
            old_class_order = ranking([float(r['class_score']) for r in old])
            new_class_order = ranking([float(r['class_score']) for r in new])
        old_metrics = group_metrics(group, old)
        current = group_metrics(group, new)
        fresh_metrics[group['id']] = current
        stored_metrics[group['id']] = old_metrics
        selected_groups.add(group['id'])
        maximum = max(differences)
        max_difference = max(max_difference, maximum)
        per_group.append({'model_id': item['model_id'], 'group_id': group['id'], 'n': len(new),
                          'max_abs_prediction_difference': maximum,
                          'mean_abs_prediction_difference': statistics.fmean(differences),
                          'max_abs_logit_difference': class_delta,
                          'prediction_tolerance_pass': maximum <= ATOL and (class_delta is None or class_delta <= ATOL),
                          'identical_full_ranking': old_order == new_order,
                          'top10_set_equal': set(old_order[:10]) == set(new_order[:10]),
                          'top100_set_equal': set(old_order[:100]) == set(new_order[:100]),
                          'classifier_top100_set_equal': (set(old_class_order[:100]) == set(new_class_order[:100])) if class_delta is not None else None,
                          'old_metrics': old_metrics, 'new_metrics': current,
                          'metric_differences': {k: current[k] - old_metrics[k] for k in current},
                          'weight_sha256': item['weight_sha256'], 'new_prediction_sha256': item['sha256']})
        for i, row in enumerate(new):
            ranking_rows.append({'model_id': item['model_id'], 'record_id': row['record_id'],
                                 'previous_rank': old_ranks[i], 'new_rank': new_ranks[i],
                                 'rank_change': old_ranks[i] - new_ranks[i],
                                 'previous_prediction': old_values[i], 'new_prediction': new_values[i]})
            plot_candidates.append({'model_id': item['model_id'], 'record_id': row['record_id'],
                                    'previous': old_values[i], 'current': new_values[i],
                                    'reference': float(row['y_true'])})
    aggregates = []
    displayed_total, displayed_mismatches = 0, 0
    table5 = next(t for t in manifest['tables'] if t['number'] == 5)
    for row in table5['rows']:
        cells = [c for c in row['cells'] if c is not None]
        if not cells or not all(set(c['groups']) <= selected_groups for c in cells):
            continue
        comparisons = []
        for c in cells:
            values = [fresh_metrics[g][c['metric']] * c['scale'] for g in c['groups']]
            old_values = [stored_metrics[g][c['metric']] * c['scale'] for g in c['groups']]
            mean, sd = statistics.fmean(values), statistics.stdev(values)
            old_mean, old_sd = statistics.fmean(old_values), statistics.stdev(old_values)
            mean_text, sd_text = f"{mean:.{c['mean_decimals']}f}", f"{sd:.{c['sd_decimals']}f}"
            displayed_total += 2
            displayed_mismatches += (mean_text != c['expected_mean']) + (sd_text != c['expected_sd'])
            comparisons.append({'metric': c['metric'], 'column': c['column'], 'percent': c['scale'] == 100,
                                'paper_mean': old_mean, 'paper_sd': old_sd,
                                'new_mean': mean, 'new_sd': sd,
                                'mean_difference': mean - old_mean,
                                'new_display': mean_text + ' ± ' + sd_text + ('%' if c['scale'] == 100 else ''),
                                'paper_display': c['expected_mean'] + ' ± ' + c['expected_sd'] + ('%' if c['scale'] == 100 else ''),
                                'display_equal': mean_text == c['expected_mean'] and sd_text == c['expected_sd'],
                                'n_repeats': len(values)})
        aggregates.append({'model': row['labels'][0], 'metrics': comparisons})
    # Fixed index sampling for the browser only; every record was compared above.
    stride = max(1, math.ceil(len(plot_candidates) / 500))
    sample = plot_candidates[::stride]
    worst = max(plot_candidates, key=lambda p: abs(p['previous'] - p['current']))
    if worst not in sample:
        sample.append(worst)
    with (output / 'candidate_rankings.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ranking_rows[0])); w.writeheader(); w.writerows(ranking_rows)
    with (output / 'metric_comparison.csv').open('w', newline='') as f:
        fields = ['model', 'metric', 'paper_mean', 'paper_sd', 'new_mean', 'new_sd', 'mean_difference', 'display_equal']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader()
        for row in aggregates:
            for m in row['metrics']:
                w.writerow({'model': row['model'], **m})
    return {'per_group': per_group, 'aggregates': aggregates, 'scatter': sample,
            'comparison_reference': {'manifest_sha256': digest(reference_dir / 'manifest.json'),
                                     'predictions_sha256': manifest['input']['sha256']},
            'scatter_total': len(plot_candidates), 'scatter_sampling': 'fixed index stride; includes greatest-difference record',
            'summary': {'groups_tested': len(per_group), 'predictions_tested': receipt['predictions'],
                        'forward_batches': receipt['forward_batches'], 'prediction_atol': ATOL,
                        'max_abs_prediction_difference': max_difference, 'max_abs_logit_difference': max_logit_difference,
                        'prediction_tolerance_pass': all(g['prediction_tolerance_pass'] for g in per_group),
                        'identical_top100_groups': sum(g['top100_set_equal'] for g in per_group),
                        'identical_full_ranking_groups': sum(g['identical_full_ranking'] for g in per_group),
                        'display_statistics_checked': displayed_total, 'display_mismatches': displayed_mismatches}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=CORE_MODELS, default=list(CORE_MODELS))
    parser.add_argument('--reference-data', type=Path, default=REPOSITORY / 'reproduce/data')
    args = parser.parse_args(argv)
    if len(args.models) != len(set(args.models)):
        parser.error('Duplicate model IDs are not allowed')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {'state': 'running', 'phase': 'loading', 'models_total': len(args.models),
             'models_completed': 0, 'predictions_completed': 0, 'predictions_total': len(args.models) * 400,
             'started_at': datetime.now(timezone.utc).isoformat()}
    def progress(**update):
        state.update(update);state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.output / 'progress.json', state)
        if state.get('elapsed_seconds', 0) > 600:
            raise TimeoutError('The fixed 600-second retest limit was reached')
    progress()
    try:
        receipt = fresh_predictions(args.output, args.models, progress)
        result = compare_predictions(args.output, receipt, args.reference_data, progress)
        result.update({'mode': 'fresh checkpoint inference; no training', 'started_at': state['started_at'],
                       'completed_at': datetime.now(timezone.utc).isoformat(),
                       'elapsed_seconds': time.monotonic() - started, 'receipt': receipt,
                       'scope': 'Table-5 open/periodic/multitask fixed evaluation only. Repeated record predictions are not independent materials. Not a full-paper retraining or fresh external validation.'})
        atomic_json(args.output / 'result.json', result)
        progress(state='completed', phase='completed')
        print(json.dumps(result['summary']), flush=True)
        return 0 if result['summary']['prediction_tolerance_pass'] else 2
    except Exception as exc:
        state.update(state='failed', phase='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output / 'progress.json', state)
        print(state['error'], file=sys.stderr, flush=True)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
