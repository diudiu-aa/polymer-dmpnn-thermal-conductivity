"""Train new paired models from random initialization, then evaluate once.

The split IDs, 600 fitting labels, 400 evaluation records and 50-epoch recipe
stay fixed. A new recorded training seed controls initialization/order/dropout.
Published .pt checkpoints are not required or read by this workflow.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import secrets
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retest_models import atomic_json, compare_predictions, digest

ARMS = ('open', 'periodic', 'multitask')


def summarize_trials(per_group):
    rows = []
    for arm, label in [('open', 'Open D-MPNN'), ('periodic', 'Periodic D-MPNN'),
                       ('multitask', 'Periodic + multitask, regression ranking')]:
        groups = [g for g in per_group if f'_{arm}_' in g['model_id']]
        if not groups:
            continue
        metrics = []
        for key in ('mae', 'r2', 'tail_mae', 'precision_at_10', 'precision_at_100'):
            scale = 100 if key.startswith('precision') else 1
            old = [g['old_metrics'][key] * scale for g in groups]
            new = [g['new_metrics'][key] * scale for g in groups]
            old_mean, new_mean = statistics.fmean(old), statistics.fmean(new)
            old_sd = statistics.stdev(old) if len(old) > 1 else None
            new_sd = statistics.stdev(new) if len(new) > 1 else None
            digits = 1 if scale == 100 else 4 if key == 'r2' else 5
            def display(mean, sd):
                return (f'{mean:.{digits}f}' + (f' ± {sd:.{digits}f}' if sd is not None else '')
                        + ('%' if scale == 100 else ''))
            metrics.append({'metric': key, 'percent': scale == 100, 'paper_mean': old_mean,
                            'paper_sd': old_sd, 'new_mean': new_mean, 'new_sd': new_sd,
                            'mean_difference': new_mean - old_mean, 'n_repeats': len(groups),
                            'paper_display': display(old_mean, old_sd), 'new_display': display(new_mean, new_sd)})
        rows.append({'model': label, 'metrics': metrics})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, choices=(1, 5), default=1)
    p.add_argument('--training-seed', type=int, help='Record/reuse a specific new training start; default draws a fresh one')
    args = p.parse_args(argv)
    if args.training_seed is not None and not 0 <= args.training_seed < 2**31 - 100000:
        p.error('Invalid training seed')
    args.output.mkdir(parents=True, exist_ok=False)
    base_seed = args.training_seed if args.training_seed is not None else secrets.randbelow(2**31 - 200000) + 100000
    started, started_at = time.monotonic(), datetime.now(timezone.utc).isoformat()
    state = {'state': 'running', 'phase': 'loading', 'started_at': started_at,
             'models_total': args.repeats * len(ARMS), 'models_completed': 0,
             'epochs_total': 50, 'epoch': 0, 'optimizer_steps_total': 0,
             'training_start': base_seed, 'curves': {}, 'predictions_completed': 0}
    def progress(**updates):
        state.update(updates);state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.output / 'progress.json', state)
        if state['elapsed_seconds'] > 1800:
            raise TimeoutError('The 30-minute training limit was reached')
    progress()
    try:
        import numpy as np
        import torch
        from rdkit import rdBase
        from fixed_inputs import load_inputs
        from polymer_dmpnn.paper import paper_registry
        from train_graph_models import arguments, train_configuration
        registry = paper_registry()
        if digest(ROOT / 'reproduce/inputs/manifest.json') != registry['reference_input_manifest_sha256']:
            raise ValueError('Fixed inputs differ from the published source')
        records, splits, manifest = load_inputs()
        split_seeds = list(range(32101, 32101 + args.repeats))
        training_seeds = [base_seed + 100 * i for i in range(args.repeats)]
        plan = {'mode': 'fresh random-initialization training; fixed published splits',
                'split_seeds': split_seeds, 'training_seeds': training_seeds, 'arms': ARMS,
                'epochs': 50, 'fit_labels': 600, 'evaluation_records': 400,
                'pretrained_weights_loaded': False,
                'comparison': 'one matched original split' if args.repeats == 1 else 'five matched original splits',
                'input_manifest_sha256': digest(ROOT / 'reproduce/inputs/manifest.json'),
                'runner_sha256': digest(__file__), 'trainer_sha256': digest(ROOT / 'reproduce/train_graph_models.py')}
        atomic_json(args.output / 'run_configuration.json', plan)
        generated, training_receipts = [], []
        steps_done = 0
        for repeat, (split_seed, training_seed) in enumerate(zip(split_seeds, training_seeds)):
            for arm in ARMS:
                key = f'table5_B600_{arm}_{split_seed}'
                model_output = args.output / key
                history = []
                state['curves'][key] = history
                progress(phase='preparing', model_id=key, repeat=repeat+1, epoch=0,
                         current_training_seed=training_seed)
                def on_epoch(entry):
                    history.append(entry)
                    progress(phase='evaluating' if entry['epoch'] == 50 else 'training',
                             epoch=entry['epoch'], train_objective=entry['train_objective'],
                             optimizer_steps_total=steps_done + entry['optimizer_steps'])
                configuration = arguments(['--experiment', 'table5', '--arm', arm, '--seed', str(split_seed),
                                           '--training-seed', str(training_seed), '--run', '--output', str(model_output)])
                train_configuration(configuration, on_epoch)
                completed = json.loads((model_output / 'complete.json').read_text())
                if completed['steps'] != 1900 or completed['initial_state_sha256'] == completed['final_state_sha256']:
                    raise RuntimeError('Training did not produce the expected optimizer updates')
                steps_done += completed['steps']
                ids = splits[str(split_seed)]['table5']['validation']
                predicted = list(csv.DictReader((model_output / 'predictions.csv').open()))
                if [r['record_id'] for r in predicted] != ids:
                    raise ValueError('Evaluation order changed')
                normalized = [{'record_id': r['record_id'], 'smiles': records[r['record_id']]['smiles'],
                               'y_true': r['true'], 'prediction': r['prediction'], 'class_score': r['class_score']}
                              for r in predicted]
                normalized_file = args.output / (key + '.csv')
                with normalized_file.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(normalized[0]));writer.writeheader();writer.writerows(normalized)
                generated.append({'model_id': key, 'file': normalized_file.name, 'n': len(predicted),
                                  'sha256': digest(normalized_file), 'weight_sha256': completed['weight_sha256']})
                training_receipts.append({'model_id': key, 'split_seed': split_seed, 'training_seed': training_seed,
                                          **completed})
                print(json.dumps({'event': 'fresh_training_completed', **training_receipts[-1]}), flush=True)
                progress(phase='model_completed', epoch=0, models_completed=len(generated), optimizer_steps_total=steps_done,
                         predictions_completed=len(generated) * 400)
                gc.collect()
        receipt = {'models': generated, 'predictions': len(generated)*400,
                   'phase': 'fresh training and evaluation completed before original predictions were loaded',
                   'forward_batches': len(generated)*25, 'optimizer_steps': steps_done,
                   'pretrained_weights_loaded': False, 'training_runs': training_receipts,
                   'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                                   'numpy': np.__version__, 'rdkit': rdBase.rdkitVersion, 'device': 'CPU', 'threads': 1}}
        atomic_json(args.output / 'fresh_training_receipt.json', receipt)
        compared = compare_predictions(args.output, receipt, ROOT / 'reproduce/data', progress)
        for group in compared['per_group']:
            group['matches_original_fixed_weight_predictions'] = group.pop('prediction_tolerance_pass')
            group['fixed_weight_match_expected'] = False
        aggregates = summarize_trials(compared['per_group'])
        # Variation from a newly trained model is an output, never a failure criterion.
        result = {'mode': 'from-scratch training with new recorded randomness', 'started_at': started_at,
                  'completed_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
                  'configuration': plan, 'receipt': receipt, 'curves': state['curves'],
                  'aggregates': aggregates, 'per_group': compared['per_group'], 'scatter': compared['scatter'],
                  'comparison_reference': compared['comparison_reference'],
                  'summary': {'models_trained': len(generated), 'repeats': args.repeats,
                              'predictions_tested': len(generated)*400, 'optimizer_steps': steps_done,
                              'max_abs_prediction_difference': compared['summary']['max_abs_prediction_difference'],
                              'same_top100_groups': compared['summary']['identical_top100_groups'],
                              'training_starts': training_seeds, 'split_seeds': split_seeds,
                              'all_weights_updated': all(r['initial_state_sha256'] != r['final_state_sha256'] for r in training_receipts)},
                  'scope': 'New randomness on fixed public Table-5 splits, not new independent material validation. One repeat compares matched split-level results, not the paper five-repeat mean.'}
        with (args.output / 'metric_comparison.csv').open('w', newline='') as f:
            fields = ['model','metric','n_repeats','paper_mean','paper_sd','new_mean','new_sd','mean_difference']
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore');writer.writeheader()
            for row in aggregates:
                for metric in row['metrics']:
                    writer.writerow({'model': row['model'], **metric})
        atomic_json(args.output / 'result.json', result)
        progress(state='completed', phase='completed')
        return 0
    except Exception as exc:
        state.update(state='failed', phase='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output / 'progress.json', state)
        print(state['error'], file=sys.stderr, flush=True)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
