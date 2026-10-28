"""Train one frozen manuscript graph configuration from bundled fixed inputs.

No split discovery, tuning or model selection is performed. The default dry-run
validates the exact roles and configuration; --run explicitly starts 50 epochs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from fixed_inputs import load_inputs
from polymer_dmpnn.graph import PeriodicDMPNN, batch_graphs, build_trimer_graph
from polymer_dmpnn.paper import MultitaskDMPNN


def learning_rate(step, n, epochs=50):
    per = math.ceil(n / 16)
    warm, total = 2 * per, epochs * per
    return (1e-4 + step * 9e-4 / warm if step <= warm
            else 1e-3 * .1 ** ((step - warm) / (total - warm)))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=['table3', 'table4', 'table5'], default='table3')
    p.add_argument('--seed', type=int, choices=range(32101, 32106), default=32101)
    p.add_argument('--training-seed', type=int, help='Optional new training randomness; fixed data split still uses --seed')
    p.add_argument('--arm', choices=['open', 'periodic', 'periodic_rank', 'balanced_control', 'multitask'], default='periodic')
    p.add_argument('--budget', type=int, choices=[60, 180, 600, 1800], default=600)
    p.add_argument('--output', type=Path, help='New directory, required only with --run')
    p.add_argument('--run', action='store_true', help='Explicitly start the fixed 50-epoch training')
    args = p.parse_args(argv)
    allowed = {'table3': ['open', 'periodic', 'periodic_rank'],
               'table4': ['open', 'periodic'],
               'table5': ['open', 'periodic', 'balanced_control', 'multitask']}
    if args.arm not in allowed[args.experiment]:
        p.error('Arm not part of the selected manuscript experiment')
    if args.experiment != 'table4' and args.budget != 600:
        p.error('Tables 3 and 5 use 600 fitting records')
    if args.run and args.output is None:
        p.error('--run requires a new --output directory')
    if args.training_seed is not None and not 0 <= args.training_seed < 2**31 - 40000:
        p.error('--training-seed must be between 0 and 2**31-40001')
    return args


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_configuration(args, callback=None):
    training_seed = args.seed if args.training_seed is None else args.training_seed
    records, splits, manifest = load_inputs()
    split = splits[str(args.seed)][args.experiment]
    train = split['train'][:args.budget]
    valid = split['validation']
    target = np.array([float(records[i]['kappa_W_mK']) for i in train])
    mu, sd = float(target.mean()), float(target.std())
    q90, q95 = [float(np.sort(target)[math.ceil(q * len(target)) - 1]) for q in [.9, .95]]
    config = {'experiment': args.experiment, 'seed': args.seed, 'split_seed': args.seed,
              'training_seed': training_seed, 'arm': args.arm,
              'fit_budget': len(train), 'evaluation_n': len(valid), 'epochs': 50,
              'batch_size': 16, 'target_mean': mu, 'target_sd': sd,
              'train_q90': q90, 'train_q95': q95,
              'source_revision': manifest['source']['revision'],
              'input_manifest_sha256': hashlib.sha256((Path(__file__).parent / 'inputs/manifest.json').read_bytes()).hexdigest(),
              'run_started': args.run}
    if not args.run:
        print(json.dumps(config, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cache = {}
    for rid in train + valid:
        smiles = records[rid]['smiles']
        if smiles not in cache:
            cache[smiles] = build_trimer_graph(smiles, periodic_closure=args.arm != 'open')
    rng = np.random.default_rng(training_seed + 10000)
    permutations = [rng.permutation(len(train)) for _ in range(50)]
    binary = target >= q90
    hi, lo = np.flatnonzero(binary), np.flatnonzero(~binary)
    balanced_batches = []
    if args.arm == 'balanced_control':
        rng = np.random.default_rng(training_seed + 1337)
        for _ in range(50):
            epoch = []
            for _ in range(math.ceil(len(train) / 16)):
                ix = np.concatenate([rng.choice(hi, 4, replace=True), rng.choice(lo, 12, replace=True)])
                rng.shuffle(ix)
                epoch.append(ix)
            balanced_batches.append(epoch)
    torch.manual_seed(training_seed)
    model = MultitaskDMPNN() if args.arm == 'multitask' else PeriodicDMPNN(133, 14, hidden_size=128, depth=3, dropout=.1)
    initial_state_sha256 = state_digest(model)
    torch.manual_seed(training_seed + 20000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    start, step = time.monotonic(), 0
    history = []
    for epoch in range(50):
        loss_total, observations = 0.0, 0
        batches = (balanced_batches[epoch] if args.arm == 'balanced_control' else
                   [permutations[epoch][j:j + 16] for j in range(0, len(train), 16)])
        for ix in batches:
            batch = batch_graphs([cache[records[train[i]]['smiles']] for i in ix])
            y = torch.as_tensor((target[ix] - mu) / sd, dtype=torch.float32)
            step += 1
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step, len(train))
            optimizer.zero_grad()
            out = model(batch)
            if args.arm == 'multitask':
                prediction, score = out
                loss = ((prediction - y) ** 2).mean() + .2 * nn.functional.binary_cross_entropy_with_logits(
                    score, torch.as_tensor(binary[ix], dtype=torch.float32),
                    pos_weight=torch.tensor(float(len(lo) / len(hi))))
            else:
                loss = (out - y) ** 2
                if args.arm == 'balanced_control':
                    loss = loss * torch.as_tensor(np.where(binary[ix], (len(hi) / len(train)) / .25,
                                                          (len(lo) / len(train)) / .75), dtype=torch.float32)
                loss = loss.mean()
                if args.arm == 'periodic_rank':
                    raw = torch.as_tensor(target[ix], dtype=torch.float32)
                    delta = raw[:, None] - raw[None, :]
                    eligible = (torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1)
                                & (torch.abs(delta) >= .02)
                                & ((raw[:, None] >= q90) | (raw[None, :] >= q90)))
                    if eligible.any():
                        difference = out[:, None] - out[None, :]
                        loss = loss + .3 * nn.functional.softplus(-torch.sign(delta[eligible]) * difference[eligible]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; run stopped')
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(ix)
            observations += len(ix)
        entry = {'epoch': epoch + 1, 'train_objective': loss_total / observations,
                 'optimizer_steps': step, 'learning_rate': optimizer.param_groups[0]['lr'],
                 'elapsed_seconds': time.monotonic() - start}
        history.append(entry)
        with (args.output / 'training_curve.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(entry)); w.writeheader(); w.writerows(history)
        if callback is not None:
            callback(entry)
        print(f'epoch={epoch+1}/50 elapsed_seconds={time.monotonic()-start:.1f}', flush=True)
    checkpoint = {'state_dict': model.state_dict(), 'seed': args.seed, 'arm': args.arm,
                  'target_mean': mu, 'target_sd': sd, 'fit_budget': len(train)}
    if args.training_seed is not None:
        checkpoint['training_seed'] = training_seed
    torch.save(checkpoint, args.output / 'weights.pt')
    # Evaluation happens after fitting; no validation-based stopping or selection.
    model.eval()
    values, scores = [], []
    with torch.inference_mode():
        for off in range(0, len(valid), 16):
            out = model(batch_graphs([cache[records[i]['smiles']] for i in valid[off:off+16]]))
            if args.arm == 'multitask':
                out, logits = out
                scores.extend(logits.numpy().tolist())
            values.extend((out.numpy() * sd + mu).tolist())
    with (args.output / 'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['record_id', 'true', 'prediction', 'class_score'])
        for j, rid in enumerate(valid):
            w.writerow([rid, records[rid]['kappa_W_mK'], values[j], scores[j] if scores else ''])
    (args.output / 'complete.json').write_text(json.dumps({'completed': True, 'steps': step,
        'training_seed': training_seed, 'initial_state_sha256': initial_state_sha256,
        'final_state_sha256': state_digest(model), 'pretrained_weights_loaded': False,
        'weight_sha256': hashlib.sha256((args.output / 'weights.pt').read_bytes()).hexdigest(),
        'elapsed_seconds': time.monotonic() - start}, indent=2) + '\n')
    return 0


def main(argv=None):
    return train_configuration(arguments(argv))

if __name__ == '__main__':
    raise SystemExit(main())
