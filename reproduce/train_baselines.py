"""Table 5 training-cluster CV for Morgan Ridge / ExtraTrees on fixed inputs."""
import argparse
import csv
import json
from pathlib import Path

from fixed_inputs import load_inputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=range(32101, 32106), default=32101)
    p.add_argument('--arm', choices=['ridge', 'extra_trees'], default='ridge')
    p.add_argument('--output', type=Path, help='New output directory')
    p.add_argument('--run', action='store_true', help='Start training-only grouped CV and final fit')
    args = p.parse_args(argv)
    records, splits, _ = load_inputs()
    split = splits[str(args.seed)]['table5']
    if not args.run:
        print(json.dumps({'experiment': 'table5', 'seed': args.seed, 'arm': args.arm,
                          'fit_n': 600, 'evaluation_n': 400, 'cv': 'GroupKFold(3)', 'run_started': False}, indent=2))
        return 0
    if args.output is None:
        p.error('--run requires a new --output directory')
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold
    args.output.mkdir(parents=True, exist_ok=False)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cache = {}
    for rid in split['train'] + split['validation']:
        smi = records[rid]['smiles']
        if smi not in cache:
            values = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(smi)), values)
            cache[smi] = values
    train, valid = split['train'], split['validation']
    X = np.stack([cache[records[i]['smiles']] for i in train])
    V = np.stack([cache[records[i]['smiles']] for i in valid])
    y = np.array([float(records[i]['kappa_W_mK']) for i in train])
    groups = np.array([records[i]['cluster_id'] for i in train])
    folds = list(GroupKFold(3).split(X, y, groups))
    grid = ([{'alpha': x} for x in [.1, 1., 10., 100.]] if args.arm == 'ridge' else
            [{'min_samples_leaf': leaf, 'max_features': frac} for leaf in [1, 3] for frac in [.3, 1.]])
    def build(parameters):
        return (Ridge(**parameters) if args.arm == 'ridge' else
                ExtraTreesRegressor(n_estimators=150, random_state=args.seed, n_jobs=1, **parameters))
    candidates = []
    for parameters in grid:
        scores = []
        for fit, evaluate in folds:
            model = build(parameters)
            model.fit(X[fit], y[fit])
            scores.append(float(mean_absolute_error(y[evaluate], model.predict(X[evaluate]))))
        candidates.append({'parameters': parameters, 'fold_mae': scores, 'mean_mae': float(np.mean(scores))})
    winner = min(range(len(candidates)), key=lambda j: candidates[j]['mean_mae'])
    model = build(grid[winner])
    model.fit(X, y)
    predictions = model.predict(V)
    with (args.output / 'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['record_id', 'true', 'prediction'])
        # Preserve the original float32 shortest decimal serialization for Ridge.
        w.writerows((rid, records[rid]['kappa_W_mK'], pred) for rid, pred in zip(valid, predictions))
    (args.output / 'selection.json').write_text(json.dumps({'seed': args.seed, 'arm': args.arm,
        'selected': grid[winner], 'inner_cv': candidates, 'prediction_dtype': str(predictions.dtype)}, indent=2) + '\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
