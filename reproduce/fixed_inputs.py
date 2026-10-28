"""Verified portable curated inputs shared by evaluation and training."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent / 'inputs'

def load_inputs():
    manifest = json.loads((INPUT_DIR / 'manifest.json').read_text())
    for name, spec in manifest['files'].items():
        if hashlib.sha256((INPUT_DIR / name).read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
    with gzip.open(INPUT_DIR / 'records.csv.gz', 'rt', newline='') as f:
        rows = list(csv.DictReader(f))
    records = {r['record_id']: r for r in rows}
    if len(records) != len(rows) or len(rows) != manifest['files']['records.csv.gz']['records']:
        raise ValueError('Duplicate or missing input records')
    splits = json.loads((INPUT_DIR / 'splits.json').read_text())
    for seed, experiments in splits.items():
        for experiment, split in experiments.items():
            train, valid = split['train'], split['validation']
            if (len(set(train)) != len(train) or len(set(valid)) != len(valid)
                    or set(train) & set(valid)):
                raise ValueError(f'Record overlap: {seed}/{experiment}')
            for key in ('cluster_id', 'isolation_group', 'source_uuid'):
                if {records[i][key] for i in train} & {records[i][key] for i in valid}:
                    raise ValueError(f'{key} overlap: {seed}/{experiment}')
    return records, splits, manifest
