# Reproduce manuscript Tables 1–5

Run this command from a downloaded repository with Python 3.10 or newer:

```bash
python reproduce/paper_tables.py --verify --output reproduced-paper-tables
```

The output directory must not already exist. The script uses only Python's
standard library, the bundled `data/manifest.json`, and
`data/saved_predictions.csv.gz`. It does not require installation of the model
package, a network connection, or any original project directory. An absolute
path to `paper_tables.py` also works from another working directory.

This command recalculates **saved-prediction metrics**. It does not retrain
models or run their inference. The inputs contain 117 model/seed/budget groups
and 40,848 saved evaluation rows, including repeated structures across models
and seeds. Those counts are not numbers of independent polymers or cohorts.
The manifest records source-file SHA256 values, group identities, original
prediction dtypes, fixed training q95, aggregation definitions, and exact
manuscript display targets. Source details are in [data/README.md](data/README.md);
source licensing and attribution are maintained in the root
[THIRD_PARTY_DATA.md](../THIRD_PARTY_DATA.md).

The command writes:

- `table_1.csv` through `table_5.csv` and equivalent Markdown tables;
- `per_seed_metrics.csv`, recalculated from the record-level inputs;
- `display_statistics.csv`, all 182 means and sample standard deviations,
  including unrounded computed values and comparisons at manuscript precision;
- `verification.json`, the integrity and numerical verification summary.

`--verify` checks all 182 displayed statistics and saved per-seed reference
metrics. Any mismatch exits with status 1. SHA256 checks of the frozen manifest
and compressed input always run, even without `--verify`. A changed release
must intentionally update its manifest fingerprint in the script. Expected
statistics are verification targets, never the source of generated results.

The calculation conventions are:

- Table 2 uses equal weight per structural cluster for MAE, and record-level
  R², restricted to the 579 labeled OPC V records. It does not substitute
  record-MAE for cluster-macro MAE.
- Table 3/5 tails use saved nearest-rank training-set q95; evaluation labels do
  not determine the threshold. Precision@K is predicted/reference top-K
  overlap, with stable saved-row-order tie breaking.
- Tables average metrics calculated separately for each seed and report
  sample SD (`ddof=1`). This differs from averaging predictions before
  calculating an ensemble's MAE or R². The inference CLI returns an ensemble
  prediction and is not a table-generation command.
- Ridge's saved decimals represent float32 values. The script restores IEEE
  float32 first, then scores in float64. Other predictions are scored in
  float64. Tanimoto-GPR's old seed reference metrics were printed to ten
  decimal places; the explicit tolerance accounts for that source precision.
  Manuscript display comparisons use exact displayed text.
- The 600-fit point in Table 4 reuses the corresponding frozen Table-3
  predictions, as in the original analysis; no duplicate fit is performed.

Run the reproduction tests with:

```bash
python -m unittest discover -s tests -p test_paper_reproduction.py -v
```

The tests run a copy of the script/data from an unrelated temporary directory,
verify all 182 statistics and paired record/truth identities, test OPC cluster
weighting, and check that changes to predictions or the threshold manifest
are rejected.

## Evaluate the saved main-study graph checkpoints

Install the package (`python -m pip install .`) first, then:

```bash
python reproduce/evaluate_checkpoints.py --output checkpoint-verification
```

This runs all 65 saved graph fits on their fixed 400-record evaluation sets.
Select individual manifest IDs with `--models ID [ID ...]`. Both regression
predictions and multitask logits are compared with the saved per-record
results. The default absolute tolerance is `1e-5`; a mismatch stops with exit 1.
Native checkpoint identity, SHA256 and target standardization are checked.

## Train from fixed curated inputs

`inputs/records.csv.gz` contains 14,112 distinct records from the historical
PolyOmics train partition. `inputs/splits.json` preserves ordered fitting and
evaluation roles for five seeds. Hashes, dataset snapshot and source hashes
are in `inputs/manifest.json`; licensing is in the root THIRD_PARTY_DATA.md.
No original private project tree or manual preparation of role tables is
required. Tables 1–2 use the saved-prediction reproduction path above.

```bash
# Validate exact roles/configuration without starting training.
python reproduce/train_graph_models.py --experiment table3 --arm periodic --seed 32101

# Table 3: arms open / periodic / periodic_rank, 600 fit + 400 evaluation.
python reproduce/train_graph_models.py --experiment table3 --arm periodic --seed 32101 --run --output retrained-table3

# Table 4: arms open / periodic, nested budgets 60 / 180 / 600 / 1800.
python reproduce/train_graph_models.py --experiment table4 --budget 60 --arm periodic --seed 32101 --run --output retrained-table4

# Table 5: arms open / periodic / balanced_control / multitask, 600 + 400.
python reproduce/train_graph_models.py --experiment table5 --arm multitask --seed 32101 --run --output retrained-table5

# Table 5 tuned classical baselines; 3-fold GroupKFold is training-only.
python -m pip install '.[reproduce]'
python reproduce/train_baselines.py --arm ridge --seed 32101 --run --output retrained-ridge
python reproduce/train_baselines.py --arm extra_trees --seed 32101 --run --output retrained-extra-trees
```

Use a new output directory for every explicit run. Seeds are 32101–32105.
Without `--run`, inputs/configuration are checked and no fitting starts.
`strict_cluster_compare.py` is now a compatibility alias to the Table-3
fixed-input runner; the broken private `--project-root` interface was removed.
The scripts do not tune on validation data, adjust recipes or launch multiple
experiments implicitly. Each graph fit uses the original 50 epochs, batch 16,
Noam-style schedule, target scaling and seed offsets. Table 4's 600-fit point
is the same Table-3 configuration and need not be retrained a second time.

After fitting, `predictions.csv` contains the ordered reference/prediction
pairs; `configuration.json` records training q90/q95 and source identity.
The graph checkpoint uses `state_dict` and original target scaling; no pickle
model objects or executable imports from a private project are needed.

Saved-result verification is exact at the manuscript's display precision.
Full training parity depends on numerical libraries and platform: the original
CPU one-thread environment is recorded in `inputs/manifest.json`, and fresh
training output is separate from the authoritative saved-result bundle.
The package does not claim to reconstruct all historical raw-data curation,
GIN/GPR training, or MD simulation production. All evaluations remain
retrospective computational evaluations, not blind experimental validation.

## Recompute a separately completed run directory

The older utility remains available when a run directory contains evaluated
predictions, `fitted_prediction_lock.json`, and `results.json`:

```bash
python reproduce/recompute_metrics.py --run-dir /path/to/run-output --verify-existing
```

It requires the `reproduce` optional dependencies. For Tables 1–5 from the
repository's bundled data, use `paper_tables.py` instead.


## Fresh inference and metric/ranking comparison in one run

```bash
python reproduce/retest_models.py --output fresh-model-retest
```

Runs the 15 fixed Table-5 open/periodic/multitask checkpoints. New prediction CSVs
and their receipt are written **before** loading stored predictions for comparison.
The result includes per-group metrics/rank differences, 34 covered displayed
statistics, candidate_rankings.csv, metric_comparison.csv, environment/input/weight
identities and result.json. All 6,000 outputs are checked; the browser scatter
uses fixed-index sampling only. --models selects a subset of these 15 IDs, e.g.
for a short run; it does not change weights, thresholds or splits. --reference-data
can point to a copy of the frozen public reference package; integrity checks still
apply. No fitting occurs. The local browser uses the same runner asynchronously
and allows stopping; it does not automatically start this computation on tab open.


## New random-initialization training trials

```bash
python reproduce/train_from_scratch.py --repeats 1 --output fresh-training
python reproduce/train_from_scratch.py --repeats 5 --training-seed 123456 --output repeatable-training
```

No released .pt file is needed. The original fixed Table-5 split(s), 600 fitting
labels, 400 evaluation records, three arms and 50-epoch recipe remain unchanged.
--training-seed controls only initialization/order/dropout, while split seeds
32101..32105 retain their original roles. The default draws a new recorded start.
Original predictions are read only after all new models and predictions are saved.
A single repeat is compared to its matching original split with no SD; five
repeats report sample SD. New errors, ranks and differences are reported even
when worse. The workflow does not tune on evaluation data or retry for a better
score. It writes optimizer steps, initial/final tensor hashes, curves, new weights,
new predictions and comparison outputs. Use the browser for live progress/cancel.
The legacy train_graph_models.py without --training-seed retains exact original
randomness; supplying that new flag leaves --seed as the data-partition selector.


## Generate the paper core results with the original settings

```bash
python reproduce/train_from_scratch.py --paper-settings --output paper-main-results
```

Uses the original five split seeds and training seeds 32101..32105, the original
50-epoch recipe, and no released checkpoint loads. Generates paper_main_results.csv
from the new predictions and paper_consistency.json for the 34 covered Table-5
display statistics. Covers open/periodic/multitask and classifier ranking only.
Exit 0 means the covered displayed statistics match; exit 2 retains genuine
outputs with display differences. Per-record prediction tolerance is reported
separately and is not substituted for display-level agreement. This is the
default browser mode; new random-start experiments remain optional.
