# Reproducing the five-seed cluster comparison

`strict_cluster_compare.py` is the training and evaluation entry point for the
five-seed open, periodic and periodic-plus-ranking comparison. This is distinct
from the installable inference-only command. It expects the **prepared** source
tables under the following project-root layout:

- `outputs/r03-total-label-budget-2026-10-18/inputs/cluster_holdout_records.csv`
- `outputs/r03-total-label-budget-2026-10-18/inputs/metadata_no_kappa.csv`
- `outputs/r03-total-label-budget-2026-10-18/data/{cluster_holdout,random_group}_s{20261111,20261112,20261113}_L{100,300,1000,3000}_representatives_before_labels.csv`
- `outputs/data-expansion-2026-10-15/raw/general_polymers_with_sp_abbe_dynamic-dielectric.csv`

These source tables are not redistributed in this package. The public raw data
source is PolyOmics dataset DOI 10.57967/hf/7475, snapshot
`041e5834ea1a48682fae12dc39ccd723bcd4f771`; the curated metadata, structural
clusters and historical role files must be obtained/prepared separately. Their
content identities and split manifest are written to the results directory.
The runner excludes recorded historical roles, draws from the historical train
partition, and uses shared held-out structure clusters. Each seed trains on
600 labels and evaluates 400; there is no calibration set. It fits 15 models on
CPU and saves weights, predictions and metrics. Check licensing and source
attribution before distributing the prepared tables.

Install the additional evaluation dependency and run from the repository root:

```bash
python -m pip install '.[reproduce]'
python reproduce/strict_cluster_compare.py --project-root /path/to/prepared-project --output /path/to/new-output
```

The output directory must not already exist. The fit and result procedure is
reproducible from the required prepared inputs and environment; a clean-room
reconstruction of the curated inputs from raw data is not yet packaged, and
cross-platform numerical parity of training has not been established. Do not
interpret these repeated, same-source simulations as blind experimental testing.

## Recompute the saved metrics

Given a completed run directory containing the saved evaluated predictions and
`fitted_prediction_lock.json`, the following command independently recomputes
MAE, RMSE, R², tail error, Spearman correlation and Precision@10/20/100:

```bash
python reproduce/recompute_metrics.py --run-dir /path/to/run-output --verify-existing
```

This command was checked locally against all 15 saved results. It does not
retrain, access the original dataset, or independently establish generalization.
