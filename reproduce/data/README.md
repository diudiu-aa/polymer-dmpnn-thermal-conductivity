# Saved evaluation data for table recalculation

`saved_predictions.csv.gz` contains saved record-level predictions and already
evaluated thermal-conductivity references needed for main Tables 1–5.
`manifest.json` records source paths (historical identifiers, not runtime
dependencies), source SHA256 values, model/seed/budget identities, dtypes,
thresholds, metric definitions, and expected statistics. Prepared 2026-10-28.

There are 117 prediction groups and 40,848 evaluation rows. The same record can
appear under multiple model/seed/budget groups. Group IDs and saved row order
are retained; counts do not imply independent materials, datasets or
replications. Table 4's 600-fit point reuses the corresponding Table-3 groups.

PolyOmics values are MD thermal conductivities near 300 K and 1 atm, W/(m·K),
from snapshot `041e5834ea1a48682fae12dc39ccd723bcd4f771`, DOI
[10.57967/hf/7475](https://doi.org/10.57967/hf/7475). Table 1 joins saved
D-MPNN V predictions to already evaluated V reference values by exact record
ID; equality of selected ID sets was checked during packaging. Table 3–5
truth comes from their saved evaluated prediction CSVs. This package does not
reopen the original raw database or the original held-out test labels.

OPC data are the 579 already evaluated labeled V records selected from the
saved 2,364-row OPC prediction table. C rows and missing targets are excluded.
Original structural-cluster IDs are retained for cluster-macro MAE. Source:
`alexliu99/neurips-open-polymer-prediction-2025-test-data`; report:
[arXiv:2512.08896](https://arxiv.org/abs/2512.08896). This is a postcompetition
public computational cohort, not experimental validation.

The gzip includes group ID, record ID, OPC cluster ID where applicable,
evaluation reference value, prediction, and optional classifier score. This directory does not contain the full source database, a training-label
table, PoLyInfo exports, restricted experimental labels, manuscript documents
or internal project logs. The separately bundled fixed training/evaluation
inputs are in [../inputs/](../inputs/). Fixed q95
thresholds are metadata from the original training sets.

Source data retain their applicable license and attribution terms. The release
owner records these separately in the root [THIRD_PARTY_DATA.md](../../THIRD_PARTY_DATA.md).
This provenance record grants no additional rights, and the software license
must not be assumed to replace source-data terms.

This package reproduces metrics from saved predictions. It does not generate
predictions, retrain models, reconstruct curated inputs from raw data, or
establish independent generalization. The manuscript's evidence limitations
remain applicable.
