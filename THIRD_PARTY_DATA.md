# Data and model provenance

The MIT license in `LICENSE` covers this repository's software and released
trained weights. Source datasets retain their own licenses. This repository
contains curated excerpts and derived predictions, **not the full databases**.

## PolyOmics

- Authors/data provider: Yoshihiro Hayashi and the PolyOmics/RadonPy contributors.
- Dataset: [PolyOmics](https://huggingface.co/datasets/yhayashi1986/PolyOmics),
  DOI [10.57967/hf/7475](https://doi.org/10.57967/hf/7475).
- Fixed source revision: `041e5834ea1a48682fae12dc39ccd723bcd4f771`.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), stated in
  the [fixed-revision data card](https://huggingface.co/datasets/yhayashi1986/PolyOmics/blob/041e5834ea1a48682fae12dc39ccd723bcd4f771/README.md).
- Changes: a subset was selected; repeat units were canonicalized, structure
  isolation/cluster identifiers assigned, fixed roles selected, and model
  predictions added. Original UUIDs are retained in `reproduce/inputs/records.csv.gz`.
- `reproduce/data/saved_predictions.csv.gz`: already evaluated reference labels
  and saved predictions for Tables 1 and 3–5.
- `reproduce/inputs/`: 14,112 distinct fixed-role records for graph training and
  checkpoint evaluation in Tables 3–5; all originate from the historical train
  partition. This is a role-complete subset, not a new external dataset.
- Units: W/(m K); simulated nominally unstretched amorphous bulk near 300 K,
  1 atm. These are simulation references, not new experimental measurements.

## Open Polymer Challenge postcompetition dataset

- Dataset provider: Alex Liu / Open Polymer Challenge organizers.
- [NeurIPS Open Polymer Prediction 2025 Test Data](https://www.kaggle.com/datasets/alexliu99/neurips-open-polymer-prediction-2025-test-data), version 1.
- License: MIT, verified from the dataset's public metadata (`licenseName: MIT`)
  on 2026-10-28; [metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/alexliu99/neurips-open-polymer-prediction-2025-test-data).
- [Competition report](https://arxiv.org/abs/2512.08896).
- Included only as saved evaluation labels/predictions and cluster IDs for the
  previously analyzed 579-record V cohort in Table 2. The public postcompetition
  cohort is retrospective; it is not a new blind or experimental validation.
- The frozen data manifest records source filenames/hashes and the preserved
  evaluation order. No competition-private service access is required.

No PoLyInfo export, restricted experimental table, publisher PDF, manuscript,
private project log, or complete raw database is redistributed here. Source
licenses and attribution must accompany redistributed derived data.
