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

## Additional optimization and APFE analyses

- `reproduce/optimization/data/` adds evaluated PolyOmics subsets and derived predictions with the same fixed source revision and CC BY 4.0 attribution above. Its group and statistic counts are separate from the original Tables 1–5 bundle.
- Ten `paper_weights/optimization_B3000_*.pt` files are newly released trained weights under the software MIT license, with explicit configuration in `optimization_manifest.json`. Original weight bytes/defaults are unchanged.
- `reproduce/apfe/` bundles our derived predictions, source IDs, inclusion decisions and aggregate metric references. It does **not** bundle the original per-candidate reference labels or author PDFs. The command fetches the fixed public author CSV from SJTU-MI/APFEforPI and checks its checksum. Cite Huang et al., JMCA 2023, DOI 10.1039/D3TA03370H and the RadonPy work. The author repository's software license is not asserted to license the underlying data.

## Controlled completion supplement

`reproduce/completion/` contains curated PolyOmics identity/reference excerpts and our derived predictions for 15 HK01 objective fits and 10 additional budget-control fits, plus the required accepted earlier scale subset. These source-derived excerpts retain the fixed PolyOmics revision, CC BY4.0 attribution and changes described above and in the [module notice](reproduce/completion_v0_9.tar.gz); they are not relabeled as MIT. The 25 exact new trained checkpoints are covered separately by the repository software/weights MIT license. The module does not redistribute individual APFE author references; external plotting retrieves or receives the checksum-locked author CSV, as in `reproduce/apfe`.
