# Polymer D-MPNN thermal-conductivity models and paper reproduction

CPU inference, saved-result reproduction of **manuscript Tables 1–5**, and
fixed-input training for the main graph experiments. Targets are simulated
thermal conductivity of nominally unstrained amorphous polymers near **300 K,
1 atm**, in **W/(m K)**. [中文说明](README_zh-CN.md).

## Reproduce all five tables — no model installation needed

```bash
git clone https://github.com/diudiu-aa/polymer-dmpnn-thermal-conductivity.git
cd polymer-dmpnn-thermal-conductivity
python reproduce/paper_tables.py --verify --output reproduced-paper-tables
```

Python 3.10+ and its standard library suffice. The command recalculates 182
means/sample standard deviations from 117 saved prediction groups (40,848
rows), verifies them against the manuscript, and writes five CSV/Markdown
tables, per-seed metrics and `verification.json`. All inputs are included.
No private project directory, account, network, or raw database is needed.
The output directory must be new. Expected statistics verify the calculation;
they are not copied as generated results.

## Install and predict

```bash
python -m pip install .
polymer-kappa --smiles '*CC*'
polymer-kappa --batch examples/polymers.csv --format csv --output predictions.csv
```

Use Python 3.10–3.13 on a platform supported by RDKit/PyTorch. Exactly two
single-bonded, degree-one `*` connection points in one connected repeat unit
are required. Quote SMILES in the shell. `--column` changes the CSV structure
column. Invalid entries remain as error rows (exit 2 if any row fails).

The default remains the original three-seed L1000 periodic ensemble (Table 1
model family; 600 fitting records per seed). Its output includes individual
predictions, mean, and population SD. **Ensemble prediction metrics differ
from the paper's mean of separately calculated seed metrics.** The SD is not
a calibrated prediction interval.

Select an actual individual main-study checkpoint explicitly:

```bash
polymer-kappa --list-models
polymer-kappa --paper-model table5_B600_multitask_32101 --smiles '*CC*'
polymer-kappa --paper-model table5_B600_periodic_32101 --batch examples/polymers.csv
```

The 65 individual graph checkpoints cover Tables 3–5: open/periodic/ranking,
nested learning budgets, and the balanced/multitask comparison. They retain
the original bytes, target scaling, seed, and SHA256. Multitask classification
outputs are raw ranking logits, not calibrated probabilities. The regression
model does not guarantee high-conductivity screening or experimental accuracy.

## Verify weights or train from fixed inputs

```bash
# Re-run all 65 saved checkpoints on their fixed 400-record evaluation sets.
python reproduce/evaluate_checkpoints.py --output checkpoint-verification

# Inspect a fixed training configuration; no fitting by default.
python reproduce/train_graph_models.py --experiment table5 --arm multitask --seed 32101

# Explicit 50-epoch CPU reproduction of that single configuration.
python reproduce/train_graph_models.py --experiment table5 --arm multitask --seed 32101 --run --output retrained-multitask
```

The fixed training inputs and roles are bundled. There is no `--project-root`
requirement. For Ridge/ExtraTrees training-only grouped tuning, install
`python -m pip install '.[reproduce]'` and see [reproduce/README.md](reproduce/README.md).

| Reproduction level | Included / verified scope |
|---|---|
| Saved-prediction metrics | All five main tables; 182 displayed mean/SD values |
| Saved graph checkpoint evaluation | All 65 Table 3–5 fits, fixed roles, scales, original predictions |
| Training from fixed curated inputs | Graph recipes for Tables 3–5 and tuned Table 5 Ridge/ExtraTrees; explicit `--run` |
| Entire raw-to-curated historical study | Not claimed: historical cluster/role construction and Table 1 GIN/GPR training are not rebuilt by these commands |

Table 4's 600-fit point reuses Table 3. Table 2 is a retrospective public OPC
cohort; its MAE weights clusters equally, not records. Five-seed samples can
share records and evaluation clusters. A successful saved-result replay is
not an independent replication of the original MD simulations.

## Tests, environment and versioning

```bash
python -m unittest discover -s tests -v
```

CI tests installation, inference, published table reproduction and checkpoint
parity on Python 3.10/3.11/3.12. The original model environment was Python
3.12.14 / PyTorch 2.14.1 / NumPy 2.5.4 / RDKit 2026.03.3, CPU, one thread.
Other supported versions are allowed; RDKit sanitization and numerical
libraries may change outputs. Saved predictions make table reproduction
independent of those differences. Checkpoint parity tolerance is absolute
`1e-5 W/(m K)`; training bit-for-bit parity across library/platform versions
is not promised. Record the Git commit with every reproduced result.

## Sources and licenses

Code and project weights: MIT. Curated PolyOmics excerpts: CC BY 4.0;
OPC source data: MIT. Original UUIDs, fixed snapshot, hashes and modifications
are documented in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) and manifests.
No PoLyInfo exports, restricted experimental tables, paper PDFs or private
project logs are included. No archival DOI has been assigned to this repo.

- PolyOmics: [DOI 10.57967/hf/7475](https://doi.org/10.57967/hf/7475).
- Yang et al., D-MPNN, *JCIM* 2019: [10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237).
- Antoniuk et al., periodic polymer graphs, *JCIM* 2022: [10.1021/acs.jcim.2c00875](https://doi.org/10.1021/acs.jcim.2c00875).

This repository evaluates existing representation/learning ideas on the
thermal-conductivity task; it does not claim to invent periodic polymer graphs.

### Local release validation

In the recorded original CPU environment, all 65 checkpoint evaluations
(26,000 predictions) matched the original saved values exactly. Representative
unchanged-recipe retraining also matched for a 60-label periodic graph model,
the 600-label multitask model, and tuned Ridge (seed 32101). These checks are
recorded in [release_validation.json](reproduce/release_validation.json);
they do not claim that every configuration was retrained on every platform.
