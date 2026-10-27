# Polymer D-MPNN thermal-conductivity predictor

This repository contains a small, CPU-ready inference package for the saved R04 periodic-trimer D-MPNN ensemble. It predicts the simulated thermal conductivity of a nominally unstrained, isotropic amorphous polymer near **300 K and 1 atm**, in W/(m·K). Three model weights (about 1 MB in total) are included; no training data are bundled.

The models were fitted to PolyOmics/RadonPy molecular-dynamics data. Their outputs estimate the simulation target, not measured experimental conductivity. The public OPC assessment is retrospective; it is not a blind or experimental validation. Treat predictions for unfamiliar structures and high-conductivity candidates as screening estimates requiring further verification.

## Install

Use Python 3.10–3.13 and a platform supported by PyTorch and RDKit:

```bash
git clone https://github.com/diudiu-aa/polymer-dmpnn-thermal-conductivity.git
cd polymer-dmpnn-thermal-conductivity
python -m pip install .
```

For development, `python -m pip install -e .` installs the same dependencies. The runtime requirements are also listed in `requirements.txt`.

## Predict

Repeat-unit SMILES must contain **exactly two degree-one `*` connection points**, both connected by single bonds, in one connected structure. Quote the SMILES in a shell:

```bash
polymer-kappa --smiles '*CC*'
python -m polymer_dmpnn --smiles '*CC(*)C'
```

For a CSV file with a `smiles` column and an optional `id` column:

```bash
polymer-kappa --batch examples/polymers.csv --format csv --output predictions.csv
```

Plain UTF-8 text files with one SMILES per nonempty line are also accepted. Use `--column` to choose a different CSV column and `--format json` for JSON output. Invalid entries are preserved as error rows; the CLI exits with status 2 if any entry fails.

Each successful result includes the mean of three predictions, the individual seed predictions, and their population standard deviation. The seed standard deviation describes only the three saved models; **it is not a calibrated prediction interval or a guarantee of accuracy**. The package does not expose the earlier provisional conformal interval.

## Tests and compatibility

```bash
python -m unittest discover -s tests -v
```

The locked training environment used Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.4 and RDKit 2026.03.3. Six local tests also passed with RDKit 2025.09.6 on the same Python/PyTorch environment, including four reference structures. A clean macOS ARM64 Python 3.12 installation selected RDKit 2026.09.1 and passed single and batch smoke tests. The strict RDKit version rejection has been removed. Other versions are allowed but emit a warning because graph sanitization and features can change between versions; verify compatibility for your structures. See `tests/golden_predictions.json` for reference predictions from the original locked implementation. The GitHub Actions matrix completed successfully for Python 3.10, 3.11 and 3.12 on commit 64f3f57.

## Provenance and citation

The model ensemble uses three saved L1000 periodic-trimer fits, with 600 fit records per seed and 200 records each assigned to calibration and evaluation in the original design. Weight hashes and per-seed target standardization values are recorded in `polymer_dmpnn/weights/manifest.json`. The implementation follows the directed-message-passing family and the periodic repeat-unit representation; it does not claim to introduce either idea.

- PolyOmics dataset: [DOI 10.57967/hf/7475](https://doi.org/10.57967/hf/7475), [dataset card](https://huggingface.co/datasets/yhayashi1986/PolyOmics).
- D-MPNN reference: Yang et al., *J. Chem. Inf. Model.* 2019, [DOI 10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237).
- Periodic polymer graph reference: Antoniuk et al., *J. Chem. Inf. Model.* 2022, [DOI 10.1021/acs.jcim.2c00875](https://doi.org/10.1021/acs.jcim.2c00875).

See `NOTICE.md` for data attribution and the scope of this release. The repository provides inference and selected model-comparison utilities; it does not redistribute the source data or the complete raw-to-curated workflow. No archival DOI has been assigned.


For the paired five-seed training/evaluation entry point and required prepared input tables, see [reproduce/README.md](reproduce/README.md). The full raw-to-curated reconstruction remains to be packaged.
