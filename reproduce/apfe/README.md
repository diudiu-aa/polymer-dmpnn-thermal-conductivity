# APFE cross-publication candidate evaluation / 跨论文候选评价

This supplement evaluates ten frozen 3,000-label models on 93 retained candidates from [Huang et al., JMCA 2023](https://doi.org/10.1039/D3TA03370H). Nominal amorphous RadonPy conditions are traced through the [authors' tutorial](https://arxiv.org/pdf/2403.15887), official SI and example code. These are screened, high-value-enriched computational candidates. The analysis is retrospective and does not establish blind, experimental, or general-population validation.

## Recompute saved predictions / 复算指标

From the repository root, using Python 3.10+:

```bash
python reproduce/apfe/replay.py --download-reference --output reproduced-apfe
```

No machine-learning installation is needed for basic saved-prediction replay. The program downloads the authors' fixed 45 KB CSV, verifies its SHA, joins the frozen 93 IDs, and recomputes all per-seed metrics. It does not redistribute the author's individual references in this package. For an offline run, provide the exact original CSV with `--reference-csv /path/to/Simulation_data.csv`.

Recompute paired candidate-record bootstrap intervals as well:

```bash
python -m pip install numpy
python reproduce/apfe/replay.py --download-reference --intervals --output reproduced-apfe
```

## Run the models again / 真正重新预测

```bash
python -m pip install .
python reproduce/apfe/replay.py --download-reference --run-models --intervals --output apfe-fresh-inference
```

This command loads all ten released checkpoints and runs 930 new predictions, using the author structures. It then recomputes the metrics against the fixed references. It does not retrain. The per-prediction parity tolerance is 1e-5 W/(m K). Every difference in reported metrics is listed in `verification.json`; numerical differences are not silently replaced by stored values. `fresh_predictions.csv` contains this run's predictions.

固定模型、相同结构再次推理通常得到相同或非常接近的预测。这与“重新训练”不同；重新训练的随机性不由本命令检验。程序会实际调用模型，重新生成930个预测，不是展示旧表。

## Expected interpretation / 结果如何解释

- 104 original candidates; 11 removed before scoring for known source-unit overlap or conservative fingerprint identity; 93 remain, all supported by both graph arms.
- Periodic/open five-model mean MAE: 0.09482 / 0.09534 W/(m K), R²: 0.0418 / 0.0358.
- The paired overall-error interval crosses zero. Periodic superiority is not established on this cohort. Both arms substantially underestimate the high-value references.
- Mean Spearman correlations: 0.6446 / 0.6377. Top-10 overlap: 42% / 46%; Top-20: 59% / 50%. N=93, so Top-100 is undefined.
- Candidate-record bootstrap intervals condition on the fixed models and observed enriched cohort. They are not the internal experiments' shared-cluster intervals.
- Finite-unit canonical and fingerprint checks do not exhaust arbitrary repeat-unit shift/doubling equivalence; identical per-row MD settings are not claimed.

`predictions.csv` contains derived model predictions; `cohort.csv` contains IDs and derived inclusion decisions. The author source is retrieved from commit `bf01ab210d465f0f81960c1e5b69f9c3a2d77a6a`, SHA-256 `2bd251615eaaefa30322c9ec35db9bdcfb1d31835b030e14cda3e148c49a883f`. Cite the original work and RadonPy. Do not infer a license for the author's data from this repository's software MIT license.
