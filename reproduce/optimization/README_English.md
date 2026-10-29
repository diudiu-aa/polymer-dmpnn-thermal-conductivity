# Controlled model optimization: saved-result replay

From the repository root:

```bash
python reproduce/optimization/replay.py --output reproduced-optimization
# Optional: also regenerate all 48 conditional paired intervals
python -m pip install numpy
python reproduce/optimization/replay.py --intervals --output reproduced-optimization
```

Basic metric replay uses only the Python standard library. The bundle contains 70 model–seed groups, 28,000 saved prediction rows from 1,738 distinct record IDs: graph controls (20 groups), final standard/context7 Tanimoto kernels (10 groups) and label/optimization-budget controls (40 groups). Models and records recur across experiments; group/row counts are not independent polymer counts.

The script verifies file integrity and paired identities, recomputes 1,050 per-seed metrics, writes five-seed mean/sample-SD tables and 40 Figure-7 source-data rows. With NumPy, it also regenerates 48 paired uncertainty intervals and checks 228 interval/count values. Prediction order breaks ranking ties stably; tail thresholds are frozen training quantiles, separately from fixed kappa >=0.30.

This is saved-result replay, not retraining or model inference. Original Tables 1–5 remain in the separate `reproduce/paper_tables.py` package (117 groups, 182 displayed values). Actual fresh inference from the new checkpoints is available through `python -m polymer_dmpnn.optimization` and the sibling `apfe/replay.py --run-models` command.

The depth candidate did not improve the original D3 model. Expanding periodic training from 600 to 3,000 labels at 50 epochs reduced MAE from 0.02213 to 0.01868, but also increased optimizer updates. At 1,900 matched updates, the latter MAE was 0.02112. The periodic-minus-open overall-error interval crossed zero at 3,000 labels. These results do not establish universal periodic superiority or a new state of the art.

Source labels originate from [PolyOmics](https://doi.org/10.57967/hf/7475), fixed revision `041e5834ea1a48682fae12dc39ccd723bcd4f771`, CC BY 4.0. Changes include subset selection, assigned cluster IDs, predictions and frozen thresholds. Source licenses are separate from the software MIT license; see `THIRD_PARTY_DATA.md`. See `README.md` for the Chinese explanation and exact statistic definitions.
