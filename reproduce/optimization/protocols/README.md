# Frozen research protocols and trajectories

These records document the actual selection process behind the added analyses. They are not another model search.

- `graph_frozen_ids.json.gz`: original 600 fitting IDs, training-cluster folds, and five final fitting/evaluation identities.
- `graph_protocol.json`, `graph_final_choice.json`, `graph_cv_*.csv`: D=2/3/5, open lengths 3/5/7 and the extra L7D5 combination, fair depth budgets and locked selections. D includes initialization; D-1 edge updates reach up to D directed bonds for fixed input features.
- `graph_learning_curves.csv`: all 37 full trajectories (1,850 epoch records); `graph_fit_costs.csv` records parameters, steps, fitting and prediction time. Loss decrease is not proof of global convergence.
- `scaling_frozen_ids.json.gz`: final corrected 600/1800/3000 nested identities and fixed architecture/budgets. Earlier discarded pilot results are excluded; their existence and label-exposure correction are reported in the paper SI.
- Kernel protocol/CV files preserve all seven-alpha/three-fold choices. The initial file includes the archived five-unit context. The corrected seven-unit context is the final comparison; the correction follows graph-radius requirements, not evaluation-score selection.

PROJECT_ROOT denotes the original project root in source-provenance references. The files themselves are portable metadata. Source-derived identities retain PolyOmics attribution/CC BY 4.0; software and trained-weight licenses are separate. All targets and structures for the original core fits are in the existing `reproduce/inputs` subset. The larger training sets can be traced to the open fixed PolyOmics source through these IDs. This archive does not claim every new exploratory fit has been independently rerun from scratch by this public package; current verified commands cover saved-result replay, released-weight inference and the original core-model retraining route.

中文：此目录公开冻结划分、内部选参分数、成本及完整训练曲线。它补上研究过程溯源，不等于已经从头重训了全部新增探索模型。修正前的无效试验不作为科学结果，相关处理已在SI披露。
