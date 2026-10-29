import csv,gzip,importlib.util,json,unittest
from pathlib import Path
import numpy as np
import torch
from polymer_dmpnn.optimization import OptimizationPredictor,registry
from polymer_dmpnn.graph import build_trimer_graph
ROOT=Path(__file__).resolve().parents[1]
class OptimizationModelsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  with gzip.open(ROOT/'reproduce/inputs/records.csv.gz','rt',newline='') as f:cls.smiles={r['record_id']:r['smiles'] for r in csv.DictReader(f)}
  with gzip.open(ROOT/'reproduce/optimization/data/predictions.csv.gz','rt',newline='') as f:cls.rows=list(csv.DictReader(f))
 def test_ten_explicit_models_match_saved_predictions(self):
  self.assertEqual(len(registry()['models']),10)
  for seed in range(32101,32106):
   for arm in ('open','periodic'):
    with self.subTest(seed=seed,arm=arm):
     m=OptimizationPredictor(arm,seed);rs=[r for r in self.rows if r['group']==f'scale/epochs/3000/{arm}/{seed}'][:3]
     self.assertEqual(len(rs),3);graphs=[build_trimer_graph(self.smiles[r['record_id']],periodic_closure=arm=='periodic') for r in rs]
     np.testing.assert_allclose(m.predict_graphs(graphs),[float(r['prediction']) for r in rs],atol=1e-5,rtol=0)
 def test_invalid_model_fails_clearly(self):
  with self.assertRaises(ValueError):OptimizationPredictor('periodic',99)
 def test_invalid_smiles_is_reported(self):
  self.assertFalse(OptimizationPredictor().predict_one('')['success'])
if __name__=='__main__':unittest.main()
