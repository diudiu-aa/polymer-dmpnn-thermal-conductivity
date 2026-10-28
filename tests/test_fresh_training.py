"""Train actual new models without access to any published .pt checkpoint."""
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from polymer_dmpnn.paper import MultitaskDMPNN
from polymer_dmpnn.graph import PeriodicDMPNN, build_trimer_graph, batch_graphs
import torch


class FreshTrainingTests(unittest.TestCase):
    def test_paired_training_works_without_any_published_model_weights(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp)/'public-code-and-data', Path(temp)/'new-models'
            shutil.copytree(root/'reproduce', source/'reproduce', ignore=shutil.ignore_patterns('__pycache__'))
            (source/'polymer_dmpnn').mkdir()
            for p in (root/'polymer_dmpnn').glob('*.py'):
                shutil.copy2(p, source/'polymer_dmpnn'/p.name)
            (source/'polymer_dmpnn/paper_weights').mkdir()
            shutil.copy2(root/'polymer_dmpnn/paper_weights/manifest.json', source/'polymer_dmpnn/paper_weights/manifest.json')
            self.assertFalse(list(source.rglob('*.pt')))
            proc = subprocess.run([sys.executable,'-I',str(source/'reproduce/train_from_scratch.py'),
                '--training-seed','123456','--repeats','1','--output',str(output)],
                cwd=temp,capture_output=True,text=True,timeout=420)
            self.assertEqual(proc.returncode,0,proc.stdout[-4000:]+proc.stderr[-4000:])
            result=json.loads((output/'result.json').read_text())
            self.assertEqual(result['summary']['models_trained'],3)
            self.assertEqual(result['summary']['optimizer_steps'],5700)
            self.assertEqual(result['summary']['predictions_tested'],1200)
            self.assertEqual(result['configuration']['split_seeds'],[32101])
            self.assertEqual(result['configuration']['training_seeds'],[123456])
            self.assertFalse(result['receipt']['pretrained_weights_loaded'])
            self.assertGreater(result['summary']['max_abs_prediction_difference'],1e-7)
            self.assertTrue(result['summary']['all_weights_updated'])
            for arm in ('open','periodic','multitask'):
                key=f'table5_B600_{arm}_32101'
                self.assertEqual(len(result['curves'][key]),50)
                model_dir=output/key
                ck=torch.load(model_dir/'weights.pt',weights_only=True,map_location='cpu')
                self.assertEqual(ck['training_seed'],123456)
                model=MultitaskDMPNN() if arm=='multitask' else PeriodicDMPNN(133,14,hidden_size=128,depth=3,dropout=.1)
                model.load_state_dict(ck['state_dict']);model.eval();torch.set_num_threads(1)
                rows=list(csv.DictReader((output/(key+'.csv')).open()))
                self.assertEqual(len(rows),400)
                with torch.inference_mode():
                    pred=model(batch_graphs([build_trimer_graph(r['smiles'],periodic_closure=arm!='open') for r in rows[:16]]))
                    if arm=='multitask':pred,_=pred
                actual=(pred.numpy()*ck['target_sd']+ck['target_mean']).tolist()
                self.assertLessEqual(max(abs(a-float(r['prediction'])) for a,r in zip(actual,rows[:16])),1e-5)
            for row in result['aggregates']:
                for metric in row['metrics']:
                    self.assertEqual(metric['n_repeats'],1)
                    self.assertIsNone(metric['new_sd'])
                    self.assertIsNone(metric['paper_sd'])
                    self.assertNotIn('±',metric['new_display'])
            # No old weights were present; the three new checkpoints are outside source.
            self.assertFalse(list(source.rglob('*.pt')))

if __name__=='__main__':unittest.main()
