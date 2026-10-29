import csv,gzip,hashlib,importlib.util,io,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('replay',ROOT/'replay.py'); replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
class ReplayTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)/'bundle';shutil.copytree(ROOT,self.base,ignore=shutil.ignore_patterns('__pycache__'))
 def mutate(self,change,rehash=False):
  p=self.base/'data/predictions.csv.gz'
  with gzip.open(p,'rt',newline='') as f:rows=list(csv.DictReader(f))
  change(rows);b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);p.write_bytes(gzip.compress(b.getvalue().encode(),mtime=0))
  if rehash:
   path=self.base/'manifest.json';m=json.loads(path.read_text());m['files']['data/predictions.csv.gz']=hashlib.sha256(p.read_bytes()).hexdigest();path.write_text(json.dumps(m))
 def test_clean_unrelated_working_directory(self):
  work=Path(self.tmp.name)/'unrelated';work.mkdir()
  r=subprocess.run([sys.executable,'-I',str(self.base/'replay.py'),'--output',str(work/'result')],cwd=work,capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr);report=json.loads(r.stdout);self.assertEqual(report['groups'],70);self.assertEqual(report['metric_checks'],1050)
 def test_prediction_tamper_rejected(self):
  self.mutate(lambda rows:rows[0].update(prediction='0.999'))
  with self.assertRaisesRegex(ValueError,'Integrity'):replay.load(self.base)
 def test_paired_truth_mismatch_even_after_rehash(self):
  self.mutate(lambda rows:rows[0].update(truth='0.999'),True)
  with self.assertRaisesRegex(ValueError,'identity'):replay.load(self.base)
 def test_row_order_corruption(self):
  self.mutate(lambda rows:rows[0].update(order='1'),True)
  with self.assertRaisesRegex(ValueError,'group/order'):replay.load(self.base)
 def test_nonfinite_rejected(self):
  self.mutate(lambda rows:rows[0].update(prediction='nan'),True)
  with self.assertRaisesRegex(ValueError,'Non-finite'):replay.load(self.base)
 def test_threshold_mutation_rejected_by_metric_reference(self):
  def change(rows):
   for r in rows:
    if r['group']==rows[0]['group']:r['q95']='100'
  self.mutate(change,True)
  with self.assertRaises((ValueError,TypeError)):replay.run(self.base,Path(self.tmp.name)/'out')
 def test_stable_ties_and_true_topk_intersection(self):
  rs=[dict(truth=float(400-i),prediction=1.0,q95=399.,base600_q95=399.) for i in range(400)]
  m=replay.metrics(rs);self.assertEqual(m['hits_at_10'],10);self.assertEqual(m['tail_n'],2)
 def test_reference_values_are_actually_checked(self):
  p=self.base/'data/expected_metrics.json';x=json.loads(p.read_text());x[next(iter(x))]['mae']=999;p.write_text(json.dumps(x))
  p2=self.base/'manifest.json';m=json.loads(p2.read_text());m['files']['data/expected_metrics.json']=hashlib.sha256(p.read_bytes()).hexdigest();p2.write_text(json.dumps(m))
  with self.assertRaisesRegex(ValueError,'Metric mismatch'):replay.run(self.base,Path(self.tmp.name)/'out')
if __name__=='__main__':unittest.main()
