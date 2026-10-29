"""Explicit larger-budget models; the original default predictor is unchanged."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math
from importlib import resources
from pathlib import Path
import numpy as np
import torch
from .graph import PeriodicDMPNN,build_trimer_graph,batch_graphs

def registry():
 return json.loads(resources.files('polymer_dmpnn').joinpath('paper_weights/optimization_manifest.json').read_text())

class OptimizationPredictor:
 """A fixed 3,000-label model, never selected/updated using the external cohort."""
 def __init__(self,arm='periodic',seed=32101):
  key=f'B3000_{arm}_{seed}';entries=registry()['models']
  if key not in entries:raise ValueError('Use arm open/periodic and a released seed from 32101 to 32105')
  self.model_id=key;self.spec=entries[key];s=self.spec
  if s['length']!=3:raise ValueError('Unsupported explicitly configured repeat length')
  blob=resources.files('polymer_dmpnn').joinpath(s['file']).read_bytes()
  if hashlib.sha256(blob).hexdigest()!=s['sha256']:raise ValueError('Checkpoint checksum mismatch')
  c=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
  for k in ('target_mean','target_sd'):
   if float(c[k])!=s[k] or not math.isfinite(s[k]):raise ValueError('Target normalization mismatch')
  if s['target_sd']<=0:raise ValueError('Target scale must be positive')
  self.model=PeriodicDMPNN(s['atom_features'],s['bond_features'],hidden_size=s['hidden'],depth=s['depth_including_initialization'],dropout=s['dropout'])
  self.model.load_state_dict(c['state_dict'],strict=True);self.model.eval()
  if sum(p.numel() for p in self.model.parameters())!=s['parameter_count']:raise ValueError('Model parameter count mismatch')
 def predict_graphs(self,graphs):
  with torch.inference_mode():p=self.model(batch_graphs(graphs)).numpy()*self.spec['target_sd']+self.spec['target_mean']
  if not np.isfinite(p).all():raise ValueError('Non-finite prediction')
  return p.astype(np.float64)
 def predict_one(self,smiles):
  if not isinstance(smiles,str) or not smiles.strip():return {'success':False,'smiles':smiles,'error':'Provide a repeat-unit SMILES with two * attachment sites'}
  try:
   p=self.predict_graphs([build_trimer_graph(smiles.strip(),periodic_closure=self.spec['arm']=='periodic')])[0]
   return {'success':True,'smiles':smiles,'model_id':self.model_id,'model_name':f"{self.spec['arm'].title()} D-MPNN, 3000 fitting labels",'seed':self.spec['seed'],'predicted_kappa_W_mK':float(p)}
  except (ValueError,TypeError,RuntimeError) as e:return {'success':False,'smiles':smiles,'error':str(e)}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['open','periodic'],default='periodic');p.add_argument('--seed',type=int,default=32101);p.add_argument('--list-models',action='store_true');inputs=p.add_mutually_exclusive_group();inputs.add_argument('--smiles');inputs.add_argument('--input',type=Path);p.add_argument('--smiles-column',default='smiles');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.list_models:print(json.dumps(registry(),indent=2));return
 if a.smiles is None and a.input is None:p.error('Provide --smiles or --input, or use --list-models')
 torch.set_num_threads(1)
 try:model=OptimizationPredictor(a.arm,a.seed)
 except (ValueError,OSError,RuntimeError) as e:p.exit(2,f'Model could not be loaded: {e}\n')
 if a.smiles is not None:
  r=model.predict_one(a.smiles);print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['success'] else 2)
 try:
  with a.input.open(encoding='utf-8-sig',newline='') as f:
   reader=csv.DictReader(f)
   if a.smiles_column not in (reader.fieldnames or []):p.error('CSV does not contain the requested SMILES column')
   results=[{'input_row':i,**model.predict_one(r[a.smiles_column])} for i,r in enumerate(reader,start=2)]
  if not a.output:p.error('Batch prediction requires --output')
  columns=['input_row','smiles','success','model_id','model_name','seed','predicted_kappa_W_mK','error']
  with a.output.open('w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(results)
  print(json.dumps({'rows':len(results),'successful':sum(r['success'] for r in results),'output':str(a.output)}))
 except OSError as e:p.exit(2,f'File error: {e}\n')
if __name__=='__main__':main()
