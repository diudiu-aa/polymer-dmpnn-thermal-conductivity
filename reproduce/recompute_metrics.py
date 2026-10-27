"""Recompute saved paired model metrics without fitting or using the source dataset."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def calc(y,p,q):
 err=np.abs(y-p);mask=y>=q;den=np.sum((y-y.mean())**2)
 result={'n':len(y),'mae':float(err.mean()),'rmse':float(np.sqrt(np.mean((y-p)**2))),'r2':float(1-np.sum((y-p)**2)/den),'train_q95':q,'tail_n':int(mask.sum()),'tail_mae':float(err[mask].mean()) if mask.any() else None,'spearman_all':float(spearmanr(y,p).statistic)}
 for k in (10,20,100):result[f'precision_at_{k}']=float(len(set(np.argsort(-p,kind='stable')[:k])&set(np.argsort(-y,kind='stable')[:k]))/k)
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--verify-existing',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args();root=args.run_dir
 lock=json.loads((root/'fitted_prediction_lock.json').read_text());computed=[]
 for fit in lock['fits']:
  seed=fit['seed'];arm=fit['arm'];p=root/f'evaluated_{seed}_{arm}.csv'
  with p.open(newline='') as f:rows=list(csv.DictReader(f))
  y=np.asarray([float(r['true']) for r in rows]);pred=np.asarray([float(r['prediction']) for r in rows])
  if not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError(f'invalid predictions {p}')
  computed.append({'seed':seed,'arm':arm,'metrics':calc(y,pred,fit['q95_train'])})
 if args.verify_existing:
  previous={(x['seed'],x['arm']):x['metrics'] for x in json.loads((root/'results.json').read_text())['results']}
  if len(previous)!=len(computed):raise ValueError('result count mismatch')
  for row in computed:
   ref=previous[(row['seed'],row['arm'])]
   for key,value in row['metrics'].items():
    if value is None:
     if ref[key] is not None:raise ValueError(f'mismatch {key}')
    elif not math.isclose(value,ref[key],rel_tol=1e-11,abs_tol=1e-12):raise ValueError(f'mismatch {row["seed"]} {row["arm"]} {key}')
 payload={'result_count':len(computed),'verified_against_existing':args.verify_existing,'results':computed}
 if args.output:args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 print(f"recomputed {len(computed)} results; verified_against_existing={args.verify_existing}")
if __name__=='__main__':main()
