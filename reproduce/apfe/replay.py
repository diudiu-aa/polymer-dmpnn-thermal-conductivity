"""Fetch fixed author reference data and replay APFE candidate metrics, without training."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,io,json,math,statistics,argparse,urllib.request
BASE=Path(__file__).resolve().parent

def ranks(x):
 order=sorted(range(len(x)),key=x.__getitem__);out=[None]*len(x);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  for k in range(i,j):out[order[k]]=(i+j-1)/2+1
  i=j
 return out

def run(source,output,live=False,with_intervals=False):
 manifest=json.loads((BASE/'manifest.json').read_text())
 for name,h in manifest['files'].items():
  if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=h:raise ValueError('Bundled input checksum mismatch: '+name)
 if hashlib.sha256(source).hexdigest()!=manifest['source_sha256']:raise ValueError('Author source checksum mismatch; do not silently use an updated dataset')
 with (BASE/'cohort.csv').open() as f:ids=[r['apfe_id'] for r in csv.DictReader(f) if r['eligible']=='True']
 source_rows=csv.DictReader(io.StringIO(source.decode('utf-8-sig').split('\n',1)[1]));labels={r['ID']:r['thermal_conductivity'] for r in source_rows}
 truth={i:float(labels[i]) for i in ids}
 with (BASE/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
 with (BASE/'expected_metrics.csv').open() as f:refs={(r['seed'],r['arm']):r for r in csv.DictReader(f)}
 max_prediction_error=0.0
 if live:
  import sys,torch
  sys.path.insert(0,str(BASE.parents[1]))
  from polymer_dmpnn.optimization import OptimizationPredictor
  from polymer_dmpnn.graph import build_trimer_graph
  torch.set_num_threads(1)
  structures={r['ID']:r['SMILES'] for r in csv.DictReader(io.StringIO(source.decode('utf-8-sig').split('\n',1)[1]))}
  for seed in range(32101,32106):
   for arm in ('open','periodic'):
    model=OptimizationPredictor(arm,seed);rs=[r for r in rows if r['seed']==str(seed) and r['arm']==arm]
    for start in range(0,len(rs),16):
     batch=rs[start:start+16];pred=model.predict_graphs([build_trimer_graph(structures[r['apfe_id']],periodic_closure=arm=='periodic') for r in batch])
     for r,v in zip(batch,pred):
      error=abs(float(v)-float(r['prediction']));max_prediction_error=max(max_prediction_error,error)
      if error>1e-5:raise ValueError('Fresh checkpoint prediction exceeds 1e-5 W/(m K) parity tolerance')
      r['prediction']=repr(float(v))

 g=defaultdict(list)
 for r in rows:g[r['seed'],r['arm']].append(r)
 results=[];checks=0;errors={};qs={};metric_differences=[]
 for key,rs in g.items():
  if [r['apfe_id'] for r in rs]!=ids:raise ValueError('Candidate identity/order mismatch')
  y=[truth[r['apfe_id']] for r in rs];p=[float(r['prediction']) for r in rs];e=[abs(a-b) for a,b in zip(y,p)];q=float(rs[0]['training_q95']);mu=statistics.fmean(y);yr=ranks(y);pr=ranks(p);ym=statistics.fmean(yr);pm=statistics.fmean(pr)
  rho=sum((a-ym)*(b-pm) for a,b in zip(yr,pr))/math.sqrt(sum((a-ym)**2 for a in yr)*sum((b-pm)**2 for b in pr))
  m={'seed':key[0],'arm':key[1],'n':len(rs),'mae':statistics.fmean(e),'r2':1-sum((a-b)**2 for a,b in zip(y,p))/sum((a-mu)**2 for a in y),'spearman':rho,'signed_bias':statistics.fmean(b-a for a,b in zip(y,p)),'training_q95':q}
  errors[key]=e;qs[key]=q
  for prefix,t in [('fixed030',.3),('tail',q)]:
   v=[b for a,b in zip(y,e) if a>=t];m[prefix+'_n']=len(v);m[prefix+'_mae']=statistics.fmean(v)
  for k in (10,20,50,100):
   hit=len(set(sorted(range(len(y)),key=lambda i:-y[i])[:k])&set(sorted(range(len(p)),key=lambda i:-p[i])[:k])) if k<=len(y) else None
   m[f'hits_at_{k}']=hit;m[f'precision_at_{k}']=hit/k if hit is not None else None
  h=[i for i,a in enumerate(p) if a>=.3];tp=sum(y[i]>=.3 for i in h);m.update(predicted_high_n=len(h),threshold_true_positive=tp,threshold_precision=tp/len(h) if h else None,threshold_recall=tp/sum(a>=.3 for a in y))
  for name,value in m.items():
   if name in ('seed','arm'):continue
   expected=refs[key][name]
   if value is None:
    if expected!='':raise ValueError('Unexpected unavailable metric')
   elif not math.isclose(value,float(expected),rel_tol=1e-10,abs_tol=1e-12):
    if not live:raise ValueError(f'Metric mismatch: {key}/{name}')
    metric_differences.append({'seed':key[0],'arm':key[1],'metric':name,'actual':value,'reference':float(expected),'difference':value-float(expected)})
   checks+=1
  results.append(m)
 interval_checks=0;interval_rows=[]
 if with_intervals:
  try:import numpy as np
  except ImportError:raise ValueError('Interval replay requires NumPy')
  y=np.array([truth[i] for i in ids]);draws=np.random.default_rng(610292).integers(0,len(ids),size=(2000,len(ids)))
  for e in manifest['expected_paired']:
   per=[];values=[]
   for seed in range(32101,32106):
    key=(str(seed),'periodic');d=np.array(errors[key])-np.array(errors[str(seed),'open']);mask=np.ones(len(ids),dtype=bool) if e['metric']=='all' else y>=(.3 if e['metric']=='fixed030' else qs[key]);per.append(float(d[mask].mean()));den=mask[draws].sum(axis=1);num=np.where(mask[draws],d[draws],0).sum(axis=1);values.append(np.divide(num,den,out=np.full(2000,np.nan),where=den>0))
   values=np.array(values);valid=np.isfinite(values).all(axis=0);low,high=np.quantile(values[:,valid].mean(axis=0),[.025,.975]);point=float(np.mean(per))
   for x,z in [(point,e['periodic_minus_open']),(low,e['ci95_conditional_candidate_record_resampling'][0]),(high,e['ci95_conditional_candidate_record_resampling'][1]),(int(valid.sum()),e['valid_draws']),(int((~valid).sum()),e['empty_draws'])]:
    if not live and not math.isclose(x,z,abs_tol=1e-11,rel_tol=1e-9):raise ValueError('Paired interval mismatch')
    interval_checks+=1
   interval_rows.append(dict(metric=e['metric'],difference=point,lower=float(low),upper=float(high),valid=int(valid.sum())))
 output.mkdir(parents=True,exist_ok=True)
 with (output/'per_seed_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 if live:
  with (output/'fresh_predictions.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (output/'paired_intervals.json').write_text(json.dumps(interval_rows,indent=2)+'\n')
 report={'status':'PASS' if not metric_differences else 'PREDICTION_PARITY_PASS_WITH_REPORTED_METRIC_DIFFERENCES','groups':len(results),'candidates':len(ids),'rows':len(rows),'metric_checks':checks,'source_sha256':manifest['source_sha256'],'scope':manifest['scope'],'fit':0,'forward_batches':60 if live else 0,'fresh_inference':live,'max_prediction_difference_W_mK':max_prediction_error,'metric_differences':metric_differences,'interval_checks':interval_checks}
 (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--download-reference',action='store_true');g.add_argument('--reference-csv',type=Path);p.add_argument('--output',type=Path,default=Path('reproduced-apfe'));p.add_argument('--run-models',action='store_true',help='Actually predict all 930 values from released weights; needs installed ML package');p.add_argument('--intervals',action='store_true');a=p.parse_args()
 try:
  manifest=json.loads((BASE/'manifest.json').read_text());payload=urllib.request.urlopen(manifest['source_url'],timeout=30).read() if a.download_reference else a.reference_csv.read_bytes();print(json.dumps(run(payload,a.output,a.run_models,a.intervals),indent=2))
 except (ValueError,OSError,KeyError) as e:p.exit(2,f'APFE reproduction failed: {e}\n')
if __name__=='__main__':main()
