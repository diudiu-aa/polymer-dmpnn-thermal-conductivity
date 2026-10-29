"""Portable replay of newly added polymer-model analyses (Python >=3.10)."""
from pathlib import Path
from collections import defaultdict
from statistics import fmean,stdev
import argparse,csv,gzip,hashlib,json,math
BASE=Path(__file__).resolve().parent

def load(base=BASE):
 m=json.loads((base/'manifest.json').read_text())
 for name,sha in m['files'].items():
  p=base/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError(f'Integrity check failed: {name}')
 with gzip.open(base/'data/predictions.csv.gz','rt',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=m['rows']:raise ValueError('Unexpected row count')
 groups=defaultdict(list); identity={}
 for r in rows:
  for k in ('truth','prediction','q95','base600_q95'):
   r[k]=float(r[k])
   if not math.isfinite(r[k]):raise ValueError(f'Non-finite {k}')
  r['order']=int(r['order']);groups[r['group']].append(r)
  # The same polymer has one frozen truth/cluster in this entire bundle.
  token=(r['truth'],r['cluster_id'])
  if r['record_id'] in identity and identity[r['record_id']]!=token:raise ValueError('Paired truth/cluster identity mismatch')
  identity[r['record_id']]=token
 if len(groups)!=m['groups']:raise ValueError('Unexpected group count')
 for gid,rs in groups.items():
  if len(rs)!=400 or [r['order'] for r in rs]!=list(range(400)) or len({r['record_id'] for r in rs})!=400:raise ValueError(f'Invalid group/order: {gid}')
  if len({(r['q95'],r['base600_q95']) for r in rs})!=1:raise ValueError('Inconsistent frozen thresholds')
 return m,dict(groups)

def metrics(rs):
 y=[r['truth'] for r in rs];p=[r['prediction'] for r in rs];err=[abs(a-b) for a,b in zip(y,p)];mu=fmean(y)
 out=dict(n=len(rs),mae=fmean(err),r2=1-sum((a-b)**2 for a,b in zip(y,p))/sum((a-mu)**2 for a in y))
 for prefix,threshold in [('tail',rs[0]['q95']),('fixed_030',.30),('base600',rs[0]['base600_q95'])]:
  selected=[e for e,a in zip(err,y) if a>=threshold];out[prefix+'_n']=len(selected);out[prefix+'_mae']=fmean(selected) if selected else None
 truth_order=sorted(range(len(y)),key=lambda i:-y[i]);prediction_order=sorted(range(len(y)),key=lambda i:-p[i])
 for k in (10,20,50,100):
  hit=len(set(truth_order[:k])&set(prediction_order[:k]));out[f'hits_at_{k}']=hit;out[f'precision_at_{k}']=hit/k
 return out

def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def intervals(groups,base):
 try:import numpy as np
 except ImportError:raise ValueError('Interval replay needs NumPy: python -m pip install numpy')
 clusters=sorted({r['cluster_id'] for rs in groups.values() for r in rs});ix={c:i for i,c in enumerate(clusters)};nc=len(clusters);seeds=range(32101,32106)
 def paired(new,ref,metric,weights):
  sums=np.zeros((5,nc));counts=np.zeros_like(sums)
  for j,seed in enumerate(seeds):
   a=groups[new.format(seed=seed)];b=groups[ref.format(seed=seed)]
   if [r['record_id'] for r in a]!=[r['record_id'] for r in b]:raise ValueError('Paired record order mismatch')
   for r,s in zip(a,b):
    if r['truth']!=s['truth'] or r['cluster_id']!=s['cluster_id']:raise ValueError('Paired identity mismatch')
    threshold={'all':-math.inf,'fixed030':.30,'tail':r['q95'],'base600_q95':r['base600_q95']}[metric]
    if r['truth']<threshold:continue
    c=ix[r['cluster_id']];sums[j,c]+=abs(r['truth']-r['prediction'])-abs(s['truth']-s['prediction']);counts[j,c]+=1
  den=weights@counts.T;num=weights@sums.T;valid=(den>0).all(axis=1)
  values=(num[valid]/den[valid]).mean(axis=1);lo,hi=np.quantile(values,[.025,.975]);point=float((sums.sum(axis=1)/counts.sum(axis=1)).mean())
  return dict(difference=point,lower=float(lo),upper=float(hi),valid=int(valid.sum()),empty=int((~valid).sum()))
 result=[];checks=0
 def check(actual,expected,label):
  nonlocal checks
  if not math.isclose(actual,expected,abs_tol=1e-11,rel_tol=1e-9):raise ValueError(f'Interval mismatch: {label}: {actual} != {expected}')
  checks+=1
 # Recreate exactly the originally frozen shared-cluster sampling draws.
 rng=np.random.default_rng(321012910);weights=np.array([np.bincount(rng.integers(0,nc,nc),minlength=nc) for _ in range(2000)])
 with (base/'data/graph_intervals.csv').open() as f:refs=list(csv.DictReader(f))
 for r in refs:
  new,ref=r['comparison'].split('_minus_');mask={'mae':'all','tail_mae':'tail','fixed_030_mae':'fixed030'}[r['metric']]
  v=paired('graph/'+new+'/{seed}','graph/'+ref+'/{seed}',mask,weights)
  for a,b in [('difference','mean_of_five_seed_differences'),('lower','shared_cluster_bootstrap_95_low'),('upper','shared_cluster_bootstrap_95_high'),('valid','bootstrap_n')]:check(v[a],float(r[b]),'graph/'+a)
  result.append(dict(experiment='graph',comparison=r['comparison'],metric=mask,**v))
 weights=np.random.default_rng(20261029).multinomial(nc,[1/nc]*nc,size=10000)
 for r in json.loads((base/'data/kernel_context_intervals.json').read_text()):
  mask={'mae':'all','tail_mae':'tail','fixed_high_mae':'fixed030'}[r['metric']];v=paired('kernel/boundary_context7/{seed}','graph/original_periodic/{seed}',mask,weights)
  for a,b in [('difference','difference'),('lower','interval_lower'),('upper','interval_upper'),('valid','valid_draws')]:check(v[a],r[b],'kernel/'+a)
  result.append(dict(experiment='kernel',comparison='context7_minus_periodic',metric=mask,**v))
 rng=np.random.default_rng(610291)
 for r in json.loads((base/'data/scaling_intervals.json').read_text()):
  # Original code drew 46 cluster strings once per replicate, sequentially per contrast.
  draws=rng.choice(clusters,size=(2000,nc),replace=True);weights=np.array([[int((draw==c).sum()) for c in clusters] for draw in draws])
  new='scale/'+'/'.join(map(str,r['new']))+'/{seed}';ref='scale/'+'/'.join(map(str,r['reference']))+'/{seed}'
  for mask in ('all','fixed030','base600_q95'):
   v=paired(new,ref,mask,weights);e=r[mask]
   for a,b in [('difference','difference_new_minus_reference'),('valid','valid_draw_count'),('empty','empty_draw_count')]:check(v[a],e[b],'scaling/'+a)
   check(v['lower'],e['ci95'][0],'scaling/lower');check(v['upper'],e['ci95'][1],'scaling/upper')
   result.append(dict(experiment='scale',comparison='/'.join(map(str,r['new']))+'_minus_'+'/'.join(map(str,r['reference'])),metric=mask,**v))
 return result,checks

def run(base,output,with_intervals=False):
 manifest,groups=load(base);expected=json.loads((base/'data/expected_metrics.json').read_text());computed={};checks=0
 if set(groups)!=set(expected):raise ValueError('Reference group set mismatch')
 for gid,rs in groups.items():
  computed[gid]=metrics(rs)
  for k,v in expected[gid].items():
   if computed[gid][k] is None or not math.isclose(computed[gid][k],v,rel_tol=1e-10,abs_tol=1e-12):raise ValueError(f'Metric mismatch: {gid}/{k}')
   checks+=1
 # Complete all checks before writing any successful output.
 ci,ci_checks=intervals(groups,base) if with_intervals else ([],0)
 output.mkdir(parents=True,exist_ok=True)
 writecsv(output/'per_seed_metrics.csv',[dict(group=g,**m) for g,m in computed.items()])
 aggregate=defaultdict(list)
 for gid,m in computed.items():aggregate[gid.rsplit('/',1)[0]].append(m)
 summary=[]
 for name,values in aggregate.items():
  if len(values)!=5:raise ValueError('A summary must contain all five seeds')
  z={'configuration':name,'seeds':5}
  for k in values[0]:
   v=[m[k] for m in values];z[k+'_mean']=fmean(v);z[k+'_sd']=stdev(v)
  summary.append(z)
 writecsv(output/'summary_metrics.csv',summary)
 figure=[]
 for gid,m in computed.items():
  if not gid.startswith('scale/'):continue
  _,mode,n,arm,seed=gid.split('/');figure.append(dict(mode=mode,n=int(n),arm=arm,seed=int(seed),mae=m['mae'],r2=m['r2'],fixed030_mae=m['fixed_030_mae'],precision_at_10=m['precision_at_10'],precision_at_100=m['precision_at_100']))
 writecsv(output/'figure7_metrics.csv',figure)
 if ci:writecsv(output/'paired_cluster_intervals.csv',ci)
 report=dict(status='PASS',groups=len(groups),rows=manifest['rows'],unique_records=manifest['unique_record_ids'],metric_checks=checks,interval_checks=ci_checks,interval_replay_completed=with_intervals,fit_calls=0,forward_calls=0,scope=manifest['scope'])
 (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('reproduced-optimization'));p.add_argument('--intervals',action='store_true',help='also recompute 48 shared-cluster uncertainty intervals (requires NumPy)');args=p.parse_args()
 try:print(json.dumps(run(BASE,args.output,args.intervals),indent=2))
 except (ValueError,OSError,KeyError) as e:p.exit(2,f'Reproduction failed: {e}\n')
if __name__=='__main__':main()
