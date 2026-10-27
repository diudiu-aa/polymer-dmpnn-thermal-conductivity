"""Five-seed open/periodic/ranking comparison on disjoint structure clusters.

Only PolyOmics records in the historical training partition are used. Samples
and graph validity are fixed before labels; prediction files are fixed before
validation labels. Previously used record groups are excluded where recorded.
"""
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
parser=argparse.ArgumentParser(description="Reproduce the paired five-seed cluster comparison from prepared source tables.")
parser.add_argument("--project-root",required=True,type=Path,help="Directory containing the prepared outputs/ input tables")
parser.add_argument("--output",required=True,type=Path,help="New directory for prediction and weight outputs")
args=parser.parse_args()
ROOT=args.project_root.resolve()
from polymer_dmpnn.graph import PeriodicDMPNN,build_trimer_graph,batch_graphs
R03=ROOT/'outputs/r03-total-label-budget-2026-10-18';DATA=R03/'data';INDEX=R03/'inputs/cluster_holdout_records.csv';META=R03/'inputs/metadata_no_kappa.csv';RAW=ROOT/'outputs/data-expansion-2026-10-15/raw/general_polymers_with_sp_abbe_dynamic-dielectric.csv';OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)

SEEDS=(32101,32102,32103,32104,32105); ARMS=('open','periodic','periodic_rank'); EPOCHS=50;BATCH=16;NTRAIN=600;NVAL=400;RANK_LAMBDA=.3;MIN_GAP=.02

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def nearest(x,q):return float(np.sort(x)[math.ceil(q*len(x))-1])
def lr(step,n):
 per=math.ceil(n/BATCH);warm=2*per;total=EPOCHS*per
 return 1e-4+step*9e-4/warm if step<=warm else 1e-3*(.1**((step-warm)/(total-warm)))
def labels_for(ids,metadata):
 by_uuid={}
 for rid in ids:by_uuid.setdefault(metadata[rid]['uuid'],[]).append(rid)
 assert all(by_uuid)
 result={};bad=[]
 with RAW.open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   u=r['UUID']
   if u not in by_uuid:continue
   try:
    x=float(r['thermal_conductivity']);assert math.isfinite(x) and x>0
   except (ValueError,AssertionError):bad.extend(by_uuid[u]);continue
   for rid in by_uuid[u]:
    if rid in result:raise ValueError('duplicate UUID in raw data '+u)
    result[rid]=x
 if bad or set(result)!=set(ids):raise ValueError(f'selected targets missing/nonfinite: bad={len(bad)} missing={len(set(ids)-set(result))}')
 return result
def metrics(y,p,threshold):
 ae=abs(y-p);sel=y>=threshold;den=np.sum((y-y.mean())**2);out={'n':len(y),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean((y-p)**2))),'r2':float(1-np.sum((y-p)**2)/den),'train_q95':threshold,'tail_n':int(sel.sum()),'tail_mae':float(ae[sel].mean()) if sel.any() else None,'spearman_all':float(spearmanr(y,p).statistic)}
 for k in (10,20,100):out[f'precision_at_{k}']=float(len(set(np.argsort(-p,kind='stable')[:k])&set(np.argsort(-y,kind='stable')[:k]))/k)
 return out
def run():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);t0=time.monotonic();excluded_groups=set()
 for proto in ('cluster_holdout','random_group'):
  for seed in (20261111,20261112,20261113):
   for budget in (100,300,1000,3000):
    with (DATA/f'{proto}_s{seed}_L{budget}_representatives_before_labels.csv').open(newline='') as f:
     excluded_groups.update(x['isolation_group'] for x in csv.DictReader(f))
 idx={}
 with INDEX.open(newline='') as f:
  for x in csv.DictReader(f):
   if x['partition']=='train' and x['isolation_group'] not in excluded_groups:idx[x['record_id']]=x
 metadata={}
 with META.open(newline='') as f:
  for x in csv.DictReader(f):
   if x['record_id'] in idx and x['original_UUID'] and x['smiles_canonical']:metadata[x['record_id']]={'uuid':x['original_UUID'],'smiles':x['smiles_canonical'],'cluster':idx[x['record_id']]['cluster_id'],'group':idx[x['record_id']]['isolation_group']}
 reps={}
 for rid,x in metadata.items():
  if x['group'] not in reps or rid<reps[x['group']]:reps[x['group']]=rid
 group_clusters={}; cluster_groups={}
 for group,rid in reps.items():
  c=metadata[rid]['cluster'];group_clusters[group]=c;cluster_groups.setdefault(c,[]).append(group)
 clusters=sorted(cluster_groups);assert len(clusters)>80
 heldout_order=np.array(clusters,dtype=object);np.random.default_rng(32100).shuffle(heldout_order);fixed_val_clusters=set(heldout_order[:max(1,int(round(.25*len(heldout_order))))])
 cache={};invalid={};splits={}
 def build(rid):
  s=metadata[rid]['smiles']
  if s in invalid:return False
  if s not in cache:
   try:cache[s]={'open':build_trimer_graph(s,periodic_closure=False),'periodic':build_trimer_graph(s,periodic_closure=True)}
   except Exception as e:invalid[s]=str(e)[:200];return False
  return True
 for seed in SEEDS:
  rng=np.random.default_rng(seed);vc=fixed_val_clusters;tc=set(clusters)-vc;assert not vc&tc
  train_pool=[reps[g] for c in tc for g in cluster_groups[c]];val_pool=[reps[g] for c in vc for g in cluster_groups[c]];train_pool.sort();val_pool.sort();rng.shuffle(train_pool);rng.shuffle(val_pool)
  tr=[];va=[]
  for rid in train_pool:
   if build(rid):tr.append(rid)
   if len(tr)==NTRAIN:break
  for rid in val_pool:
   if build(rid):va.append(rid)
   if len(va)==NVAL:break
  if len(tr)!=NTRAIN or len(va)!=NVAL:raise RuntimeError(f'insufficient valid graphs {seed} {len(tr)} {len(va)}')
  assert not {metadata[i]['cluster'] for i in tr}&{metadata[i]['cluster'] for i in va}
  assert not {metadata[i]['group'] for i in tr}&{metadata[i]['group'] for i in va}
  splits[str(seed)]={'train':tr,'validation':va,'train_cluster_ids':sorted({metadata[i]['cluster'] for i in tr}),'validation_cluster_ids':sorted({metadata[i]['cluster'] for i in va})}
  print('selected',seed,'train',len(tr),'val',len(va),'clusters',len(splits[str(seed)]['train_cluster_ids']),len(splits[str(seed)]['validation_cluster_ids']),flush=True)
 lock={'scope':'strict high-level cluster-disjoint resplit of historical train partition; excluded recorded prior group roles; not guaranteed unseen across all past work; no old test labels','budget_definition':'600 fit + 400 evaluation labels, no calibration; differs from R04 600/200/200','seed_count':len(SEEDS),'split_rule':'assign the same 25% of eligible clusters to held-out validation for all five seeds; randomize record samples and initialization by seed; one canonical representative per isolation group; label-blind selection and graph validation','arms':ARMS,'rank_rule':'within-batch pair with member >= training q90 and label gap >=0.02; softplus coefficient 0.3','seed_splits':splits,'invalid_smiles_count':len(invalid),'metadata_sha256':digest(META),'index_sha256':digest(INDEX),'raw_source_sha256':digest(RAW),'code_sha256':digest(Path(__file__))}
 save(OUT/'selection_lock.json',lock)
 train_ids=set(i for spl in splits.values() for i in spl['train']);val_ids=set(i for spl in splits.values() for i in spl['validation']);assert train_ids.isdisjoint(val_ids);all_training_labels=labels_for(train_ids,metadata)
 fits=[];predictions={}
 for seed in SEEDS:
  spl=splits[str(seed)];tr=spl['train'];va=spl['validation'];y=np.array([all_training_labels[i] for i in tr]);mu=float(y.mean());sd=float(y.std());q90=nearest(y,.9);q95=nearest(y,.95);rng=np.random.default_rng(seed+10000);perms=[rng.permutation(len(tr)) for _ in range(EPOCHS)]
  for arm in ARMS:
   torch.manual_seed(seed);model=PeriodicDMPNN(133,14,hidden_size=128,depth=3,dropout=.1);torch.manual_seed(seed+20000);opt=torch.optim.Adam(model.parameters(),lr=1e-4);model.train();step=0;npairs=0;now=time.monotonic()
   for perm in perms:
    for off in range(0,len(perm),BATCH):
     ix=perm[off:off+BATCH];batch=batch_graphs([cache[metadata[tr[i]]['smiles']]['open' if arm=='open' else 'periodic'] for i in ix]);target=torch.as_tensor((y[ix]-mu)/sd,dtype=torch.float32);step+=1
     for pg in opt.param_groups:pg['lr']=lr(step,len(tr))
     opt.zero_grad();pred=model(batch);loss=((pred-target)**2).mean()
     if arm=='periodic_rank':
      raw=torch.as_tensor(y[ix],dtype=torch.float32);delta=raw[:,None]-raw[None,:];eligible=torch.triu(torch.ones_like(delta,dtype=torch.bool),diagonal=1)&(torch.abs(delta)>=MIN_GAP)&((raw[:,None]>=q90)|(raw[None,:]>=q90));npairs+=int(eligible.sum())
      if eligible.any():score=pred[:,None]-pred[None,:];loss=loss+RANK_LAMBDA*torch.nn.functional.softplus(-torch.sign(delta[eligible])*score[eligible]).mean()
     if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
     loss.backward();opt.step()
   model.eval();z=[]
   with torch.no_grad():
    for off in range(0,len(va),BATCH):z.extend((model(batch_graphs([cache[metadata[i]['smiles']]['open' if arm=='open' else 'periodic'] for i in va[off:off+BATCH]])).numpy()*sd+mu).tolist())
   z=np.array(z);assert np.isfinite(z).all();prediction=OUT/f'prediction_{seed}_{arm}.csv'
   with prediction.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['record_id','prediction']);w.writerows(zip(va,z))
   path=OUT/f'weights_{seed}_{arm}.pt';torch.save({'state_dict':model.state_dict(),'target_mean':mu,'target_sd':sd,'seed':seed,'arm':arm},path)
   fit={'seed':seed,'arm':arm,'fit_n':len(tr),'q90_train':q90,'q95_train':q95,'rank_pair_count':npairs,'elapsed_seconds':time.monotonic()-now,'prediction_sha256':digest(prediction),'weights_sha256':digest(path)};fits.append(fit);predictions[(seed,arm)]=z
   print('finished fit',seed,arm,'pairs',npairs,'seconds',round(fit['elapsed_seconds'],1),flush=True)
 save(OUT/'fitted_prediction_lock.json',{'selection_sha256':digest(OUT/'selection_lock.json'),'fits':fits})
 # Only after all model predictions are fixed access validation numeric values.
 val_labels=labels_for(val_ids,metadata);results=[]
 for seed in SEEDS:
  va=splits[str(seed)]['validation'];y=np.array([val_labels[i] for i in va]);q95=next(x['q95_train'] for x in fits if x['seed']==seed)
  for arm in ARMS:
   z=predictions[(seed,arm)];m=metrics(y,z,q95);results.append({'seed':seed,'arm':arm,'metrics':m})
   with (OUT/f'evaluated_{seed}_{arm}.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['record_id','true','prediction']);w.writerows(zip(va,y,z))
   print('result',seed,arm,json.dumps(m),flush=True)
 save(OUT/'results.json',{'scope':'five-seed strict structure-cluster disjoint on historical PolyOmics train partition; retrospective project lineage, not external experiment or independent cohorts','selection_lock_sha256':digest(OUT/'selection_lock.json'),'fit_lock_sha256':digest(OUT/'fitted_prediction_lock.json'),'results':results,'elapsed_seconds':time.monotonic()-t0});print('DONE',flush=True)
if __name__=='__main__':run()
