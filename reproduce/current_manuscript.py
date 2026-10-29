"""Recompute current main Tables 1-3 and optionally redraw all five main figures.

Saved-prediction replay only. This entry never trains or performs model inference.
"""
import argparse,csv,json,subprocess,sys,hashlib,tarfile
from pathlib import Path

BASE=Path(__file__).resolve().parent
ARCHIVE_SHA256='c37313a7fd09a50b96be7f613e342975bf8f83f6c9a1732e65ec138e3af2809f'
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def table(p,records):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def run(script,*args):subprocess.run([sys.executable,str(BASE/script),*map(str,args)],check=True)
def prepare_resources():
    archive=BASE/'completion_v0_9.tar.gz'
    if not archive.is_file():
        if (BASE/'completion/manifest.json').is_file() and (BASE/'current_figures/method_structure.json').is_file():return
        raise FileNotFoundError('Download the complete repository including reproduce/completion_v0_9.tar.gz')
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA256:raise ValueError('Completion archive checksum mismatch')
    if (BASE/'completion/manifest.json').is_file() and (BASE/'current_figures/method_structure.json').is_file():return
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        if sum(m.size for m in members)>80_000_000:raise ValueError('Oversized resource archive')
        for m in members:
            name=Path(m.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0] not in ('completion','current_figures') or not m.isfile():raise ValueError('Unsafe resource member')
            target=BASE/name;target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(m) as src,target.open('wb') as dst:dst.write(src.read())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path);ap.add_argument('--prepare-only',action='store_true',help='Verify and unpack the checksum-locked resource archive without calculation')
    ap.add_argument('--figures',action='store_true',help='Replot main Figures1-5; needs the figures extra')
    ap.add_argument('--download-reference',action='store_true',help='Allow Figure5 to retrieve the fixed author APFE reference CSV')
    ap.add_argument('--reference-csv',type=Path,help='Alternatively supply the checksum-matched author CSV for Figure5')
    a=ap.parse_args();prepare_resources()
    if a.prepare_only:
        print(json.dumps({'status':'RESOURCES_READY','fit':0,'forward':0}));return
    if a.output is None:ap.error('--output is required except with --prepare-only')
    if a.figures and not(a.download_reference or a.reference_csv):ap.error('--figures requires --download-reference or --reference-csv for the APFE panel')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    run('paper_tables.py','--verify','--output',out/'original-release')
    run('optimization/replay.py','--intervals','--output',out/'optimization')
    run('completion/replay.py','--output',out/'completion')
    first=rows(out/'original-release/table_3.csv');second=rows(out/'original-release/table_5.csv')
    result=[];maekey='MAE (W m⁻¹ K⁻¹)'
    for model in ['Open D-MPNN','Periodic D-MPNN']:
        r=next(r for r in first if r['Model']==model);result.append({'Partition':'First','Model':model,'MAE':r[maekey]})
    for model in ['Open D-MPNN, post hoc','Periodic D-MPNN','Training-CV Ridge','Training-CV ExtraTrees']:
        r=next(r for r in second if r['Model']==model);result.append({'Partition':'Second','Model':model.replace(', post hoc',''),'MAE':r[maekey]})
    opt=rows(out/'optimization/summary_metrics.csv')
    for key,label in [('kernel/unit','Repeat-unit KRR'),('kernel/boundary_context7','Boundary-context KRR')]:
        r=next(r for r in opt if r['configuration']==key);result.append({'Partition':'Second','Model':label,'MAE':f"{float(r['mae_mean']):.5f} ± {float(r['mae_sd']):.5f}"})
    table(out/'main_table_1.csv',result)
    budget=rows(out/'completion/B_summary_metrics.csv');result=[]
    for n,u in [(600,1900),(3000,1900),(600,9400),(3000,9400)]:
        row={'Fitting labels':n,'Updates':u}
        for arm in ['open','periodic']:
            r=next(r for r in budget if int(r['labels'])==n and int(r['updates'])==u and r['arm']==arm and r['endpoint']=='mae')
            row[arm+' MAE']=f"{float(r['mean']):.5f} ± {float(r['seed_sd']):.5f}"
        result.append(row)
    table(out/'main_table_2.csv',result)
    obj=rows(out/'completion/new25_summary_metrics.csv');result=[]
    for arm in ['mse','q90_weight2','multitask']:
        row={'Model':arm}
        for key,n,mult in [('mae',5,1),('r2',4,1),('high_mae',5,1),('high_bias',5,1),('top100_overlap',1,100)]:
            r=next(r for r in obj if r['family']=='hk01' and r['arm']==arm and r['endpoint']==key)
            row[key]=f"{mult*float(r['mean']):.{n}f} ± {mult*float(r['seed_sd']):.{n}f}"
        result.append(row)
    table(out/'main_table_3.csv',result)
    if a.figures:
        run('current_figures/method_overview.py','--output',out/'figure1')
        run('current_figures/representation.py','--output',out/'figure2')
        args=['--replay-output',out/'completion','--output',out/'figures3-5','--figure','all']
        if a.download_reference:args+=['--download-reference']
        if a.reference_csv:args+=['--reference-csv',a.reference_csv.resolve()]
        run('completion/plot.py',*args)
    receipt={'status':'PASS','main_tables':[1,2,3],'main_figures':[1,2,3,4,5] if a.figures else [],'fit_calls':0,'model_forward_calls':0,'scope':'Current main-table and saved-analysis replay; figure1 constructs graphs only; not fresh training or independent validation.'}
    (out/'current_manuscript_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
