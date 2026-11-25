"""FAV -> PGF common B/C completion: frozen P2 mappings, no training or refitting."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__='scripts.trainable_or_3way.fav_pgf_bc'
import argparse
import numpy as np
from scripts.trainable_or_3way.reinforcement_p2.common import *
from scripts.trainable_or_3way.reinforcement_p2.inputs import preflight
from scripts.trainable_or_3way.reinforcement_p2.extract import infer_cohort
from scripts.trainable_or_3way.reinforcement.inputs import scores
from scripts.trainable_or_3way.reinforcement.statistics import metric_table,paired_changes
from scripts.trainable_or_3way.reinforcement.analysis import csv_text,strata
from .inputs import FORMAT,P2,CROSS,SUMMARY,OUTPUT,prepare,verify_preserved,sha_array

NAMES={'M0':'Baseline','M1':'an artifact-only baseline with learned score adjustment',
       'M2':'Emotion OR','M3':'Hierarchical OR','M4':'Proposed',
       'M5':'Artifact + visual emotion','M6':'Artifact + audio emotion'}

def bind_inputs(output,run,data,models):
    p=output/'inputs/PGF_test.npz'
    if p.exists():
        old=load_npz(p)
        require(set(old)==set(data) and all(sha_array(old[k])==sha_array(data[k]) for k in data),'Normalized input changed')
    else:atomic_npz(p,**data)
    require(digest({k:sha_array(v) for k,v in sorted(data.items())})==run['normalized_content_sha256'],'Input binding differs')
    # New lock refers to existing P2 models byte-for-value; there is no fitting call.
    immutable(output/'fusion/FUSION.json',models)
    require(models==read(Path(run['p2'])/'fusion/FUSION.json'),'P2 fusion mapping changed')
    immutable(output/'fusion/LOCKED_BEFORE_TEST.json',dict(run=digest(run),source='FAV',
        models_sha256=sha(output/'fusion/FUSION.json'),original_models_sha256=run['fusion']['p2_models_sha256'],
        conditions=['B','C'],threshold=.5,target_training=False,refit=False))

def evaluate(output,run,paths,models):
    lock=read(output/'fusion/LOCKED_BEFORE_TEST.json')
    require(lock['run']==digest(run) and lock['models_sha256']==sha(output/'fusion/FUSION.json')
            and read(output/'fusion/FUSION.json')==models,'Frozen fusion changed')
    result=[]
    for c in ('B','C'):
        d=load_npz(paths[c]);all_scores,valid=scores(d['raw'],models[c])
        for panel in ('full','common_emotion'):
            check_stop(output)
            folder=output/'evaluation'/f'FAV__{c}__PGF__test'/panel
            identity=digest(dict(run=digest(run),condition=c,panel=panel,raw_sha256=sha(paths[c]),fusion=lock['models_sha256']))
            if not completed(folder,identity):
                mask=np.ones(len(valid),bool) if panel=='full' else valid
                p={m:v[mask].copy() for m,v in all_scores.items()}
                for m in ('M2','M3'):p[m][~valid[mask]]=p['M1'][~valid[mask]]
                y=d['labels'][mask,0]
                require(set(y.tolist())=={0,1},'Evaluation needs both classes')
                bar=Progress('FAV-PGF '+c+' '+panel,len(y),folder)
                report=dict(source='FAV',target='PGF',condition=c,cohort=panel,n=len(y),
                    missing_emotion=int((~valid[mask]).sum()),metrics=metric_table(y,p),threshold=.5,
                    fusion_refit=False,common_p2_protocol=True,bootstrap=False)
                atomic_json(folder/'METRICS.json',report)
                atomic_json(folder/'STRATA.json',strata(y,d['labels'][mask,1:],p))
                atomic_json(folder/'PAIRED_ERRORS.json',paired_changes(y,p,d['sample_id'][mask]))
                predictions=[dict(sample_id=str(sid),cluster=str(group),clip_label=int(label),emotion_complete=bool(v),
                    **{m:float(p[m][i]) for m in p})
                    for i,(sid,group,label,v) in enumerate(zip(d['sample_id'][mask],d['cluster'][mask],y,valid[mask]))]
                atomic_text(folder/'predictions.csv',csv_text(predictions))
                finish(folder,identity,['METRICS.json','STRATA.json','PAIRED_ERRORS.json','predictions.csv'])
                bar.update(len(y),force=True)
            result.append(read(folder/'METRICS.json'))
    return result

def summarize(output,run,results):
    rows=[]
    for r in results:
        for method,m in r['metrics'].items():
            rows.append(dict(source='FAV',target='PGF',condition=r['condition'],cohort=r['cohort'],
                method=method,name=NAMES[method],**m))
    atomic_json(output/'SUMMARY.json',dict(rows=rows,scope='FAV->PGF B/C',bootstrap=False))
    atomic_text(output/'SUMMARY.csv',csv_text(rows))
    old=read(run['summary'])
    merged=[dict(r) for r in old['rows'] if r['scope']=='common_BC']
    names={'M0':'baseline','M1':'artifact_affine','M4':'proposed'}
    for r in rows:
        if r['cohort']!='full' or r['method'] not in names:continue
        merged.append(dict(source='FAV',target='PGF',condition=r['condition'],method=names[r['method']],scope='common_BC',
            **{k:r[k] for k in ('n','real','fake','roc_auc','fake_f1','real_fpr')},
            metrics_path=str(output/'evaluation'/f"FAV__{r['condition']}__PGF__test"/'full/METRICS.json')))
    expected={(s,t,c,m) for s in ('FAV','AVDF1M','PGF') for t in ('FAV','AVDF1M','PGF')
              for c in ('B','C') for m in names.values()}
    keys=[(r['source'],r['target'],r['condition'],r['method']) for r in merged]
    require(len(keys)==54 and set(keys)==expected,'The full 18-condition/54-row comparison is incomplete or duplicated')
    atomic_json(output/'ALL_DATASETS_BC.json',dict(rows=merged,completed_conditions=18,
        cross_directions=6,in_domain_directions=3,common_bn_precision_policy=True,
        identical_source_training_recipe=False,pgf_bootstrap=False,
        historical_summary_preserved=run['summary']))
    atomic_text(output/'ALL_DATASETS_BC.csv',csv_text(merged))
    lines=['# FAV -> PGF: common B/C evaluation','',
        'Existing P2 FAV-dev mappings and FAV emotion scores reused. No target fitting or encoder training.',
        'B: stored X3D BN; C: per-video X3D BN. FP32 / TF32 off. Audio and emotion scores shared.',
        'Metrics below are percentages. F1/FPR threshold: 0.5; AUC is threshold-independent.','',
        '| Condition | Method | ROC-AUC | Fake F1 | Real FPR |','|---|---|---:|---:|---:|']
    for r in rows:
        if r['cohort']=='full' and r['method'] in names:
            lines.append(f"| {r['condition']} | {r['name']} | {100*r['roc_auc']:.2f} | {100*r['fake_f1']:.2f} | {100*r['real_fpr']:.2f} |")
    lines+=['','ALL_DATASETS_BC contains 6 cross directions and 3 in-domain directions, each under B/C.',
        'This is common inference, not identical source training. No PGF confidence intervals or extended bootstrap family are claimed.',
        'Update manuscript tables, interpretation and pending-evaluation wording after reviewing these results.']
    atomic_text(output/'REPORT.md','\n'.join(lines)+'\n')

def verify_stores(output,run):
    from scripts.trainable_or_3way.reinforcement_p2.storage import ScoreStore
    from scripts.trainable_or_3way.reinforcement_p2.runtime import validate_score
    rows=read(output/'cohorts/PGF_test.json')
    for mod in ('video','audio'):
        store=ScoreStore(output/'scores/PGF/test'/mod,dict(run=digest(run),cohort='PGF/test',modality=mod),readonly=True)
        try:
            _,done=store.verify(rows,lambda v:validate_score(v,mod))
            require(done,'Incomplete '+mod+' scores')
        finally:store.close()

def execute(output,run,cohorts,models):
    if completed(output,digest(run)):
        verify_preserved(run['preserved_inputs']);verify_stores(output,run)
        print('[COMPLETE] Verified; nothing to repeat',flush=True);return
    preflight(output,run,cohorts)
    check_stop(output)
    require((output/'fusion/LOCKED_BEFORE_TEST.json').is_file(),'Lock must precede target inference')
    paths=infer_cohort(output,run,'PGF/test',cohorts['PGF/test'])
    atomic_json(output/'STATUS.json',dict(status='RUNNING',stage='evaluate'))
    results=evaluate(output,run,paths,models)
    verify_preserved(run['preserved_inputs'])
    summarize(output,run,results)
    files=[]
    for p in sorted(output.rglob('*')):
        if not p.is_file() or p==output/'COMPLETED.json':continue
        if p.name in ('STATUS.json','progress.json','.writer.lock','STOP_REQUESTED','LAST_ERROR.json'):continue
        if p.name.startswith('scores.sqlite') or p.suffix=='.partial':continue
        files.append(p.relative_to(output).as_posix())
    finish(output,digest(run),files)
    print('[COMPLETE] '+str(output/'REPORT.md'),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','run','status','verify'])
    p.add_argument('--output',default=str(OUTPUT));p.add_argument('--p2',default=str(P2))
    p.add_argument('--historical',default=str(CROSS));p.add_argument('--summary',default=str(SUMMARY))
    p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--decode-workers',type=int,default=2);p.add_argument('--prefetch',type=int,default=2)
    p.add_argument('--local-paths',default='configs/local_paths.yaml')
    p.add_argument('--ffmpeg',default='.venv/Scripts/ffmpeg.exe')
    return p

def run(args):
    output=resolve(args.output)
    if args.command=='status':
        paths=[output/'STATUS.json',*sorted(output.glob('scores/*/*/*/progress.json'))]
        if not (output/'RUN.json').exists():print('[STATUS] Not initialized');return
        for p in paths:
            if p.exists():print(p.relative_to(output),read(p),flush=True)
        return
    require(output.is_relative_to(ROOT/'reports') and output!=ROOT/'reports','Use a dedicated reports output')
    safe_output(output,[resolve(args.p2),resolve(args.historical),resolve(args.summary).parent,ROOT/'checkpoints'])
    if output.exists():
        if (output/'RUN.json').exists():require(read(output/'RUN.json').get('format')==FORMAT,'Different experiment output')
        else:require(not any(p.name!='.writer.lock' for p in output.iterdir()),'Nonempty output; use a new directory')
    with writer_lock(output):
        if args.command=='verify':
            require((output/'RUN.json').is_file(),'Run not initialized')
            saved=read(output/'RUN.json');require(completed(output,digest(saved)),'Run is incomplete')
            verify_preserved(saved['preserved_inputs']);verify_stores(output,saved)
            print('[VERIFY] PASS: results, source mappings and preserved inputs',flush=True);return
        owned=False
        try:
            description,cohorts,data,models=prepare(args)
            immutable(output/'RUN.json',description);owned=True
            immutable(output/'cohorts/PGF_test.json',cohorts['PGF/test'])
            bind_inputs(output,description,data,models)
            if args.command=='preflight':
                preflight(output,description,cohorts)
                atomic_json(output/'STATUS.json',dict(status='PREFLIGHT_PASS',inference_started=False))
                print('[PREFLIGHT] PASS; no model forward started',flush=True)
            else:
                execute(output,description,cohorts,models)
                atomic_json(output/'STATUS.json',dict(status='COMPLETE',stage='FAV_to_PGF_BC'))
        except BaseException as exc:
            if owned:atomic_json(output/'STATUS.json',dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                error=str(exc),resume='Repeat the same run command and output; committed clips retained'))
            raise

if __name__=='__main__':
    try:run(parser().parse_args())
    except KeyboardInterrupt:
        print('[INTERRUPTED] Re-run the same command to resume',flush=True);sys.exit(130)
    except Exception as exc:
        print('[FAILED] '+str(exc)+'; existing results retained',file=sys.stderr,flush=True);raise
