"""Source-dev refit of trainable final fusion with immutable workstation B/C scores."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__='scripts.trainable_or_3way.workstation_fusion'
import argparse,csv,json,time,traceback
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from .inputs import DEFAULT_ARTIFACT,DEFAULT_EMOTION,verify_emotion_parent,completed,load_cohort
from scripts.trainable_or_3way.common import ROOT,read,sha,digest,require,immutable,atomic_json,WriterLock,status,score_or
from scripts.trainable_or_3way.evaluate import export,verify_evaluation,combine
from scripts.trainable_or.fusion import fit

METHODS=('baseline','artifact_affine','trainable_final')
DEFAULT_OUTPUT=ROOT/'reports/workstation_trainable_final_fusion_v1/source_AVDF1M/seed_42'

def implementation():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in (
        'scripts/trainable_or/fusion.py','scripts/trainable_or/evaluation.py','scripts/trainable_or_3way/evaluate.py',
        'scripts/trainable_or_3way/common.py','scripts/trainable_or_3way/storage.py','scripts/trainable_or_3way/data.py',
        'scripts/hierarchical_score_fusion_GRU_Temproal_Attention/common.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths) if not p.name.startswith('test_')}

def fit_source(data,condition):
    require(all(r.dataset=='AVDF1M' and r.split=='dev' for r in data['records']),'Fit accepts only AVDF1M source dev')
    raw=data['raw'][condition];y=np.array([r.clip_label for r in data['records']]);valid=np.isfinite(raw[:,2:]).all(1)
    artifact=score_or(raw[:,0],raw[:,1]);emotion=score_or(raw[valid,2],raw[valid,3])
    status('FIT '+condition,'source dev only; complete emotion '+str(int(valid.sum()))+'/'+str(len(y)))
    return dict(source='AVDF1M',source_dev_n=len(y),complete_emotion_dev_n=int(valid.sum()),
        trainable_final=fit(artifact[valid],emotion,y[valid],split='dev',regularization=.01),
        artifact_affine=fit(artifact,None,y,split='dev',regularization=.01))

def paired(records,scores):
    y=np.array([r.clip_label for r in records]);out={}
    for control in ('baseline','artifact_affine'):
        before=scores[control]>=.5;after=scores['trainable_final']>=.5
        out[control]=dict(corrected=int(((before!=y)&(after==y)).sum()),regressed=int(((before==y)&(after!=y)).sum()),
            extra_real_false_positives=int(((y==0)&~before&after).sum()),real_false_positives_removed=int(((y==0)&before&~after).sum()),
            extra_fake_detections=int(((y==1)&~before&after).sum()),fake_detections_lost=int(((y==1)&before&~after).sum()))
    return out

def verify_saved(folder,records,fusion):
    result=verify_evaluation(folder,records,fusion)
    with (folder/'predictions.csv').open(encoding='utf-8',newline='') as f:saved=list(csv.DictReader(f))
    y=np.array([int(r['clip_label']) for r in saved]);cols={k:np.array([float(r[k]) for r in saved]) for k in ('Av','Aa','Ev','Ea',*METHODS)}
    a=1-(1-cols['Av'])*(1-cols['Aa']);valid=np.isfinite(cols['Ev'])&np.isfinite(cols['Ea']);e=1-(1-cols['Ev'])*(1-cols['Ea'])
    expected={}
    for method in ('artifact_affine','trainable_final'):
        model=fusion[method];w=model['weights'];eps=model['epsilon']
        def logit(x):
            p=np.clip(x,eps,1-eps);return np.log(p/(1-p))
        z=w[0]*logit(a)+w[-1]
        if method=='trainable_final':z[valid]+=w[1]*logit(e[valid])
        expected[method]=expit(z)
        if method=='trainable_final':expected[method][~valid]=expected['artifact_affine'][~valid]
        require(np.allclose(expected[method],cols[method],atol=1e-12,rtol=0),'Independent fusion formula mismatch')
    report=read(folder/'metrics.json')
    for method in METHODS:
        p=cols[method];pred=p>=.5;tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum());m=report['full_cohort'][method]
        require(m['confusion_matrix']==[[tn,fp],[fn,tp]],'Independent confusion matrix differs')
        require(abs(m['f1']-2*tp/(2*tp+fp+fn))<1e-12 and abs(m['real_fpr']-fp/(tn+fp))<1e-12,'Independent F1/FPR differs')
        require(abs(m['roc_auc']-roc_auc_score(y,p))<1e-12,'Independent AUC differs')
    return dict(result,independent_fusion_formula=True,independent_metric_arithmetic=True)

def verify(out):
    run,done=completed(out,'identity');lock=read(out/'LOCKED_BEFORE_TEST.json')
    require(lock['run_identity']==digest(run) and lock['fusion_sha256']==sha(out/'FUSION.json'),'Fusion lock changed')
    for p,h in run['implementation'].items():require(sha(ROOT/p)==h,'Evaluation implementation changed')
    return dict(status='PASS',targets=run['targets'],conditions=['B','C'],files=len(done['files']))

def wait_ready(folder,out,key):
    while not (folder/'COMPLETED.json').exists():
        if (out/'STOP_REQUESTED').exists():raise KeyboardInterrupt('Stop requested while waiting for artifact completion')
        state=read(folder/'STATUS.json')
        require(state['status'] not in ('FAILED','INTERRUPTED'),'Artifact inference stopped: '+str(state))
        progress=read(folder/'progress.json') if (folder/'progress.json').exists() else {}
        atomic_json(out/'STATUS.json',dict(status='WAITING_FOR_ARTIFACT',cohort=key,artifact_status=state,artifact_progress=progress))
        status('WAIT ARTIFACT',key+' '+str(progress.get('completed','?'))+'/'+str(progress.get('total','?'))+'; source-dev fusion is already locked')
        time.sleep(30)

def execute(args):
    start=time.monotonic();out=args.output.resolve();ar=args.artifact_root.resolve();er=args.emotion_root.resolve()
    require(out.is_relative_to(ROOT/'reports') and not out.is_relative_to(ar) and not out.is_relative_to(er),'Use a separate reports output')
    require(not ar.is_relative_to(out) and not er.is_relative_to(out),'Output cannot contain input reports')
    require(len(set(args.targets))==len(args.targets)>0,'Duplicate or missing targets')
    if args.command=='verify':status('VERIFY',str(verify(out)));return
    status('INPUTS','Verify completed workstation artifacts and AVDF1M-trained emotion heads')
    parent,selected=verify_emotion_parent(er);input_receipts={}
    for name in ['dev_AVDF1M']+['test_'+t for t in args.targets]:
        header=read(ar/name/'RUN.json')
        require(header['source']=='AVDF1M' and not header['smoke_only'],'Unexpected artifact source')
        input_receipts[name]=dict(run=sha(ar/name/'RUN.json'),cohort=sha(ar/name/'COHORT.json'),expected_identity=digest(header))
    run=dict(format='workstation_emotion_trainable_final_v1',source='AVDF1M',targets=args.targets,conditions=['B','C'],
        artifact_root=str(ar),emotion_root=str(er),artifact_inputs=input_receipts,
        emotion_run_sha256=sha(er/'RUN.json'),emotion_completed_sha256=sha(er/'COMPLETED.json'),
        emotion_models={p:h for p,h in selected['models'].items() if p.startswith('models/emotion/')},
        source_subset=parent['source_subset'],implementation=implementation(),
        threshold=.5,regularization=.01,epsilon=1e-6,fitting='AVDF1M dev only; reused model-selection dev',
        target_training=False,independent_calibration=False,previous_targets_observed=True,
        pretrained_or_gru_retraining=False,backbone_inference=False,
        missing_emotion='Source-fitted artifact_affine fallback; common-emotion metrics also reported',
        controls=['raw OR baseline','artifact-only affine calibration','emotion trainable final fusion'])
    out.mkdir(parents=True,exist_ok=True)
    with WriterLock(out):
        owned=False
        try:
            if (out/'RUN.json').exists():require(read(out/'RUN.json')==run,'Run settings changed; use another output')
            else:require(not any(p.name!='.writer.lock' for p in out.iterdir()),'Output is not empty')
            immutable(out/'RUN.json',run);owned=True
            if (out/'COMPLETED.json').exists():status('RESUME',str(verify(out)));return
            atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='source_dev_join'))
            dev=load_cohort(ar,er,parent,selected,'AVDF1M','dev')
            fusion_path=out/'FUSION.json'
            if fusion_path.exists():
                fusion=read(fusion_path);require(fusion['run_identity']==digest(run) and fusion['source_dev_proof']==dev['proof'],'Source dev or model identity changed')
                status('RESUME','Verified saved source-dev fusion; no refit')
            else:
                fusion=dict(run_identity=digest(run),source_dev_proof=dev['proof'],conditions={c:fit_source(dev,c) for c in ('B','C')})
                atomic_json(fusion_path,fusion)
            immutable(out/'LOCKED_BEFORE_TEST.json',dict(run_identity=digest(run),fusion_sha256=sha(fusion_path),threshold=.5,
                emotion_models=run['emotion_models'],artifact_inputs=input_receipts,target_training=False,
                selection='source dev only; prior test exposure acknowledged; no target thresholds fitted'))
            summary=dict(source='AVDF1M',targets=args.targets,conditions={},limitations=['source dev reused after checkpoint selection','targets observed in previous experiments','single seed; no statistical significance claim'],
                pending=['AVDF1M full test excluded: artifact audio length failure remains unchanged'] if 'AVDF1M' not in args.targets else [])
            for dataset,split in [('AVDF1M','dev')]+[(t,'test') for t in args.targets]:
                key=dataset+'/'+split
                atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='evaluate',cohort=key))
                if split=='test':wait_ready(ar/('test_'+dataset),out,key)
                data=dev if split=='dev' else load_cohort(ar,er,parent,selected,dataset,split)
                require(data['compatibility']==dev['compatibility'],'Artifact weights/runtime/recipe differ across cohorts')
                for c in ('B','C'):
                    folder=out/'evaluation'/c/dataset/split;model=fusion['conditions'][c]
                    if (folder/'COMPLETED.json').exists():
                        done=read(folder/'COMPLETED.json');require(done['run_identity']==digest(run) and done['fusion_sha256']==sha(fusion_path),'Evaluation lock differs')
                        for name,h in done['files'].items():require(sha(folder/name)==h,'Completed result changed')
                        verify_saved(folder,data['records'],model);report=read(folder/'metrics.json')
                        status('RESUME',c+' '+key+' already complete')
                    else:
                        report=export(folder,data['records'],data['raw'][c],model,'AVDF1M',digest(run))
                        baseline=report['full_cohort']['baseline'];old=data['artifact_report']['conditions'][c]['baseline_or']
                        for k,j in (('roc_auc','roc_auc_probability'),('f1','fake_f1'),('real_fpr','real_fpr')):
                            require(abs(baseline[k]-old[j])<1e-12,'Reused baseline metric differs')
                        require(baseline['confusion_matrix']==old['confusion_matrix'],'Baseline confusion matrix differs')
                        scores,_=combine(data['raw'][c],model)
                        report.update(split=split,bn_condition=c,role='source calibration resubstitution' if split=='dev' else 'held-out project test',
                            calibration_on_target=False,paired=paired(data['records'],scores),input_proof=data['proof'])
                        report['delta_final_pp']={control:{k:100*(report['full_cohort']['trainable_final'][k]-report['full_cohort'][control][k]) for k in ('roc_auc','f1','real_fpr','fake_recall','balanced_accuracy')} for control in ('baseline','artifact_affine')}
                        atomic_json(folder/'metrics.json',report)
                        atomic_json(folder/'VERIFICATION.json',verify_saved(folder,data['records'],model))
                        atomic_json(folder/'COMPLETED.json',dict(run_identity=digest(run),fusion_sha256=sha(fusion_path),files={n:sha(folder/n) for n in ('metrics.json','predictions.csv','VERIFICATION.json')}))
                    summary['conditions'].setdefault(c,{})[key]=report
                    atomic_json(out/'PARTIAL_SUMMARY.json',summary)
                    status('RESULT',c+' '+key+' '+str({m:report['full_cohort'][m] for m in METHODS}))
            summary['elapsed_seconds']=time.monotonic()-start
            atomic_json(out/'SUMMARY.json',summary)
            paths=['RUN.json','FUSION.json','LOCKED_BEFORE_TEST.json','SUMMARY.json']
            paths += [p.relative_to(out).as_posix() for p in sorted((out/'evaluation').rglob('*')) if p.is_file() and p.name in ('metrics.json','predictions.csv','VERIFICATION.json','COMPLETED.json')]
            atomic_json(out/'COMPLETED.json',dict(identity=digest(run),files={p:sha(out/p) for p in paths}))
            status('VERIFY',str(verify(out)))
            atomic_json(out/'STATUS.json',dict(status='COMPLETE',targets=args.targets,conditions=['B','C'],elapsed_seconds=time.monotonic()-start))
            status('COMPLETE',str(out))
        except BaseException as exc:
            if owned:atomic_json(out/'STATUS.json',dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc)))
            raise

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','verify'))
    p.add_argument('--artifact-root',type=Path,default=DEFAULT_ARTIFACT);p.add_argument('--emotion-root',type=Path,default=DEFAULT_EMOTION)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--targets',nargs='+',choices=('FAV','PGF','AVDF1M'),default=['FAV','PGF','AVDF1M'])
    return p
if __name__=='__main__':
    try:execute(parser().parse_args())
    except KeyboardInterrupt:status('INTERRUPTED','Repeat identical command to resume');raise SystemExit(130)
    except Exception:traceback.print_exc();raise SystemExit(1)
