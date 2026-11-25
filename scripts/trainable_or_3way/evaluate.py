"""Locked zero-shot score fusion, full coverage and independent saved-result checks."""
import csv, os
from pathlib import Path
import numpy as np
from .common import *
from .storage import Store
from scripts.trainable_or.fusion import predict as fuse
from scripts.trainable_or.evaluation import metrics

METHODS=('baseline','emotion_or','hierarchical_or','trainable_final','artifact_affine')

def safe_metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    mask=(y>=0)&np.isfinite(p)
    if set(y[mask].tolist())!={0,1}:
        return dict(n=int(mask.sum()),roc_auc=None,f1=None,real_fpr=None,
                    unknown_labels=int((y<0).sum()))
    return dict(metrics(y[mask],p[mask]),unknown_labels=int((y<0).sum()))

def combine(raw,fusion):
    a=score_or(raw[:,0],raw[:,1])
    valid=np.isfinite(raw[:,2:]).all(1)
    e=np.full(len(raw),np.nan); e[valid]=score_or(raw[valid,2],raw[valid,3])
    affine=fuse(fusion['artifact_affine'],a)
    trainable=affine.copy()
    trainable[valid]=fuse(fusion['trainable_final'],a[valid],e[valid])
    return dict(baseline=a,emotion_or=e,hierarchical_or=score_or(a,e),
                trainable_final=trainable,artifact_affine=affine),valid

def read_raw(records,stamps,stores):
    result=np.empty((len(records),4),np.float64)
    for i,r in enumerate(records):
        for j,mod in enumerate(('video','audio')):
            t=stores[mod].get(r.sample_id,stamps[r.sample_id])
            require(t is not None,'Missing raw score: '+r.sample_id)
            result[i,j]=float(t['artifact']); result[i,j+2]=float(t['emotion'])
    return result

def summarize(records,raw,fusion):
    scores,valid=combine(raw,fusion); y=labels(records)
    report=dict(full_cohort={k:safe_metrics(y,p) for k,p in scores.items() if k not in ('emotion_or','hierarchical_or')},
        common_emotion_cohort={k:safe_metrics(y[valid],p[valid]) for k,p in scores.items()},
        branch_metrics={key:safe_metrics(labels(records,mod),raw[:,j])
                        for j,(key,mod) in enumerate((('Av','video'),('Aa','audio'),('Ev','video'),('Ea','audio')))},
        coverage=dict(planned=len(records),evaluated=len(records),common_emotion=int(valid.sum()),
                      artifact_fallback=int((~valid).sum())),
        threshold=.5,all_fake_f1=float(2*(y==1).sum()/(len(y)+(y==1).sum())))
    return report,scores,valid

def export(folder,records,raw,fusion,source,identity):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    report,scores,valid=summarize(records,raw,fusion)
    report.update(source=source,target=records[0].dataset,split='test',identity=identity,
                  zero_shot=source!=records[0].dataset,target_training=False)
    final=folder/'predictions.csv'; temp=folder/'predictions.csv.partial'
    bar=Progress('EXPORT '+records[0].dataset,len(records),folder)
    with temp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['sample_id','group_id','clip_label','video_label','audio_label',
                         'Av','Aa','Ev','Ea',*METHODS,'emotion_complete'])
        for i,r in enumerate(records):
            writer.writerow([r.sample_id,r.group_id,*[-1 if v is None else v for v in
                (r.clip_label,r.video_label,r.audio_label)],*raw[i].tolist(),
                *[scores[k][i] for k in METHODS],int(valid[i])])
            bar.update(i+1)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,final)
    atomic_json(folder/'metrics.json',report)
    return report

def verify_evaluation(folder,records,fusion):
    folder=Path(folder)
    with (folder/'predictions.csv').open(encoding='utf-8',newline='') as stream:
        saved=list(csv.DictReader(stream))
    require(len(saved)==len(records),'Saved evaluation size differs')
    raw=np.array([[float(x[k]) for k in ('Av','Aa','Ev','Ea')] for x in saved])
    for row,r in zip(saved,records):
        require(row['sample_id']==r.sample_id and row['group_id']==r.group_id,'Saved cohort mismatch')
        require([int(row[k]) for k in ('clip_label','video_label','audio_label')]==
                [-1 if v is None else v for v in (r.clip_label,r.video_label,r.audio_label)],'Saved labels mismatch')
    report,scores,valid=summarize(records,raw,fusion)
    stored=read(folder/'metrics.json')
    for key,value in report.items(): require(stored[key]==value,'Saved metric mismatch: '+key)
    for key,p in scores.items():
        require(np.allclose(np.array([float(x[key]) for x in saved]),p,rtol=0,atol=1e-12,equal_nan=True),
                'Saved fusion arithmetic mismatch: '+key)
    return dict(status='PASS',n=len(records),fusion_recomputed=True,metrics_recomputed=True)

def verify_run(out):
    out=Path(out); run=read(out/'RUN.json'); done=read(out/'COMPLETED.json')
    require(done['identity']==digest(run),'Completion run mismatch')
    for path,h in done['files'].items(): require(sha(out/path)==h,'Output modified: '+path)
    lock=read(out/'LOCKED_BEFORE_TEST.json')
    require(lock['identity']==done['identity'] and lock['fusion_sha256']==sha(out/'FUSION.json'),'Fusion lock mismatch')
    for path,h in lock['models'].items(): require(sha(out/path)==h,'Locked model mismatch')
    from src.dataloader.records import SampleRecord
    for target in run['targets']:
        folder=out/'evaluation'/target
        with (folder/'predictions.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        require(len({r['sample_id'] for r in rows})==len(rows),'Duplicate evaluated ID')
        # The separately locked descriptor binds IDs/groups/labels in order.
        columns=[(r['sample_id'],r['group_id'],*[int(r[k]) for k in
                  ('clip_label','video_label','audio_label')]) for r in rows]
        require(digest(columns)==run['cohorts'][target+'/test']['evaluation_sha256'],'Evaluation cohort changed')
        records=[SampleRecord(target,r['sample_id'],Path('unused'),'test',
                    *[None if int(r[k])<0 else int(r[k]) for k in
                      ('clip_label','video_label','audio_label')],r['group_id']) for r in rows]
        verify_evaluation(folder,records,read(out/'FUSION.json'))
    return dict(status='PASS',source=run['config']['source'],targets=run['targets'])
