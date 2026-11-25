"""Verified score-only inputs for new-workstation artifact + existing AVDF1M emotion."""
from pathlib import Path
import csv
import numpy as np
from src.dataloader.records import SampleRecord
from scripts.trainable_or_3way.common import ROOT, read, sha, digest, require, stamp, Progress
from scripts.trainable_or_3way.data import record_identity, media_identity
from scripts.trainable_or_3way.storage import Store

DEFAULT_ARTIFACT=ROOT/'reports/workstation_baseline_full_audio_v2'
DEFAULT_EMOTION=ROOT/'reports/trainable_or_3way_fav_size_matched_v1/source_AVDF1M/seed_42'

def completed(folder,identity_key):
    folder=Path(folder);run=read(folder/'RUN.json');done=read(folder/'COMPLETED.json')
    require(done[identity_key]==digest(run),'Completed run identity differs: '+str(folder))
    for name,h in done['files'].items():require(sha(folder/name)==h,'Completed input changed: '+str(folder/name))
    return run,done

def verify_emotion_parent(folder):
    run,done=completed(folder,'identity'); selected=read(folder/'MODELS_SELECTED.json')
    require(run['config']['source']=='AVDF1M' and not run['target_training'],'Unexpected emotion training source')
    require(selected['identity']==digest(run),'Emotion selection identity differs')
    for mod in ('video','audio'):
        name=f'models/emotion/{mod}/best.pt'
        require(selected['models'][name]==done['files'][name]==sha(folder/name),'Emotion checkpoint differs')
    subset=run['source_subset'];p=Path(subset['path'])
    require(sha(p)==subset['sha256'],'Source subset selection changed')
    for name,h in subset['files'].items():require(sha(p.parent/name)==h,'Source subset file changed')
    # Verify the original trained emotion heads behind the existing reuse receipt.
    reuse=read(folder/'EMOTION_REUSE.json')['parent'];origin=Path(reuse['path'])
    require(sha(origin/'COMPLETED.json')==reuse['completed_sha256'],'Original emotion run receipt differs')
    origin_run,origin_done=completed(origin,'identity')
    require(digest(origin_run)==reuse['run_identity'],'Original emotion run identity differs')
    require(origin_run['source_subset']['sha256']==subset['sha256'],'Emotion training used another subset')
    for mod in ('video','audio'):
        name=f'models/emotion/{mod}/best.pt'
        require(sha(origin/name)==selected['models'][name],'Original/reused emotion head differs')
    return run,selected

def labels_tuple(r):
    return tuple(-1 if r[k] is None or r[k]=='' else int(r[k]) for k in ('clip_label','video_label','audio_label'))

def align_predictions(rows,predictions):
    by={p['sample_id']:p for p in predictions}
    require(len(by)==len(predictions)==len(rows) and set(by)=={r['sample_id'] for r in rows},'Prediction ID set mismatch')
    aligned=[]
    for r in rows:
        p=by[r['sample_id']]
        require(p['group_id']==r['group_id'] and labels_tuple(p)==labels_tuple(r),'Prediction groups/labels differ')
        aligned.append(p)
    return aligned

def load_cohort(artifact_root,emotion_root,parent_run,selected,dataset,split):
    folder=artifact_root/(('dev_' if split=='dev' else 'test_')+dataset)
    ar,done=completed(folder,'run')
    require(read(folder/'STATUS.json')['status']=='COMPLETE','Artifact inference incomplete: '+str(folder))
    require(ar['source']=='AVDF1M' and ar['dataset']==dataset and ar['split']==split and not ar['smoke_only'],'Wrong artifact cohort')
    require(ar['conditions']==['B','C'],'Both fixed B/C conditions are required')
    rows=read(folder/'COHORT.json')['rows']
    meta=[(r['sample_id'],r['group_id'],r['split'],r['media_path'],r['clip_label'],r['video_label'],r['audio_label'],r['stamp']) for r in rows]
    require(digest(meta)==ar['cohort_sha256'] and len(rows)==ar['n']==done['n'],'Artifact cohort identity mismatch')
    records=[SampleRecord(dataset,r['sample_id'],Path(r['media_path']),split,r['clip_label'],r['video_label'],r['audio_label'],r['group_id']) for r in rows]
    stamps={r['sample_id']:r['stamp'] for r in rows}
    spec=parent_run['cohorts'][dataset+'/'+split]
    require(record_identity(records)==spec['records_sha256'] and media_identity(records,stamps)==spec['media_sha256'],'Emotion/artifact input cohort differs')
    require(digest([(r['sample_id'],r['group_id'],*labels_tuple(r)) for r in rows])==spec['evaluation_sha256'],'Emotion labels differ')
    bar=Progress('JOIN '+dataset+'/'+split,len(rows));emotion=np.empty((len(rows),2));receipts={}
    for mod_index,mod in enumerate(('video','audio')):
        cache=emotion_root/'scores'/dataset/split/mod;receipt=read(cache/'COMPLETED.json');ident=receipt['identity']
        require(ident['run']==digest(parent_run) and ident['modality']==mod and ident['models']==selected['models'],'Emotion score provenance differs')
        require(ident['records_sha256']==spec['records_sha256'],'Emotion score records differ')
        store=Store(cache,ident,readonly=True)
        try:
            require(receipt['n']==len(rows) and store.ids()==set(stamps) and receipt['index_sha256']==store.index_hash(),'Incomplete or changed emotion cache')
            for i,r in enumerate(records):
                require(stamp(r)==stamps[r.sample_id],'Media changed since extraction: '+r.sample_id)
                value=store.get(r.sample_id,stamps[r.sample_id]);e=float(value['emotion'])
                require(np.isnan(e) or (np.isfinite(e) and 0<=e<=1),'Invalid emotion score')
                emotion[i,mod_index]=e
                if mod_index==1:bar.update(i+1)
        finally:store.close()
        receipts[mod]=dict(path=str(cache),receipt_sha256=sha(cache/'COMPLETED.json'),identity=ident)
    with (folder/'predictions.csv').open(encoding='utf-8',newline='') as f:pred=align_predictions(rows,list(csv.DictReader(f)))
    raw={}
    for c in ('B','C'):
        scores=np.array([[float(p[c+'_Av']),float(p['Aa'])] for p in pred])
        require(np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(),'Invalid artifact probabilities')
        require(np.allclose(1-(1-scores[:,0])*(1-scores[:,1]),[float(p[c+'_baseline_or']) for p in pred],rtol=0,atol=1e-14),'Artifact OR arithmetic differs')
        raw[c]=np.column_stack((scores,emotion))
    return dict(records=records,raw=raw,artifact_report=read(folder/'metrics.json'),
        proof=dict(artifact_folder=str(folder),artifact_run_sha256=sha(folder/'RUN.json'),artifact_completed_sha256=sha(folder/'COMPLETED.json'),
            predictions_sha256=sha(folder/'predictions.csv'),emotion_scores=receipts,cohort_sha256=ar['cohort_sha256'],n=len(rows)),compatibility=ar['compatibility'])
