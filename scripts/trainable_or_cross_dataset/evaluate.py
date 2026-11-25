"""Frozen fusion and resumable export on a strictly joined target cohort."""
import csv,os,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import require,score_or,atomic_json,sha,read,Progress,status
from scripts.hierarchical_score_fusion_GRU_Tempooral_Attention_cross_dataset.inputs import labels_of,verify_files
from scripts.trainable_or.fusion import predict as predict_fusion
from .storage import ResultStore

METHODS=('artifact_or','emotion_or','hierarchical_or','trainable_final','artifact_affine','previous_gated_or')

def metric(y,p):
    y=np.asarray(y);p=np.asarray(p,np.float64)
    require(y.shape==p.shape and np.isin(y,[0,1]).all() and np.isfinite(p).all()
            and ((p>=0)&(p<=1)).all(),'Invalid metric input')
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=.5,labels=[0,1]).ravel())
    real,fake=tn+fp,fn+tp;two=real>0 and fake>0
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    rf1=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return dict(n=len(y),real=real,fake=fake,roc_auc=float(roc_auc_score(y,p)) if two else None,
                average_precision=float(average_precision_score(y,p)) if two else None,
                f1=f1 if len(y) else None,real_f1=rf1 if len(y) else None,
                macro_f1=(f1+rf1)/2 if len(y) else None,
                balanced_accuracy=.5*(tn/real+tp/fake) if two else None,
                real_fpr=fp/real if real else None,fake_recall=tp/fake if fake else None,
                confusion_matrix=[[tn,fp],[fn,tp]],threshold=.5)

def fused_scores(artifact_scores,emotion,models):
    av,aa,gated=np.asarray(artifact_scores,np.float64).T
    ev,ea=np.asarray(emotion['video'],np.float64),np.asarray(emotion['audio'],np.float64)
    require(ev.shape==ea.shape==av.shape,'Score shapes differ')
    require(np.isfinite(artifact_scores).all() and ((artifact_scores>=0)&(artifact_scores<=1)).all(),'Invalid artifact scores')
    for p in (ev,ea):
        require((np.isnan(p)|((p>=0)&(p<=1)&np.isfinite(p))).all(),'Invalid emotion score')
    complete=np.isfinite(ev)&np.isfinite(ea)
    a=score_or(av,aa);e=score_or(np.nan_to_num(ev,nan=0.),np.nan_to_num(ea,nan=0.))
    affine=predict_fusion(models['artifact_affine'],a)
    learned=affine.copy()
    # The learned two-input layer saw both emotions during source calibration.
    # Missing either emotion uses the SOURCE-fitted artifact-only control.
    if complete.any():
        learned[complete]=predict_fusion(models['trainable_final'],a[complete],e[complete])
    return dict(artifact_or=a,emotion_or=e,hierarchical_or=score_or(a,e),
                trainable_final=learned,artifact_affine=affine,previous_gated_or=gated),complete

def summarize(labels,artifact_scores,emotion,models):
    labels=np.asarray(labels,np.int64)
    require(labels.shape==(len(artifact_scores),3) and np.isin(labels,[-1,0,1]).all(),'Invalid labels')
    scores,complete=fused_scores(artifact_scores,emotion,models);known=labels[:,0]>=0
    def cohort(mask):
        result={k:metric(labels[mask,0],p[mask]) for k,p in scores.items()}
        result['delta_trainable_pp']={}
        for ref in ('artifact_or','hierarchical_or','artifact_affine','previous_gated_or'):
            result['delta_trainable_pp'][ref]={key:100*(result['trainable_final'][key]-result[ref][key])
                if result['trainable_final'][key] is not None and result[ref][key] is not None else None
                for key in ('roc_auc','f1','macro_f1','balanced_accuracy','real_fpr')}
        return result
    branches={}
    for name,j,p in [('x3d',1,artifact_scores[:,0]),('aasist',2,artifact_scores[:,1]),
                     ('video_emotion',1,emotion['video']),('audio_emotion',2,emotion['audio'])]:
        valid=np.isfinite(p);mask=(labels[:,j]>=0)&valid
        branches[name]=dict(metric(labels[mask,j],p[mask]),unknown_labels=int((labels[:,j]<0).sum()),missing_score=int((~valid).sum()))
    return dict(primary_complete_emotion=cohort(known&complete),secondary_full_cohort=cohort(known),
                branches=branches,coverage=dict(planned=len(labels),evaluated=int(known.sum()),complete_emotion=int(complete.sum()),
                video_emotion_missing=int((~np.isfinite(emotion['video'])).sum()),
                audio_emotion_missing=int((~np.isfinite(emotion['audio'])).sum()),
                artifact_affine_fallback=int((~complete&known).sum()))),scores,complete

def make_rows(data,emotion,scores,complete):
    rows=[]
    for i,r in enumerate(data['records']):
        labels=labels_of(r)
        rows.append(dict(sample_id=r.sample_id,group=r.group_id,
             clip_label=labels[0],video_label=labels[1],audio_label=labels[2],
             Av=float(data['scores'][i,0]),Aa=float(data['scores'][i,1]),
             Ev=float(emotion['video'][i]) if np.isfinite(emotion['video'][i]) else None,
             Ea=float(emotion['audio'][i]) if np.isfinite(emotion['audio'][i]) else None,
             **{k:float(scores[k][i]) for k in METHODS},
             included_primary=bool(complete[i] and labels[0]>=0),
             used_artifact_fallback=bool(not complete[i]),
             video_valid_tokens=int(emotion['video_counts'][i]),audio_valid_tokens=int(emotion['audio_counts'][i])))
    return rows

def evaluate(data,emotion,models,folder,lock,identity,batch_size=512,interrupt_after_batches=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETED.json').exists():
        done=verify_files(folder)
        require(done['identity']==identity and done['lock_sha256']==sha(lock),'Completed evaluation identity differs')
        status('RESUME',data['target']+': completed evaluation verified')
        return read(folder/'metrics.json')
    require(batch_size>0,'Invalid fusion batch size')
    labels=np.array([labels_of(r) for r in data['records']],np.int64)
    report,scores,complete=summarize(labels,data['scores'],emotion,models)
    rows=make_rows(data,emotion,scores,complete);planned={r['sample_id'] for r in rows}
    require(len(planned)==len(rows)>0,'Duplicate/empty evaluation IDs')
    started=time.perf_counter();new=0
    store=ResultStore(folder/'predictions.sqlite',identity)
    try:
        saved=store.rows();require(set(saved)<=planned,'Unexpected committed ID')
        pending=[]
        for row in rows:
            if row['sample_id'] in saved:
                require(saved[row['sample_id']]==row,'Committed prediction changed')
            else:pending.append(row)
        bar=Progress('FUSION '+data['target']+' new',len(pending))
        for batch_no,start in enumerate(range(0,len(pending),batch_size),1):
            chunk=pending[start:start+batch_size];store.put_many(chunk);new+=len(chunk)
            bar.update(new,f'resumed={len(saved)} completed={len(saved)+new}/{len(rows)}')
            atomic_json(folder/'progress.json',dict(stage='fusion',planned=len(rows),completed=len(saved)+new,
                        resumed=len(saved),new=new,elapsed_seconds=time.perf_counter()-started))
            if interrupt_after_batches==batch_no:raise InterruptedError('Synthetic interruption after committed batch')
        checked=store.rows()
        require(len(checked)==len(rows) and all(checked[r['sample_id']]==r for r in rows),'Committed score verification failed')
        index_sha=store.checksum()
    finally:store.close()
    temp=folder/'predictions.csv.partial'
    with temp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        bar=Progress('EXPORT '+data['target'],len(rows))
        for i,row in enumerate(rows,1):writer.writerow(row);bar.update(i)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,folder/'predictions.csv')
    report.update(source='FAV',target=data['target'],split='test',zero_shot=True,threshold=.5,
                  lock_sha256=sha(lock),target_training=False,
                  selection='frozen FAV heads and full FAV-dev fusion; no target fitting or threshold tuning',
                  target_status='previously evaluated; not an untouched holdout',
                  fusion='A=OR(Av,Aa); E=OR(Ev,Ea); F=sigmoid(wA*logit(A)+wE*logit(E)+b)',
                  missing_policy='primary both emotions; full cohort falls back to source-fitted artifact-affine when either is missing',
                  previous_gated_note='historical face-crop input; descriptive comparison, not controlled fusion-only ablation')
    atomic_json(folder/'metrics.json',report)
    atomic_json(folder/'progress.json',dict(stage='complete',completed=len(rows),resumed=len(rows)-new,new=new))
    atomic_json(folder/'COMPLETED.json',dict(identity=identity,lock_sha256=sha(lock),n=len(rows),
                payload_index_sha256=index_sha,files={n:sha(folder/n) for n in ('metrics.json','predictions.csv')}))
    status('EVALUATE',data['target']+': COMPLETE '+str(len(rows)))
    return report
