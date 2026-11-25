"""Metrics and independent on-disk verification; no model selection from test."""
from pathlib import Path
import csv
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,f1_score,average_precision_score,confusion_matrix,balanced_accuracy_score
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import read,require,sha,digest
METHODS=('artifact_or','emotion_or','hierarchical_or','trainable_final','artifact_affine')

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p,np.float64)
    require(y.shape==p.shape and set(y.tolist())=={0,1},'Need two binary evaluation classes')
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probabilities')
    pred=p>=.5
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    return dict(n=len(y),real=tn+fp,fake=fn+tp,roc_auc=float(roc_auc_score(y,p)),
                f1=float(f1_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
                real_f1=float(f1_score(y,pred,pos_label=0)),average_precision=float(average_precision_score(y,p)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),real_fpr=fp/(tn+fp),
                fake_recall=tp/(tp+fn),confusion_matrix=cm.tolist(),threshold=.5)

def verify_outputs(folder,check_completion=True):
    folder=Path(folder);summary=read(folder/'metrics.json')
    if check_completion:
        done=read(folder/'COMPLETED.json')
        for name,h in done['files'].items():
            require(sha(folder/name)==h,'Completed file changed: '+name)
    lock=read(folder/'LOCKED_BEFORE_TEST.json')
    for mod,h in lock['heads'].items():
        require(sha(folder/'heads'/mod/'best.pt')==h,'Locked head changed')
    require(sha(folder/'FUSION.json')==lock['fusion_sha256'],'Locked fusion changed')
    with (folder/'test/predictions.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==len({r['sample_id'] for r in rows})==3126,'Incomplete/duplicate test IDs')
    require(digest([r['sample_id'] for r in rows])==summary['cohorts']['test']['ordered_ids_sha256'],'Test cohort order differs')
    labels=[[int(r[k]) for k in ('clip_label','video_label','audio_label')] for r in rows]
    require(digest(labels)==summary['cohorts']['test']['labels_sha256'],'Test labels differ')
    y=np.array([int(r['clip_label']) for r in rows])
    scores={k:np.array([float(r[k]) for r in rows]) for k in (*METHODS,'Av','Aa','Ev','Ea')}
    recomputed={}
    for name in METHODS:
        r=metrics(y,scores[name]);recorded=summary['metrics'][name]
        require(r==recorded,'Metric mismatch: '+name)
        tn,fp,fn,tp=np.asarray(r['confusion_matrix']).ravel()
        require(abs(r['f1']-2*tp/(2*tp+fp+fn))<1e-12,'F1 arithmetic mismatch')
        recomputed[name]=r
    a=1-(1-scores['Av'])*(1-scores['Aa']);e=1-(1-scores['Ev'])*(1-scores['Ea'])
    for actual,expected in ((scores['artifact_or'],a),(scores['emotion_or'],e),
                            (scores['hierarchical_or'],1-(1-a)*(1-e))):
        require(np.array_equal(actual,expected),'OR arithmetic mismatch')
    fused=read(folder/'FUSION.json')
    for method,model in [('trainable_final',fused['trainable_final']),('artifact_affine',fused['artifact_affine'])]:
        def logit(p):
            p=np.clip(p,model['epsilon'],1-model['epsilon'])
            return np.log(p)-np.log1p(-p)
        w=model['weights']
        z=w[0]*logit(a)+w[-1]
        if model['emotion_used']:z+=w[1]*logit(e)
        require(np.allclose(expit(z),scores[method],rtol=0,atol=1e-12),'Learned fusion mismatch')
    comparisons={}
    for name in ('hierarchical_or','artifact_or','artifact_affine'):
        before=scores[name]>=.5;after=scores['trainable_final']>=.5
        comparisons[name]=dict(corrected=int(((before!=y)&(after==y)).sum()),
                               regressed=int(((before==y)&(after!=y)).sum()),
                               prediction_disagreements=int((before!=after).sum()))
    for mod,selection in summary['selection'].items():
        history=read(folder/'heads'/mod/'history.json')
        require(len(history)==summary['config']['epochs'],'Not all epochs completed')
        require(max(history,key=lambda h:h['dev']['roc_auc'])['epoch']==selection['best_epoch'],
                'Best epoch not selected by dev')
    return dict(status='PASS',test_n=len(rows),methods=len(METHODS),fusion_recomputed=True,
                metric_arithmetic_verified=True,paired= comparisons)
