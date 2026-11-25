"""Validate FAV models and completed scores without fitting on target data."""
import csv,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import ROOT,read,sha,digest,require,status,Progress
from scripts.hierarchical_score_fusion_GRU_Tempooral_Attention_cross_dataset import inputs as legacy
from scripts.trainable_or.fusion import predict as fusion_predict

SOURCE=ROOT/'reports/trainable_or_fav_v1/seed_42'
SCORES=ROOT/'reports/hierarchical_score_fusion_gru_temporal_attention_cross_dataset_v1/source_FAV/seed_42'
OUTPUT=ROOT/'reports/trainable_or_cross_dataset_v1/source_FAV/seed_42'
ARTIFACT=legacy.ARTIFACT_RUN
MODALITIES=legacy.MODALITIES

def state_fingerprint(path,modality):
    s=torch.load(path,map_location='cpu',weights_only=True)
    require(s['format']=='emotion_gru_temporal_attention_v1' and s['modality']==modality,'Wrong GRU checkpoint')
    h=hashlib.sha256()
    for name,t in sorted(s['model'].items()):
        require(torch.isfinite(t).all().item(),'Nonfinite model tensor')
        t=t.detach().cpu().contiguous()
        h.update(digest([name,str(t.dtype),list(t.shape)]).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def require_same_heads(current,previous):
    proof={}
    for mod in MODALITIES:
        new=state_fingerprint(Path(current)/'heads'/mod/'best.pt',mod)
        old=state_fingerprint(Path(previous)/'heads'/mod/'best.pt',mod)
        require(new==old,'Cannot reuse scores: '+mod+' GRU tensors differ; choose --mode infer in a new output')
        proof[mod]=new
    return proof

def load_source(folder):
    folder=Path(folder).resolve()
    done=legacy.verify_files(folder)
    run=read(folder/'RUN.json');lock=read(folder/'LOCKED_BEFORE_TEST.json')
    require(run['format']=='full_fav_trainable_final_v1','Need full trainable_or FAV source')
    require(done['identity']==lock['identity']==digest(run),'Trainable source identity mismatch')
    require(done['source']==done['target']=='FAV' and done['full_cohort'],'Need full FAV source')
    require(run['config']['full_train'] and not run['full_encoder_retraining'],'Unexpected source training scope')
    for name,h in run['implementation'].items():
        require(sha(ROOT/name)==h,'Source implementation changed: '+name)
    parent=legacy.source_model(Path(run['source']))
    require(run['source_completion_sha256']==parent['receipt']['completion_sha256'],'Parent FAV source changed')
    require(run['config']['input_recipe']==parent['run']['config']['video_input']
            and run['config']['audio_input']==parent['run']['config']['audio_input'],'Source encoder input changed')
    fused=read(folder/'FUSION.json')
    require(sha(folder/'FUSION.json')==lock['fusion_sha256'],'Frozen fusion changed')
    require(fused['head_selection_sha256']==sha(folder/'HEADS_SELECTED.json'),'Fusion head selection changed')
    for key,emotion in [('trainable_final',True),('artifact_affine',False)]:
        model=fused[key]
        require(model['split']=='dev' and model['n']==3147 and model['threshold']==.5 and model['success']
                and model['emotion_used']==emotion,'Fusion was not fitted on the full source dev')
        fusion_predict(model,np.array([.5]),np.array([.5]) if emotion else None)
    heads={}
    for mod in MODALITIES:
        head=read(folder/'heads'/mod/'COMPLETED.json')
        require(sha(folder/'heads'/mod/'best.pt')==head['checkpoint_sha256']==lock['heads'][mod],'Source head changed')
        require(head['train_n']==14724 and head['epochs_completed']==run['config']['epochs'],'Incomplete full head training')
        heads[mod]=dict(identity=head['identity'],checkpoint_sha256=head['checkpoint_sha256'])
    # Adapter for the existing frozen inference/asset pipeline. The heads point
    # to the NEW full trainable_or run; preprocessing and artifact provenance are unchanged.
    bridge=dict(folder=str(folder),run=parent['run'],assets=parent['assets'],
                lock=dict(heads=heads,artifact_sources=parent['lock']['artifact_sources']))
    receipt=dict(run_sha256=sha(folder/'RUN.json'),completion_sha256=sha(folder/'COMPLETED.json'),
                 lock_sha256=sha(folder/'LOCKED_BEFORE_TEST.json'),fusion_sha256=sha(folder/'FUSION.json'),
                 heads=heads,parent_receipt=parent['receipt'])
    bridge.update(receipt=receipt,fusion=fused,parent=parent)
    return bridge

def reuse_source(folder,source,producer,targets):
    folder=Path(folder).resolve()
    done=legacy.verify_files(folder);run=read(folder/'RUN.json');lock=read(folder/'LOCKED_BEFORE_TARGET.json')
    require(run['format']=='frozen_fav_temporal_cross_v1','Unsupported historical score run')
    require(done['identity']==lock['identity']==digest(run),'Historical score identity mismatch')
    require(done['source']=='FAV' and done['zero_shot'] and not done['smoke_only'],'Need completed zero-shot scores')
    require(set(targets)<=set(done['targets']),'Historical target not completed')
    require(run['source_receipt']==source['parent']['receipt']==lock['source_receipt'],'Historical head provenance differs')
    require(run['source_implementation']==source['parent']['run']['implementation'],'Historical encoder implementation differs')
    require(run['implementation']==legacy.implementation(),'Historical inference implementation changed')
    require(run['artifact_receipt']==lock['artifact_receipt'],'Historical artifact receipt changed')
    for key in ('run_sha256','lock_sha256'):
        require(run['artifact_receipt'][key]==producer['receipt'][key],'Historical artifact source differs')
    proof=require_same_heads(source['folder'],source['parent']['folder'])
    cfg=run['config']
    require(cfg['prediction_batch_size']==1 and cfg['precision']=='FP32'
            and cfg['video_input']==source['run']['config']['video_input'],'Historical inference recipe differs')
    seals={}
    for target in targets:
        require(run['artifact_receipt']['targets'][target]==producer['receipt']['targets'][target],'Historical artifact target differs')
        result=legacy.verify_files(folder/'evaluation'/target)
        require(result['lock_sha256']==sha(folder/'LOCKED_BEFORE_TARGET.json'),'Historical target lock mismatch')
        caches={mod:read(folder/'cache'/target/mod/'COMPLETED.json') for mod in MODALITIES}
        require(caches['video']['identity']['cohort']==caches['audio']['identity']['cohort'],'Historical cohorts differ')
        for mod,cache in caches.items():
            require(cache['identity']['run']==done['identity'] and cache['identity']['target']==target
                    and cache['identity']['modality']==mod,'Historical score cache belongs to another run')
        seals[target]=dict(predictions_sha256=sha(folder/'evaluation'/target/'predictions.csv'),
                           cohort=caches['video']['identity']['cohort'],n=caches['video']['n'])
        require(caches['audio']['n']==seals[target]['n'],'Historical modality count differs')
    return dict(folder=str(folder),receipt=dict(run_sha256=sha(folder/'RUN.json'),
                completion_sha256=sha(folder/'COMPLETED.json'),tensor_fingerprints=proof,targets=seals,
                recorded_inference=cfg,recorded_packages=run['packages']))

def join_emotion_csv(path,data):
    """Join by exact IDs; never invent missing rows, labels, or artifact scores."""
    records=data['records'];lookup={r.sample_id:i for i,r in enumerate(records)};n=len(records)
    require(n>0 and len(lookup)==n,'Empty or duplicate expected IDs')
    values=np.full((n,2),np.nan);counts=np.zeros((n,2),np.int64);seen=set()
    bar=Progress('READ REUSABLE SCORES '+data['target'],n)
    with Path(path).open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            sid=row['sample_id'];require(sid in lookup and sid not in seen,'Unexpected/duplicate reused ID: '+sid)
            seen.add(sid);i=lookup[sid];r=records[i]
            require(row['group']==r.group_id and [int(row[k]) for k in ('clip_label','video_label','audio_label')]==legacy.labels_of(r),'Reused label/group mismatch')
            require(np.array_equal(np.array([float(row[k]) for k in ('Av','Aa','previous_gated_or')]),data['scores'][i]),'Reused artifact/gated score mismatch')
            for j,(score,count) in enumerate((('Ev','video_valid_tokens'),('Ea','audio_valid_tokens'))):
                value=float(row[score]) if row[score] else np.nan
                cnt=int(row[count])
                require(cnt>=0 and (np.isfinite(value)==(cnt>0)),'Missing-score/token mismatch')
                require(np.isnan(value) or 0<=value<=1,'Invalid reused probability')
                values[i,j]=value;counts[i,j]=cnt
            require(counts[i,0]==128,'Reused video is not the 128-frame center-crop recipe')
            bar.update(len(seen))
    require(len(seen)==n,'Historical CSV is incomplete')
    return dict(video=values[:,0],audio=values[:,1],video_counts=counts[:,0],audio_counts=counts[:,1])
