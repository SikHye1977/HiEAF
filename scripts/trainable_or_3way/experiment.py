"""Train AVDF1M or PGF once, evaluate source test and both external test sets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='scripts.trainable_or_3way'
import argparse, gc, time, traceback, shutil
import numpy as np
import torch
from .common import *
from .data import (source_metadata,targets,record_identity,media_identity,
                   verify_media,REGISTRY,ELIGIBILITY,counts)
from .models import (local_assets,artifact,EmotionFlowHead,Tokens,validate_tokens,
                     artifact_tensor,emotion_score,load_best,TOKEN_RECIPE,VIDEO,AUDIO)
from .storage import Store,complete_store
from .train import Task,fit,DETERMINISM_POLICY
from .numerics import NUMERICS_POLICY
from .sampling import subset_path,read_bundle
from .continuation import plan_continuation,install_continuation
from .resume_compat import resume_description, RECEIPT as IMPORT_PATCH_RECEIPT
from .evaluate import read_raw,export,verify_evaluation,verify_run
from src.dataloader.metadata import load_records
from scripts.trainable_or.fusion import fit as fit_fusion

DEFAULT_CACHE=Path('D:/datasets/experiment_cache/trainable_or_3way_v1')

def descriptor(rows,stamps):
    return dict(n=len(rows),records_sha256=record_identity(rows),
                media_sha256=media_identity(rows,stamps),counts=counts(rows),
                evaluation_sha256=digest([(r.sample_id,r.group_id,*[-1 if v is None else v
                    for v in (r.clip_label,r.video_label,r.audio_label)]) for r in rows]))

def prepare(args):
    require(args.source in SOURCES,'FAV training is excluded')
    for k in ('artifact_epochs','head_epochs','head_batch_size','accumulation','checkpoint_steps','frame_batch','threads','decode_samples'):
        require(getattr(args,k)>0,'Positive value required: '+k)
    require(args.artifact_lr>0 and args.head_lr>0,'Learning rate must be positive')
    require(args.device=='cpu' or (args.device.startswith('cuda') and torch.cuda.is_available()),'Requested device unavailable')
    torch.set_num_threads(args.threads)
    status('MEDIA POLICY','Current Windows data; source-workstation hashes are not required')
    selected_path=subset_path(args)
    subset=read_bundle(selected_path) if selected_path else None
    require(not (subset and args.resume_from),
            'Subset training starts from foundation initialization; --resume-from is not allowed')
    if subset:
        status('FIXED SUBSET',str(subset['receipt']['sizes'])+' sampling_seed='+str(subset['receipt']['seed'])+'; test unchanged')
    asset=local_assets()
    source,proofs=source_metadata(args.source,subset=subset)
    cohorts={args.source+'/'+s:rows for s,rows in source.items()}
    for target in targets(args.source)[1:]:
        status('METADATA',target+' test (evaluation only)')
        cohorts[target+'/test']=load_records(target,'test',task='paired')
        check_classes(cohorts[target+'/test'])
    stamps,media_validation=verify_media(cohorts,decode_samples=args.decode_samples)
    for name,rows in cohorts.items(): status('COHORT',name+' '+str(counts(rows))+' n='+str(len(rows)))
    cache_parent=args.cache_root.resolve()
    while not cache_parent.exists(): cache_parent=cache_parent.parent
    free=shutil.disk_usage(cache_parent).free
    require(free>2*1024**3,'Less than 2 GiB free on the token cache drive')
    video_bytes=sum(len(rows) for rows in cohorts.values())*128*(1280+8)*4
    status('STORAGE',f'cache_free={free/1024**3:.1f} GiB; video_token_payload_if_all_new~{video_bytes/1024**3:.1f} GiB plus audio/index; existing caches may be reused')
    cfg={k:getattr(args,k) for k in ('source','seed','artifact_epochs','head_epochs','head_batch_size',
        'artifact_lr','head_lr','accumulation','checkpoint_steps','frame_batch','threads','device','amp')}
    cfg.update(threshold=.5,calibration='full source dev reused after source-dev head selection',
        artifact_initialization=dict(video='public Kinetics X3D_m; fresh binary head',audio='random official AASIST'),
        source_class_weighting='known-valid source train negative/positive ratio; natural sampling',
        video_input=VIDEO,audio_input=AUDIO,token_recipe=TOKEN_RECIPE,full_data=subset is None,
        determinism=DETERMINISM_POLICY,numerics=NUMERICS_POLICY)
    run=dict(format='trainable_or_3way_v1',config=cfg,assets=asset,
        targets=targets(args.source),cohorts={k:descriptor(v,stamps) for k,v in cohorts.items()},
        registry_sha256=sha(REGISTRY),eligibility_sha256=sha(ELIGIBILITY),media_validation=media_validation,
        pgf_overlay_sha256=digest(proofs),implementation=file_set(),
        cache_root=str(args.cache_root.resolve()),torch=str(torch.__version__),numpy=np.__version__,
        fav_source_excluded=True,target_training=False,independent_calibration=False,
        previous_targets_observed=True)
    if subset:
        run['source_subset']=subset['receipt']
        cfg['calibration']='fixed source dev subset reused after source-dev head selection'
    if args.resume_from:
        run['continuation']=plan_continuation(args.resume_from,run)
        status('CONTINUATION READY',str(run['continuation']))
    return run,cohorts,stamps,proofs

def token_cache(args,run,rows,mod,stamps,progress_root):
    # No source/seed/label field: exactly matching frozen inputs are reusable.
    implementation=run['implementation']
    dependencies={k:v for k,v in implementation.items()
        if k.startswith(('src/','scripts/hierarchical_score_fusion_GRU_','scripts/artifact_emotion_gated_attetnion_optimized/'))
        or k=='scripts/trainable_or_3way/models.py'}
    ident=dict(kind='frozen_temporal_tokens_v1',dataset=rows[0].dataset,split=rows[0].split,
               modality=mod,media_sha256=media_identity(rows,stamps),recipe=TOKEN_RECIPE,
               assets={k:v for k,v in run['assets'].items() if not k.startswith('x3d')},
               implementation=dependencies,frame_batch=args.frame_batch,device=args.device)
    folder=args.cache_root.resolve()/rows[0].dataset/rows[0].split/mod/digest(ident)[:20]
    producer=Tokens(run['assets'],mod,args.device,args.frame_batch)
    complete_store(folder,ident,rows,stamps,producer,lambda t:validate_tokens(t,mod),
                   f'EXTRACT {rows[0].dataset} {rows[0].split} {mod}',progress_root)
    del producer
    gc.collect()
    if args.device.startswith('cuda'): torch.cuda.empty_cache()
    return folder,ident

def score_cache(out,rows,stamps,mod,artifact_model,head,cache,model_hashes,device,identity):
    token_folder,token_identity=cache
    token_store=Store(token_folder,token_identity,readonly=True)
    folder=out/'scores'/rows[0].dataset/rows[0].split/mod
    ident=dict(run=identity,records_sha256=record_identity(rows),modality=mod,
               models=model_hashes,tokens=sha(token_folder/'COMPLETED.json'))
    def produce(r):
        with torch.inference_mode():
            av=float(artifact_model(artifact_tensor(r,mod,device)).fake_probability[0].cpu())
            t=token_store.get(r.sample_id,stamps[r.sample_id])
            ev=emotion_score(head,t,device)
        return dict(artifact=np.asarray(av),emotion=np.asarray(ev))
    def validate(value):
        a=float(value['artifact']);e=float(value['emotion'])
        require(np.isfinite(a) and 0<=a<=1,'Invalid artifact score')
        require(np.isnan(e) or (np.isfinite(e) and 0<=e<=1),'Invalid emotion score')
    try:
        complete_store(folder,ident,rows,stamps,produce,validate,
            f'SCORE {rows[0].dataset} {rows[0].split} {mod}',out)
    finally: token_store.close()
    return folder,ident

def raw_from_specs(rows,stamps,specs):
    stores={m:Store(*specs[m],readonly=True) for m in ('video','audio')}
    try: return read_raw(rows,stamps,stores)
    finally:
        for s in stores.values(): s.close()

def execute(args,run,cohorts,stamps,proofs):
    out=args.output.resolve();identity=digest(run); cfg=run['config'];source=args.source
    source_rows={s:cohorts[source+'/'+s] for s in ('train','dev','test')}
    def stage(name,detail=None):
        atomic_json(out/'progress.json',dict(stage=name,source=source,detail_progress=str(detail) if detail else None))
        status('STAGE',name)
    immutable(out/'RUN.json',run)
    if (out/'COMPLETED.json').exists():
        verify_run(out); status('COMPLETE','Already complete; verified, no retraining');return
    install_continuation(out,run)
    atomic_json(out/'STATUS.json',dict(status='RUNNING',source=source))
    if proofs: immutable(out/'PGF_LABEL_EVIDENCE.json',proofs)
    caches={};identities={};model_hashes={}
    for mod in ('video','audio'):
        identity_art=digest(dict(run=identity,kind='artifact',modality=mod))
        folder=out/'models/artifact'/mod
        stage('train_artifact_'+mod,folder/'progress.json')
        seed_all(args.seed+(mod=='audio'))
        model=artifact(mod)
        train=Task(source_rows['train'],mod,'artifact',args.device,stamps)
        dev=Task(source_rows['dev'],mod,'artifact',args.device,stamps)
        fit(model,train,dev,folder,cfg,identity_art)
        identities['artifact/'+mod]=identity_art
        del model,train,dev; gc.collect()
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
        for split in ('train','dev'):
            caches[source+'/'+split+'/'+mod]=token_cache(args,run,source_rows[split],mod,stamps,out)
        stores={s:Store(*caches[source+'/'+s+'/'+mod],readonly=True) for s in ('train','dev')}
        try:
            identity_head=digest(dict(run=identity,kind='emotion',modality=mod,
                 caches={s:sha(caches[source+'/'+s+'/'+mod][0]/'COMPLETED.json') for s in stores}))
            stage('train_emotion_'+mod,out/'models/emotion'/mod/'progress.json')
            seed_all(args.seed+(mod=='audio')+100)
            model=EmotionFlowHead(mod)
            train=Task(source_rows['train'],mod,'emotion',args.device,stamps,stores['train'])
            dev=Task(source_rows['dev'],mod,'emotion',args.device,stamps,stores['dev'])
            fit(model,train,dev,out/'models/emotion'/mod,cfg,identity_head)
            identities['emotion/'+mod]=identity_head
            del model,train,dev
        finally:
            for store in stores.values():store.close()
        gc.collect()
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    for key in identities:
        path='models/'+key+'/best.pt';model_hashes[path]=sha(out/path)
    selection=dict(identity=identity,model_identities=identities,models=model_hashes)
    immutable(out/'MODELS_SELECTED.json',selection)
    def scores_for(rows):
        specs={}
        for mod in ('video','audio'):
            cache_key=rows[0].dataset+'/'+rows[0].split+'/'+mod
            if cache_key not in caches:
                caches[cache_key]=token_cache(args,run,rows,mod,stamps,out)
            art=load_best(out/'models/artifact'/mod/'best.pt','artifact',mod,
                          identities['artifact/'+mod],args.device)
            head=load_best(out/'models/emotion'/mod/'best.pt','emotion',mod,
                          identities['emotion/'+mod],args.device)
            specs[mod]=score_cache(out,rows,stamps,mod,art,head,caches[cache_key],model_hashes,args.device,identity)
            del art,head;gc.collect()
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
        return raw_from_specs(rows,stamps,specs)
    # Fit only source dev. External labels are never arguments of fitting.
    stage('source_dev_scores')
    dev_raw=scores_for(source_rows['dev']); y=labels(source_rows['dev'])
    valid=np.isfinite(dev_raw[:,2:]).all(1)
    a=score_or(dev_raw[:,0],dev_raw[:,1]); e=score_or(dev_raw[valid,2],dev_raw[valid,3])
    require(set(y[valid].tolist())=={0,1},'Need both source dev clip classes with valid emotion')
    stage('fit_source_dev_fusion')
    fusion_path=out/'FUSION.json'
    if fusion_path.exists():
        fusion=read(fusion_path)
        require(fusion['selection_sha256']==sha(out/'MODELS_SELECTED.json'),'Fusion/model mismatch')
    else:
        fusion=dict(selection_sha256=sha(out/'MODELS_SELECTED.json'),source=source,
            source_dev_n=len(y),complete_emotion_dev_n=int(valid.sum()),
            trainable_final=fit_fusion(a[valid],e,y[valid],split='dev'),
            artifact_affine=fit_fusion(a,None,y,split='dev'))
        atomic_json(fusion_path,fusion)
    immutable(out/'LOCKED_BEFORE_TEST.json',dict(identity=identity,models=model_hashes,
        fusion_sha256=sha(fusion_path),threshold=.5,target_training=False,
        selection='source dev only; dev reused for fusion'))
    results={}
    for target in run['targets']:
        # No optimizer is constructed after the source selection/fusion lock.
        rows=cohorts[target+'/test'];folder=out/'evaluation'/target
        receipt=folder/'COMPLETED.json'
        if receipt.exists():
            old=read(receipt)
            require(old['identity']==identity,'Evaluation run mismatch')
            for name,h in old['files'].items():require(sha(folder/name)==h,'Evaluation changed')
            verify_evaluation(folder,rows,fusion);results[target]=read(folder/'metrics.json')
            status('RESUME',target+' evaluation already complete');continue
        raw=scores_for(rows)
        results[target]=export(folder,rows,raw,fusion,source,identity)
        check=verify_evaluation(folder,rows,fusion);atomic_json(folder/'VERIFICATION.json',check)
        atomic_json(receipt,dict(identity=identity,files={name:sha(folder/name) for name in
            ('metrics.json','predictions.csv','VERIFICATION.json')}))
        status('RESULT',source+' -> '+target+' '+str(results[target]['full_cohort']))
    atomic_json(out/'SUMMARY.json',results)
    scope=('Fixed source train/dev subset: '+str(run['source_subset']['sizes'])+
           '; sampling seed '+str(run['source_subset']['seed'])+'; full test cohorts retained.'
           if run.get('source_subset') else 'Full eligible source train/dev; full test cohorts retained.')
    lines=['# Trainable Final Fusion: '+source,'',scope,'',
        'Source train/dev only. Threshold 0.5. FAV source experiments excluded.','',
        '| Source -> Target | Method | ROC-AUC (%) | Fake F1 (%) | Real FPR (%) |',
        '|---|---|---:|---:|---:|']
    for target,r in results.items():
        for method in ('baseline','trainable_final','artifact_affine'):
            m=r['full_cohort'][method]
            lines.append('| '+source+' -> '+target+' | '+method+' | '+
                ' | '.join('NA' if m[k] is None else f'{100*m[k]:.4f}' for k in ('roc_auc','f1','real_fpr'))+' |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=['RUN.json','MODELS_SELECTED.json','FUSION.json','LOCKED_BEFORE_TEST.json','SUMMARY.json','RESULTS.md']
    files+=list(model_hashes)
    for key in identities:
        files+=['models/'+key+'/'+n for n in ('history.json','COMPLETED.json')]
    for target in run['targets']:
        files+=['evaluation/'+target+'/'+n for n in ('metrics.json','predictions.csv','VERIFICATION.json','COMPLETED.json')]
    if proofs:files.append('PGF_LABEL_EVIDENCE.json')
    if run.get('continuation'):files+=['CONTINUATION.json','CONTINUATION_CHECKPOINT.pt']
    if (out/IMPORT_PATCH_RECEIPT).exists():files.append(IMPORT_PATCH_RECEIPT)
    atomic_json(out/'COMPLETED.json',dict(identity=identity,files={n:sha(out/n) for n in files}))
    verify_run(out)
    atomic_json(out/'progress.json',dict(stage='complete',source=source,targets=run['targets']))
    atomic_json(out/'STATUS.json',dict(status='COMPLETE',source=source,targets=run['targets']))
    status('PIPELINE COMPLETE',str(out))

def run(args):
    if args.command=='verify':
        require(args.output is not None,'verify requires --output')
        result=verify_run(args.output); status('VERIFY',str(result)); return result
    selected_path=subset_path(args)
    require(not (selected_path and args.resume_from),
            'Subset training starts from foundation initialization; --resume-from is not allowed')
    if args.command=='inspect-labels':
        records,proofs=source_metadata(args.source,subset=read_bundle(selected_path) if selected_path else None)
        result={k:dict(n=len(v),labels=counts(v)) for k,v in records.items()}
        status('LABELS',str(result));return result
    if args.resume_from:
        require(args.output is not None,'--resume-from requires an explicit new --output')
        source_path=args.resume_from.resolve();destination=args.output.resolve()
        require(not destination.is_relative_to(source_path) and not source_path.is_relative_to(destination),
                'Continuation output must be separate from the source directory')
    if args.output is None:
        family='trainable_or_3way_fav_size_v1' if selected_path else 'trainable_or_3way_v1'
        args.output=ROOT/'reports'/family/('source_'+args.source)/('seed_'+str(args.seed))
    # No historical report directory can be used as an output.
    require(args.output.resolve()!=ROOT and args.output.resolve()!=args.cache_root.resolve(),'Unsafe output directory')
    if (args.output/'RUN.json').exists():
        require(args.resume or args.command=='preflight','Existing run: repeat with --resume')
        previous=read(args.output/'RUN.json')
        require(previous.get('format')=='trainable_or_3way_v1','Output belongs to another experiment')
        require(bool(previous.get('source_subset'))==bool(selected_path),
                'Full-data/subset output mismatch; use a separate output')
        if selected_path:
            require(previous['source_subset']['sha256']==sha(selected_path),
                    'Fixed subset changed; use a separate output')
    elif args.output.exists():
        require(not any(p.name!='.writer.lock' for p in args.output.iterdir()),'Output is not empty')
    description,cohorts,stamps,proofs=prepare(args)
    import_patch=None
    if (args.output/'RUN.json').exists():
        description,import_patch=resume_description(read(args.output/'RUN.json'),description)
    if (args.output/IMPORT_PATCH_RECEIPT).exists():
        require(import_patch is not None and read(args.output/IMPORT_PATCH_RECEIPT)==import_patch,
                'AASIST import compatibility receipt changed')
    if import_patch:
        status('RESUME PATCH','Verified AASIST import-only fix; completed models and token cache identities retained')
    if args.command=='preflight':
        status('PREFLIGHT PASS',str(description['cohorts'])); return description
    args.output.mkdir(parents=True,exist_ok=True)
    with WriterLock(args.output):
        if import_patch: immutable(args.output/IMPORT_PATCH_RECEIPT,import_patch)
        try: execute(args,description,cohorts,stamps,proofs)
        except BaseException as error:
            atomic_json(args.output/'STATUS.json',dict(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                       error=str(error),resume='Repeat identical run command with --resume'))
            raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preflight','run','verify','inspect-labels'))
    p.add_argument('--source',choices=SOURCES,default='AVDF1M')
    sampling=p.add_mutually_exclusive_group()
    sampling.add_argument('--source-subset',type=Path,
        help='Fixed selection.json; AVDF1M defaults to the shipped FAV-size seed42 subset')
    sampling.add_argument('--full-source-data',action='store_true',
        help='Explicitly use all source train/dev data (PGF already uses full data by default)')
    p.add_argument('--output',type=Path);p.add_argument('--cache-root',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--artifact-epochs',type=int,default=20);p.add_argument('--head-epochs',type=int,default=20)
    p.add_argument('--head-batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--artifact-lr',type=float,default=1e-4);p.add_argument('--head-lr',type=float,default=5e-4)
    p.add_argument('--checkpoint-steps',type=int,default=100)
    p.add_argument('--frame-batch',type=int,default=64);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--decode-samples',type=int,default=2,
        help='Positive number of clips per dataset/split to decode in preflight (default: 2); all files get availability/stat checks')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-from',type=Path,
        help='Continue the known legacy AMP failure checkpoint into a separate output; keeps the original run unchanged')
    return p

if __name__=='__main__':
    try: run(parser().parse_args())
    except KeyboardInterrupt:
        status('INTERRUPTED','Committed clips and optimizer steps retained; repeat with --resume.')
        sys.exit(130)
    except Exception:
        traceback.print_exc();status('FAILED','No successful completion for the failed stage.');sys.exit(1)
