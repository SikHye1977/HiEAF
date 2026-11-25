"""Checked continuation of the known first-epoch AMP failure into a fresh run."""
from pathlib import Path
import copy, math
import torch
from .common import *
from .numerics import finite_state, NUMERICS_POLICY

LEGACY = {
 'scripts/trainable_or_3way/experiment.py':'13aa48b7c899bb2e928eb6083c7c0066215a6ba893b22d82b557d48de981bd14',
 'scripts/trainable_or_3way/train.py':'0bf32f780cc21e02759f49d667fe06884ff6d2c7fb8aaf152f28eceda5348195',
 'scripts/trainable_or_3way/test_experiment.py':'d726c8e232e361a08498d1a653ff1515cf5eb16b77edec860314d9f2ec9ae80e'}
ADDED = {'scripts/trainable_or_3way/'+name for name in
         ('numerics.py','continuation.py','test_numerics.py','test_continuation.py')}
CHECKPOINT = 'models/artifact/video/last.pt'

def compatible(old,current):
    left=copy.deepcopy(old);right=copy.deepcopy(current)
    require('continuation' not in left and 'continuation' not in right,
            'Only a legacy run can be imported; resume the destination normally afterwards')
    previous=left.pop('implementation');now=right.pop('implementation')
    require(right['config'].pop('numerics')==NUMERICS_POLICY,'Unknown numerical recovery policy')
    require(left==right,'Continuation config/data/assets/cohort differs from source run')
    changed={k for k in set(previous)|set(now) if previous.get(k)!=now.get(k)}
    require(changed<=set(LEGACY)|ADDED,'Unrelated code changed: '+str(sorted(changed-set(LEGACY)-ADDED)))
    for name in changed:
        if name in LEGACY: require(previous.get(name)==LEGACY[name], 'Unknown legacy code: '+name)
        else: require(name not in previous and name in now,'Unexpected added-code history: '+name)

def plan_continuation(source,current):
    source=Path(source).resolve()
    old=read(source/'RUN.json')
    require(old.get('format')=='trainable_or_3way_v1','Unsupported source run')
    compatible(old,current)
    failure=read(source/'STATUS.json')
    require(failure['status']=='FAILED' and 'non-finite' in failure.get('error',''),
            'Source must be the failed nonfinite-gradient run')
    require(not (source/'COMPLETED.json').exists() and not (source/'MODELS_SELECTED.json').exists(),
            'Cannot import a completed/selected run')
    require({p.relative_to(source).as_posix() for p in (source/'models').rglob('*.pt')}=={CHECKPOINT},
            'Only the initial video artifact checkpoint may exist')
    path=source/CHECKPOINT
    before=sha(path)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    require(before==sha(path),'Source checkpoint changed while reading')
    identity=digest(old)
    model_identity=digest(dict(run=identity,kind='artifact',modality='video'))
    require(saved['identity']==model_identity,'Source checkpoint/run mismatch')
    s=saved['state'];cfg=old['config']
    known=sum(old['cohorts'][cfg['source']+'/train']['counts']['video'][str(k)] for k in (0,1))
    require(s['epoch']==0 and not s['history'] and s['best_model'] is None,
            'This import supports only the first artifact training epoch')
    require(0<s['cursor']<known and s['cursor']==s['updates']*cfg['accumulation']
            and s['count']==s['cursor'],'Invalid optimizer-boundary cursor')
    require(cfg['amp'] and saved['scaler']['scale']>0 and math.isfinite(saved['scaler']['scale']),
            'Expected a valid AMP checkpoint')
    require(finite_state(saved['model']) and finite_state(saved['optimizer'])
            and math.isfinite(s['loss_sum']), 'Source checkpoint contains invalid values')
    require(saved['scheduler'] is not None and saved['rng'],'Incomplete training state')
    return dict(source_dir=str(source),source_run_sha256=sha(source/'RUN.json'),
        source_status_sha256=sha(source/'STATUS.json'),source_checkpoint_sha256=before,
        source_model_identity=model_identity,cursor=s['cursor'],optimizer_updates=s['updates'],
        policy='continue the same source training after AMP overflow handling fix')

def install_continuation(out,run):
    plan=run.get('continuation')
    if plan is None:return
    out=Path(out);source=Path(plan['source_dir'])
    require(out.resolve()!=source.resolve(),'Continuation output must differ from source')
    identity=digest(dict(run=digest(run),kind='artifact',modality='video'))
    imported=out/'CONTINUATION_CHECKPOINT.pt';receipt=out/'CONTINUATION.json'
    if receipt.exists():
        record=read(receipt)
        require(record['plan']==plan and record['identity']==identity,'Continuation receipt changed')
        require(sha(imported)==record['imported_sha256'],'Imported checkpoint changed')
    else:
        # Source is read-only; hold its writer lock during the verified snapshot copy.
        with WriterLock(source):
            for name,key in [('RUN.json','source_run_sha256'),('STATUS.json','source_status_sha256'),
                             (CHECKPOINT,'source_checkpoint_sha256')]:
                require(sha(source/name)==plan[key],'Continuation source changed: '+name)
            saved=torch.load(source/CHECKPOINT,map_location='cpu',weights_only=True)
            require(saved['identity']==plan['source_model_identity'],'Source identity changed')
            saved['identity']=identity
            save_torch(imported,saved)
            atomic_json(receipt,dict(plan=plan,identity=identity,imported_sha256=sha(imported)))
    last=out/CHECKPOINT
    if not last.exists():
        require(not (last.parent/'progress.json').exists(),'Training checkpoint missing after import')
        saved=torch.load(imported,map_location='cpu',weights_only=True)
        last.parent.mkdir(parents=True,exist_ok=True)
        save_torch(last,saved)
        status('CONTINUE CHECKPOINT',f'cursor={plan["cursor"]} optimizer_step={plan["optimizer_updates"]}; model/optimizer/scaler/scheduler/RNG retained')
