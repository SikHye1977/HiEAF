"""Source-only training; atomic optimizer-boundary resume including RNG and BN."""
import math, random, signal, time, shutil
from contextlib import contextmanager
import numpy as np
import torch
from .common import *
from .models import artifact_tensor
from .numerics import NUMERICS_POLICY, gradients_finite, clip_finite_gradients
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.data import collate
from scripts.trainable_or.evaluation import metrics

DETERMINISM_POLICY = dict(
    default='strict deterministic algorithms; fixed seeds and saved RNG states',
    exception='CUDA artifact/video backward only: deterministic algorithms with warn_only=True',
    cuda_bitwise_reproducibility_guaranteed=False)


@contextmanager
def backward_policy(task):
    """Allow unsupported CUDA pooling backward only inside video-artifact backprop."""
    if not (task.kind == 'artifact' and task.mod == 'video'
            and torch.device(task.device).type == 'cuda'):
        yield
        return
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['python']); n=s['numpy']
    np.random.set_state((n[0],np.array(n[1],np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])

class Task:
    def __init__(self,records,mod,kind,device,stamps,store=None):
        self.mod,self.kind,self.device,self.stamps,self.store=mod,kind,device,stamps,store
        self.records=[]
        self.unknown=self.missing=0
        scan=Progress('SCAN '+kind+' '+mod+' '+records[0].split,len(records))
        for n,r in enumerate(records,1):
            scan.update(n)
            if getattr(r,mod+'_label') is None: self.unknown+=1; continue
            if kind=='emotion':
                t=store.get(r.sample_id,stamps[r.sample_id])
                require(t is not None,'Missing frozen token cache')
                if not t['valid'].any(): self.missing+=1; continue
            self.records.append(r)
        self.y=labels(self.records,mod)
        require(set(self.y.tolist())=={0,1},'Need both valid source classes')
        status('TRAIN COHORT',f'{kind} {mod} known_valid={len(self.records)} unknown={self.unknown} missing_emotion={self.missing}')
    def __len__(self): return len(self.records)
    def forward(self,model,indices):
        rows=[self.records[i] for i in indices]
        for r in rows: require(stamp(r)==self.stamps[r.sample_id],'Training media changed: '+r.sample_id)
        if self.kind=='artifact':
            require(len(rows)==1,'Artifact microbatch must be 1; use accumulation')
            logits=model(artifact_tensor(rows[0],self.mod,self.device)).logits
        else:
            items=[]
            for i,r in enumerate(rows):
                t=self.store.get(r.sample_id,self.stamps[r.sample_id]); valid=t['valid']
                items.append(dict(index=i,positions=np.flatnonzero(valid),
                    embedding=torch.from_numpy(t['embedding'][valid]).float(),
                    probabilities=torch.from_numpy(t['probabilities'][valid]).float(),
                    label=getattr(r,self.mod+'_label')))
            e,p,m,_,_,_=collate(items)
            logits=model(e.to(self.device),p.to(self.device),m.to(self.device))[0]
        y=torch.as_tensor(self.y[indices],device=self.device)
        return logits,y

@torch.inference_mode()
def validate(model,task,batch,folder,label):
    model.eval(); ps=[]
    bar=Progress(label,len(task),folder)
    for i in range(0,len(task),batch):
        ix=list(range(i,min(i+batch,len(task))))
        logits,_=task.forward(model,ix)
        p=logits.softmax(-1)[:,1] if task.kind=='artifact' and task.mod=='audio' else logits.sigmoid()
        ps.extend(p.cpu().tolist()); bar.update(i+len(ix))
    return metrics(task.y,np.array(ps))

def train_group(model,task,order,cursor,end,batch,amp,scaler,opt,pos_weight,cw,
                state,folder,checkpoint,stop):
    """Retry AMP overflow without dropping samples or advancing BN/RNG twice."""
    rng_before=rng_state()
    buffers={name:value.detach().clone() for name,value in model.named_buffers()}
    indices=order[cursor*batch:min(end*batch,len(task))]
    ids=[task.records[i].sample_id for i in indices] if hasattr(task,'records') else indices
    def rollback():
        with torch.no_grad():
            for name,value in model.named_buffers(): value.copy_(buffers[name])
        restore_rng(rng_before)
        opt.zero_grad(set_to_none=True)
    def record(reason,**extra):
        event=dict(reason=reason,epoch=state['epoch']+1,cursor=cursor,
                   optimizer_updates=state['updates'],sample_ids=ids,
                   amp_overflows=state['amp_overflows'],**extra)
        atomic_json(folder/'NUMERICS.json',event)
        return event
    def fail(reason):
        rollback();record(reason);checkpoint()
        raise FloatingPointError(reason+'; sample IDs: '+str(ids))
    if state['amp_retries_at_cursor']>NUMERICS_POLICY['max_amp_retries']:
        fail('AMP retry limit already exhausted at this batch; inspect NUMERICS.json')
    while True:
        opt.zero_grad(set_to_none=True)
        group_loss=0.;group_count=0
        for step in range(cursor,end):
            ix=order[step*batch:min((step+1)*batch,len(task))]
            with torch.autocast('cuda' if task.device.startswith('cuda') else 'cpu',enabled=amp):
                logits,y=task.forward(model,ix)
                if task.kind=='artifact' and task.mod=='audio':
                    loss=(torch.nn.functional.cross_entropy(logits,y,reduction='none')*cw[y]).mean()
                else:
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y.float(),
                        pos_weight=torch.tensor(pos_weight,device=task.device))
            if not bool(torch.isfinite(loss)):
                fail('Nonfinite training loss')
            with backward_policy(task):
                scaler.scale(loss/(end-cursor)).backward()
            group_loss+=float(loss.detach())*len(ix);group_count+=len(ix)
        scaler.unscale_(opt)
        if gradients_finite(model):
            norm=clip_finite_gradients(model.parameters(),5.)
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            state['amp_retries_at_cursor']=0
            return group_loss,group_count,float(norm)
        if not amp:
            fail('Nonfinite FP32 gradients')
        before=float(scaler.get_scale())
        # unscale_ recorded invalid elements, so GradScaler skips this optimizer step.
        scaler.step(opt);scaler.update()
        after=float(scaler.get_scale())
        rollback()
        state['amp_overflows']+=1
        state['amp_retries_at_cursor']+=1
        record('AMP overflow; same batch will be retried',scale_before=before,scale_after=after,
               retry=state['amp_retries_at_cursor'])
        checkpoint()
        status('AMP RETRY',f'cursor={cursor} retry={state["amp_retries_at_cursor"]} '
               f'scale={before:g}->{after:g}; weights unchanged; same samples')
        if not (math.isfinite(after) and 0<after<before):
            fail('GradScaler did not produce a valid reduced scale')
        if state['amp_retries_at_cursor']>NUMERICS_POLICY['max_amp_retries']:
            fail('AMP retry limit exhausted; inspect NUMERICS.json')
        if stop[0]: raise KeyboardInterrupt()


def fit(model,train,dev,folder,cfg,identity,*,interrupt_after_updates=None):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    done=folder/'COMPLETED.json'
    if done.exists():
        old=read(done)
        require(old['identity']==identity and old['sha256']==sha(folder/'best.pt'),'Completed model changed')
        status('RESUME',str(folder)+' already trained'); return old
    kind,mod,device=train.kind,train.mod,train.device
    if kind=='artifact' and mod=='video' and device.startswith('cuda'):
        status('DETERMINISM','CUDA X3D backward: unsupported deterministic operations warn instead of stopping; seeds/RNG checkpoints retained; bitwise GPU equality not guaranteed')
    epochs=cfg['artifact_epochs'] if kind=='artifact' else cfg['head_epochs']
    batch=1 if kind=='artifact' else cfg['head_batch_size']
    accumulation=cfg['accumulation'] if kind=='artifact' else 1
    lr=cfg['artifact_lr'] if kind=='artifact' else cfg['head_lr']
    seed=cfg['seed']+(mod=='audio')+(100 if kind=='emotion' else 0)
    # Construction must also be seeded by the caller.
    model=model.to(device).requires_grad_(True)
    optimizer_type=torch.optim.Adam if kind=='artifact' and mod=='audio' else torch.optim.AdamW
    opt=optimizer_type(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs) if kind=='artifact' else None
    amp=kind=='artifact' and mod=='video' and device.startswith('cuda') and cfg['amp']
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    counts=np.bincount(train.y,minlength=2)
    pos_weight=float(counts[0]/counts[1])
    cw=torch.tensor(len(train)/(2*counts),device=device,dtype=torch.float32)
    state=dict(epoch=0,cursor=0,updates=0,loss_sum=0.,count=0,best_auc=-1.,
               best_epoch=0,best_model=None,history=[])
    last=folder/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=True)
        require(saved['identity']==identity,'Training identity changed')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        if scheduler: scheduler.load_state_dict(saved['scheduler'])
        state=saved['state']; restore_rng(saved['rng'])
    state.setdefault('amp_overflows',0)
    state.setdefault('amp_retries_at_cursor',0)
    status('RESUME STATE',f'epoch={state["epoch"]+1} cursor={state["cursor"]} optimizer_step={state["updates"]} AMP_scale={scaler.get_scale():g}')
    def commit():
        require(shutil.disk_usage(folder).free>2*1024**3,'Insufficient checkpoint disk space; previous checkpoint retained')
        save_torch(last,dict(identity=identity,model=cpu_state(model),optimizer=opt.state_dict(),
            scaler=scaler.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
            state=state,rng=rng_state()))
    total_batches=math.ceil(len(train)/batch)
    stop=[False]
    previous=signal.getsignal(signal.SIGINT)
    def on_interrupt(signum,frame):
        stop[0]=True
        status('INTERRUPT REQUEST','Finishing the current optimizer step, then saving.')
    for epoch in range(state['epoch'],epochs):
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        bar=Progress(f'TRAIN {kind} {mod} epoch {epoch+1}/{epochs}',total_batches,folder,
                     resumed=state['cursor'],unit='batches')
        complete_epoch_this_invocation=state['cursor']==0
        model.train(); start_time=time.monotonic()
        signal.signal(signal.SIGINT,on_interrupt)
        try:
            cursor=state['cursor']
            while cursor<total_batches:
                end=min(cursor+accumulation,total_batches)
                group_loss,group_count,grad_norm=train_group(
                    model,train,order,cursor,end,batch,amp,scaler,opt,pos_weight,cw,
                    state,folder,commit,stop)
                state['loss_sum']+=group_loss;state['count']+=group_count
                cursor=end; state['cursor']=cursor; state['updates']+=1
                bar.update(cursor,f'optimizer_step={state["updates"]} loss={state["loss_sum"]/state["count"]:.6f} AMP_scale={scaler.get_scale():g} amp_retries={state["amp_overflows"]}')
                if state['updates']%cfg['checkpoint_steps']==0 or cursor==total_batches or stop[0]:
                    commit()
                if stop[0] or (interrupt_after_updates is not None and state['updates']>=interrupt_after_updates):
                    commit(); raise KeyboardInterrupt()
        finally:
            signal.signal(signal.SIGINT,previous)
        # End-of-train checkpoint lets interrupted dev evaluation restart without retraining.
        commit()
        report=validate(model,dev,batch,folder,f'DEV {kind} {mod} epoch {epoch+1}/{epochs}')
        if report['roc_auc']>state['best_auc']:
            state.update(best_auc=report['roc_auc'],best_epoch=epoch+1,best_model=cpu_state(model))
        state['history'].append(dict(epoch=epoch+1,loss=state['loss_sum']/state['count'],
            dev=report,seconds_this_invocation=time.monotonic()-start_time,
            full_epoch_timed=complete_epoch_this_invocation))
        if scheduler: scheduler.step()
        state.update(epoch=epoch+1,cursor=0,loss_sum=0.,count=0)
        commit(); atomic_json(folder/'history.json',state['history'])
        # Invocation may contain only a resumed suffix: never extrapolate it to a full epoch.
        timings=[h['seconds_this_invocation'] for h in state['history'] if h.get('full_epoch_timed')]
        remaining=float(np.mean(timings))*(epochs-epoch-1) if timings else None
        status('TRAIN ETA',f'{kind} {mod} remaining_train_and_dev~{duration(remaining)}; excludes later extraction/evaluation')
        status('EPOCH COMPLETE',f'{kind} {mod} {epoch+1}/{epochs} dev_auc={report["roc_auc"]:.6f} '
               f'best_epoch={state["best_epoch"]}; remaining_epochs={epochs-epoch-1}')
    save_torch(folder/'best.pt',dict(identity=identity,kind=kind,modality=mod,
        model=state['best_model'],best_epoch=state['best_epoch'],source=cfg['source']))
    result=dict(identity=identity,sha256=sha(folder/'best.pt'),best_epoch=state['best_epoch'],
        dev_auc=state['best_auc'],epochs=epochs,train_n=len(train),dev_n=len(dev),
        train_unknown=train.unknown,train_emotion_missing=train.missing,pos_weight=pos_weight,
        amp_overflows=state['amp_overflows'],numerics=NUMERICS_POLICY)
    atomic_json(done,result); return result
