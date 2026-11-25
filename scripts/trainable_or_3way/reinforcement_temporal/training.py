"""FP32 head-only training with optimizer-boundary checkpoints and resumable prediction."""
import math, signal, time
import numpy as np
import torch
from scripts.trainable_or_3way.reinforcement.statistics import Metric
from .common import *
from .model import TemporalHead, compact, batch

class Task:
    def __init__(self, cache, *, labeled=True):
        self.cache=cache;self.mod=cache.mod
        col=1 if self.mod=="video" else 2
        self.indices=[]; self.lengths=[]
        self.unknown=self.missing=0
        bar=Progress("SCAN "+self.mod,len(cache.ids))
        for i,sid in enumerate(cache.ids):
            n=int(cache.counts[i]) if cache.counts is not None else int(cache.get(sid)["valid"].sum())
            self.lengths.append(n)
            if n==0: self.missing+=1
            if cache.cohort["labels"][i,col]<0: self.unknown+=1
            if n and (not labeled or cache.cohort["labels"][i,col]>=0): self.indices.append(i)
            bar.update(i+1)
        self.y=cache.cohort["labels"][self.indices,col].astype(np.float32)
        if labeled: require(set(self.y.tolist())=={0.,1.},"Need both valid source classes")
    def __len__(self): return len(self.indices)
    def forward(self,model,indices,device):
        rows=[self.indices[i] for i in indices]
        items=[compact(self.cache.get(self.cache.ids[i]),self.cache.ids[i],self.mod) for i in rows]
        z=model(*batch(items,device))[0]
        return z,torch.as_tensor(self.y[indices],device=device)

@torch.inference_mode()
def validate(model, task, device, batch_size, label, folder):
    model.eval();z=[]
    bar=Progress(label,len(task),folder)
    for i in range(0,len(task),batch_size):
        ix=list(range(i,min(i+batch_size,len(task))))
        logit,_=task.forward(model,ix,device)
        require(bool(torch.isfinite(logit).all()),"Nonfinite dev logit")
        z.extend(logit.sigmoid().cpu().tolist());bar.update(i+len(ix))
    return Metric(task.y,z).calculate()

def fit(train, dev, variant, seed, source, folder, identity, *, device, config=TRAIN,
        stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if completed(folder,identity):
        print("[REUSE HEAD] "+str(folder),flush=True);return
    mod=train.mod;branch_seed=int(seed)+100+(mod=="audio")
    seed_all(branch_seed)
    model=TemporalHead(mod,variant).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    counts=np.bincount(train.y.astype(int),minlength=2);pos=float(counts[0]/counts[1])
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device=device))
    state=dict(epoch=0,cursor=0,updates=0,loss_sum=0.,seen=0,best_auc=-1.,
               best_epoch=0,best_model=None,history=[])
    last=folder/"last.pt"
    if last.exists():
        saved=torch.load(last,map_location="cpu",weights_only=True)
        require(saved["identity"]==identity,"Training identity changed")
        model.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"])
        state=saved["state"];restore_rng(saved["rng"])
    def save():
        save_torch(last,dict(identity=identity,model=cpu_state(model),optimizer=opt.state_dict(),
                            state=state,rng=rng_state()))
    stop=[False];previous=signal.getsignal(signal.SIGINT)
    def interrupt(signum,frame):
        stop[0]=True
        print("[STOP REQUEST] Saving after the current optimizer step.",flush=True)
    batch_size=config["batch_size"];total=math.ceil(len(train)/batch_size)
    for epoch in range(state["epoch"],config["epochs"]):
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(branch_seed+epoch)).tolist()
        timed=state["cursor"]==0;start=time.monotonic()
        bar=Progress(f"TRAIN {source} {variant} seed{seed} {mod} epoch {epoch+1}/{config['epochs']}",
                     total,folder,resumed=state["cursor"])
        model.train();signal.signal(signal.SIGINT,interrupt)
        try:
            for cursor in range(state["cursor"],total):
                ix=order[cursor*batch_size:(cursor+1)*batch_size]
                opt.zero_grad(set_to_none=True)
                z,y=train.forward(model,ix,device)
                loss=lossfn(z,y)
                require(bool(torch.isfinite(loss)),"Nonfinite FP32 loss; last committed batch preserved")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip"],error_if_nonfinite=True)
                opt.step()
                state["cursor"]=cursor+1;state["updates"]+=1
                state["loss_sum"]+=float(loss.detach())*len(ix);state["seen"]+=len(ix)
                bar.update(cursor+1,f"loss={state['loss_sum']/state['seen']:.6f} update={state['updates']}")
                if state["updates"]%config["checkpoint_steps"]==0 or stop[0]: save()
                if stop[0] or (stop_after is not None and state["updates"]>=stop_after):
                    save();raise KeyboardInterrupt()
        finally: signal.signal(signal.SIGINT,previous)
        save()  # Dev interruption repeats dev only.
        report=validate(model,dev,device,batch_size,f"DEV {source} {variant} {mod} epoch{epoch+1}",folder)
        if report["roc_auc"]>state["best_auc"]:
            state.update(best_auc=report["roc_auc"],best_epoch=epoch+1,best_model=cpu_state(model))
        state["history"].append(dict(epoch=epoch+1,loss=state["loss_sum"]/state["seen"],dev=report,
                                     seconds=time.monotonic()-start,full_epoch_timed=timed))
        state.update(epoch=epoch+1,cursor=0,loss_sum=0.,seen=0)
        save();atomic_json(folder/"history.json",state["history"])
        timings=[h["seconds"] for h in state["history"] if h["full_epoch_timed"]]
        eta=float(np.mean(timings))*(config["epochs"]-epoch-1) if timings else None
        print(f"[EPOCH COMPLETE] {source}/{variant}/{mod} {epoch+1}/{config['epochs']} "
              f"dev_auc={report['roc_auc']:.6f} best_epoch={state['best_epoch']} "
              f"remaining_train_dev~{duration(eta)}; excludes other stages",flush=True)
    save_torch(folder/"best.pt",dict(model=state["best_model"],identity=identity,source=source,
                                  modality=mod,variant=variant,seed=seed,best_epoch=state["best_epoch"]))
    atomic_json(folder/"TRAINING.json",dict(config=config,source=source,seed=seed,branch_seed=branch_seed,
                modality=mod,variant=variant,best_epoch=state["best_epoch"],dev_auc=state["best_auc"],
                pos_weight=pos,train_n=len(train),dev_n=len(dev),unknown=train.unknown,missing=train.missing,
                trainable_parameters=sum(p.numel() for p in model.parameters())))
    finish(folder,identity,["best.pt","history.json","TRAINING.json"])

def load_head(path,mod,variant,device):
    state=torch.load(path,map_location="cpu",weights_only=True)
    require(state["modality"]==mod,"Checkpoint modality mismatch")
    head=TemporalHead(mod,variant)
    head.load_state_dict(state["model"],strict=True)
    return head.to(device).eval().requires_grad_(False)

@torch.inference_mode()
def predict(cache, head_path, variant, transform, folder, *, device, batch_size=32):
    folder=Path(folder)
    identity=digest(dict(cache=cache.proof,head=sha(head_path),variant=variant,transform=transform,
                         code=training_code_identity(),batch_size=batch_size,device=str(device)))
    if completed(folder,identity): return load_npz(folder/"scores.npz")
    immutable(folder/"RUN.json",dict(identity=identity,transform=transform,
              permutation="sha256([shuffle seed,sample ID,modality]) first 64 bits; NumPy default_rng",
              original_cache_unchanged=True))
    path=folder/"state.npz";n=len(cache.ids)
    ps=np.full(n,np.nan);zs=ps.copy();lengths=np.zeros(n,np.int64);cursor=0
    if path.exists():
        s=load_npz(path);require(str(s["identity"])==identity and np.array_equal(s["sample_id"],cache.ids),
                               "Prediction resume input mismatch")
        ps,zs,lengths,cursor=s["probability"],s["logit"],s["lengths"],int(s["cursor"])
    def save():
        atomic_npz(path,identity=np.array(identity),sample_id=np.array(cache.ids),
                   probability=ps,logit=zs,lengths=lengths,cursor=np.array(cursor))
    model=load_head(head_path,cache.mod,variant,device)
    bar=Progress(f"PREDICT {cache.mod} {variant} {transform}",n,folder,resumed=cursor)
    # A batch is committed to memory only after complete forward; interrupted work is replayed.
    try:
        while cursor<n:
            end=min(cursor+batch_size,n); items=[];indices=[];local_lengths=[]
            for i in range(cursor,end):
                sid=cache.ids[i];item=compact(cache.get(sid),sid,cache.mod,transform)
                local_lengths.append(len(item["embedding"]))
                if local_lengths[-1]: items.append(item);indices.append(i)
            if items:
                z=model(*batch(items,device))[0]
                require(bool(torch.isfinite(z).all()),"Nonfinite prediction")
                zz=z.cpu().numpy().astype(np.float64);pp=z.sigmoid().cpu().numpy().astype(np.float64)
                zs[indices]=zz;ps[indices]=pp
            lengths[cursor:end]=local_lengths;cursor=end
            if cursor%(batch_size*25)==0 or cursor==n: save()
            bar.update(cursor)
    except BaseException:
        save();raise
    atomic_npz(folder/"scores.npz",sample_id=np.array(cache.ids),probability=ps,logit=zs,lengths=lengths)
    # The exact compact valid-token permutations, including short sequences, are materialized.
    from .model import permutation
    offsets=np.r_[0,np.cumsum(lengths)]
    orders=np.concatenate([permutation(s,cache.mod,transform,int(t)) for s,t in zip(cache.ids,lengths)])
    atomic_npz(folder/"permutations.npz",sample_id=np.array(cache.ids),offsets=offsets,positions=orders)
    finish(folder,identity,["scores.npz","permutations.npz","RUN.json"])
    return load_npz(folder/"scores.npz")
