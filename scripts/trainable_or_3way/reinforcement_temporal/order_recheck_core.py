"""P3-a only: one feature read serves all fixed sequence permutations."""
import numpy as np
import torch
from .common import *
from .model import compact, batch, permutation
from .training import load_head
from .analysis import fused, metrics_nullable
from .statistics import metric_table

POLICY = dict(reference="fresh ordered inference, same weights/features/fusion, common FP32",
              matmul_tf32=False,cudnn_tf32=False,training=False,calibration_refitted=False,
              historical_comparison="audit only; no enlarged numerical tolerance",
              permutations=list(TRANSFORMS),interpretation="descriptive order sensitivity")

def check_precision():
    require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
            "P3-a requires both TF32 flags off")

@torch.inference_mode()
def predict_all(cache,head_path,folder,*,device,batch_size,stop_after=None):
    check_precision();folder=Path(folder)
    identity=digest(dict(cache=cache.proof,head=sha(head_path),transforms=list(TRANSFORMS),
                        code=code_identity(),policy=POLICY,batch_size=batch_size,device=str(device)))
    if completed(folder,identity):
        print("[REUSE] "+str(folder),flush=True)
        return load_npz(folder/"scores.npz")
    n=len(cache.ids);cursor=0
    ps=np.full((n,len(TRANSFORMS)),np.nan);zs=ps.copy();lengths=np.zeros(n,np.int64)
    state_path=folder/"state.npz"
    immutable(folder/"RUN.json",dict(identity=identity,cache=cache.proof,head=str(head_path),
              head_sha256=sha(head_path),policy=POLICY,batch_size=batch_size,
              sample_ids_sha256=digest(cache.ids)))
    if state_path.exists():
        saved=load_npz(state_path);require(str(saved["identity"])==identity,"Prediction inputs changed")
        ps,zs,lengths,cursor=saved["probabilities"],saved["logits"],saved["lengths"],int(saved["cursor"])
        require(ps.shape==zs.shape==(n,len(TRANSFORMS)) and lengths.shape==(n,) and 0<=cursor<=n,
                "Corrupt resume state")
    model=load_head(head_path,cache.mod,"T2",device)
    bar=Progress("P3-a "+cache.mod+" all 5 orders",n,folder,resumed=cursor)
    def save():
        # IDs are hash-bound once; avoid compressing large Unicode IDs at every checkpoint.
        atomic_npz(state_path,identity=np.array(identity),cursor=np.array(cursor),
                   probabilities=ps,logits=zs,lengths=lengths)
    try:
        while cursor<n:
            end=min(cursor+batch_size,n)
            tokens=[cache.get(sid) for sid in cache.ids[cursor:end]]
            local_lengths=np.array([int(t["valid"].sum()) for t in tokens],np.int64)
            valid=np.flatnonzero(local_lengths>0)
            local_ps=np.full((end-cursor,len(TRANSFORMS)),np.nan);local_zs=local_ps.copy()
            for j,transform in enumerate(TRANSFORMS):
                if len(valid):
                    items=[compact(tokens[i],cache.ids[cursor+i],cache.mod,transform) for i in valid]
                    z=model(*batch(items,device))[0]
                    require(bool(torch.isfinite(z).all()),"Nonfinite head output")
                    local_zs[valid,j]=z.cpu().numpy()
                    local_ps[valid,j]=z.sigmoid().cpu().numpy()
            # Commit all 5 orders together; a partial batch is replayed after interruption.
            ps[cursor:end]=local_ps;zs[cursor:end]=local_zs;lengths[cursor:end]=local_lengths
            cursor=end
            if cursor%(batch_size*25)==0 or cursor==n:save()
            bar.update(cursor,"features read once; no encoder/training")
            if stop_after is not None and cursor>=stop_after:
                save();raise InterruptedError("Synthetic requested interruption")
    except BaseException:
        save();raise
    missing=lengths==0
    require(np.isnan(ps[missing]).all() and np.isfinite(ps[~missing]).all(),"Unexpected missing coverage")
    for j in range(len(TRANSFORMS)):
        require(np.allclose(ps[lengths==1,j],ps[lengths==1,0],atol=2e-6,rtol=0),
                "One-token order should not change the prediction")
    atomic_npz(folder/"scores.npz",sample_id=np.array(cache.ids),probabilities=ps,logits=zs,lengths=lengths,
               transforms=np.array(TRANSFORMS))
    files=["RUN.json","scores.npz"]
    offsets=np.r_[0,np.cumsum(lengths)];effective={}
    for transform in TRANSFORMS:
        orders=[permutation(s,cache.mod,transform,int(t)) for s,t in zip(cache.ids,lengths)]
        effective[transform]=int(sum(not np.array_equal(o,np.arange(len(o))) for o in orders))
        name="permutation_"+transform+".npz"
        atomic_npz(folder/name,offsets=offsets,positions=np.concatenate(orders).astype(np.int32))
        files.append(name)
    atomic_json(folder/"COVERAGE.json",dict(n=n,missing=int(missing.sum()),one=int((lengths==1).sum()),
                two=int((lengths==2).sum()),three_plus=int((lengths>=3).sum()),actual_order_changed=effective,
                permutation_positions="compact valid-token indices; native probabilities travel with embedding"))
    files.append("COVERAGE.json")
    finish(folder,identity,files)
    return load_npz(folder/"scores.npz")

def score_difference(old,new):
    require(old.shape==new.shape and np.array_equal(np.isfinite(old),np.isfinite(new)),
            "Historical/current coverage mismatch")
    mask=np.isfinite(old);d=np.abs(old[mask]-new[mask])
    return dict(n=int(mask.sum()),max_abs=float(d.max()) if len(d) else None,
                mean_abs=float(d.mean()) if len(d) else None,
                decisions_changed_at_half=int(((old[mask]>=.5)!=(new[mask]>=.5)).sum()),
                within_old_tolerance=bool(np.allclose(old,new,atol=2e-5,rtol=2e-5,equal_nan=True)))

def evaluate_pair(inputs,source,target,pairs,folder):
    folder=Path(folder);data=inputs.raw[source,"B",target,"test"]
    for mod in ("video","audio"):
        require(np.array_equal(pairs[mod]["sample_id"],data["sample_id"])
                and pairs[mod]["transforms"].tolist()==list(TRANSFORMS),"Score IDs/order types differ")
    ordered=np.column_stack([pairs[m]["probabilities"][:,0] for m in ("video","audio")])
    # This FP32 ordered baseline is the comparator; historical values remain untouched.
    reference=dict(sample_id=data["sample_id"],labels=data["labels"],ordered=ordered)
    reference_path=folder/"ORDERED_FP32_REFERENCE.npz"
    if reference_path.exists():
        old_reference=load_npz(reference_path)
        require(all(np.array_equal(old_reference[k],v,equal_nan=v.dtype.kind=="f")
                    for k,v in reference.items()),"FP32 ordered reference changed")
    else:atomic_npz(reference_path,**reference)
    results=[]
    for condition in ("B","C"):
        data=inputs.raw[source,condition,target,"test"];model=inputs.historical_fusion(source,condition)
        identity=digest(dict(source=source,target=target,condition=condition,model=model,policy=POLICY,
            heads=inputs.heads[source],input=inputs.proofs[source,condition,target,"test"],
            reference_sha256=sha(folder/"ORDERED_FP32_REFERENCE.npz"),code=code_identity(),
            predictions={m:hash_array(pairs[m]["probabilities"]) for m in pairs}))
        job=folder/condition
        if completed(job,identity):
            results.append(read(job/"DIAGNOSTIC.json"));continue
        historical=data["raw"][:,2:]
        audit={m:score_difference(historical[:,i],ordered[:,i]) for i,m in enumerate(("video","audio"))}
        hscore=fused(data["raw"],historical,model)[0]
        base=fused(data["raw"],ordered,model)[0]
        audit["final_score"]=score_difference(hscore,base)
        audit["final_metrics"]=metric_table(data["labels"][:,0],dict(historical=hscore,ordered_fp32=base))
        scores={};branch={}
        for j,transform in enumerate(TRANSFORMS):
            e=np.column_stack([pairs[m]["probabilities"][:,j] for m in ("video","audio")])
            require(np.array_equal(np.isfinite(e),np.isfinite(ordered)),"Order changed missingness")
            for mode in (("both",) if transform=="ordered" else ("both","video","audio")):
                changed=ordered.copy()
                if mode in ("both","video"):changed[:,0]=e[:,0]
                if mode in ("both","audio"):changed[:,1]=e[:,1]
                scores[transform+"__"+mode]=fused(data["raw"],changed,model)[0]
            branch[transform]={}
            for i,mod in enumerate(("video","audio")):
                keep=(data["labels"][:,i+1]>=0)&np.isfinite(e[:,i])
                branch[transform][mod]=metrics_nullable(data["labels"][keep,i+1],{mod:e[keep,i]})[mod]
        y=data["labels"][:,0]
        metrics=metric_table(y,scores);changes={}
        for name,p in scores.items():
            changes[name]=dict(mean_abs_score_change=float(np.mean(np.abs(p-base))),
                decisions_changed=int(((p>=.5)!=(base>=.5)).sum()),
                real_fp_added=int(((y==0)&(p>=.5)&(base<.5)).sum()),
                real_fp_removed=int(((y==0)&(p<.5)&(base>=.5)).sum()),
                delta={k:metrics[name][k]-metrics["ordered__both"][k] for k in
                       ("roc_auc","fake_f1","real_fpr","fake_recall")})
        strata={}
        for mod in ("video","audio"):
            n=pairs[mod]["lengths"]
            strata[mod]={label:dict(n=int(keep.sum()),metrics=metrics_nullable(y[keep],{k:p[keep] for k,p in scores.items()}))
                for label,keep in (("missing",n==0),("one",n==1),("two",n==2),("three_plus",n>=3))}
        result=dict(source=source,target=target,condition=condition,metrics=metrics,branches=branch,
                    length_strata=strata,changes=changes,historical_numerics=audit,policy=POLICY,
                    checkpoint_selection=False,target_fitting=False,calibration_refitted=False)
        atomic_json(job/"DIAGNOSTIC.json",result)
        atomic_npz(job/"predictions.npz",sample_id=data["sample_id"],labels=data["labels"],**scores)
        finish(job,identity,["DIAGNOSTIC.json","predictions.npz"])
        results.append(result)
    return results

def hash_array(a):
    import hashlib
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
