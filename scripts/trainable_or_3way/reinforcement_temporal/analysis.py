"""Source-dev fitting, fixed-head order diagnostics and paired temporal comparisons."""
from pathlib import Path
import numpy as np
from scripts.trainable_or.fusion import fit as fit_fusion, predict as fusion_predict
from .common import *
from .training import predict
from .statistics import metric_table, bootstrap, BOOT_METRICS

def calibrate(raw, emotion, labels, *, source, dataset, split):
    require(source==dataset and split=="dev","Fusion must be fitted on source dev only")
    a=score_or(raw[:,0],raw[:,1]);valid=np.isfinite(emotion).all(1)
    require(set(labels.tolist())=={0,1} and set(labels[valid].tolist())=={0,1},"Need both dev classes")
    return dict(artifact_affine=fit_fusion(a,None,labels,split="dev"),
                trainable_final=fit_fusion(a[valid],score_or(emotion[valid,0],emotion[valid,1]),
                                          labels[valid],split="dev"))

def fused(raw,emotion,model):
    a=score_or(raw[:,0],raw[:,1])
    p=fusion_predict(model["artifact_affine"],a)
    out=p.copy();valid=np.isfinite(emotion).all(1)
    if valid.any():
        out[valid]=fusion_predict(model["trainable_final"],a[valid],score_or(emotion[valid,0],emotion[valid,1]))
    return out,valid,dict(M0=a,M1=p)

def predict_pair(inputs, dataset, split, paths, variant, folder, device, batch_size, transform="ordered"):
    results={}
    for mod in ("video","audio"):
        cache=inputs.cache(dataset,split,mod)
        try:
            results[mod]=predict(cache,paths[mod],variant,transform,Path(folder)/mod/transform,
                                 device=device,batch_size=batch_size)
        finally: cache.close()
    require(np.array_equal(results["video"]["sample_id"],results["audio"]["sample_id"]),
            "Modality prediction IDs do not align")
    return results

def pair_array(pair):
    return np.column_stack([pair[m]["probability"] for m in ("video","audio")])

def fit_locked(inputs, source, seed, variant, pair, head_paths, folder):
    folder=Path(folder);dev=inputs.cohorts[source,"dev"]
    emotion=pair_array(pair)
    binding=dict(source=source,seed=seed,variant=variant,split="dev",
                 heads={m:sha(p) for m,p in head_paths.items()},
                 emotion=hash_array(emotion),
                 dev={c:inputs.proofs[source,c,source,"dev"] for c in ("B","C")})
    identity=digest(binding)
    if completed(folder,identity): return read(folder/"FUSION.json")
    models={c:calibrate(inputs.raw[source,c,source,"dev"]["raw"],emotion,dev["labels"][:,0],
                         source=source,dataset=source,split="dev") for c in ("B","C")}
    atomic_json(folder/"FUSION.json",models)
    atomic_json(folder/"LOCKED_BEFORE_TEST.json",dict(identity=identity,inputs=binding,
                fusion_sha256=sha(folder/"FUSION.json"),threshold=.5,
                target_training=False,independent_calibration=False,prior_targets_observed=True))
    finish(folder,identity,["FUSION.json","LOCKED_BEFORE_TEST.json"])
    return models

def metrics_nullable(y,scores):
    if len(y)==0: return {k:dict(n=0) for k in scores}
    return metric_table(y,scores)

def evaluate_heads(inputs, source, target, seed, condition, predictions, models, folder, replicates):
    folder=Path(folder)
    data=inputs.raw[source,condition,target,"test"]
    raw=data["raw"];ys=data["labels"][:,0]
    probabilities={};valids=[];branches={}
    for variant in VARIANTS:
        pair=predictions[variant];emotion=pair_array(pair)
        p,valid,baseline=fused(raw,emotion,models[variant][condition])
        probabilities[variant]=p;valids.append(valid)
        branches[variant]={}
        for i,mod in enumerate(("video","audio")):
            mask=(data["labels"][:,i+1]>=0)&np.isfinite(emotion[:,i])
            branches[variant][mod]=metrics_nullable(data["labels"][mask,i+1],{mod:emotion[mask,i]})[mod]
    require(all(np.array_equal(valids[0],v) for v in valids),"Temporal methods have different missing coverage")
    # M1 is fitted on all source-dev artifact scores, identically for every temporal model.
    for v in VARIANTS[1:]:
        require(models[v][condition]["artifact_affine"]==models["T0"][condition]["artifact_affine"],
                "Artifact-affine control differs between temporal variants")
    probabilities.update(baseline)
    outputs=[]
    for cohort,mask in (("full",np.ones(len(ys),bool)),("common_emotion",valids[0])):
        job=folder/cohort
        identity=digest(dict(source=source,target=target,seed=seed,condition=condition,cohort=cohort,
            data=inputs.proofs[source,condition,target,"test"],models=models,
            predictions={k:hash_array(v) for k,v in probabilities.items()},replicates=replicates,
            code=evaluation_code_identity()))
        if completed(job,identity):
            outputs.append(read(job/"METRICS.json"));continue
        p={k:v[mask] for k,v in probabilities.items()};y=ys[mask]
        report=dict(source=source,target=target,seed=seed,condition=condition,cohort=cohort,
                    n=int(mask.sum()),missing_emotion=int((~valids[0]).sum()),
                    metrics=metric_table(y,p),branches=branches,threshold=.5)
        atomic_json(job/"METRICS.json",report)
        atomic_npz(job/"predictions.npz",sample_id=data["sample_id"][mask],labels=data["labels"][mask],
                   cluster=data["cluster"][mask],**p)
        b=bootstrap(y,p,data["cluster"][mask],job,identity,replicates=replicates,
                    seed=1701+int(digest([source,target,condition,seed,cohort])[:6],16))
        require(b["status"]=="COMPLETE","Bootstrap did not reach required valid replicates")
        atomic_json(job/"BOOTSTRAP.json",b)
        finish(job,identity,["METRICS.json","predictions.npz","BOOTSTRAP.json","bootstrap_state.npz"])
        outputs.append(report)
    return outputs

def hash_array(a):
    import hashlib
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def order_diagnostic(inputs,source,target,folder,device,batch_size):
    folder=Path(folder);paths={m:r["path"] for m,r in inputs.heads[source].items()}
    # Every altered sequence uses the SAME historical head and SAME source-dev fusion model.
    pairs={t:predict_pair(inputs,target,"test",paths,"T2",folder/"predictions",device,batch_size,t)
           for t in TRANSFORMS}
    ordered=pair_array(pairs["ordered"])
    reports=[]
    for condition in ("B","C"):
        data=inputs.raw[source,condition,target,"test"]
        expected=data["raw"][:,2:]
        require(np.array_equal(np.isfinite(expected),np.isfinite(ordered))
                and np.allclose(expected,ordered,atol=2e-5,rtol=2e-5,equal_nan=True),
                "Historical ordered predictions do not match preserved emotion scores; stop before perturbation interpretation")
        model=inputs.historical_fusion(source,condition)
        binding=dict(source=source,target=target,condition=condition,head=inputs.heads[source],
                     fusion=model,data=inputs.proofs[source,condition,target,"test"],transforms=list(TRANSFORMS),
                     code=evaluation_code_identity())
        job=folder/condition;identity=digest(binding)
        if completed(job,identity):
            reports.append(read(job/"DIAGNOSTIC.json"));continue
        scores={};branch_scores={}
        for t in TRANSFORMS:
            e=pair_array(pairs[t])
            for mode in (("both",) if t=="ordered" else ("both","video","audio")):
                changed=ordered.copy()
                if mode in ("both","video"): changed[:,0]=e[:,0]
                if mode in ("both","audio"): changed[:,1]=e[:,1]
                name=t+"__"+mode;scores[name]=fused(data["raw"],changed,model)[0]
            branch_scores[t]={}
            for i,mod in enumerate(("video","audio")):
                mask=(data["labels"][:,i+1]>=0)&np.isfinite(e[:,i])
                branch_scores[t][mod]=metrics_nullable(data["labels"][mask,i+1],{mod:e[mask,i]})[mod]
        y=data["labels"][:,0];length={}
        for mod in ("video","audio"):
            n=pairs["ordered"][mod]["lengths"]
            length[mod]={label:dict(n=int(mask.sum()),metrics=metrics_nullable(y[mask],{k:v[mask] for k,v in scores.items()}))
                         for label,mask in (("missing",n==0),("one",n==1),("two",n==2),("three_plus",n>=3))}
        changes={}
        ordered_p=scores["ordered__both"]
        for name,p in scores.items():
            changes[name]=dict(mean_absolute_score_change=float(np.mean(np.abs(p-ordered_p))),
                decisions_changed=int(((p>=.5)!=(ordered_p>=.5)).sum()),
                real_fp_added=int(((y==0)&(p>=.5)&(ordered_p<.5)).sum()),
                real_fp_removed=int(((y==0)&(p<.5)&(ordered_p>=.5)).sum()))
        result=dict(source=source,target=target,condition=condition,
                    metrics=metric_table(y,scores),branches=branch_scores,length_strata=length,changes=changes,
                    interpretation="Descriptive fixed-head order sensitivity; not proof of emotion inconsistency",
                    one_token_order_testable=False,calibration_refitted=False,
                    ordered_max_abs_difference=float(np.nanmax(np.abs(expected-ordered))))
        atomic_json(job/"DIAGNOSTIC.json",result)
        atomic_npz(job/"predictions.npz",sample_id=data["sample_id"],labels=data["labels"],**scores)
        finish(job,identity,["DIAGNOSTIC.json","predictions.npz"])
        reports.append(result)
    return reports

def aggregate(output,scope,rows):
    """Family expansion reuses saved draws; it does not refit heads or resample old jobs."""
    output=Path(output);scope_id=digest(scope)[:16];folder=output/"analyses"/scope_id
    immutable(folder/"SCOPE.json",scope)
    cross=[(s,t) for s in scope["sources"] for t in scope["targets"] if s!=t]
    family_size=max(1,len(cross)*2*2*len(scope["seeds"]))
    summaries=[];family=[]
    for r in rows:
        for method,metrics in r["metrics"].items():
            summaries.append({k:r[k] for k in ("source","target","seed","condition","cohort")}|dict(method=method,**metrics))
        if r["cohort"]!="full" or r["source"]==r["target"]: continue
        job=output/"evaluations"/r["source"]/f"seed_{r['seed']}"/r["target"]/r["condition"]/"full"
        b=read(job/"BOOTSTRAP.json");state=load_npz(job/"bootstrap_state.npz")
        for name,stats in b["comparisons"].items():
            left,right,col=stats["roc_auc"]["draws_column"]
            values=state["values"][:,left,col]-state["values"][:,right,col]
            q=.05/(2*family_size)
            family.append({k:r[k] for k in ("source","target","seed","condition")}|dict(comparison=name,
                delta_auc=stats["roc_auc"]["delta"],ci_family=np.quantile(values,[q,1-q]).tolist()))
    averages=[]
    groups={}
    for r in summaries:
        key=tuple(r[k] for k in ("source","target","condition","cohort","method"))
        groups.setdefault(key,[]).append(r)
    for key,group in groups.items():
        entry=dict(zip(("source","target","condition","cohort","method"),key));entry["seeds"]=[r["seed"] for r in group]
        for metric in ("roc_auc","fake_f1","real_fpr","fake_recall"):
            values=[r[metric] for r in group if r[metric] is not None]
            entry[metric]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None)
        averages.append(entry)
    atomic_json(folder/"SUMMARY.json",dict(scope=scope,rows=summaries,seed_summary=averages,
                auc_family_size=family_size,auc_family=family,
                family_complete=len(family)==family_size,
                caution="Head-seed variability conditional on fixed encoders/artifact checkpoints and fixed split; not full-model reproducibility"))
    import csv,io
    buffer=io.StringIO(newline="")
    keys=["source","target","seed","condition","cohort","method","n","real","fake","roc_auc","fake_f1","real_fpr","fake_recall"]
    writer=csv.DictWriter(buffer,fieldnames=keys,extrasaction="ignore");writer.writeheader();writer.writerows(summaries)
    atomic_text(folder/"SUMMARY.csv",buffer.getvalue())
    finish(folder,digest(scope),["SCOPE.json","SUMMARY.json","SUMMARY.csv"])
    return folder
