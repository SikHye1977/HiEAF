"""Condition-specific source-dev calibration and paired analysis with verified AVDF reuse."""
from pathlib import Path
import numpy as np
from .common import *
from scripts.trainable_or_3way.reinforcement.inputs import scores,METHODS
from scripts.trainable_or_3way.reinforcement.statistics import metric_table,bootstrap,paired_changes
from scripts.trainable_or_3way.reinforcement.analysis import csv_text,strata,figures

def calibrate_condition(raw,labels,*,dataset,split):
    from scripts.trainable_or.fusion import fit
    require(dataset=="FAV" and split=="dev","P2 fitting may use FAV source dev only")
    raw=np.asarray(raw,dtype=np.float64)
    y=np.asarray(labels,dtype=np.int64)
    require(raw.shape==(len(y),4) and set(y.tolist())=={0,1},"Invalid FAV dev calibration input")
    require(np.isfinite(raw[:,:2]).all(),"Artifact dev scores cannot be missing")
    valid=np.isfinite(raw[:,2:]).all(axis=1)
    require(set(y[valid].tolist())=={0,1},"Complete-emotion dev lacks both classes")
    a=1-(1-raw[:,0])*(1-raw[:,1])
    e=1-(1-raw[:,2])*(1-raw[:,3])
    return dict(artifact_affine=fit(a,None,y,split="dev"),
                trainable_final=fit(a[valid],e[valid],y[valid],split="dev"),
                M5=fit(a[valid],raw[valid,2],y[valid],split="dev"),
                M6=fit(a[valid],raw[valid,3],y[valid],split="dev"))

def calibrate(output,run,paths):
    output=Path(output)
    folder=output/"fusion"
    identity=digest(dict(run=digest(run),stage="source_dev_fit",inputs={c:sha(p) for c,p in paths.items()}))
    if completed(folder,identity):
        print("[P2 REUSE] Source-dev B/C fusion lock",flush=True)
        return read(folder/"FUSION.json")
    models={}
    for c in ("B","C"):
        d=load_npz(paths[c])
        models[c]=calibrate_condition(d["raw"],d["labels"][:,0],dataset="FAV",split="dev")
    atomic_json(folder/"FUSION.json",models)
    atomic_json(folder/"LOCKED_BEFORE_TEST.json",dict(run=digest(run),source="FAV",conditions=["B","C"],
                models_sha256=sha(folder/"FUSION.json"),dev_inputs={c:sha(p) for c,p in paths.items()},
                threshold=.5,target_training=False,independent_calibration=False,prior_targets_observed=True))
    finish(folder,identity,["FUSION.json","LOCKED_BEFORE_TEST.json"])
    return models

def evaluate_one(output,run,target,condition,data_path,model):
    output=Path(output)
    lock=read(output/"fusion/LOCKED_BEFORE_TEST.json")
    require(lock["run"]==digest(run) and lock["models_sha256"]==sha(output/"fusion/FUSION.json"),
            "Source fusion lock changed")
    require(read(output/"fusion/FUSION.json")[condition]==model,"Fusion model differs from locked source dev")
    d=load_npz(data_path)
    all_scores,valid=scores(d["raw"],model)
    key="FAV__"+condition+"__"+target+"__test"
    results=[]
    for cohort in ("full","common_emotion"):
        check_stop(output)
        folder=output/"evaluation"/key/cohort
        identity=digest(dict(run=digest(run),cohort=key,panel=cohort,raw_sha256=sha(data_path),
                             source_fusion_sha256=lock["models_sha256"],bootstrap=run["bootstrap"]))
        if completed(folder,identity):
            print("[P2 REUSE] "+key+"/"+cohort,flush=True)
            report=read(folder/"METRICS.json");boot=read(folder/"BOOTSTRAP.json")
        else:
            mask=np.ones(len(valid),bool) if cohort=="full" else valid
            require(mask.any(),"Empty analysis cohort")
            p={m:v[mask].copy() for m,v in all_scores.items()}
            if cohort=="full":
                for m in ("M2","M3"):p[m][~valid]=all_scores["M1"][~valid]
            y=d["labels"][mask,0]
            require(set(y.tolist())=={0,1} and len(set(d["cluster"][mask]))>1,"Invalid statistical cohort")
            report=dict(source="FAV",target=target,condition=condition,cohort=cohort,n=len(y),
                        groups=len(set(d["cluster"][mask])),missing_emotion=int((~valid[mask]).sum()),
                        metrics=metric_table(y,p),threshold=.5)
            atomic_json(folder/"METRICS.json",report)
            atomic_json(folder/"STRATA.json",strata(y,d["labels"][mask,1:],p))
            atomic_json(folder/"PAIRED_ERRORS.json",paired_changes(y,p,d["sample_id"][mask]))
            predictions=[dict(sample_id=str(s),cluster=str(g),clip_label=int(label),emotion_complete=bool(v),
                         **{m:float(p[m][i]) for m in METHODS})
                         for i,(s,g,label,v) in enumerate(zip(d["sample_id"][mask],d["cluster"][mask],y,valid[mask]))]
            atomic_text(folder/"predictions.csv",csv_text(predictions))
            figures(folder,y,p)
            b=run["bootstrap"]
            seed=(b["seed"]+int(digest(key+"/"+cohort)[:8],16))%(2**32)
            boot=bootstrap(y,p,d["cluster"][mask],folder,identity,replicates=b["replicates"],
                           seed=seed,max_attempts=b["max_attempts"],checkpoint_every=b["checkpoint_every"])
            boot["simultaneous_family"]="Four cross source/BN M4-M1 AUC comparisons; family complete only after P2 root completion"
            atomic_json(folder/"BOOTSTRAP.json",boot)
            require(boot["status"]=="COMPLETE","Insufficient valid bootstrap draws; earlier stages retained")
            finish(folder,identity,["METRICS.json","STRATA.json","PAIRED_ERRORS.json","predictions.csv",
                                   "curves.png","BOOTSTRAP.json","bootstrap_state.npz"])
        results.append(dict(key=key,folder=str(folder),metrics=report,bootstrap=boot,reused=False))
    return results

def reuse_avdf(output,run):
    """Copy only verified small result files; never refit AVDF models or rerun its inference."""
    parent=Path(run["parent"])
    result=[]
    for target in ("FAV","AVDF1M"):
        for c in ("B","C"):
            key="AVDF1M__"+c+"__"+target+"__test"
            for cohort in ("full","common_emotion"):
                old=parent/"p1"/key/cohort
                done=read(old/"COMPLETED.json")
                verify_files(old,done["files"])
                m,b=read(old/"METRICS.json"),read(old/"BOOTSTRAP.json")
                require((m["source"],m["target"],m["condition"],m["cohort"])==("AVDF1M",target,c,cohort)
                        and b["status"]=="COMPLETE"
                        and b["valid_replicates"]==run["bootstrap"]["replicates"],
                        "Wrong/incomplete AVDF1M analysis")
                folder=Path(output)/"reused_evaluation"/key/cohort
                identity=digest(dict(run=digest(run),stage="reuse",path=str(old),receipt_sha256=sha(old/"COMPLETED.json")))
                files=["METRICS.json","BOOTSTRAP.json","STRATA.json","PAIRED_ERRORS.json"]
                if not completed(folder,identity):
                    for name in files:
                        # Numeric results remain exact; the original bootstrap family note remains historical.
                        atomic_json(folder/name,read(old/name))
                    atomic_json(folder/"REUSE.json",dict(parent=str(old),completion_sha256=sha(old/"COMPLETED.json"),
                                file_hashes=done["files"],new_fitting=False,new_inference=False,
                                original_family_note_is_historical=True))
                    finish(folder,identity,files+["REUSE.json"])
                result.append(dict(key=key,folder=str(folder),metrics=m,bootstrap=b,reused=True))
    return result

def summarize(output,run,results):
    require(len(results)==16 and len({(v["key"],v["metrics"]["cohort"]) for v in results})==16,
            "Need eight source/target/BN conditions with full/common cohorts")
    rows=[];primary={}
    for r in results:
        m=r["metrics"];b=r["bootstrap"]
        require(b["status"]=="COMPLETE","Unfinished bootstrap job")
        for method,v in m["metrics"].items():
            rows.append(dict(source=m["source"],target=m["target"],condition=m["condition"],cohort=m["cohort"],
                 method=method,name=METHODS[method]+(" + missing fallback" if m["cohort"]=="full" and method in ("M2","M3") else ""),reused=r["reused"],
                 **{k:v[k] for k in ("n","real","fake","roc_auc","fake_f1","real_fpr","fake_recall",
                                     "balanced_accuracy","macro_f1","average_precision","all_fake_f1")}))
        if m["source"]!=m["target"] and m["cohort"]=="full":
            primary[r["key"]]=b["comparisons"]["M4-M1"]["roc_auc"]
    expected={s+"__"+c+"__"+t+"__test" for s,t in (("FAV","AVDF1M"),("AVDF1M","FAV")) for c in ("B","C")}
    require(set(primary)==expected,"Primary comparison family incomplete")
    output=Path(output)
    atomic_json(output/"SUMMARY.json",dict(rows=rows,primary_family_complete=True,primary_delta_auc=primary,
                 common_bn_precision_policy=True,identical_source_training_recipe=False,
                 limits=["Source-specific audio padding/batch and emotion head training differences remain",
                         "Prior target observation and source-dev reuse", "Single training seed; bootstrap is conditional",
                         "No new proof of cross-dataset identity independence or temporal causality"]))
    atomic_text(output/"SUMMARY.csv",csv_text(rows))
    lines=["# P2 FAV / AVDF1M B/C evaluation","",
           "FAV artifact scores: new FP32 B/C inference, same input/weights, condition-specific FAV dev fitting.",
           "AVDF1M: verified P1 B/C results reused without new inference or refitting.",
           "B=saved X3D BN; C=single-video BN without updating saved buffers. Dropout off. AASIST saved BN FP32.",
           "FAV audio minimum 284672, batch1; AVDF1M minimum 510976, short batch4/long batch1. Whole long waveform retained.",
           "B/C policy is aligned; source training recipes/padding are not claimed identical.",
           "Threshold .5. Full M2-M6 use the same artifact-affine fallback when either emotion is missing.",
           "Only source dev fits coefficients. Prior test exposure/dev reuse remain.",
           "Primary family: four full-cohort cross M4-M1 AUC comparisons; 95% and 98.75% individual intervals.",
           "Bootstrap does not include model training/selection variability; FAV fused Real test n=60.",
           "Temporal order effects and cross-dataset identity overlap remain separate follow-up work.","",
           "| Source | Target | BN | Cohort | Method | AUC % | Fake F1 % | Real FPR % | Fake recall % |",
           "|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| "+" | ".join([r["source"],r["target"],r["condition"],r["cohort"],r["name"]]+
                      [f"{100*r[k]:.4f}" for k in ("roc_auc","fake_f1","real_fpr","fake_recall")])+" |")
    lines += ["","## Primary M4-M1 AUC differences (percentage points)","",
              "| Direction/condition | Delta | 95% CI | 98.75% CI |","|---|---:|---|---|"]
    for k,v in sorted(primary.items()):
        fmt=lambda vals:"["+", ".join(f"{100*x:+.4f}" for x in vals)+"]"
        lines.append(f"| {k} | {100*v['delta']:+.4f} | {fmt(v['ci95'])} | {fmt(v['ci98_75'])} |")
    atomic_text(output/"REPORT.md","\n".join(lines)+"\n")
