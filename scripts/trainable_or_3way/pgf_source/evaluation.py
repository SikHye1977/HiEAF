"""Resumable B/C scores; source-only final fusion and independently checked metrics."""
from concurrent.futures import ThreadPoolExecutor
import csv,io
from .common import *
from .runtime import decode,row_for,Engine,validate_score
from scripts.trainable_or_3way.reinforcement_p2.storage import ScoreStore
from scripts.trainable_or_3way.reinforcement_temporal.analysis import calibrate,fused,metrics_nullable
from scripts.trainable_or_3way.reinforcement.statistics import metric_table

def score_artifact(output,run,rows,stamps,mod,checkpoint,engine_factory=Engine,decode_fn=decode,stop_after=None):
    key=rows[0].dataset+"/"+rows[0].split
    folder=Path(output)/"artifact_scores"/key/mod
    identity=dict(run=digest(run),cohort=key,modality=mod,model=sha(checkpoint))
    planned=[row_for(r,stamps) for r in rows]
    engine=None
    with writer_lock(folder):
        store=ScoreStore(folder,identity)
        try:
            for r in rows:require(stamp(r)==stamps[r.sample_id],"Media changed before score resume")
            existing,done=store.verify(planned,lambda value:validate_score(value,mod))
            if not done:
                pending=[r for r in planned if r["sample_id"] not in existing]
                if pending:
                    engine=engine_factory(checkpoint,mod,run["config"]["device"],identity["model"])
                    bar=Progress("SCORE "+key+" "+mod,len(rows),folder,resumed=len(existing))
                    with ThreadPoolExecutor(max_workers=run["decode_workers"]) as pool:
                        futures={}
                        def submit(i):
                            if i<len(pending):futures[i]=pool.submit(decode_fn,pending[i],mod,run["ffmpeg"]["path"])
                        try:
                            for i in range(min(run["prefetch"],len(pending))):submit(i)
                            for i,row in enumerate(pending):
                                decoded=futures.pop(i).result();submit(i+run["prefetch"])
                                value=engine.predict(decoded);validate_score(value,mod)
                                from .runtime import stamp_path
                                require(stamp_path(row["media_path"])==row["stamp"],"Media changed during prediction")
                                store.put(row["sample_id"],row["stamp"],value)
                                del decoded;bar.update(len(existing)+i+1,"FP32 shared B/C" if mod=="video" else "whole audio; batch1")
                                if stop_after is not None and i+1>=stop_after:raise KeyboardInterrupt()
                        except Exception as error:
                            atomic_json(folder/"LAST_ERROR.json",dict(error=str(error),sample_id=row["sample_id"] if "row" in locals() else None))
                            raise
                        finally:
                            for f in futures.values():f.cancel()
                    engine.close();engine=None
                store.finish(planned)
            else:status("REUSE ARTIFACT SCORES",key+" "+mod)
            return [store.get(r["sample_id"],r["stamp"]) for r in planned]
        finally:
            if engine is not None:engine.close()
            store.close()

def raw_arrays(rows,video,audio,emotion):
    require(len(video)==len(audio)==len(rows),"Score coverage differs")
    ids=np.array([r.sample_id for r in rows])
    for mod in ("video","audio"):
        require(np.array_equal(emotion[mod]["sample_id"],ids),"Emotion score ID alignment differs")
    ev,ea=[emotion[m]["probability"] for m in ("video","audio")]
    result={}
    for i in range(len(rows)):require(video[i]["media_sha256"]==audio[i]["media_sha256"],"Modalities used different files")
    for c in ("B","C"):
        result[c]=np.column_stack([[v[c+"_probability"] for v in video],[v["probability"] for v in audio],ev,ea])
    return result

def fit_source(rows,raw,folder,binding):
    require(rows and all(r.dataset=="PGF" and r.split=="dev" for r in rows),"Fit only PGF dev")
    identity=digest(binding);folder=Path(folder)
    if completed(folder,identity):return read(folder/"FUSION.json")
    y=labels(rows)
    model={c:calibrate(raw[c],raw[c][:,2:],y,source="PGF",dataset="PGF",split="dev") for c in ("B","C")}
    atomic_json(folder/"FUSION.json",model)
    atomic_json(folder/"LOCKED_BEFORE_TEST.json",dict(identity=identity,binding=binding,source="PGF",
        fusion_sha256=sha(folder/"FUSION.json"),threshold=.5,target_training=False,independent_calibration=False))
    finish(folder,identity,["FUSION.json","LOCKED_BEFORE_TEST.json"])
    return model

def evaluate(rows,raw,models,folder,binding):
    require(all(r.split=="test" for r in rows),"Evaluation must use test")
    folder=Path(folder);identity=digest(binding)
    if completed(folder,identity):return read(folder/"METRICS.json")
    reports={};arrays=cohort(rows)
    for c in ("B","C"):
        final,valid,base=fused(raw[c],raw[c][:,2:],models[c])
        scores=dict(baseline=base["M0"],artifact_affine=base["M1"],trainable_final=final)
        y=labels(rows)
        branches={name:metrics_nullable(labels(rows,mod)[mask],{name:raw[c][mask,i]})[name]
            for i,(name,mod) in enumerate((("Av","video"),("Aa","audio"),("Ev","video"),("Ea","audio")))
            for mask in [(labels(rows,mod)>=0)&np.isfinite(raw[c][:,i])]}
        reports[c]=dict(source="PGF",target=rows[0].dataset,condition=c,split="test",threshold=.5,
            full=metric_table(y,scores),common_emotion=metrics_nullable(y[valid],{k:p[valid] for k,p in scores.items()}),
            branches=branches,coverage=dict(planned=len(rows),evaluated=len(rows),common_emotion=int(valid.sum()),
                   fallback=int((~valid).sum()),video_emotion_missing=int((~np.isfinite(raw[c][:,2])).sum()),
                   audio_emotion_missing=int((~np.isfinite(raw[c][:,3])).sum())),
            target_training=False,zero_shot=rows[0].dataset!="PGF")
        arrays["raw_"+c]=raw[c]
        for method,p in scores.items():arrays[c+"__"+method]=p
    atomic_npz(folder/"predictions.npz",**arrays)
    atomic_json(folder/"METRICS.json",reports)
    saved=load_npz(folder/"predictions.npz")
    for c in ("B","C"):
        values=metric_table(saved["labels"][:,0],{m:saved[c+"__"+m] for m in ("baseline","artifact_affine","trainable_final")})
        require(values==reports[c]["full"],"Saved metric verification failed")
    atomic_json(folder/"VERIFIED.json",dict(status="PASS",n=len(rows),metrics_recomputed=True))
    finish(folder,identity,["predictions.npz","METRICS.json","VERIFIED.json"])
    return reports

def summary(output,reports):
    rows=[]
    for target in TARGETS:
        for condition in ("B","C"):
            for method,m in reports[target][condition]["full"].items():
                rows.append(dict(source="PGF",target=target,condition=condition,method=method,**m))
    atomic_json(Path(output)/"SUMMARY.json",dict(rows=rows,training_source="PGF",targets=list(TARGETS),
          statistical_inference="point metrics only; bootstrap/temporal ablations are a separate extension"))
    fields=["source","target","condition","method","n","real","fake","roc_auc","fake_f1","real_fpr","fake_recall"]
    s=io.StringIO(newline="");w=csv.DictWriter(s,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    atomic_text(Path(output)/"SUMMARY.csv",s.getvalue())
    lines=["# PGF source: Baseline and Trainable Final Fusion","",
           "Source train/dev only; fixed threshold 0.5; B=saved X3D BN; C=per-video X3D BN.",
           "Single seed; source dev reused for selection/calibration. Not an exact 4-GPU reproduction.","",
           "| Source → Target | BN | Model | ROC-AUC % | Fake F1 % | Real FPR % | Fake Recall % |",
           "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| PGF → "+r["target"]+" | "+r["condition"]+" | "+r["method"]+" | "+
           " | ".join("NA" if r[k] is None else f"{100*r[k]:.4f}" for k in ("roc_auc","fake_f1","real_fpr","fake_recall"))+" |")
    atomic_text(Path(output)/"RESULTS.md","\n".join(lines)+"\n")
