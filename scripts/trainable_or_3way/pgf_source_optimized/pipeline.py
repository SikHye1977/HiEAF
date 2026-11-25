"""Train only PGF, lock on its dev, then evaluate all three frozen test cohorts."""
from ..pgf_source.common import *
from . import artifact_training as art_train
from ..pgf_source.features import ensure_cache,open_cache
from ..pgf_source.evaluation import score_artifact,raw_arrays,fit_source,evaluate,summary
from scripts.trainable_or_3way.models import artifact
from scripts.trainable_or_3way_matched.memory import enable_audio_checkpointing
from scripts.trainable_or_3way.reinforcement_temporal import training as head_train

def verify_output(output):
    output=Path(output)
    run=read(output/"RUN.json");require(run["format"]==FORMAT,"Wrong run type")
    done=read(output/"COMPLETED.json");require(done["identity"]==digest(run),"Wrong completion identity")
    verify_files(output,done["files"])
    for key,entry in read(output/"TOKEN_CACHES.json").items():
        p=Path(entry["path"])
        require(sha(p/"COMPLETED.json")==entry["receipt_sha256"],"Feature receipt changed: "+key)
    require(read(output/"STATUS.json")["status"]=="COMPLETE","Pipeline incomplete")
    models=read(output/"fusion/FUSION.json")
    for target in TARGETS:
        d=load_npz(output/"evaluation"/target/"predictions.npz")
        require(digest({k:d[k].tolist() for k in ("sample_id","labels","cluster")})==run["cohorts"][target+"/test"]["cohort_sha256"],
                "Saved IDs/labels/groups differ")
        from ..pgf_source.evaluation import fused,metric_table
        reports=read(output/"evaluation"/target/"METRICS.json")
        for c in ("B","C"):
            raw=d["raw_"+c];p,valid,base=fused(raw,raw[:,2:],models[c])
            for name,values in dict(baseline=base["M0"],artifact_affine=base["M1"],trainable_final=p).items():
                require(np.array_equal(values,d[c+"__"+name]),"Saved fusion differs")
            require(metric_table(d["labels"][:,0],{m:d[c+"__"+m] for m in ("baseline","artifact_affine","trainable_final")})
                    ==reports[c]["full"],"Saved metrics differ")
    status("VERIFY PASS","PGF -> PGF/FAV/AVDF1M; locked fusion, models, cohorts and committed results")

def execute(args,run,cohorts,stamps,proofs):
    out=resolve(args.output);identity=digest(run)
    immutable(out/"RUN.json",run)
    if (out/"COMPLETED.json").exists():verify_output(out);return
    immutable(out/"PGF_LABEL_EVIDENCE.json",proofs)
    stages=[0];total=34
    def stage(name,detail):
        stages[0]+=1;step(out,name,stages[0],total,detail)
    cfg=run["config"];model_paths={};model_ids={};cache_paths={};cache_receipts={}
    source={s:cohorts["PGF/"+s] for s in ("train","dev","test")}
    # Target descriptors never enter model training identity.
    train_binding=dict(config=cfg,assets=run["assets"],code=run["implementation"],
        runtime=run["runtime"],source={s:run["cohorts"]["PGF/"+s] for s in ("train","dev")})
    for mod in ("video","audio"):
        folder=out/"models/artifact"/mod
        mid=digest(dict(training=train_binding,kind="artifact",modality=mod))
        stage("train_artifact_"+mod,folder/"progress.json")
        if (folder/"COMPLETED.json").exists():
            rec=read(folder/"COMPLETED.json")
            require(rec["identity"]==mid and rec["sha256"]==sha(folder/"best.pt"),"Completed artifact changed")
            status("REUSE ARTIFACT",mod)
        else:
            seed_all(cfg["seed"]+(mod=="audio"))
            model=artifact(mod)
            if mod=="audio":enable_audio_checkpointing(model)
            train=art_train.Task(source["train"],mod,"artifact",cfg["device"],stamps,input_config=run["input_optimization"])
            dev=art_train.Task(source["dev"],mod,"artifact",cfg["device"],stamps,input_config=run["input_optimization"])
            art_train.fit(model,train,dev,folder,cfg,mid)
            del model,train,dev;free_gpu()
        model_paths["artifact/"+mod]=folder/"best.pt";model_ids["artifact/"+mod]=mid
    def cache_for(rows,mod):
        key=rows[0].dataset+"/"+rows[0].split+"/"+mod
        if key not in cache_paths:
            cache_paths[key]=ensure_cache(out,run,rows,mod,stamps)
            cache_receipts[key]=dict(path=str(cache_paths[key]),receipt_sha256=sha(cache_paths[key]/"COMPLETED.json"))
            atomic_json(out/"TOKEN_CACHES.json",cache_receipts)
        return cache_paths[key]
    for mod in ("video","audio"):
        for split in ("train","dev"):
            stage("tokens_PGF_"+split+"_"+mod,out/"STATUS.json")
            cache_for(source[split],mod)
    for mod in ("video","audio"):
        folder=out/"models/emotion"/mod
        stage("train_emotion_"+mod,folder/"progress.json")
        mid=digest(dict(training=train_binding,kind="emotion",variant="T2",modality=mod,
             caches={s:cache_receipts["PGF/"+s+"/"+mod] for s in ("train","dev")}))
        if not completed(folder,mid):
            tr=open_cache(cache_for(source["train"],mod),source["train"],mod)
            dv=open_cache(cache_for(source["dev"],mod),source["dev"],mod)
            try:
                head_train.fit(head_train.Task(tr),head_train.Task(dv),"T2",cfg["seed"],"PGF",folder,mid,
                               device=cfg["device"],config=cfg["emotion_head"])
            finally:tr.close();dv.close();free_gpu()
        model_paths["emotion/"+mod]=folder/"best.pt";model_ids["emotion/"+mod]=mid
    selected=dict(models={k:dict(path=str(p),sha256=sha(p),identity=model_ids[k]) for k,p in model_paths.items()},
                  source="PGF",selection="source dev ROC-AUC; earliest tie")
    immutable(out/"MODELS_SELECTED.json",selected)
    def scores_for(rows):
        key=rows[0].dataset+"/"+rows[0].split;arts={};emotions={}
        for mod in ("video","audio"):
            stage("score_artifact_"+key+"_"+mod,out/"artifact_scores"/key/mod/"progress.json")
            arts[mod]=score_artifact(out,run,rows,stamps,mod,model_paths["artifact/"+mod])
        for mod in ("video","audio"):
            folder=out/"emotion_scores"/key/mod
            stage("score_emotion_"+key+"_"+mod,folder/"progress.json")
            cache=open_cache(cache_for(rows,mod),rows,mod)
            try:
                emotions[mod]=head_train.predict(cache,model_paths["emotion/"+mod],"T2","ordered",folder,
                            device=cfg["device"],batch_size=run["predict_batch_size"])
            finally:cache.close();free_gpu()
        return raw_arrays(rows,arts["video"],arts["audio"],emotions)
    dev_raw=scores_for(source["dev"])
    stage("lock_PGF_dev_fusion",out/"fusion")
    binding=dict(run=identity,selection_sha256=sha(out/"MODELS_SELECTED.json"),
                 dev=run["cohorts"]["PGF/dev"])
    # Missing emotion may be NaN; use array hashes instead of JSON in the binding.
    from scripts.trainable_or_3way.reinforcement_temporal.analysis import hash_array
    binding["dev_scores"]={c:hash_array(dev_raw[c]) for c in ("B","C")}
    models=fit_source(source["dev"],dev_raw,out/"fusion",binding)
    fusion_sha=sha(out/"fusion/FUSION.json");reports={}
    for target in TARGETS:
        rows=cohorts[target+"/test"]
        for mod in ("video","audio"):
            stage("tokens_"+target+"_test_"+mod,out/"STATUS.json")
            cache_for(rows,mod)
        require(sha(out/"fusion/FUSION.json")==fusion_sha,"Locked fusion changed")
        raw=scores_for(rows)
        stage("evaluate_"+target,out/"evaluation"/target)
        from scripts.trainable_or_3way.reinforcement_temporal.analysis import hash_array
        reports[target]=evaluate(rows,raw,models,out/"evaluation"/target,
             dict(run=identity,fusion=fusion_sha,target=target,raw={c:hash_array(raw[c]) for c in ("B","C")}))
        status("RESULT",target+" "+str({c:reports[target][c]["full"]["trainable_final"] for c in ("B","C")}))
    require(stages[0]==total,"Internal stage count mismatch")
    summary(out,reports)
    # Referenced historical token receipts are read-only; assert unchanged at the end.
    for entry in run["reuse"].values():
        require(sha(Path(entry["path"])/"COMPLETED.json")==entry["receipt_sha256"],"Historical token receipt changed")
    atomic_json(out/"STATUS.json",dict(status="COMPLETE",source="PGF",targets=list(TARGETS),
        stages=total,conditions=6,head_seeds=[cfg["seed"]],ablation_variants=["T2"]))
    files=["CONTINUATION.json","MIGRATION.json","migration_seed.pt","RUN.json","PGF_LABEL_EVIDENCE.json","MODELS_SELECTED.json","TOKEN_CACHES.json",
           "SUMMARY.json","SUMMARY.csv","RESULTS.md","STATUS.json"]
    files.extend(p.relative_to(out).as_posix() for p in (out/"parent_snapshot").rglob("*") if p.is_file())
    for p in sorted(out.rglob("COMPLETED.json")):
        if p.parent==out:continue
        rec=read(p);files.append(p.relative_to(out).as_posix())
        if "files" in rec:files.extend((p.parent/name).relative_to(out).as_posix() for name in rec["files"])
        elif "sha256" in rec:
            files.extend((p.parent/name).relative_to(out).as_posix() for name in ("best.pt","history.json"))
    finish(out,identity,sorted(set(files)))
    verify_output(out);status("PIPELINE COMPLETE",str(out))
