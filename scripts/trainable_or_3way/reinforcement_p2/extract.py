"""Bounded CPU prefetch and per-clip durable GPU inference; never skip decode errors."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .common import *
from .storage import ScoreStore
from .runtime import Decoder,Engine,validate_score

def score_modality(output,run,name,rows,modality,decoder_factory=Decoder,engine_factory=Engine):
    output=Path(output)
    folder=output/"scores"/name/modality
    identity=dict(run=digest(run),cohort=name,modality=modality)
    store=ScoreStore(folder,identity)
    engine=None
    try:
        for row in rows:
            require(stamp(row["media_path"])==row["stamp"],"Media changed before resume: "+row["sample_id"])
        existing,done=store.verify(rows,lambda v:validate_score(v,modality))
        if done:
            print("[P2 REUSE] "+name+"/"+modality+" complete",flush=True)
            return
        pending=[r for r in rows if r["sample_id"] not in existing]
        if not pending:
            store.finish(rows)
            return
        check_stop(output)
        decoder=decoder_factory(run)
        engine=engine_factory(run,modality)
        bar=Progress("P2 EXTRACT "+name+" "+modality,len(rows),folder,resumed=len(existing))
        count=len(existing)
        pool=ThreadPoolExecutor(max_workers=run["decode_workers"])
        futures={}
        def submit(i):
            if i<len(pending):futures[i]=pool.submit(decoder.decode,pending[i],modality)
        try:
            for i in range(min(run["prefetch"],len(pending))):submit(i)
            for i,row in enumerate(pending):
                check_stop(output)
                try:
                    decoded=futures.pop(i).result()
                    submit(i+run["prefetch"])
                    value=engine.predict(decoded)
                    validate_score(value,modality)
                    require(stamp(row["media_path"])==row["stamp"],"Media changed during prediction")
                    store.put(row["sample_id"],row["stamp"],value)
                except Exception as exc:
                    atomic_json(folder/"LAST_ERROR.json",dict(sample_id=row["sample_id"],error=str(exc)))
                    raise
                del decoded
                count+=1
                bar.update(count,detail="FP32; shared B/C decode" if modality=="video" else "FP32; batch1 whole waveform")
        finally:
            for future in futures.values():future.cancel()
            pool.shutdown(wait=True,cancel_futures=True)
        engine.close()
        engine=None
        store.finish(rows)
        bar.update(len(rows),force=True)
    finally:
        if engine is not None:engine.close()
        store.close()

def export_cohort(output,run,name,rows):
    """Join new B/C artifact scores with exact saved FAV emotion scores by ID."""
    output=Path(output)
    folder=output/"raw"/name
    identity=digest(dict(run=digest(run),cohort=name,stage="raw_export"))
    if completed(folder,identity):
        return {c:folder/(c+".npz") for c in ("B","C")}
    original=load_npz(run["cohorts"][name]["path"])
    require(original["sample_id"].tolist()==[r["sample_id"] for r in rows],"Export order mismatch")
    gathered={}
    for mod in ("video","audio"):
        store=ScoreStore(output/"scores"/name/mod,dict(run=digest(run),cohort=name,modality=mod),readonly=True)
        try:
            _,done=store.verify(rows,lambda v:validate_score(v,mod))
            require(done,"Artifact scores incomplete: "+name+"/"+mod)
            gathered[mod]=[store.get(r["sample_id"],r["stamp"]) for r in rows]
        finally:store.close()
    for i,r in enumerate(rows):
        require(gathered["video"][i]["media_sha256"]==gathered["audio"][i]["media_sha256"],
                "Video/audio were decoded from different media: "+r["sample_id"])
    files=[]
    for c in ("B","C"):
        raw=original["raw"].copy()
        raw[:,0]=[v[c+"_probability"] for v in gathered["video"]]
        raw[:,1]=[v["probability"] for v in gathered["audio"]]
        # Ev/Ea, labels, groups, IDs remain exactly those verified in P1.
        require(np.array_equal(raw[:,2:],original["raw"][:,2:],equal_nan=True),"Emotion reuse changed")
        data={k:v for k,v in original.items() if k!="raw"}
        data.update(raw=raw,video_logit=np.array([v[c+"_logit"] for v in gathered["video"]]),
                    audio_logits=np.array([v["logits"] for v in gathered["audio"]]))
        atomic_npz(folder/(c+".npz"),**data)
        files.append(c+".npz")
    atomic_json(folder/"REUSE.json",dict(emotion_source=run["cohorts"][name],emotion_scores_unchanged=True,
                 new_artifact_conditions=["B","C"],audio_shared=True))
    finish(folder,identity,files+["REUSE.json"])
    return {c:folder/(c+".npz") for c in ("B","C")}

def infer_cohort(output,run,name,rows):
    for mod in ("video","audio"):
        atomic_json(Path(output)/"STATUS.json",dict(status="RUNNING",stage="extract",cohort=name,modality=mod))
        score_modality(output,run,name,rows,mod)
    return export_cohort(output,run,name,rows)
