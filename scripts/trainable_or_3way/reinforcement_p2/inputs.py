"""Exact P1 cohorts and current Windows files; never resplit or change labels."""
from pathlib import Path
import numpy as np
from .common import *

def compare_records(data,records,dataset,split):
    by={r.sample_id:r for r in records}
    require(len(by)==len(records)==len(data["sample_id"]) and set(by)==set(data["sample_id"]),
            "Current loader and P1 cohort differ: "+dataset+"/"+split)
    result=[]
    for i,sid in enumerate(data["sample_id"]):
        r=by[str(sid)]
        labels=[-1 if v is None else v for v in (r.clip_label,r.video_label,r.audio_label)]
        require(r.dataset==dataset and r.split==split and labels==data["labels"][i].tolist()
                and r.group_id==str(data["cluster"][i]),"P1/frozen metadata mismatch: "+str(sid))
        result.append(dict(sample_id=str(sid),dataset=dataset,split=split,group_id=r.group_id,
                           labels=labels,media_path=str(r.media_path.resolve()),stamp=stamp(r.media_path)))
    return result

def checkpoint_config(path,expected,modality):
    import torch
    require(Path(path).is_file() and sha(path)==expected,"FAV checkpoint changed: "+str(path))
    # These exact user-owned historical checkpoints were already hash-bound in P0.
    state=torch.load(path,map_location="cpu",weights_only=False)
    require(state.get("format") in ("x3a_ddp_v1","x3a_single_v1"),"Wrong artifact checkpoint format")
    cfg=state["provenance"]["config"]
    require(cfg["data"]["dataset"]=="FAV","Target-trained artifact weights are forbidden")
    if modality=="video":
        v=cfg["video"]
        require(cfg["model"]["name"]=="x3d_m" and v["num_frames"]==128 and v["crop_size"]==256
                and v["resize_short_side"]==256 and v["normalization"]==dict(mean=[.45]*3,std=[.225]*3),
                "FAV video recipe changed")
    else:
        require(cfg["audio"]["sample_rate"]==16000 and cfg["audio"]["target_num_samples"]==284672
                and cfg["model"]["fake_class_index"]==1
                and cfg["model"]["class_mapping"]==dict(real=0,fake=1),"FAV audio recipe/mapping changed")
    return cfg

def check_peer_protocol(inventory):
    source=inventory["source_runs"]["AVDF1M"]
    require(source["source"]=="AVDF1M" and source["conditions"]==["B","C"]
            and not source["target_training"],"Unexpected AVDF1M source run")
    protocols=inventory["protocols"]
    for name in ("dev_AVDF1M","test_FAV","test_AVDF1M"):
        p=protocols[name]
        audio=p["audio_recipe"];env=p["runtime"]
        require(audio["minimum_samples"]==510976 and audio["sample_rate"]==16000
                and audio["fp32"] and audio["saved_batchnorm"] and not audio["truncation"]
                and not audio["windowing"] and not audio["normalization"]
                and not env["matmul_tf32"] and not env["cudnn_tf32"],
                "Existing AVDF1M B/C protocol is not compatible")
    bundle=protocols["dev_AVDF1M"]["bundle"]
    # P0's generic path matcher missed x3d_bn_audit_*/weights; verify the exact known path.
    for mod,rel in (("x3d","x3d_bn_audit_20261102_001/weights/best_state_dict.pth"),
                    ("aasist","aasist/weights/best_state_dict.pth")):
        p=Path(bundle["path"])/rel
        require(p.is_file() and sha(p)==bundle[mod],"Referenced AVDF1M weight missing/changed: "+str(p))
    return {k:protocols[k] for k in ("dev_AVDF1M","test_FAV","test_AVDF1M")}

def prepare(args):
    from src.dataloader.metadata import load_records
    from .runtime import configure
    parent=resolve(args.parent)
    proof=verify_parent(parent)
    inventory=read(parent/"p0/INVENTORY.json")
    peer=check_peer_protocol(inventory)
    p1run=read(parent/"RUN.json")
    registry=resolve(p1run["config"]["registry"])
    local=resolve(args.local_paths)
    eligibility=ROOT/"reports/phase0_v1/eligibility_v4.jsonl"
    ffmpeg=resolve(args.ffmpeg)
    require(ffmpeg.is_file() and local.is_file() and eligibility.is_file(),"Local paths/FFmpeg/eligibility missing")
    require(args.decode_workers>=1 and 1<=args.prefetch<=4 and args.threads>=1,"Invalid worker/prefetch/thread count")
    assets={}
    for mod in ("video","audio"):
        entry=next(p for p in inventory["protocols"]["checkpoint_inventory"] if p["source"]=="FAV" and p["modality"]==mod)
        path=resolve(entry["path"])
        cfg=checkpoint_config(path,entry["expected_sha256"],mod)
        assets[mod]=dict(path=str(path),sha256=entry["expected_sha256"],config=cfg)
    env=configure(args.device,args.threads)
    cohorts={}; sources={}
    for name in COHORTS:
        dataset,split=name.split("/")
        key="FAV__historical__"+dataset+"__"+split
        c=inventory["cohorts"][key]
        p=parent/"p0"/c["normalized_file"]
        require(sha(p)==c["normalized_sha256"],"P1 normalized input changed")
        data=load_npz(p)
        print("[P2 METADATA] "+name,flush=True)
        records=load_records(dataset,split,task="paired",registry=registry,eligibility_path=eligibility,
                             local_paths=local,sample_ids=data["sample_id"].tolist())
        cohorts[name]=compare_records(data,records,dataset,split)
        sources[name]=dict(path=str(p),sha256=sha(p),n=len(records),cohort_sha256=digest(cohorts[name]))
    for target in ("FAV","AVDF1M"):
        fav=load_npz(sources[target+"/test"]["path"])
        for c in ("B","C"):
            peer_data=load_npz(parent/"p0"/inventory["cohorts"]["AVDF1M__"+c+"__"+target+"__test"]["normalized_file"])
            i,j=np.argsort(fav["sample_id"]),np.argsort(peer_data["sample_id"])
            require(np.array_equal(fav["sample_id"][i],peer_data["sample_id"][j])
                    and np.array_equal(fav["labels"][i],peer_data["labels"][j])
                    and np.array_equal(fav["cluster"][i],peer_data["cluster"][j]),
                    "Source directions use different target cohorts")
    spec=read(parent/"p0/ANALYSIS_SPEC.json")
    require(spec["threshold"]==.5 and spec["regularization"]==.01 and spec["epsilon"]==1e-6,"Unexpected P1 fusion spec")
    run=dict(format="reinforcement_p2_bc_v1",source="FAV",targets=["FAV","AVDF1M"],conditions=["B","C"],
             parent=str(parent),parent_proof=proof,assets=assets,cohorts=sources,
             implementation=file_set(),runtime=env,packages=package_versions(),
             metadata=dict(registry=str(registry),registry_sha256=sha(registry),local_paths=str(local),
                           local_paths_sha256=sha(local),eligibility_sha256=sha(eligibility)),
             ffmpeg=dict(path=str(ffmpeg),sha256=sha(ffmpeg)),video=VIDEO,audio=AUDIO,bn=BN,
             decode_workers=args.decode_workers,prefetch=args.prefetch,bootstrap=spec["bootstrap"],
             fusion=dict(epsilon=1e-6,regularization=.01,threshold=.5,fit="FAV dev separately for B/C",
                         missing="any emotion missing -> same condition artifact-affine"),
             avdf1m_reused_protocol=peer,prior_targets_observed=True,independent_calibration=False,
             encoder_training=False,emotion_inference=False,current_windows_data_policy=True,
             source_audio_difference="FAV min 284672 batch1; AVDF1M min 510976, short batch4/long batch1")
    return run,cohorts

def selected_preflight(rows):
    selected=[]
    for pair in ((0,0),(0,1),(1,0),(1,1),(-1,-1)):
        group=[r for r in rows if tuple(r["labels"][1:])==pair]
        selected.extend(sorted(group,key=lambda r:digest([1701,r["sample_id"]]))[:1])
    require(selected,"No bounded preflight samples")
    return selected

def preflight(output,run,cohorts):
    from .runtime import Decoder
    output=Path(output)
    identity=digest(dict(run=digest(run),stage="preflight"))
    folder=output/"preflight"
    if completed(folder,identity):
        print("[P2 PREFLIGHT] Reusing verified decode checks; no model forward",flush=True)
        return
    decoder=Decoder(run)
    checks=[]
    selected=[(name,r) for name,rows in cohorts.items() for r in selected_preflight(rows)]
    bar=Progress("P2 bounded decode preflight",len(selected),folder)
    for i,(name,r) in enumerate(selected):
        check_stop(output)
        v,vm=decoder.decode(r,"video")
        a,am=decoder.decode(r,"audio")
        checks.append(dict(cohort=name,sample_id=r["sample_id"],video_shape=list(v.shape),audio_samples=len(a),
                           video=vm,audio=am))
        del v,a
        bar.update(i+1)
    atomic_json(folder/"PREFLIGHT.json",dict(status="PASS",model_forward=False,checks=checks,
                policy="Current Windows availability/nonempty, IDs/splits/labels and bounded decoding; full production decode remains required"))
    finish(folder,identity,["PREFLIGHT.json"])
