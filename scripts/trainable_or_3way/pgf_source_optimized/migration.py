"""Explicit parent-to-child continuation; checkpoint content changes only identity."""
import copy
import shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.trainable_or_3way.pgf_source.common import *
from scripts.trainable_or_3way.pgf_source import inputs as old_inputs
from scripts.trainable_or_3way.pgf_source.common import implementation as old_implementation
from scripts.trainable_or_3way.reinforcement.common import _write_atomic

BASE = Path(__file__).resolve().parent
DEFAULT_PARENT = ROOT/"reports/trainable_or_pgf_v1/source_PGF/seed_42"
DEFAULT_OUTPUT = ROOT/"reports/trainable_or_pgf_optimized_v1/source_PGF/seed_42"
DEFAULT_FRAMES = ROOT/"reports/input_cache/pgf_uint8_v1"
VERSION = "pgf_input_optimization_continuation_v1"


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b)
    if isinstance(a, dict):
        return isinstance(b,dict) and a.keys() == b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a == b


def model_identity(run):
    binding = dict(config=run["config"],assets=run["assets"],code=run["implementation"],runtime=run["runtime"],
                   source={s:run["cohorts"]["PGF/"+s] for s in ("train","dev")})
    return digest(dict(training=binding,kind="artifact",modality="video"))


def code():
    value=old_implementation()
    for p in BASE.glob("*.py"):
        if not p.name.startswith("test_"):value[p.relative_to(ROOT).as_posix()]=sha(p)
    return value


def paths(parent, output, frame_cache):
    parent,output,frame_cache=map(resolve,(parent,output,frame_cache))
    safe_output(output,[parent,frame_cache])
    require(output.is_relative_to(ROOT/"reports") and output != ROOT/"reports", "Use dedicated output under reports")
    require(frame_cache.is_relative_to(ROOT/"reports/input_cache") and frame_cache != ROOT/"reports/input_cache",
            "Use a dedicated SSD frame cache under reports/input_cache")
    require(parent.is_dir() and (parent/"RUN.json").is_file(), "Missing parent run")
    return parent,output,frame_cache


def original_args(parent, run):
    c=run["config"]
    return SimpleNamespace(output=parent,cache_root=Path(run["cache_root"]),device=c["device"],seed=c["seed"],
        artifact_epochs=c["artifact_epochs"],head_epochs=c["emotion_head"]["epochs"],
        head_batch_size=c["emotion_head"]["batch_size"],checkpoint_steps=c["checkpoint_steps"],
        frame_batch=run["frame_batch"],predict_batch_size=run["predict_batch_size"],threads=c["threads"],
        decode_samples=run["media_validation"]["decode_samples"],decode_workers=run["decode_workers"],prefetch=run["prefetch"])


def proposed(parent, output, frame_cache, workers=2, depth=2):
    parent,output,frame_cache=paths(parent,output,frame_cache)
    require(1<=workers<=2 and workers<=depth<=4, "Use 1-2 workers and bounded prefetch 1-4")
    prior=read(parent/"RUN.json")
    require(prior["format"]==FORMAT and prior["config"]["source"]=="PGF", "Wrong parent format/source")
    require(prior["implementation"]==old_implementation(), "Parent implementation changed")
    require(read(parent/"STATUS.json")["status"]=="INTERRUPTED", "Parent must be paused at a saved boundary")
    require(not (parent/"COMPLETED.json").exists(), "Parent already complete")
    source=parent/"models/artifact/video/last.pt"
    require(source.is_file(), "Missing resumable video checkpoint")
    require(not (parent/"models/artifact/video/COMPLETED.json").exists() and
            not (parent/"models/artifact/audio").exists(), "This migration is for paused X3D before audio starts")
    files={p.relative_to(parent).as_posix():sha(p) for p in parent.rglob("*") if p.is_file() and p.name!=".writer.lock"}
    optimization=dict(version="uint8_fifo_v1",frame_cache=str(frame_cache),workers=workers,depth=depth,reserve_gib=20,
        cache_selection="PGF real train + all dev; frame index metadata for fake train",
        group_replay="retain at most one accumulation group of uint8 inputs; retry consumes no new FIFO items",
        normalization="identical CPU FP32 operations",sample_order="unchanged",bn="unchanged")
    continuation=dict(format=VERSION,parent=str(parent),parent_run_sha256=sha(parent/"RUN.json"),
                      parent_files=files,parent_checkpoint_sha256=files["models/artifact/video/last.pt"],
                      input_optimization=optimization,implementation=code())
    run=copy.deepcopy(prior)
    run.update(implementation=code(),input_optimization=optimization,continuation=continuation)
    return run,continuation


def validate_current(parent, run):
    args=original_args(parent,read(Path(parent)/"RUN.json"))
    setup(args.device,args.threads)
    current,cohorts,stamps,proofs=old_inputs.prepare(args)
    require(current["config"]==run["config"], "Recipe changed")
    for key in current:
        if key!="implementation":require(current[key]==run[key],"Original run field changed: "+key)
    cached_count=sum(r.video_label==0 for r in cohorts["PGF/train"])+len(cohorts["PGF/dev"])
    frame_root=Path(run["input_optimization"]["frame_cache"])
    existing=frame_root
    while not existing.exists():existing=existing.parent
    occupied=sum(p.stat().st_size for p in frame_root.rglob("*.npy")) if frame_root.exists() else 0
    needed=max(0,cached_count*128*256*256*3-occupied)+run["input_optimization"]["reserve_gib"]*1024**3
    require(shutil.disk_usage(existing).free > needed,"Insufficient SSD space for planned RGB cache and reserve")
    status("INPUT CACHE",f"{cached_count} hot clips, ~{cached_count*24/1024:.2f} GiB uint8; lazy fill, 20 GiB reserve")
    return cohorts,stamps,proofs


def copy_exact(source, target, expected):
    source,target=Path(source),Path(target)
    require(sha(source)==expected,"Parent file changed: "+str(source))
    if target.exists():require(sha(target)==expected,"Snapshot changed: "+str(target))
    else:
        def write(f):
            with source.open("rb") as stream:shutil.copyfileobj(stream,f,4*1024*1024)
        _write_atomic(target,write)
    require(sha(source)==sha(target)==expected,"Copy integrity mismatch")


def migrate(parent, output, run, continuation):
    parent,output=Path(parent),Path(output)
    allowed={".writer.lock","RUN.json","CONTINUATION.json","parent_snapshot","models","migration_seed.pt","MIGRATION.json","STATUS.json"}
    if (output/"MIGRATION.json").exists():
        verify_migration(output,run)
        return read(output/"MIGRATION.json")
    require(not output.exists() or set(p.name for p in output.iterdir()) <= allowed,"Output has unrelated files")
    require(not output.exists() or not any(p.name not in (".writer.lock",) for p in output.iterdir()) or
            (output/"CONTINUATION.json").exists(),"Unrecognized partial migration")
    immutable(output/"CONTINUATION.json",continuation)
    immutable(output/"RUN.json",run)
    for name,checksum in continuation["parent_files"].items():
        copy_exact(parent/name,output/"parent_snapshot"/name,checksum)
    source=output/"parent_snapshot/models/artifact/video/last.pt"
    saved=torch.load(source,map_location="cpu",weights_only=True)
    require(saved["identity"]==model_identity(read(parent/"RUN.json")),"Parent checkpoint identity mismatch")
    state=saved["state"]
    require(state["amp_retries_at_cursor"]==0 and state["cursor"]%8==0,"Checkpoint is not a clean update boundary")
    require(state["count"]==state["cursor"] and state["epoch"] < run["config"]["artifact_epochs"],"Invalid resume cursor")
    require(all(k in saved for k in ("model","optimizer","scaler","scheduler","rng")),"Incomplete state")
    child=dict(saved,identity=model_identity(run))
    seed=output/"migration_seed.pt"
    if seed.exists():require(tree_equal(child,torch.load(seed,map_location="cpu",weights_only=True)),"Migration seed changed")
    else:save_torch(seed,child)
    restored=torch.load(seed,map_location="cpu",weights_only=True)
    preserved={k:tree_equal(saved[k],restored[k]) for k in saved if k!="identity"}
    require(all(preserved.values()),"Checkpoint content changed during migration")
    dest=output/"models/artifact/video"
    copy_exact(seed,dest/"last.pt",sha(seed))
    for name,checksum in continuation["parent_files"].items():
        if name.startswith("models/artifact/video/") and name!="models/artifact/video/last.pt":
            copy_exact(parent/name,output/name,checksum)
    receipt=dict(format=VERSION,parent_checkpoint_sha256=sha(source),child_seed_sha256=sha(seed),
        parent_identity=saved["identity"],child_identity=restored["identity"],preserved=preserved,
        resume_epoch=state["epoch"]+1,cursor=state["cursor"],optimizer_updates=state["updates"],
        learning_rate=saved["optimizer"]["param_groups"][0]["lr"],amp_scale=saved["scaler"].get("scale",1.),
        source_files=continuation["parent_files"],run_sha256=sha(output/"RUN.json"))
    atomic_json(output/"MIGRATION.json",receipt)
    atomic_json(output/"STATUS.json",dict(status="READY_TO_RESUME",stage="migrated_artifact_video",
        resume_epoch=receipt["resume_epoch"],cursor=receipt["cursor"],optimizer_updates=receipt["optimizer_updates"]))
    verify_migration(output,run)
    return receipt


def verify_migration(output, run=None):
    output=Path(output)
    current=read(output/"RUN.json")
    if run is not None:require(current==run,"Continuation code/options/inputs changed")
    require(current["implementation"]==code(),"Optimized implementation changed")
    rec=read(output/"MIGRATION.json")
    require(rec["run_sha256"]==sha(output/"RUN.json"),"Migration run changed")
    require(read(output/"CONTINUATION.json")==current["continuation"],"Continuation manifest changed")
    for name,checksum in rec["source_files"].items():
        require(sha(output/"parent_snapshot"/name)==checksum,"Parent snapshot changed: "+name)
    require(sha(output/"migration_seed.pt")==rec["child_seed_sha256"],"Immutable migration seed changed")
    before=torch.load(output/"parent_snapshot/models/artifact/video/last.pt",map_location="cpu",weights_only=True)
    after=torch.load(output/"migration_seed.pt",map_location="cpu",weights_only=True)
    require(all(tree_equal(before[k],after[k]) for k in before if k!="identity"),"Preserved state mismatch")
    require(after["identity"]==model_identity(current),"Child identity changed")
    status("MIGRATION VERIFIED",f"epoch={rec['resume_epoch']} cursor={rec['cursor']} optimizer_step={rec['optimizer_updates']}; all state preserved")
    return rec
