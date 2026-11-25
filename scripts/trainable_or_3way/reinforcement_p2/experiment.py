"""P2: FAV artifact B/C inference, source-dev fitting, paired analysis, AVDF result reuse."""
import os,sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
sys.dont_write_bytecode=True
from pathlib import Path
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__="scripts.trainable_or_3way.reinforcement_p2"
import argparse
from .common import *
from .inputs import prepare,preflight
from .extract import infer_cohort
from .evaluate import calibrate,evaluate_one,reuse_avdf,summarize

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["preflight","run","status","verify"])
    p.add_argument("--output",default=str(OUTPUT_DEFAULT))
    p.add_argument("--parent",default=str(P1_DEFAULT))
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--decode-workers",type=int,default=2)
    p.add_argument("--prefetch",type=int,default=2)
    p.add_argument("--local-paths",default="configs/local_paths.yaml")
    p.add_argument("--ffmpeg",default=".venv/Scripts/ffmpeg.exe")
    return p

def execute(output,run,cohorts):
    identity=digest(run)
    if completed(output,identity):
        print("[P2 COMPLETE] Verified; nothing to repeat",flush=True)
        return
    preflight(output,run,cohorts)
    dev=infer_cohort(output,run,"FAV/dev",cohorts["FAV/dev"])
    atomic_json(output/"STATUS.json",dict(status="RUNNING",stage="source_dev_fusion"))
    models=calibrate(output,run,dev)
    results=[]
    for target in ("FAV","AVDF1M"):
        name=target+"/test"
        paths=infer_cohort(output,run,name,cohorts[name])
        for c in ("B","C"):
            atomic_json(output/"STATUS.json",dict(status="RUNNING",stage="statistics",target=target,condition=c))
            results+=evaluate_one(output,run,target,c,paths[c],models[c])
    check_stop(output)
    results+=reuse_avdf(output,run)
    summarize(output,run,results)
    require(verify_parent(run["parent"])==run["parent_proof"],"Original P0/P1 changed during P2")
    files=[]
    # SQLite has its own logical checksum, verified on resume; no byte hash of live WAL files.
    for p in sorted(output.rglob("*")):
        if not p.is_file():continue
        if p.name in ("STATUS.json","progress.json",".writer.lock","STOP_REQUESTED","LAST_ERROR.json"):continue
        if p.suffix==".partial" or p.name.startswith("scores.sqlite"):continue
        if p==output/"COMPLETED.json":continue
        files.append(p.relative_to(output).as_posix())
    finish(output,identity,files)
    print("[P2 COMPLETE] "+str(output/"REPORT.md"),flush=True)

def run(args):
    output=resolve(args.output)
    if args.command=="status":
        for p in [output/"STATUS.json",*sorted(output.glob("scores/*/*/*/progress.json")),
                  *sorted(output.glob("evaluation/*/*/progress.json"))]:
            if p.exists():print(p.relative_to(output),read(p),flush=True)
        return
    if args.command=="verify":
        from .storage import ScoreStore
        from .runtime import validate_score
        require((output/"RUN.json").is_file(),"P2 has not been initialized")
        with writer_lock(output):
            saved=read(output/"RUN.json")
            require(saved["format"]=="reinforcement_p2_bc_v1","Not a P2 output")
            require(completed(output,digest(saved)),"P2 is not complete yet; use status or resume run")
            require(verify_parent(saved["parent"])==saved["parent_proof"],"Parent changed")
            for name in COHORTS:
                rows=read(output/"cohorts"/(name.replace("/","_")+".json"))
                for mod in ("video","audio"):
                    store=ScoreStore(output/"scores"/name/mod,dict(run=digest(saved),cohort=name,modality=mod),readonly=True)
                    try:
                        _,done=store.verify(rows,lambda v:validate_score(v,mod))
                        require(done,"Incomplete store")
                    finally:store.close()
            print("[VERIFY] PASS: P2 outputs, per-clip stores and preserved P0/P1",flush=True)
        return
    parent=resolve(args.parent)
    safe_output(output,[parent,ROOT/"checkpoints"])
    require(output.is_relative_to(ROOT/"reports") and output!=ROOT/"reports","Use a dedicated reports output folder")
    if output.exists():
        if (output/"RUN.json").exists():
            require(read(output/"RUN.json").get("format")=="reinforcement_p2_bc_v1","Do not overwrite another experiment")
        else:
            require(not any(p.name!=".writer.lock" for p in output.iterdir()),"Nonempty output; choose a new P2 directory")
    with writer_lock(output):
        owned=False
        try:
            description,cohorts=prepare(args)
            immutable(output/"RUN.json",description)
            owned=True
            for name,rows in cohorts.items():immutable(output/"cohorts"/(name.replace("/","_")+".json"),rows)
            if args.command=="preflight":
                preflight(output,description,cohorts)
                atomic_json(output/"STATUS.json",dict(status="PREFLIGHT_PASS",stage="preflight",inference_started=False))
                print("[P2 PREFLIGHT] PASS; model forward has not started",flush=True)
            else:
                execute(output,description,cohorts)
                atomic_json(output/"STATUS.json",dict(status="COMPLETE",stage="p2",conditions=["B","C"],primary_family_complete=True))
        except BaseException as exc:
            if owned:
                atomic_json(output/"STATUS.json",dict(status="INTERRUPTED" if isinstance(exc,KeyboardInterrupt) else "FAILED",
                      error=str(exc),resume="Repeat the identical run command/output; committed clips retained"))
            raise

if __name__=="__main__":
    try:run(parser().parse_args())
    except KeyboardInterrupt:
        print("[INTERRUPTED] Re-run the same command to resume",flush=True)
        sys.exit(130)
    except Exception as exc:
        print("[FAILED] "+str(exc)+"; prior results and committed samples retained",file=sys.stderr,flush=True)
        raise
