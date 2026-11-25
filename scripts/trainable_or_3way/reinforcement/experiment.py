"""Entry point: p0 audit, p1 saved-score analysis, status, verify. Never launches training."""
import argparse
import sys
from pathlib import Path
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.trainable_or_3way.reinforcement.common import (
    ROOT, read, digest, resolve, safe_output, implementation, implementation_patch, runtime, immutable, atomic_json, writer_lock, require, completed)
from scripts.trainable_or_3way.reinforcement.inputs import DEFAULTS
from scripts.trainable_or_3way.reinforcement import audit, analysis

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("p0","p1","status","verify"):
        q=sub.add_parser(name)
        q.add_argument("--output",default="reports/fav_avdf1m_reinforcement_v1")
        if name=="p0":
            for key,value in DEFAULTS.items():
                q.add_argument("--"+key.replace("_","-"),default=value)
            q.add_argument("--include-pgf",action="store_true",help="Include preserved PGF tests as supplementary analysis")
            q.add_argument("--bootstrap",type=int,default=5000,help="Lock P1 valid replicates in P0 (default 5000)")
            q.add_argument("--max-attempts",type=int,default=50000)
            q.add_argument("--seed",type=int,default=1701)
            q.add_argument("--checkpoint-every",type=int,default=25)
    return p

def run(args):
    output=resolve(args.output)
    if args.command=="status":
        for path in (output/"STATUS.json",output/"p1/progress.json"):
            if path.exists():
                print(path,read(path),flush=True)
        if (output/"p1").exists():
            for path in sorted((output/"p1").glob("*/*/progress.json")):
                print(path.relative_to(output),read(path),flush=True)
        return
    if args.command=="p0":
        config={k:str(resolve(getattr(args,k))) for k in DEFAULTS}
        config.update(targets=["FAV","AVDF1M"]+(["PGF"] if args.include_pgf else []),
                      bootstrap=args.bootstrap,max_attempts=args.max_attempts,seed=args.seed,
                      checkpoint_every=args.checkpoint_every)
        require(args.bootstrap>=2 and args.max_attempts>=args.bootstrap and args.checkpoint_every>0 and args.seed>=0,
                "Invalid bootstrap/checkpoint options")
        run_spec=dict(format="reinforcement_code_v1",config=config,implementation=implementation(),runtime=runtime())
        if (output/"RUN.json").exists():
            previous=read(output/"RUN.json")
            require(previous["config"]==config and previous["runtime"]==run_spec["runtime"],
                    "Configuration/runtime changed; use a new output")
            implementation_patch(previous["implementation"])
            run_spec=previous
        if output.exists() and not (output/"RUN.json").exists():
            require(not any(p.name!=".writer.lock" for p in output.iterdir()),"Output is nonempty and belongs to another run")
    else:
        require((output/"RUN.json").is_file(),"Run p0 first")
        run_spec=read(output/"RUN.json")
        config=run_spec["config"]
        implementation_patch(run_spec["implementation"])
        require(run_spec["runtime"]==runtime(),"Analysis dependency versions changed; use a new output and run p0")
    safe_output(output,[resolve(config[k]) for k in ("fav_root","fav_cross_root","workstation_root")])
    require(output!=ROOT and output!=ROOT/"reports","Choose a new dedicated results directory")
    with writer_lock(output):
        immutable(output/"RUN.json",run_spec)
        identity=digest(run_spec)
        patch=implementation_patch(run_spec["implementation"])
        if patch:
            immutable(output/"IO_PATCHES"/(patch["patch_id"]+".json"),dict(original_run_identity=identity,**patch))
            print("[RESUME] Reviewed Windows save fix; original run identity and statistics preserved",flush=True)
        if args.command=="verify":
            require(completed(output/"p0",digest(dict(run=identity,stage="p0"))),"P0 incomplete")
            audit.verify_preserved(output)
            from scripts.trainable_or_3way.reinforcement.common import sha
            p1id=digest(dict(run=identity,stage="p1",p0=sha(output/"p0/COMPLETED.json")))
            if (output/"p1/COMPLETED.json").exists():
                completed(output/"p1",p1id)
            print("[VERIFY] PASS: preserved inputs and committed output checksums",flush=True)
            return
        atomic_json(output/"STATUS.json",dict(status="RUNNING",stage=args.command))
        try:
            if args.command=="p0":
                audit.execute(config,output,identity)
            else:
                analysis.execute(config,output,identity)
        except BaseException as exc:
            atomic_json(output/"STATUS.json",dict(status="INTERRUPTED" if isinstance(exc,KeyboardInterrupt) else "FAILED",
                                                  stage=args.command,error=str(exc),resume="Repeat the same command/output"))
            raise
        atomic_json(output/"STATUS.json",dict(status="COMPLETE",stage=args.command,
                    note="Stage complete; FAV B/C harmonization and cross-dataset identity audit remain pending"))

if __name__=="__main__":
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        print("[INTERRUPTED] Saved bootstrap checkpoint retained. Repeat the same command.",flush=True)
        sys.exit(130)
    except Exception as exc:
        print("[FAILED] "+str(exc)+"; no successful completion for failed stage.",file=sys.stderr,flush=True)
        raise
