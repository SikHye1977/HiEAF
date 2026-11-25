"""Migrate or resume PGF training without changing optimization or BN conditions."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
import argparse
import traceback
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__="scripts.trainable_or_3way.pgf_source_optimized"
from .migration import *


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("migrate","preflight","run","status","verify"))
    p.add_argument("--parent",type=Path,default=DEFAULT_PARENT)
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--frame-cache",type=Path,default=DEFAULT_FRAMES)
    p.add_argument("--input-workers",type=int,default=2)
    p.add_argument("--input-prefetch",type=int,default=2)
    return p


def run(args):
    parent,out,frames=paths(args.parent,args.output,args.frame_cache)
    if args.command=="status":
        value=read(out/"STATUS.json");print(json.dumps(value,ensure_ascii=False,indent=2))
        detail=value.get("detail_progress")
        if detail and Path(detail).is_file():print(json.dumps(read(detail),ensure_ascii=False,indent=2))
        return
    if args.command=="verify":
        verify_migration(out)
        if (out/"COMPLETED.json").exists():
            from .pipeline import verify_output
            verify_output(out)
        return
    with writer_lock(parent):  # refuse if parent training has resumed
        proposed_run,continuation=proposed(parent,out,frames,args.input_workers,args.input_prefetch)
        if (out/"COMPLETED.json").exists():
            verify_migration(out,proposed_run)
            from .pipeline import verify_output
            verify_output(out)
            return
        cohorts,stamps,proofs=validate_current(parent,proposed_run)
        if args.command=="preflight":
            if (out/"RUN.json").exists():require(read(out/"RUN.json")==proposed_run,"Continuation changed")
            status("PREFLIGHT PASS","Parent recipe unchanged; lossless frame cache and ordered prefetch only")
            return
        with writer_lock(out):
            migrate(parent,out,proposed_run,continuation)
            if args.command=="migrate":
                status("READY TO RESUME","State imported and verified; training has NOT started")
                return
            from scripts.trainable_or_3way_matched.memory import configure_cuda_budget
            from .pipeline import execute
            configure_cuda_budget(proposed_run["config"]["device"])
            args.output=out
            with writer_lock(frames):
                try:execute(args,proposed_run,cohorts,stamps,proofs)
                except BaseException as error:
                    current=read(out/"STATUS.json")
                    atomic_json(out/"STATUS.json",dict(current,status="INTERRUPTED" if isinstance(error,KeyboardInterrupt) else "FAILED",
                        error=str(error),resume="Repeat the identical optimized run command"))
                    raise


if __name__=="__main__":
    try:run(parser().parse_args())
    except KeyboardInterrupt:
        status("INTERRUPTED","Saved optimizer boundary retained; original run remains unchanged")
        sys.exit(130)
    except Exception:
        traceback.print_exc();status("FAILED","Committed checkpoints retained");sys.exit(1)
