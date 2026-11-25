"""PGF train/dev -> PGF, FAV, AVDF1M test; no historical output mutation."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE","1")
import sys,argparse,traceback
from pathlib import Path
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__="scripts.trainable_or_3way.pgf_source"
from .common import *

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("preflight","run","verify","status"))
    p.add_argument("--output",type=Path)
    p.add_argument("--cache-root",type=Path,default=CACHE)
    p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=42)
    p.add_argument("--artifact-epochs",type=int,default=20);p.add_argument("--head-epochs",type=int,default=20)
    p.add_argument("--head-batch-size",type=int,default=8)
    p.add_argument("--checkpoint-steps",type=int,default=100)
    p.add_argument("--frame-batch",type=int,default=64);p.add_argument("--predict-batch-size",type=int,default=32)
    p.add_argument("--threads",type=int,default=4);p.add_argument("--decode-samples",type=int,default=2)
    p.add_argument("--decode-workers",type=int,default=2);p.add_argument("--prefetch",type=int,default=2)
    return p

def run(args):
    if args.output is None:args.output=OUTPUT.parent/("seed_"+str(args.seed))
    out=resolve(args.output)
    if args.command=="status":
        require((out/"STATUS.json").exists(),"No run status yet: "+str(out))
        value=read(out/"STATUS.json");print(json.dumps(value,ensure_ascii=False,indent=2))
        detail=value.get("detail_progress")
        if detail and Path(detail).is_file() and Path(detail).resolve()!=out/"STATUS.json":
            print(json.dumps(read(detail),ensure_ascii=False,indent=2))
        print("Saved progress snapshot; this does not certify that a process is currently alive.",flush=True);return
    from .pipeline import execute,verify_output
    if args.command=="verify":verify_output(out);return
    check_destination(out,args.cache_root)
    if (out/"COMPLETED.json").exists():
        verify_output(out);return
    setup(args.device,args.threads)
    from .inputs import prepare
    if args.command=="preflight":
        prepare(args);return  # No training, encoder inference or output directory creation.
    with writer_lock(out):
        run_record,cohorts,stamps,proofs=prepare(args)
        from scripts.trainable_or_3way_matched.memory import configure_cuda_budget
        configure_cuda_budget(args.device)
        try:execute(args,run_record,cohorts,stamps,proofs)
        except BaseException as error:
            current=read(out/"STATUS.json") if (out/"STATUS.json").exists() else {}
            atomic_json(out/"STATUS.json",dict(current,status="INTERRUPTED" if isinstance(error,KeyboardInterrupt) else "FAILED",
                        error=str(error),resume="Repeat the identical run command"))
            raise

if __name__=="__main__":
    try:run(parser().parse_args())
    except KeyboardInterrupt:
        status("INTERRUPTED","Committed optimizer steps/clips retained; run the same command to resume.")
        sys.exit(130)
    except Exception:
        traceback.print_exc();status("FAILED","Previous checkpoints retained; no successful completion.");sys.exit(1)
