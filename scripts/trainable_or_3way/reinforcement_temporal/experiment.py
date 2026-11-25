"""P3/P4 CLI. Nothing runs on import. All outputs are independent of historical runs."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, argparse
from pathlib import Path
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__="scripts.trainable_or_3way.reinforcement_temporal"
from .common import *
from .inputs import Inputs, P1, P2, FAV, AVDF, CACHE
from .training import Task, fit
from .analysis import predict_pair, fit_locked, evaluate_heads, order_diagnostic, aggregate

def proof_snapshot(inputs):
    return {"/".join(k):v for k,v in inputs.proofs.items()}

def preflight(inputs,output,full):
    inputs.prepare(full_hash=full)
    proof=proof_snapshot(inputs)
    identity=digest(proof)
    folder=output/"preflight"/identity
    if full:
        atomic_json(folder/"CHECKS.json",dict(inputs=proof,heads=inputs.heads,
                    frozen_metadata=True,packed_payload_hashes=True,sqlite_index_verified=True,
                    sqlite_payload_checks="bounded preflight plus every production read",
                    raw_decode=False,encoder_inference=False,
                    explanation="Completed frozen feature caches are inputs; no media recipe changes"))
        finish(folder,identity,["CHECKS.json"])
    else:
        require(completed(folder,identity),"Run preflight first for these sources/targets and current cache files")
    return inputs

def train_and_evaluate(inputs,output,args,protocol,seeds,scope):
    rows=[];device=args.device
    for source in args.sources:
        for seed in seeds:
            models={};head_paths={};test_predictions={}
            for variant in VARIANTS:
                paths={}
                for mod in ("video","audio"):
                    train_cache=inputs.cache(source,"train",mod);dev_cache=inputs.cache(source,"dev",mod)
                    folder=output/"heads"/source/f"seed_{seed}"/variant/mod
                    identity=model_identity(source,mod,variant,seed,train_cache.proof,dev_cache.proof,
                                            TRAIN,protocol["code"],protocol["runtime"])
                    try:
                        if not completed(folder,identity):
                            train=Task(train_cache);dev=Task(dev_cache)
                            fit(train,dev,variant,seed,source,folder,identity,device=device)
                        else: print("[REUSE HEAD] "+str(folder),flush=True)
                    finally: train_cache.close();dev_cache.close()
                    paths[mod]=folder/"best.pt"
                head_paths[variant]=paths
                devpair=predict_pair(inputs,source,"dev",paths,variant,
                          output/"predictions"/source/f"seed_{seed}"/variant/source/"dev",
                          device,args.predict_batch_size)
                models[variant]=fit_locked(inputs,source,seed,variant,devpair,paths,
                          output/"fusion"/source/f"seed_{seed}"/variant)
            # All source-dev model selection and calibration above precede any new target predictions.
            for target in args.targets:
                test_predictions={}
                for variant in VARIANTS:
                    test_predictions[variant]=predict_pair(inputs,target,"test",head_paths[variant],variant,
                        output/"predictions"/source/f"seed_{seed}"/variant/target/"test",
                        device,args.predict_batch_size)
                for condition in ("B","C"):
                    rows.extend(evaluate_heads(inputs,source,target,seed,condition,test_predictions,models,
                        output/"evaluations"/source/f"seed_{seed}"/target/condition,args.bootstrap))
    return aggregate(output,scope,rows)

def verify(output):
    receipts=sorted(output.rglob("COMPLETED.json"))
    require(receipts,"No completed jobs")
    for i,path in enumerate(receipts,1):
        r=read(path);verify_files(path.parent,r["files"])
        if i%20==0: print(f"[VERIFY] {i}/{len(receipts)} receipts",flush=True)
    print(f"[VERIFY] PASS {len(receipts)} completed jobs; incomplete jobs are not counted",flush=True)

def run(args):
    output=safe_output(resolve(args.output),[resolve(args.parent),resolve(args.p2),FAV,AVDF,CACHE])
    if args.command=="verify": verify(output);return
    require(len(args.sources)==len(set(args.sources)) and len(args.targets)==len(set(args.targets)),
            "Duplicate source/target")
    setup(args.device,args.threads)
    inputs=Inputs(args.sources,args.targets,p1=args.parent,p2=args.p2)
    with writer_lock(output):
        try:
            protocol=dict(format="reinforcement_temporal_v1",train=TRAIN,variants=list(VARIANTS),
                          code=training_code_identity(),runtime=dict(torch=torch.__version__,numpy=np.__version__,
                          device=args.device,threads=args.threads,predict_batch_size=args.predict_batch_size,
                          deterministic=True,tf32=False),encoder_training=False,
                          source_subset_fixed=True,prior_targets_observed=True,independent_calibration=False)
            immutable(output/"PROTOCOL.json",protocol)
            implementation=code_identity()
            immutable(output/"implementations"/(digest(implementation)+".json"),implementation)
            atomic_json(output/"STATUS.json",dict(stage=args.command,status="RUNNING"))
            inputs=preflight(inputs,output,args.command=="preflight")
            if args.command=="preflight":
                atomic_json(output/"STATUS.json",dict(stage="preflight",status="COMPLETE",experiment_started=False))
                print("[PREFLIGHT] PASS. No training/inference/bootstrap executed.",flush=True);return
            seeds=[42] if args.command=="p3" else [42,43,44]
            scope=dict(sources=args.sources,targets=args.targets,seeds=seeds,conditions=["B","C"],
                       variants=list(VARIANTS),bootstrap=args.bootstrap,primary_cohort="full",
                       comparisons=["T2-T0","T2-T1"],family="all requested cross directions x B/C x pairs x seeds",
                       code=evaluation_code_identity())
            scope_id=digest(scope)[:16]
            immutable(output/"analyses"/scope_id/"SCOPE.json",scope)
            if args.command=="p3" and args.part in ("all","order"):
                for source in args.sources:
                    for target in args.targets:
                        order_diagnostic(inputs,source,target,output/"order_diagnostics"/source/target,
                                         args.device,args.predict_batch_size)
            if args.command=="p4" or args.part in ("all","heads"):
                summary=train_and_evaluate(inputs,output,args,protocol,seeds,scope)
                print("[SUMMARY] "+str(summary/"SUMMARY.json"),flush=True)
            atomic_json(output/"STATUS.json",dict(stage=args.command,status="COMPLETE",scope_id=scope_id,
                        part=args.part if args.command=="p3" else "heads_seed42_43_44"))
            print(f"[{args.command.upper()}] COMPLETE requested jobs. Existing P0/P1/P2 preserved.",flush=True)
        except BaseException as exc:
            atomic_json(output/"STATUS.json",dict(stage=args.command,status="INTERRUPTED" if isinstance(exc,KeyboardInterrupt) else "FAILED",error=str(exc)))
            raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["preflight","p3","p4","verify"])
    p.add_argument("--sources",nargs="+",choices=["FAV","AVDF1M"],default=["FAV","AVDF1M"])
    p.add_argument("--targets",nargs="+",choices=["FAV","AVDF1M"],default=["FAV","AVDF1M"])
    p.add_argument("--part",choices=["all","order","heads"],default="all",help="P3 only")
    p.add_argument("--parent",default=str(P1));p.add_argument("--p2",default=str(P2))
    p.add_argument("--output",default=str(DEFAULT_OUTPUT));p.add_argument("--device",default="cuda:0")
    p.add_argument("--threads",type=int,default=4);p.add_argument("--predict-batch-size",type=int,default=32)
    p.add_argument("--bootstrap",type=int,default=5000)
    return p

if __name__=="__main__":
    args=parser().parse_args()
    try:
        require(args.bootstrap>=100 and args.predict_batch_size>0,"Invalid bootstrap/batch size")
        run(args)
    except KeyboardInterrupt:
        print("[INTERRUPTED] Committed jobs retained. Resume with the same command.",flush=True)
        sys.exit(130)
    except Exception as exc:
        # Do not touch a potentially protected --output before safe_output validation has passed.
        print("[FAILED] "+str(exc)+"; earlier committed jobs retained.",flush=True)
        raise
