"""Run only P3-a under one FP32 reference. Historical results stay read-only."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import sys,argparse,csv,io,time
from pathlib import Path
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__="scripts.trainable_or_3way.reinforcement_temporal"
from .common import *
from .inputs import Inputs,P1,P2
from .experiment import preflight
from .order_recheck_core import POLICY,predict_all,evaluate_pair

OUTPUT=ROOT/"reports/fav_avdf1m_reinforcement_p3a_v2"

def preservation_snapshot(origin):
    snapshot={}
    for receipt_path in sorted(origin.rglob("COMPLETED.json")):
        r=read(receipt_path);verify_files(receipt_path.parent,r["files"])
        snapshot[receipt_path.relative_to(origin).as_posix()]=sha(receipt_path)
        for name,h in r["files"].items():
            p=child(receipt_path.parent,name)
            snapshot[p.relative_to(origin).as_posix()]=h
    for name in ("STATUS.json","PROTOCOL.json"):
        snapshot[name]=sha(origin/name)
    for p in origin.rglob("*.pt"):snapshot[p.relative_to(origin).as_posix()]=sha(p)
    return snapshot

def write_summary(output,results):
    rows=[]
    text=["# P3-a common-FP32 order diagnostic","","All heads and fusion coefficients are fixed. "
          "Ordered/reversed/shuffled predictions use identical FP32 settings. No retraining or calibration.",
          "Historical scores are retained and audited separately; this is not a bitwise reproduction of their precision settings.",
          "Results are descriptive order sensitivity, not causal evidence of emotional inconsistency.","",
          "| Source → Target | BN | Ordered AUC (%) | Reversed AUC (%) | Shuffle AUC mean ± SD (%) |",
          "|---|---|---:|---:|---:|"]
    for r in results:
        for name,m in r["metrics"].items():
            rows.append(dict(source=r["source"],target=r["target"],condition=r["condition"],order=name,
                        **{k:m[k] for k in ("n","roc_auc","fake_f1","real_fpr","fake_recall")},
                        **{"delta_"+k:v for k,v in r["changes"][name]["delta"].items()}))
        o=100*r["metrics"]["ordered__both"]["roc_auc"]
        rev=100*r["metrics"]["reversed__both"]["roc_auc"]
        shuffled=[100*r["metrics"][s+"__both"]["roc_auc"] for s in TRANSFORMS if s.startswith("shuffle")]
        text.append(f"| {r['source']} → {r['target']} | {r['condition']} | {o:.4f} | {rev:.4f} | {np.mean(shuffled):.4f} ± {np.std(shuffled,ddof=1):.4f} |")
    text+=["","Shuffle SD describes three fixed permutations; it is not training-seed uncertainty.",
           "One-token clips contain no order information. See per-modality length strata and changes in each DIAGNOSTIC.json.",
           "AUC/F1/FPR comparisons are paired on the same clips, but no new significance test is claimed here."]
    atomic_json(output/"SUMMARY.json",dict(policy=POLICY,conditions=len(results),rows=rows))
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_text(output/"SUMMARY.csv",buffer.getvalue())
    atomic_text(output/"REPORT.md","\n".join(text)+"\n")

def verify_output(output):
    receipts=sorted(output.rglob("COMPLETED.json"));require(receipts,"No completed jobs")
    for p in receipts:verify_files(p.parent,read(p)["files"])
    require((output/"COMPLETED.json").is_file(),"Some jobs complete, but P3-a root is incomplete")
    preserved=read(output/"PRESERVED.json");verify_files(Path(preserved["origin"]),preserved["files"])
    print(f"[VERIFY] P3-a complete; {len(receipts)} receipts PASS; original P3-b/P4 preserved",flush=True)

def run(args):
    output=resolve(args.output);origin=resolve(args.temporal)
    safe_output(output,[origin,resolve(args.parent),resolve(args.p2)])
    if args.command=="verify":verify_output(output);return
    setup(args.device,args.threads)
    with writer_lock(output):
        if (output/"COMPLETED.json").exists():
            verify_output(output);return
        try:
            atomic_json(output/"STATUS.json",dict(stage="input_verification",status="RUNNING"))
            print("[P3-a] Preserving completed results and validating existing feature receipts",flush=True)
            snapshot=preservation_snapshot(origin)
            immutable(output/"PRESERVED.json",dict(origin=str(origin),files=snapshot))
            parent_protocol=read(origin/"PROTOCOL.json")
            require(training_code_identity()==parent_protocol["code"],"Historical training implementation changed")
            inputs=Inputs(["FAV","AVDF1M"],["FAV","AVDF1M"],p1=args.parent,p2=args.p2)
            # Read-only use of the completed full preflight; no decoder or encoder runs.
            preflight(inputs,origin,False)
            models={source:{c:inputs.historical_fusion(source,c) for c in ("B","C")} for source in inputs.sources}
            run_record=dict(format="p3a_common_fp32_v2",sources=inputs.sources,targets=inputs.targets,
                device=args.device,batch_size=args.batch_size,threads=args.threads,policy=POLICY,
                heads=inputs.heads,fusion=models,inputs={"/".join(k):v for k,v in inputs.proofs.items()},
                implementation=code_identity(),runtime=dict(torch=torch.__version__,numpy=np.__version__),
                previous_targets_observed=True,head_training=False,calibration_refitted=False,
                original_results=str(origin))
            immutable(output/"RUN.json",run_record)
            results=[];start=time.monotonic();done=0;outer=Progress("P3-a source/target pairs",4,output)
            for source in inputs.sources:
                for target in inputs.targets:
                    pair={}
                    for mod in ("video","audio"):
                        atomic_json(output/"STATUS.json",dict(stage="predict_all_orders",status="RUNNING",
                                    source=source,target=target,modality=mod,completed_pairs=done,total_pairs=4))
                        cache=inputs.cache(target,"test",mod)
                        try:
                            pair[mod]=predict_all(cache,inputs.heads[source][mod]["path"],
                                output/"predictions"/source/target/mod,device=args.device,batch_size=args.batch_size)
                        finally:cache.close()
                    atomic_json(output/"STATUS.json",dict(stage="score_diagnostics",status="RUNNING",
                                source=source,target=target,completed_pairs=done,total_pairs=4))
                    results.extend(evaluate_pair(inputs,source,target,pair,output/"diagnostics"/source/target))
                    done+=1;outer.update(done,force=True)
            write_summary(output,results)
            verify_files(origin,snapshot)
            atomic_json(output/"PRESERVATION_VERIFIED.json",dict(original_files=len(snapshot),status="PASS"))
            atomic_json(output/"STATUS.json",dict(stage="p3a",status="COMPLETE",source_target_pairs=4,
                        condition_reports=len(results),elapsed_after_preflight_seconds=time.monotonic()-start,
                        training_executed=False,calibration_refitted=False))
            files=["RUN.json","PRESERVED.json","PRESERVATION_VERIFIED.json","STATUS.json","SUMMARY.json","SUMMARY.csv","REPORT.md"]
            for p in sorted(output.rglob("COMPLETED.json")):
                if p.parent==output:continue
                files.append(p.relative_to(output).as_posix())
                files.extend((p.parent/name).relative_to(output).as_posix() for name in read(p)["files"])
            files.extend(p.relative_to(output).as_posix() for p in output.glob("diagnostics/*/*/ORDERED_FP32_REFERENCE.npz"))
            finish(output,digest(run_record),sorted(set(files)))
            print("[P3-a] COMPLETE. 8 condition reports; original P3-b/P4 unchanged.",flush=True)
        except BaseException as exc:
            atomic_json(output/"STATUS.json",dict(stage="p3a",status="INTERRUPTED" if isinstance(exc,KeyboardInterrupt) else "FAILED",
                                                 error=str(exc),resume="Run the same command"))
            raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["run","verify"])
    p.add_argument("--output",default=str(OUTPUT))
    p.add_argument("--temporal",default=str(DEFAULT_OUTPUT))
    p.add_argument("--parent",default=str(P1));p.add_argument("--p2",default=str(P2))
    p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--threads",type=int,default=4)
    return p

if __name__=="__main__":
    args=parser().parse_args()
    require(args.batch_size>0 and args.threads>0,"Invalid batch/threads")
    try:run(args)
    except KeyboardInterrupt:
        print("[INTERRUPTED] Saved completed batches; use the same command to resume.",flush=True)
        sys.exit(130)
