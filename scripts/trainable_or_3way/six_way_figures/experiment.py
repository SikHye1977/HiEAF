"""Six-direction Figure 2/3 extension. No experiment runs on import."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.trainable_or_3way.six_way_figures"
from .common import *
from .inputs import Inputs
from .pipeline import figure2, figure3a, figure3b


def inspect_status(output):
    output = Path(output)
    require((output / "RUN.json").is_file(), "No run at " + str(output))
    print(read(output / "STATUS.json") if (output / "STATUS.json").exists() else "No stage status yet", flush=True)
    candidates = sorted(output.rglob("progress.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates[:3]:
        d = read(p)
        print(f"[LAST PROGRESS] {p}\n  {d.get('stage')} {d.get('done')}/{d.get('total')} "
              f"elapsed={duration(d.get('elapsed_seconds'))} ETA={duration(d.get('eta_seconds'))}", flush=True)
    print("Saved progress only; this does not prove a process is still alive.", flush=True)


def verify(output, require_complete=True):
    output = Path(output)
    run = read(output / "RUN.json")
    require(run["format"] == FORMAT, "Wrong run type")
    receipts = sorted(output.rglob("COMPLETED.json"))
    require(receipts, "No completed stages")
    for p in receipts:
        verify_files(p.parent, read(p)["files"])
    evidence = Evidence()
    evidence.files = read(output / "PRESERVED_INPUTS.json")
    if (output / "PRESERVATION_VERIFIED.json").exists():
        for p, h in read(output / "PRESERVATION_VERIFIED.json").get("additional_inputs", {}).items():
            require(p not in evidence.files or evidence.files[p] == h, "Conflicting preserved file hash")
            evidence.files[p] = h
    evidence.verify()
    if require_complete:
        require((output / "COMPLETED.json").is_file(), "Some stages finished, but the whole extension is incomplete")
        require(read(output / "COMPLETED.json")["identity"] == digest(run), "Root completion identity differs")
        require(read(output / "fig2/SUMMARY.json")["family_complete"], "Figure 2 family incomplete")
        require(read(output / "fig3b/SUMMARY.json")["family_complete"], "Figure 3b family incomplete")
    print(f"[VERIFY] PASS {len(receipts)} receipts; preserved historical inputs unchanged", flush=True)


def finalize(output, run):
    from .export import export
    export(output)
    files = ["RUN.json", "PREFLIGHT.json", "PRESERVED_INPUTS.json"]
    if (output / "PRESERVATION_VERIFIED.json").exists():
        files.append("PRESERVATION_VERIFIED.json")
    for p in sorted(output.rglob("COMPLETED.json")):
        if p.parent == output:
            continue
        files.append(p.relative_to(output).as_posix())
        files.extend((p.parent / name).relative_to(output).as_posix() for name in read(p)["files"])
    atomic_json(output / "STATUS.json", dict(status="COMPLETE", stages=["fig2", "fig3a", "fig3b", "figures"],
        cross_directions=6, conditions=12, fig2_family=12, temporal_auc_family=72,
        artifact_training=False, encoder_inference=False))
    files.append("STATUS.json")
    finish(output, digest(run), sorted(set(files)))
    verify(output)


def run(args):
    output = destination(args.output)
    if args.command == "status":
        inspect_status(output)
        return
    if args.command == "verify":
        verify(output)
        return
    require(args.threads > 0 and args.predict_batch_size > 0, "Invalid threads/batch size")
    with writer_lock(output):
        if (output / "COMPLETED.json").exists():
            verify(output)
            return
        if args.command == "export":
            finalize(output, read(output / "RUN.json"))
            return
        setup(args.device, args.threads)
        protocol = dict(format=FORMAT, directions=[list(p) for p in DIRECTIONS], conditions=list(CONDITIONS),
            seeds=list(SEEDS), variants=list(VARIANTS), transforms=list(TRANSFORMS), train=TRAIN,
            bootstrap=REPLICATES, fig2_family=12, temporal_family=72,
            runtime=dict(device=args.device, threads=args.threads, predict_batch_size=args.predict_batch_size,
                         torch=str(torch.__version__), numpy=np.__version__, tf32=False),
            implementation=implementation(), artifact_training=False, encoder_inference=False,
            current_windows_frozen_inputs=True, source_workstation_media_hash_required=False,
            target_fitting=False, prior_targets_observed=True)
        immutable(output / "RUN.json", protocol)
        try:
            print("[INPUT] Verifying completed scores, source-dev locks, heads and frozen tokens; no media extraction", flush=True)
            inputs = Inputs().prepare(full_hash=args.command == "preflight")
            snapshot_id = digest(inputs.snapshot)
            if args.command == "preflight":
                immutable(output / "PREFLIGHT.json", dict(identity=snapshot_id, snapshot=inputs.snapshot,
                    payload_checks="packed payload hashes; SQLite index and bounded payload; every production read verified",
                    experiment_started=False))
                immutable(output / "PRESERVED_INPUTS.json", inputs.evidence.files)
                atomic_json(output / "STATUS.json", dict(status="READY", stage="preflight", experiment_started=False,
                    reused_heads=38, new_heads=16, new_bootstrap_conditions=8, new_order_directions=4))
                print("[PREFLIGHT] PASS. No training, inference, or bootstrap executed.", flush=True)
                return
            require((output / "PREFLIGHT.json").exists(), "Run preflight first")
            require(read(output / "PREFLIGHT.json")["identity"] == snapshot_id,
                    "Preflight inputs changed; preserve this output and inspect changes")
            require(read(output / "PRESERVED_INPUTS.json") == inputs.evidence.files, "Preserved input set changed")
            for stage, fn in (("fig2", figure2), ("fig3a", figure3a), ("fig3b", figure3b)):
                if args.command not in ("all", stage):
                    continue
                if (output / stage / "COMPLETED.json").exists():
                    checked_receipt(output / stage)
                    print("[REUSE STAGE] " + stage, flush=True)
                    continue
                if stage == "fig2":
                    fn(inputs, output)
                else:
                    fn(inputs, output, args)
            # Include any additional committed artifacts referenced during the execution.
            original = read(output / "PRESERVED_INPUTS.json")
            require(all(inputs.evidence.files.get(p) == h for p, h in original.items()), "Historical input changed")
            inputs.evidence.verify()
            atomic_json(output / "PRESERVATION_VERIFIED.json", dict(original_inputs=len(original),
                        verified=True, additional_inputs={p: h for p, h in inputs.evidence.files.items() if p not in original}))
            if args.command == "all":
                finalize(output, protocol)
            else:
                atomic_json(output / "STATUS.json", dict(status="STAGE_COMPLETE", stage=args.command,
                    all_complete=False, next="Run all to reuse completed stages and finish the remaining work"))
                print("[STAGE COMPLETE] " + args.command, flush=True)
        except BaseException as exc:
            atomic_json(output / "STATUS.json", dict(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                        stage=args.command, error=str(exc), resume="Use the same command; completed jobs are retained"))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["preflight", "all", "fig2", "fig3a", "fig3b", "export", "status", "verify"])
    p.add_argument("--output", default=str(OUTPUT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--predict-batch-size", type=int, default=32)
    return p


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        print("[INTERRUPTED] Use the same command to resume from committed state.", flush=True)
        sys.exit(130)
    except Exception as exc:
        print("[FAILED] " + str(exc) + "; no successful completion for the failed stage.", flush=True)
        raise
