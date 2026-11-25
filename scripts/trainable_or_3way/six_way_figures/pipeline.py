"""Incremental experiments: bootstrap, fixed-head order tests, and PGF temporal heads."""
import time
from .common import *
from .inputs import Inputs
from scripts.trainable_or_3way.reinforcement_temporal import analysis as temporal
from scripts.trainable_or_3way.reinforcement_temporal import training
from scripts.trainable_or_3way.reinforcement_temporal.statistics import bootstrap, metric_table, BOOT_METRICS
from scripts.trainable_or_3way.reinforcement_temporal.order_recheck_core import predict_all, evaluate_pair


class Jobs:
    """Stage-boundary status plus actual inner-loop ETA from reused producers."""
    def __init__(self, output, stage, total):
        self.output, self.stage, self.total = Path(output), stage, total
        self.done = 0
        self.start = time.monotonic()

    def start_job(self, label):
        atomic_json(self.output / "STATUS.json", dict(status="RUNNING", stage=self.stage,
            job_index=self.done + 1, total_jobs=self.total, detail=label,
            elapsed_seconds=time.monotonic() - self.start,
            eta_note="Measured ETA is printed by each training/prediction/bootstrap loop; jobs have unequal costs"))
        print(f"[{self.stage}] job {self.done + 1}/{self.total}: {label} | "
              f"stage_elapsed={duration(time.monotonic() - self.start)}; stage ETA shown below", flush=True)

    def advance(self):
        self.done += 1


def figure2_row(source, target, condition, metrics, differences):
    require(differences.shape == (REPLICATES, len(BOOT_METRICS)) and np.isfinite(differences).all(),
            "Incomplete bootstrap difference arrays")
    q = .05 / (2 * len(DIRECTIONS) * len(CONDITIONS))
    row = dict(source=source, target=target, condition=condition, n=metrics["M4"]["n"],
               clusters=None, replicates=REPLICATES, family_size=12, individual_ci_level=1 - 2 * q)
    for i, metric in enumerate(BOOT_METRICS):
        row["baseline_" + metric] = metrics["M0"][metric]
        row["adjusted_" + metric] = metrics["M1"][metric]
        row["proposed_" + metric] = metrics["M4"][metric]
        row["delta_" + metric] = metrics["M4"][metric] - metrics["M1"][metric]
        lo, hi = np.quantile(differences[:, i], [.025, .975])
        row[metric + "_ci95_low"], row[metric + "_ci95_high"] = float(lo), float(hi)
    lo, hi = np.quantile(differences[:, 0], [q, 1-q])
    row["auc_ci_family_low"], row["auc_ci_family_high"] = float(lo), float(hi)
    return row


def old_bootstrap(inputs, source, target, condition, data, probabilities):
    job = inputs.old_boot(source, target, condition)
    report, boot = read(job / "METRICS.json"), read(job / "BOOTSTRAP.json")
    state = load_npz(job / "bootstrap_state.npz")
    require(boot["status"] == "COMPLETE" and boot["valid_replicates"] == REPLICATES
            and int(state["valid"]) == REPLICATES, "Old bootstrap is incomplete")
    # Old states used the insertion order of METRICS.metrics, confirmed by their CSV columns.
    keys = list(report["metrics"])
    with (job / "predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require([k for k in reader.fieldnames if k in keys] == keys, "Old bootstrap column order differs")
        records = list(reader)
    by_id = {r["sample_id"]: r for r in records}
    require(len(by_id) == len(records) == len(data["sample_id"]) and set(by_id) == set(data["sample_id"]),
            "Old bootstrap cohort differs")
    for i, sid in enumerate(data["sample_id"]):
        r = by_id[sid]
        require(int(r["clip_label"]) == data["labels"][i, 0] and r["cluster"] == data["cluster"][i],
                "Old bootstrap label/group differs")
        require(all(abs(float(r[m]) - probabilities[m][i]) < 1e-12 for m in ("M0", "M1", "M4")),
                "Old bootstrap predictions differ")
    values = state["values"]
    require(values.shape == (REPLICATES, len(keys), len(BOOT_METRICS)), "Old bootstrap shape differs")
    return values[:, keys.index("M4")] - values[:, keys.index("M1")], boot["clusters"]


def figure2(inputs, output):
    root = output / "fig2"
    jobs = Jobs(output, "FIG2", 12)
    rows = []
    for source, target in DIRECTIONS:
        for c in CONDITIONS:
            jobs.start_job(f"{source}->{target} {c} paired bootstrap")
            data = inputs.raw[source, c, target, "test"]
            p, _, base = temporal.fused(data["raw"], data["raw"][:, 2:], inputs.historical_fusion(source, c))
            probabilities = dict(M0=base["M0"], M1=base["M1"], M4=p)
            metrics = metric_table(data["labels"][:, 0], probabilities)
            identity = digest(dict(input=inputs.proofs[source, c, target, "test"], source=source, target=target,
                                   condition=c, scores={k: temporal.hash_array(v) for k, v in probabilities.items()},
                                   replicates=REPLICATES, family=12, code=implementation()))
            folder = root / source / target / c
            if completed(folder, identity):
                rows.append(read(folder / "RESULT.json"))
                jobs.advance()
                continue
            if "PGF" not in (source, target):
                differences, clusters = old_bootstrap(inputs, source, target, c, data, probabilities)
                reuse = dict(origin=str(inputs.old_boot(source, target, c)), resampling=False)
                files = ["REUSE.json"]
                atomic_json(folder / "REUSE.json", reuse)
            else:
                boot = bootstrap(data["labels"][:, 0], probabilities, data["cluster"], folder, identity,
                    replicates=REPLICATES, seed=1701 + int(digest([source, target, c, "fig2"])[:7], 16),
                    pairs=(("M4", "M1"),))
                require(boot["status"] == "COMPLETE", "Insufficient bootstrap replicates")
                values = load_npz(folder / "bootstrap_state.npz")["values"]
                differences = values[:, 2] - values[:, 1]
                clusters = boot["clusters"]
                atomic_json(folder / "BOOTSTRAP.json", boot)
                atomic_npz(folder / "predictions.npz", sample_id=data["sample_id"], labels=data["labels"],
                           cluster=data["cluster"], **probabilities)
                files = ["BOOTSTRAP.json", "bootstrap_state.npz", "predictions.npz"]
            row = figure2_row(source, target, c, metrics, differences)
            row["clusters"] = clusters
            atomic_json(folder / "RESULT.json", row)
            atomic_npz(folder / "paired_differences.npz", values=differences, metrics=np.array(BOOT_METRICS))
            finish(folder, identity, files + ["RESULT.json", "paired_differences.npz"])
            rows.append(row)
            jobs.advance()
    require(len(rows) == 12, "Figure 2 family incomplete")
    atomic_json(root / "SUMMARY.json", dict(rows=rows, family_size=12, family_complete=True,
        comparator="an artifact-only baseline with learned score adjustment",
        scope="full test cohorts; paired cluster bootstrap; AUC family-wise coverage target 95%",
        fixed_models=True, training_variability_included=False))
    csv_write(root / "SUMMARY.csv", rows)
    finish(root, digest(rows), ["SUMMARY.json", "SUMMARY.csv"])


def figure3a(inputs, output, args):
    root = output / "fig3a"
    jobs = Jobs(output, "FIG3A", 6)
    reports = []
    for source, target in DIRECTIONS:
        jobs.start_job(f"{source}->{target} ordered/reversed/3 shuffles")
        if "PGF" not in (source, target):
            for c in CONDITIONS:
                folder = import_job(ORDER / "diagnostics" / source / target / c,
                                    root / "diagnostics" / source / target / c, inputs.evidence)
                reports.append(read(folder / "DIAGNOSTIC.json"))
        else:
            pairs = {}
            for mod in ("video", "audio"):
                cache = inputs.cache(target, "test", mod)
                try:
                    pairs[mod] = predict_all(cache, inputs.heads[source][mod]["path"],
                        root / "predictions" / source / target / mod,
                        device=args.device, batch_size=args.predict_batch_size)
                finally:
                    cache.close()
            reports.extend(evaluate_pair(inputs, source, target, pairs, root / "diagnostics" / source / target))
        jobs.advance()
    rows = []
    for r in reports:
        for order, metrics in r["metrics"].items():
            rows.append(dict(source=r["source"], target=r["target"], condition=r["condition"], order=order,
                **{k: metrics[k] for k in ("n", "roc_auc", "fake_f1", "real_fpr", "fake_recall")},
                **{"delta_" + k: v for k, v in r["changes"][order]["delta"].items()}))
    require(len(reports) == 12, "Figure 3a incomplete")
    atomic_json(root / "SUMMARY.json", dict(rows=rows, conditions=12, transforms=list(TRANSFORMS),
        reference="common FP32 ordered inference; historical rounding drift is audited separately",
        heads_fixed=True, fusion_fixed=True, target_fitting=False))
    csv_write(root / "SUMMARY.csv", rows)
    finish(root, digest(rows), ["SUMMARY.json", "SUMMARY.csv"])


def head_paths(inputs, output, source, seed, variant, args):
    paths = {}
    for mod in ("video", "audio"):
        key = source, seed, variant, mod
        if key in inputs.reusable_heads:
            paths[mod] = inputs.reusable_heads[key]
            print("[REUSE HEAD] " + str(paths[mod]), flush=True)
            continue
        require(source == "PGF", "Never retrain a completed FAV/AVDF1M head")
        train_cache = inputs.cache(source, "train", mod)
        dev_cache = inputs.cache(source, "dev", mod)
        folder = output / "fig3b/heads" / source / f"seed_{seed}" / variant / mod
        runtime = dict(torch=str(torch.__version__), numpy=np.__version__, device=args.device,
                       threads=args.threads, deterministic=True, tf32=False)
        identity = model_identity(source, mod, variant, seed, train_cache.proof, dev_cache.proof,
                                  TRAIN, training_code_identity(), runtime)
        try:
            if not completed(folder, identity):
                training.fit(training.Task(train_cache), training.Task(dev_cache), variant, seed,
                             source, folder, identity, device=args.device)
        finally:
            train_cache.close()
            dev_cache.close()
        paths[mod] = folder / "best.pt"
    return paths


def figure3b(inputs, output, args):
    root = output / "fig3b"
    jobs = Jobs(output, "FIG3B", 9)
    rows = []
    for source in DATASETS:
        targets = [t for s, t in DIRECTIONS if s == source]
        for seed in SEEDS:
            jobs.start_job(f"{source} seed={seed}: T0/T1/T2 heads, source-dev fusion, targets={targets}")
            paths, models = {}, {}
            for variant in VARIANTS:
                paths[variant] = head_paths(inputs, output, source, seed, variant, args)
                if source != "PGF":
                    folder = TEMPORAL / "fusion" / source / f"seed_{seed}" / variant
                    inputs.evidence.job(folder)
                    lock = read(folder / "LOCKED_BEFORE_TEST.json")
                    require(lock["inputs"]["heads"] == {m: sha(p) for m, p in paths[variant].items()},
                            "Historical temporal fusion/head mismatch")
                    require(not lock["target_training"] and lock["fusion_sha256"] == sha(folder / "FUSION.json"),
                            "Historical source-dev fusion lock differs")
                    models[variant] = read(folder / "FUSION.json")
                elif seed == 42 and variant == "T2":
                    models[variant] = inputs.models["PGF"]
                    atomic_json(root / "fusion/PGF/seed_42/T2/REUSE.json",
                                dict(origin=str(PGF / "fusion"), fitting=False,
                                     fusion_sha256=sha(PGF / "fusion/FUSION.json")))
                else:
                    pair = temporal.predict_pair(inputs, source, "dev", paths[variant], variant,
                        root / "predictions" / source / f"seed_{seed}" / variant / source / "dev",
                        args.device, args.predict_batch_size)
                    models[variant] = temporal.fit_locked(inputs, source, seed, variant, pair, paths[variant],
                        root / "fusion" / source / f"seed_{seed}" / variant)
            # Fit/lock every source-dev model before new target predictions for this seed.
            for target in targets:
                if "PGF" not in (source, target):
                    for c in CONDITIONS:
                        for cohort in ("full", "common_emotion"):
                            rel = Path("evaluations") / source / f"seed_{seed}" / target / c / cohort
                            folder = import_job(TEMPORAL / rel, root / rel, inputs.evidence)
                            rows.append(read(folder / "METRICS.json"))
                    continue
                pairs = {}
                for variant in VARIANTS:
                    if source == "PGF" and seed == 42 and variant == "T2":
                        pairs[variant] = {}
                        for mod in ("video", "audio"):
                            p = inputs.evidence.committed(PGF / "emotion_scores" / target / "test" / mod, "scores.npz")
                            pair = load_npz(p)
                            require(np.array_equal(pair["sample_id"], inputs.cohorts[target, "test"]["sample_id"]),
                                    "Reused PGF T2 prediction IDs differ")
                            pairs[variant][mod] = pair
                    else:
                        pairs[variant] = temporal.predict_pair(inputs, target, "test", paths[variant], variant,
                            root / "predictions" / source / f"seed_{seed}" / variant / target / "test",
                            args.device, args.predict_batch_size)
                for c in CONDITIONS:
                    rows.extend(temporal.evaluate_heads(inputs, source, target, seed, c, pairs, models,
                        root / "evaluations" / source / f"seed_{seed}" / target / c, REPLICATES))
            jobs.advance()
    scope = dict(sources=list(DATASETS), targets=list(DATASETS), seeds=list(SEEDS), conditions=list(CONDITIONS),
        variants=list(VARIANTS), bootstrap=REPLICATES, primary_cohort="full", comparisons=["T2-T0", "T2-T1"],
        family="six cross directions x B/C x two contrasts x three head seeds",
        in_domain_included=False, code=implementation())
    require(len(rows) == 6 * 2 * 3 * 2, "Figure 3b evaluations incomplete")
    folder = temporal.aggregate(root, scope, rows)
    summary = read(folder / "SUMMARY.json")
    require(summary["family_complete"] and summary["auc_family_size"] == 72, "Temporal AUC family incomplete")
    atomic_json(root / "SUMMARY.json", summary)
    atomic_text(root / "SUMMARY.csv", (folder / "SUMMARY.csv").read_text(encoding="utf-8"))
    finish(root, digest(scope), ["SUMMARY.json", "SUMMARY.csv"])
