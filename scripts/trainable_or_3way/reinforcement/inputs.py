"""Adapters for the three completed score runs, with strict saved-receipt checks."""
import csv
from pathlib import Path
import numpy as np
from scipy.special import expit
from .common import ROOT, require, read, sha, digest, resolve, child, Progress, load_npz

DEFAULTS = dict(
    fav_root="reports/trainable_or_fav_v1/seed_42",
    fav_cross_root="reports/trainable_or_cross_dataset_v1/source_FAV/seed_42",
    workstation_root="reports/workstation_trainable_final_fusion_v1/source_AVDF1M/seed_42",
    registry="configs/fixed_manifests.json",
)
METHODS = {
    "M0": "Artifact OR", "M1": "Artifact-affine", "M2": "Emotion OR",
    "M3": "Hierarchical OR", "M4": "Trainable Final Fusion",
    "M5": "Artifact + visual emotion", "M6": "Artifact + audio emotion",
}
EPSILON = 1e-6

def design(a, e=None):
    def logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), EPSILON, 1-EPSILON)
        return np.log(p)-np.log1p(-p)
    cols = [logit(a)]
    if e is not None:
        cols.append(logit(e))
    return np.column_stack(cols + [np.ones(len(a))])

def predict(model, a, e=None):
    require(model["format"] == "trainable_final_score_v1", "Unknown fusion model format")
    require(model["split"] == "dev" and model["epsilon"] == EPSILON
            and model["threshold"] == .5 and model["regularization"] == .01,
            "Fusion calibration protocol differs")
    require(model["emotion_used"] == (e is not None), "Fusion input dimension mismatch")
    x = design(a, e)
    w = np.asarray(model["weights"], dtype=np.float64)
    require(w.shape == (x.shape[1],) and np.isfinite(w).all(), "Invalid fusion weights")
    return expit(x @ w)

def scores(raw, models):
    a = 1-(1-raw[:, 0])*(1-raw[:, 1])
    valid = np.isfinite(raw[:, 2:]).all(axis=1)
    e = 1-(1-raw[:, 2])*(1-raw[:, 3])
    affine = predict(models["artifact_affine"], a)
    p = dict(M0=a, M1=affine, M2=e, M3=1-(1-a)*(1-e), M4=affine.copy())
    p["M4"][valid] = predict(models["trainable_final"], a[valid], e[valid])
    for key, branch in (("M5", 2), ("M6", 3)):
        if key in models:
            # Same complete-emotion gate for all augmented methods, fixed before evaluation.
            p[key] = affine.copy()
            p[key][valid] = predict(models[key], a[valid], raw[valid, branch])
    return p, valid

def receipt(folder, preserved):
    folder = Path(folder)
    run = read(folder/"RUN.json")
    done = read(folder/"COMPLETED.json")
    expected = done.get("identity", done.get("run"))
    require(expected == digest(run), "Completed run identity mismatch: " + str(folder))
    require(done.get("files"), "Empty completed input receipt: " + str(folder))
    paths = {"RUN.json": sha(folder/"RUN.json"), "COMPLETED.json": sha(folder/"COMPLETED.json"), **done["files"]}
    bar = Progress("P0 receipt " + folder.name, len(paths))
    for i, (name, expected_hash) in enumerate(paths.items()):
        p = child(folder, name)
        require(p.is_file() and sha(p) == expected_hash, "Missing/changed committed input: " + str(p))
        preserved[str(p)] = expected_hash
        bar.update(i+1)
    return run, done

def checked(path, expected, preserved):
    path = resolve(path)
    require(path.is_file() and sha(path) == expected, "Missing/changed referenced input: " + str(path))
    preserved[str(path)] = expected
    return path

def load_scores(path, models):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    require(rows, "Empty score CSV: " + str(path))
    ids = [r["sample_id"] for r in rows]
    require(len(set(ids)) == len(ids) and all(ids), "Duplicate/empty prediction IDs")
    labels = np.array([[int(r[k]) if r[k] not in ("", "None") else -1
                        for k in ("clip_label", "video_label", "audio_label")] for r in rows])
    require(np.isin(labels[:, 0], [0, 1]).all() and np.isin(labels[:, 1:], [-1, 0, 1]).all(), "Invalid labels")
    raw = np.array([[float(r[k]) if r[k] != "" else np.nan for k in ("Av", "Aa", "Ev", "Ea")] for r in rows])
    require(np.isfinite(raw[:, :2]).all(), "Artifact scores cannot be missing")
    require(not np.isinf(raw).any() and ((raw[np.isfinite(raw)] >= 0) & (raw[np.isfinite(raw)] <= 1)).all(),
            "Scores outside [0,1] or infinite")
    expected, valid = scores(raw, models)
    aliases = dict(M0=("artifact_or", "baseline"), M1=("artifact_affine",), M2=("emotion_or",),
                   M3=("hierarchical_or",), M4=("trainable_final",))
    for key, names in aliases.items():
        name = next((n for n in names if n in rows[0]), None)
        require(name is not None, "Required stored score absent: " + key)
        recorded = np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])
        mask = valid if key in ("M2", "M3") else np.ones(len(rows), bool)
        require(np.allclose(recorded[mask], expected[key][mask], atol=1e-10, rtol=0, equal_nan=False),
                "Saved score/formula differs: " + name + " in " + str(path))
    for field in ("emotion_complete", "included_primary"):
        if field in rows[0]:
            actual = np.array([r[field].lower() in ("1", "true") for r in rows])
            require(np.array_equal(actual, valid), "Stored emotion mask differs: " + field)
    if "used_artifact_fallback" in rows[0]:
        actual = np.array([r["used_artifact_fallback"].lower() in ("1", "true") for r in rows])
        require(np.array_equal(actual, ~valid), "Stored fallback mask differs")
    return dict(sample_id=np.array(ids), recorded_group=np.array([r.get("group_id", r.get("group", "")) for r in rows]),
                labels=labels, raw=raw)

def load_inputs(config):
    preserved, models, cohorts, protocols = {}, {}, {}, {}
    fav = resolve(config["fav_root"])
    cross = resolve(config["fav_cross_root"])
    ws = resolve(config["workstation_root"])
    fr, _ = receipt(fav, preserved)
    cr, _ = receipt(cross, preserved)
    wr, _ = receipt(ws, preserved)
    ff, wf = read(fav/"FUSION.json"), read(ws/"FUSION.json")
    fl, cl, wl = (read(fav/"LOCKED_BEFORE_TEST.json"), read(cross/"LOCKED_BEFORE_TARGET.json"),
                  read(ws/"LOCKED_BEFORE_TEST.json"))
    require(fl["identity"] == digest(fr) and fl["fusion_sha256"] == sha(fav/"FUSION.json"), "FAV fusion lock differs")
    require(cl["identity"] == digest(cr) and cl["fusion"] == ff, "FAV cross fusion differs from source")
    for key, name in (("run_sha256", "RUN.json"), ("completion_sha256", "COMPLETED.json"),
                      ("lock_sha256", "LOCKED_BEFORE_TEST.json"), ("fusion_sha256", "FUSION.json")):
        require(cl["source_receipt"][key] == sha(fav/name), "Cross source receipt mismatch: " + name)
    require(wl["run_identity"] == digest(wr) and wl["fusion_sha256"] == sha(ws/"FUSION.json"), "Workstation lock differs")
    require(wf["run_identity"] == digest(wr) and wr["source"] == "AVDF1M"
            and wr["conditions"] == ["B", "C"] and not wr["target_training"], "Workstation source/conditions differ")
    require(cr["source"] == "FAV" and cr["config"]["zero_shot"] and not cr["config"]["target_training"],
            "Unexpected FAV cross protocol")
    models["FAV__historical"] = ff
    for cond in ("B", "C"):
        models["AVDF1M__"+cond] = wf["conditions"][cond]
    for mod, h in fl["heads"].items():
        checked(fav/"heads"/mod/"best.pt", h, preserved)
    for name, h in wr["emotion_models"].items():
        checked(resolve(wr["emotion_root"])/name, h, preserved)
    for key, spec in wr["artifact_inputs"].items():
        folder = resolve(wr["artifact_root"])/key
        checked(folder/"RUN.json", spec["run"], preserved)
        checked(folder/"COHORT.json", spec["cohort"], preserved)
        ar, _ = receipt(folder, preserved)
        require(digest(ar) == spec["expected_identity"] and not ar["smoke_only"], "Artifact producer differs")
        protocols[key] = ar["compatibility"]
    subset = wr["source_subset"]
    selection = checked(subset["path"], subset["sha256"], preserved)
    for name, h in subset["files"].items():
        checked(child(selection.parent, name), h, preserved)
    for source, cond, target, split, path in (
        [("FAV", "historical", "FAV", s, fav/s/"predictions.csv") for s in ("dev", "test")]
        + [("FAV", "historical", t, "test", cross/"evaluation"/t/"predictions.csv")
           for t in config["targets"] if t != "FAV"]
        + [("AVDF1M", c, t, s, ws/"evaluation"/c/t/s/"predictions.csv")
           for c in ("B", "C") for t, s in [("AVDF1M", "dev")] + [(t, "test") for t in config["targets"]]]
    ):
        key = "__".join((source, cond, target, split))
        require(str(path.resolve()) in preserved, "Score CSV absent from completion receipt: " + str(path))
        data = load_scores(path, models[source+"__"+cond])
        cohorts[key] = dict(source=source, condition=cond, target=target, split=split,
                            csv=str(path), data=data)
    # Source training cohorts: actual selected IDs, never all rows merely labelled train.
    fav_cache = resolve(fr["source"])/"cache"
    checked(resolve(fr["source"])/"COMPLETED.json", fr["source_completion_sha256"], preserved)
    train_receipt = read(fav_cache/"video/train/COMPLETED.json")
    idx = checked(fav_cache/"video/train/index.npz", train_receipt["files"]["index.npz"], preserved)
    preserved[str(fav_cache/"video/train/COMPLETED.json")] = sha(fav_cache/"video/train/COMPLETED.json")
    train = load_npz(idx)
    fav_cohorts = read(fav/"TRAIN_DEV_COHORTS.json")
    for split, current in (("train", train), ("dev", cohorts["FAV__historical__FAV__dev"]["data"])):
        expected = fav_cohorts[split]
        require(len(current["sample_id"]) == expected["n"]
                and digest(current["sample_id"].tolist()) == expected["ordered_ids_sha256"]
                and digest(current["labels"].tolist()) == expected["labels_sha256"], "FAV selected training/dev cohort differs")
    training = {"FAV": [dict(sample_id=str(s), group_id=str(g), labels=list(map(int, y)), split="train")
                        for s, g, y in zip(train["sample_id"], train["group"], train["labels"])]}
    with (selection.parent/"train.jsonl").open(encoding="utf-8-sig") as f:
        import json
        training["AVDF1M"] = [json.loads(line) for line in f if line.strip()]
    with (selection.parent/"dev.jsonl").open(encoding="utf-8-sig") as f:
        dev_rows = [json.loads(line) for line in f if line.strip()]
    dev = cohorts["AVDF1M__B__AVDF1M__dev"]["data"]
    require(len(training["AVDF1M"]) == 14724 and len(dev_rows) == 3147, "Unexpected AVDF1M selected train/dev size")
    selected = {r["sample_id"]: r for r in dev_rows}
    require(len(selected) == len(dev_rows) and set(selected) == set(dev["sample_id"]), "AVDF1M selected dev ID set differs")
    for sid, label in zip(dev["sample_id"], dev["labels"]):
        require([selected[sid][k] for k in ("clip_label", "video_label", "audio_label")] == label.tolist(),
                "Selected dev labels differ")
    validate_shared_cohorts(cohorts)
    # Inventory backbone availability independently of saved-score reuse eligibility.
    assets = []
    artifact_base = ROOT/"checkpoints/experiments/x3a_reproduction/fav_indomain_v1/unified_4gpu/x3a_4gpu_001"
    for mod, name in (("video", "x3d"), ("audio", "aasist")):
        r = read(fav_cache/mod/"train/COMPLETED.json")
        expected = r["identity"]["source"]["artifact_checkpoint_sha256"]
        p = artifact_base/name/"best.pt"
        actual = sha(p) if p.is_file() else None
        assets.append(dict(source="FAV", modality=mod, path=str(p), expected_sha256=expected,
                           actual_sha256=actual, status="MATCH" if actual==expected else "MISSING_OR_DIFFERENT"))
        if actual is not None:
            preserved[str(p)] = actual
    bundle = protocols["dev_AVDF1M"]["bundle"]
    for mod in ("x3d", "aasist"):
        candidates = [p for p in resolve(bundle["path"]).rglob("best_state_dict.pth") if mod in p.parts]
        matches = [p for p in candidates if sha(p) == bundle[mod]]
        assets.append(dict(source="AVDF1M", modality=mod, expected_sha256=bundle[mod],
                           matching_paths=[str(p) for p in matches], status="MATCH" if matches else "MISSING_OR_DIFFERENT"))
        for p in matches:
            preserved[str(p)] = bundle[mod]
    protocols["checkpoint_inventory"] = assets
    # Record receipt metadata only: do not claim to verify 90+ GiB of embedding blobs.
    tokens = []
    token_roots = [fav_cache, Path("D:/datasets/experiment_cache/trainable_or_3way_v1")]
    for token_root in token_roots:
        if not token_root.exists():
            tokens.append(dict(path=str(token_root), status="MISSING"))
            continue
        for p in sorted(token_root.rglob("COMPLETED.json")):
            r = read(p)
            fields = {k: r[k] for k in ("n", "tokens", "valid_tokens", "missing", "one_valid_token",
                       "valid_length_min", "valid_length_median", "valid_length_max") if k in r}
            tokens.append(dict(path=str(p.parent), receipt_sha256=sha(p), receipt=fields,
                               token_blob_validation="NOT_PERFORMED"))
            preserved[str(p.resolve())] = sha(p)
    return cohorts, models, training, protocols, tokens, preserved, {"FAV": fr, "AVDF1M": wr}

def validate_shared_cohorts(cohorts):
    """Identical target cohorts, plus identical AASIST/emotion scores across B and C."""
    for target in sorted({c["target"] for c in cohorts.values()}):
        values = [c["data"] for c in cohorts.values() if c["target"] == target and c["split"] == "test"]
        aligned = [(d, np.argsort(d["sample_id"])) for d in values]
        first, idx = aligned[0]
        for d, order in aligned[1:]:
            require(np.array_equal(first["sample_id"][idx], d["sample_id"][order])
                    and np.array_equal(first["labels"][idx], d["labels"][order]), "Target test cohort differs across methods/sources: "+target)
    for key,c in cohorts.items():
        if c["condition"] != "B":
            continue
        other = cohorts[key.replace("__B__", "__C__", 1)]["data"]
        d = c["data"]
        i,j = np.argsort(d["sample_id"]),np.argsort(other["sample_id"])
        require(np.array_equal(d["sample_id"][i],other["sample_id"][j]), "B/C IDs differ")
        require(np.allclose(d["raw"][i,1:],other["raw"][j,1:],rtol=0,atol=0,equal_nan=True),
                "B/C must share identical AASIST and emotion scores")

def join_manifest(cohorts, training, registry_path, preserved):
    registry_path = resolve(registry_path)
    registry = read(registry_path)
    preserved[str(registry_path)] = sha(registry_path)
    datasets = sorted({c["target"] for c in cohorts.values()} | set(training))
    maps, overlaps = {}, {}
    for dataset in datasets:
        wanted = set()
        for c in cohorts.values():
            if c["target"] == dataset:
                wanted.update(c["data"]["sample_id"].tolist())
        wanted.update(r["sample_id"] for r in training.get(dataset, []))
        spec = registry["datasets"][dataset]
        path = checked(spec["manifest"], spec["sha256"], preserved)
        found = {}
        bar = Progress("P0 manifest "+dataset, path.stat().st_size)
        import json
        with path.open("rb") as f:
            count = 0
            for line in f:
                count += len(line)
                if line.strip():
                    r = json.loads(line)
                    sid = r["sample_id"]
                    if sid in wanted:
                        require(sid not in found, "Duplicate frozen ID: " + sid)
                        found[sid] = {k: r.get(k) for k in ("sample_id", "split", "group_id", "identity_id",
                            "source_id", "original_group_id", "official_split", "group_definition",
                            "clip_label", "video_label", "audio_label")}
                bar.update(count)
        require(set(found) == wanted, f"Missing {len(wanted-set(found))} IDs in frozen manifest: {dataset}")
        maps[dataset] = found
    def validate(dataset, sid, split, labels, group):
        m = maps[dataset][sid]
        expected = [-1 if m[k] is None else int(m[k]) for k in ("clip_label", "video_label", "audio_label")]
        require(m["split"] == split and expected == list(map(int, labels)), "Frozen split/labels mismatch: " + sid)
        require(bool(m["group_id"]), "Missing frozen group: " + sid)
        # Historical FAV CSV uses the same source identity without the namespace.
        require(group == m["group_id"] or (dataset == "FAV" and group == m["identity_id"]
                and m["group_id"] == "FAV:source:"+group), "Prediction/frozen group mismatch: " + sid)
        return m
    for key, c in cohorts.items():
        d = c["data"]
        matched = [validate(c["target"], str(s), c["split"], y, str(g))
                   for s, y, g in zip(d["sample_id"], d["labels"], d["recorded_group"])]
        d["cluster"] = np.array([r["group_id"] for r in matched])
        d["original_group"] = np.array([r["original_group_id"] or "" for r in matched])
    for source, rows in training.items():
        train_ids, train_groups = set(), set()
        for r in rows:
            require(r["sample_id"] not in train_ids, "Duplicate selected training ID")
            labels = r.get("labels", [r.get(k) for k in ("clip_label", "video_label", "audio_label")])
            m = validate(source, r["sample_id"], "train", labels, r["group_id"])
            train_ids.add(r["sample_id"])
            train_groups.add(m["group_id"])
        per_split = {"train": (train_ids, train_groups)}
        for split in ("dev", "test"):
            c = next(c for c in cohorts.values() if c["source"] == source and c["target"] == source and c["split"] == split)
            per_split[split] = (set(c["data"]["sample_id"]), set(c["data"]["cluster"]))
        detail = {}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            ni, ng = len(per_split[a][0] & per_split[b][0]), len(per_split[a][1] & per_split[b][1])
            require(ni == 0 and ng == 0, f"Source {source} {a}/{b} ID/group overlap: {ni}/{ng}")
            detail[a+"__"+b] = dict(overlapping_ids=ni, overlapping_verified_groups=ng)
        overlaps[source] = dict(splits={s: dict(n=len(v[0]), groups=len(v[1])) for s,v in per_split.items()}, checks=detail)
    return maps, overlaps
