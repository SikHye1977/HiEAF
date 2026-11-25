"""Bind completed audit and prediction receipts; never read media or load models."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from scripts.trainable_or_3way.reinforcement.common import (
    ROOT, require, read, sha, resolve, digest, safe_output)

DEFAULTS = dict(
    audit="reports/diagnostics/split_overlap_audit_20261119",
    p1="reports/fav_avdf1m_reinforcement_v1/p1",
    p2="reports/fav_avdf1m_reinforcement_p2_v1")
OUTPUT = "reports/fav_avdf1m_overlap_sensitivity_v1"
METHODS = ("M0", "M1", "M4")
NAMES = dict(M0="Baseline (artifact-only OR)",
             M1="an artifact-only baseline with learned score adjustment",
             M4="Proposed (emotion + trainable final fusion)")
COHORTS = ("full", "metadata_unshared", "metadata_plus_filename_unshared")
DIRECTIONS = (("FAV", "AVDF1M"), ("AVDF1M", "FAV"))
FORMAT = "known_person_overlap_sensitivity_v1"


def file_map(folder, mapping):
    return {(Path(folder)/k).resolve(): v for k, v in mapping.items()}


class Pins:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), "Missing input: " + str(path))
        actual = sha(path)
        require(expected is None or actual == expected, "Changed committed input: " + str(path))
        require(str(path) not in self.files or self.files[str(path)] == actual,
                "Input changed during reading: " + str(path))
        self.files[str(path)] = actual
        return path

    def verify(self):
        for path, expected in self.files.items():
            require(sha(path) == expected, "Preserved input changed: " + path)


def load_metadata(path):
    rows = {dataset: {} for dataset in ("FAV", "AVDF1M")}
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] not in rows:
                continue
            require(r["split"] in ("train", "dev", "test") and r["label"] in (0, 1),
                    "Invalid metadata split/label")
            require(r["sample_id"] not in rows[r["dataset"]], "Duplicate metadata ID")
            for key in ("persons", "candidates"):
                require(isinstance(r[key], list) and all(re.fullmatch(r"id\d+", p) for p in r[key]),
                        "Invalid person namespace: " + r["sample_id"])
            require(r["persons"] and r["source_person"] in r["persons"],
                    "Unknown explicit person; cannot call it unshared: " + r["sample_id"])
            require(r["group_id"], "Missing manifest cluster")
            rows[r["dataset"]][r["sample_id"]] = r
    for dataset in rows:
        require({r["split"] for r in rows[dataset].values()} == {"train", "dev", "test"},
                "Missing source/test split: " + dataset)
    return rows


def make_masks(metadata, source, target):
    source_rows = [r for r in metadata[source].values() if r["split"] in ("train", "dev")]
    known = set().union(*(set(r["persons"]) for r in source_rows))
    strict = set().union(*(set(r["persons"]) | set(r["candidates"]) for r in source_rows))
    target_rows = sorted((r for r in metadata[target].values() if r["split"] == "test"),
                         key=lambda r: r["sample_id"])
    require(known and target_rows, "Empty source or target")
    overlap = [sorted(known.intersection(r["persons"])) for r in target_rows]
    candidates = [sorted(strict.intersection(set(r["persons"]) | set(r["candidates"])))
                  for r in target_rows]
    masks = np.array([np.ones(len(target_rows), dtype=bool),
                      [not x for x in overlap], [not x for x in candidates]], dtype=bool)
    require(np.all(~masks[2] | masks[1]), "Candidate mask must be a subset of explicit mask")
    return dict(rows=target_rows, masks=masks, overlap=overlap, candidates=candidates,
                source_persons=len(known), source_candidate_persons=len(strict))


def load_scores(path, target_rows):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "clip_label", "cluster", "emotion_complete", *METHODS}
        require(required <= set(reader.fieldnames or []), "Prediction columns missing")
        saved = {}
        for r in reader:
            require(r["sample_id"] not in saved, "Duplicate prediction ID")
            require(r["clip_label"] in ("0", "1") and r["emotion_complete"] in ("True", "False"),
                    "Invalid prediction label or emotion flag")
            saved[r["sample_id"]] = r
    require(set(saved) == {r["sample_id"] for r in target_rows},
            "Prediction IDs differ from audited actual target test")
    ordered = [saved[r["sample_id"]] for r in target_rows]
    for r, s in zip(target_rows, ordered):
        require(r["label"] == int(s["clip_label"]) and r["group_id"] == s["cluster"],
                "Prediction label/cluster differs from audit: " + r["sample_id"])
    scores = np.array([[float(r[k]) for k in METHODS] for r in ordered], dtype=np.float64)
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
            "Non-finite or out-of-range saved probability")
    return dict(ids=np.array([r["sample_id"] for r in target_rows]),
                y=np.array([r["label"] for r in target_rows], dtype=np.int64),
                clusters=np.array([r["group_id"] for r in target_rows]), scores=scores,
                emotion=np.array([r["emotion_complete"] == "True" for r in ordered]))


def describe(data, masks):
    result = {}
    for name, mask in zip(COHORTS, masks):
        y, clusters = data["y"][mask], data["clusters"][mask]
        require(set(y.tolist()) == {0, 1} and len(np.unique(clusters)) > 1,
                "Both classes and multiple clusters required: " + name)
        result[name] = dict(n=len(y), real=int((y == 0).sum()), fake=int(y.sum()),
                            excluded=int((~mask).sum()), clusters=len(np.unique(clusters)),
                            real_clusters=len(np.unique(clusters[y == 0])),
                            fake_clusters=len(np.unique(clusters[y == 1])),
                            emotion_missing=int((~data["emotion"][mask]).sum()))
    return result


def prepare(paths):
    paths = {k: resolve(v) for k, v in paths.items()}
    pins = Pins()
    audit = paths["audit"]
    receipt = read(pins.add(audit/"COMPLETED.json"))
    require(receipt.get("status") == "PASS" and all(receipt["checks"].values()), "Audit not PASS")
    amap = file_map(ROOT, receipt["output_sha256"])
    audit_names = ("used_sample_provenance.jsonl", "candidate_person_disjoint_test_counts.json",
                   "candidate_person_disjoint_test_eligibility_NOT_APPLIED.csv", "COHORT_VERIFICATION.json")
    for name in audit_names:
        p = (audit/name).resolve()
        require(p in amap, "Audit file not bound to receipt: " + name)
        pins.add(p, amap[p])
    verification = read(audit/"COHORT_VERIFICATION.json")
    require(verification["checks"] and all(r["pass_check"] for r in verification["checks"]),
            "Audit actual-cohort verification failed")
    metadata = load_metadata(audit/audit_names[0])
    with (audit/audit_names[2]).open(encoding="utf-8-sig", newline="") as f:
        eligibility = {}
        for r in csv.DictReader(f):
            key = (r["source"], r["target"], r["sample_id"])
            require(key not in eligibility and r["keep_by_known_person_metadata"] in ("0", "1"),
                    "Duplicate or invalid audited eligibility")
            eligibility[key] = r
    counts = {(r["source"], r["target"]): r for r in read(audit/audit_names[1])}
    plans, jobs, descriptions, used_keys = {}, {}, {}, set()
    for source, target in DIRECTIONS:
        plan = make_masks(metadata, source, target)
        plans[source+"__"+target] = plan
        for i, r in enumerate(plan["rows"]):
            key = (source, target, r["sample_id"])
            require(key in eligibility, "Audited eligibility ID missing")
            e = eligibility[key]
            require(int(e["label"]) == r["label"] and
                    int(e["keep_by_known_person_metadata"]) == int(plan["masks"][1, i]) and
                    set(filter(None, e["overlapping_ids"].split(";"))) == set(plan["overlap"][i]),
                    "Recomputed filter differs from audit: " + r["sample_id"])
            used_keys.add(key)
        root = paths["p2"] if source == "FAV" else paths["p1"]
        parent = read(pins.add(root/"COMPLETED.json"))
        parent_map = file_map(root, parent["files"])
        for condition in ("B", "C"):
            name = f"{source}__{condition}__{target}__test"
            folder = root/("evaluation/" if source == "FAV" else "")/name/"full"
            leaf = (folder/"COMPLETED.json").resolve()
            require(leaf in parent_map, "Leaf not committed by parent receipt")
            record = read(pins.add(leaf, parent_map[leaf]))
            for filename in ("predictions.csv", "METRICS.json"):
                p = (folder/filename).resolve()
                require(p in parent_map and parent_map[p] == record["files"][filename],
                        "Parent/leaf receipt disagree: " + str(p))
                pins.add(p, parent_map[p])
            recorded = read(folder/"METRICS.json")
            require([recorded[k] for k in ("source", "target", "condition", "cohort")] ==
                    [source, target, condition, "full"], "Wrong prediction provenance")
            data = load_scores(folder/"predictions.csv", plan["rows"])
            desc = describe(data, plan["masks"])
            old = counts[(source, target)]
            require(old["original_n"] == desc["full"]["n"] and all(
                old["retained_"+k] == desc["metadata_unshared"][k] for k in ("n", "real", "fake")),
                "Retained cohort counts differ from audit")
            jobs[name] = dict(source=source, target=target, condition=condition,
                              data=data, masks=plan["masks"], recorded=recorded)
            descriptions[name] = desc
    require(used_keys == set(eligibility), "Unused/unexpected eligibility rows")
    pins.verify()
    return dict(paths=paths, pins=pins, plans=plans, jobs=jobs, descriptions=descriptions)


def check_output(output, input_roots):
    output = safe_output(resolve(output), input_roots)
    require((ROOT/"reports").resolve() in output.parents,
            "Use a new subdirectory inside workspace reports/")
    for parent in output.parents:
        if parent == ROOT:
            break
        require(not (parent/"RUN.json").exists() and not (parent/"COMPLETED.json").exists(),
                "Output is nested within an existing experiment")
    if output.exists() and any(output.iterdir()):
        run = output/"RUN.json"
        require(run.is_file() and read(run).get("format") == FORMAT,
                "Refusing nonempty output belonging to another workflow")
    return output


def implementation():
    here = Path(__file__).resolve().parent
    files = list(here.glob("*.py")) + [here.parent/"reinforcement"/n for n in ("common.py", "statistics.py")]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
