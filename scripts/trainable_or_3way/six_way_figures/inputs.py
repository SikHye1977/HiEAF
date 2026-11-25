"""Completed score/token adapters. No decoder, encoder, or historical writer is called."""
import hashlib
import json
import sqlite3
from .common import *
from scripts.trainable_or_3way.reinforcement_temporal.inputs import (
    Inputs as LegacyInputs, TokenCache, cohort_proof, FAV, P1, P2)
from scripts.trainable_or_3way.reinforcement_temporal.analysis import fused, hash_array
from scripts.trainable_or_3way.reinforcement.inputs import load_scores
from scripts.trainable_or_3way.reinforcement_temporal.statistics import metric_table


def validate_reusable_head(folder, source, variant, seed, mod, evidence):
    folder = Path(folder)
    evidence.job(folder)
    t = read(folder / "TRAINING.json")
    require(t["config"] == TRAIN and t["source"] == source and t["variant"] == variant
            and t["seed"] == seed and t["branch_seed"] == seed + 100 + (mod == "audio")
            and t["modality"] == mod, "Historical comparison-head recipe differs: " + str(folder))
    state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    require(all(state[k] == t[k] for k in ("source", "modality", "variant", "seed", "best_epoch")),
            "Head/checkpoint metadata differ")
    require(state["identity"] == read(folder / "COMPLETED.json")["identity"], "Head identity differs")
    history = read(folder / "history.json")
    require(len(history) == TRAIN["epochs"] and [h["epoch"] for h in history] == list(range(1, 21)),
            "Historical head did not complete 20 epochs")
    return folder / "best.pt"


def read_score_store(folder, ids, evidence, expected_run, expected_model):
    """Read PGF dev artifact scores without constructing an inference engine."""
    folder = Path(folder)
    evidence.file(folder / "COMPLETED.json")
    rec = read(folder / "COMPLETED.json")
    require(rec["identity"]["run"] == expected_run and rec["identity"]["model"] == expected_model
            and rec["identity"]["cohort"] == "PGF/dev", "PGF dev artifact producer mismatch")
    db = sqlite3.connect((folder / "scores.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        require(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0] == digest(rec["identity"]),
                "Score store identity mismatch")
        rows = db.execute("SELECT sid,stamp,payload,checksum FROM scores ORDER BY sid").fetchall()
        h = hashlib.sha256()
        values = {}
        for sid, stamp, payload, checksum in rows:
            h.update(json.dumps([sid, stamp, checksum], separators=(",", ":")).encode())
            require(hashlib.sha256(payload.encode()).hexdigest() == checksum, "Corrupt artifact score")
            values[sid] = json.loads(payload)
        require(h.hexdigest() == rec["index_sha256"] and set(values) == set(ids) and rec["n"] == len(ids),
                "Incomplete or changed artifact score store")
        return [values[s] for s in ids]
    finally:
        db.close()


class Inputs:
    def __init__(self):
        self.evidence = Evidence()
        self.legacy = LegacyInputs(["FAV", "AVDF1M"], ["FAV", "AVDF1M"])
        self.cohorts, self.raw, self.proofs, self.heads, self.folders = {}, {}, {}, {}, {}
        self.models = {}
        self.reusable_heads = {}
        self.benchmarks = {}

    def put(self, source, condition, target, split, data, proof):
        d = align(data, self.cohorts[target, split])
        require(d["raw"].shape == (len(d["sample_id"]), 4) and np.isfinite(d["raw"][:, :2]).all(),
                "Missing or malformed artifact score")
        finite = d["raw"][np.isfinite(d["raw"])]
        require(not np.isinf(d["raw"]).any() and ((finite >= 0) & (finite <= 1)).all(), "Invalid probability")
        key = source, condition, target, split
        self.raw[key], self.proofs[key] = d, dict(origin=proof, aligned=cohort_proof(d), raw=hash_array(d["raw"]))

    def prepare(self, full_hash):
        e = self.evidence
        self.legacy.prepare(full_hash=full_hash)
        self.cohorts.update(self.legacy.cohorts)
        self.raw.update(self.legacy.raw)
        self.proofs.update(self.legacy.proofs)
        self.heads.update(self.legacy.heads)
        self.folders.update(self.legacy.folders)
        for proof in self.legacy.proofs.values():
            if "sha256" in proof:
                e.file(proof["path"], proof["sha256"])
            elif "receipt_sha256" in proof:
                e.file(Path(proof["path"]) / "COMPLETED.json", proof["receipt_sha256"])
        # Pin frozen metadata and completed source selection, not source-workstation media hashes.
        for p in (ROOT / "configs/fixed_manifests.json", ROOT / "reports/phase0_v1/eligibility_v4.jsonl"):
            e.file(p)
        for source in ("FAV", "AVDF1M"):
            self.models[source] = {c: self.legacy.historical_fusion(source, c) for c in CONDITIONS}
            for item in self.heads[source].values():
                e.file(item["path"], item["sha256"])
        e.job(P2 / "fusion")
        e.committed(P1 / "p1", "FITTED_MODELS.json")
        parent = read(e.file(TEMPORAL / "PROTOCOL.json"))
        require(parent["train"] == TRAIN and parent["code"] == training_code_identity(),
                "Existing temporal training implementation/recipe differs")
        for source in ("FAV", "AVDF1M"):
            for seed in SEEDS:
                for variant in VARIANTS:
                    for mod in ("video", "audio"):
                        p = validate_reusable_head(TEMPORAL / "heads" / source / f"seed_{seed}" / variant / mod,
                                                   source, variant, seed, mod, e)
                        self.reusable_heads[source, seed, variant, mod] = p
                    e.job(TEMPORAL / "fusion" / source / f"seed_{seed}" / variant)
        # PGF metadata is reconstructed using the original evidence-backed overlay.
        from scripts.trainable_or_3way.data import source_metadata
        from scripts.trainable_or_3way.pgf_source.common import cohort
        records, label_evidence = source_metadata("PGF")
        pgf_run = read(e.committed(PGF, "RUN.json"))
        require(read(PGF / "COMPLETED.json")["identity"] == digest(pgf_run)
                and pgf_run["config"]["source"] == "PGF", "Completed PGF run identity differs")
        saved_evidence = read(e.committed(PGF, "PGF_LABEL_EVIDENCE.json"))
        require(label_evidence == saved_evidence and digest(label_evidence) == pgf_run["pgf_evidence_sha256"],
                "PGF label evidence changed")
        for p, h in {(x["evidence_file"], x["evidence_sha256"]) for x in saved_evidence}:
            e.file(p, h)
        for split, rows in records.items():
            d = cohort(rows)
            require(digest({k: v.tolist() for k, v in d.items()}) == pgf_run["cohorts"]["PGF/" + split]["cohort_sha256"],
                    "PGF frozen split/labels/groups changed")
            self.cohorts["PGF", split] = d
        # Reuse the exact same head training code, pretrained assets, and input recipe.
        require(pgf_run["config"]["emotion_head"] == TRAIN, "PGF head recipe differs")
        for k, v in training_code_identity().items():
            require(pgf_run["implementation"].get(k) == v, "PGF head code differs: " + k)
        self.models["PGF"] = read(e.committed(PGF / "fusion", "FUSION.json"))
        lock = read(e.committed(PGF / "fusion", "LOCKED_BEFORE_TEST.json"))
        require(not lock["target_training"] and lock["source"] == "PGF" and lock["threshold"] == .5
                and lock["binding"]["run"] == digest(pgf_run)
                and lock["fusion_sha256"] == sha(PGF / "fusion/FUSION.json"), "PGF fusion lock differs")
        selected = read(e.committed(PGF, "MODELS_SELECTED.json"))
        require(lock["binding"]["selection_sha256"] == sha(PGF / "MODELS_SELECTED.json"), "PGF head lock differs")
        self.heads["PGF"] = {}
        for mod in ("video", "audio"):
            p = validate_reusable_head(PGF / "models/emotion" / mod, "PGF", "T2", 42, mod, e)
            require(sha(p) == selected["models"]["emotion/" + mod]["sha256"], "PGF emotion head differs")
            self.heads["PGF"][mod] = dict(path=str(p), sha256=sha(p))
            self.reusable_heads["PGF", 42, "T2", mod] = p
        token_map = read(e.committed(PGF, "TOKEN_CACHES.json"))
        for split in ("train", "dev", "test"):
            for mod in ("video", "audio"):
                entry = token_map[f"PGF/{split}/{mod}"]
                folder = resolve(entry["path"])
                e.file(folder / "COMPLETED.json", entry["receipt_sha256"])
                cache = TokenCache(folder, mod, self.cohorts["PGF", split], full_hash)
                try:
                    identity = cache.receipt["identity"]
                    require(identity["dataset"] == "PGF" and identity["split"] == split
                            and identity["recipe"] == pgf_run["config"]["token_recipe"], "PGF token recipe differs")
                    for k in ("hse_sha256", "seed_sha256", "seed_revision"):
                        require(identity["assets"][k] == pgf_run["assets"][k] == self.legacy.fav_run["assets"][k],
                                "Pretrained emotion encoder differs")
                    for i in (0, len(cache.ids) // 2, len(cache.ids) - 1):
                        cache.get(cache.ids[i])
                    self.folders["PGF", split, mod] = folder
                    self.proofs["PGF", split, mod] = cache.proof
                finally:
                    cache.close()
        # Existing FAV->PGF common B/C raw scores.
        for c in CONDITIONS:
            p = e.committed(FAV_PGF / "raw/PGF/test", c + ".npz")
            self.put("FAV", c, "PGF", "test", load_npz(p), dict(path=str(p), sha256=sha(p)))
        # Existing AVDF1M->PGF raw scores and source-dev fusion.
        wr = read(e.committed(WS, "RUN.json"))
        wf = read(e.committed(WS, "FUSION.json"))
        wl = read(e.committed(WS, "LOCKED_BEFORE_TEST.json"))
        require(wl["run_identity"] == digest(wr) and wl["fusion_sha256"] == sha(WS / "FUSION.json")
                and not wr["target_training"], "AVDF1M producer lock differs")
        for c in CONDITIONS:
            for method in ("artifact_affine", "trainable_final"):
                require(wf["conditions"][c][method] == self.models["AVDF1M"][c][method],
                        "AVDF1M historical fusion differs across target runs")
            p = e.committed(WS, f"evaluation/{c}/PGF/test/predictions.csv")
            d = load_scores(p, wf["conditions"][c])
            self.put("AVDF1M", c, "PGF", "test", d, dict(path=str(p), sha256=sha(p)))
        # PGF-source final test arrays already contain artifact and emotion raw scores.
        for target in DATASETS:
            p = e.committed(PGF / "evaluation" / target, "predictions.npz")
            saved = load_npz(p)
            for mod in ("video", "audio"):
                # Bind reused seed42/T2 branch predictions during preflight as well.
                e.committed(PGF / "emotion_scores" / target / "test" / mod, "scores.npz")
            for c in CONDITIONS:
                d = {k: saved[k] for k in ("sample_id", "labels", "cluster")}
                d["raw"] = saved["raw_" + c]
                self.put("PGF", c, target, "test", d, dict(path=str(p), sha256=sha(p)))
        # Source dev artifact scores come from transactional read-only stores.
        dev = self.cohorts["PGF", "dev"]
        parts, emotions = {}, {}
        for mod in ("video", "audio"):
            model_id = selected["models"]["artifact/" + mod]["sha256"]
            parts[mod] = read_score_store(PGF / "artifact_scores/PGF/dev" / mod, dev["sample_id"].tolist(), e,
                                         digest(pgf_run), model_id)
            p = e.committed(PGF / "emotion_scores/PGF/dev" / mod, "scores.npz")
            emotions[mod] = load_npz(p)
            require(np.array_equal(emotions[mod]["sample_id"], dev["sample_id"]), "PGF dev emotion IDs differ")
        for c in CONDITIONS:
            require(all(v["media_sha256"] == a["media_sha256"] for v, a in zip(parts["video"], parts["audio"])),
                    "PGF dev modalities refer to different media")
            raw = np.column_stack([[v[c + "_probability"] for v in parts["video"]],
                [v["probability"] for v in parts["audio"]], emotions["video"]["probability"], emotions["audio"]["probability"]])
            require(hash_array(raw) == lock["binding"]["dev_scores"][c], "PGF dev raw scores differ from locked fit")
            self.put("PGF", c, "PGF", "dev", dict(**dev, raw=raw), lock["binding"])
        # Check final point metrics against the completed six-direction manuscript inputs.
        summary = read(e.committed(FAV_PGF, "ALL_DATASETS_BC.json"))
        for r in summary["rows"]:
            if (r["source"], r["target"]) not in DIRECTIONS:
                continue
            key = r["source"], r["condition"], r["target"], "test"
            d = self.raw[key]
            p, _, b = fused(d["raw"], d["raw"][:, 2:], self.historical_fusion(key[0], key[1]))
            m = metric_table(d["labels"][:, 0], dict(baseline=b["M0"], artifact_affine=b["M1"], proposed=p))[r["method"]]
            for k in ("n", "roc_auc", "fake_f1", "real_fpr"):
                require(abs(m[k] - r[k]) < 1e-12, "Completed table/score mismatch: " + str(key) + " " + k)
            self.benchmarks[key + (r["method"],)] = m
        self._bind_existing_outputs()
        self.snapshot = dict(files=e.files, feature_proofs={"/".join(k): self.proofs[k] for k in self.folders},
                             cohorts={"/".join(k): cohort_proof(v) for k, v in self.cohorts.items()},
                             raw={"/".join(k): self.proofs[k] for k in self.raw})
        return self

    def _bind_existing_outputs(self):
        e = self.evidence
        for source, target in DIRECTIONS:
            if "PGF" in (source, target):
                continue
            for c in CONDITIONS:
                job = self.old_boot(source, target, c)
                for name in ("METRICS.json", "BOOTSTRAP.json", "bootstrap_state.npz", "predictions.csv"):
                    e.committed(job, name)
                e.job(ORDER / "diagnostics" / source / target / c)
                for seed in SEEDS:
                    for cohort in ("full", "common_emotion"):
                        e.job(TEMPORAL / "evaluations" / source / f"seed_{seed}" / target / c / cohort)

    @staticmethod
    def old_boot(source, target, condition):
        root = P2 / "evaluation" if source == "FAV" else P1 / "p1"
        return root / f"{source}__{condition}__{target}__test" / "full"

    def cache(self, dataset, split, mod):
        return TokenCache(self.folders[dataset, split, mod], mod, self.cohorts[dataset, split], False)

    def historical_fusion(self, source, condition):
        return self.models[source][condition]
