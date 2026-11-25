"""P0: read-only input/protocol audit, normalized scores and an immutable analysis specification."""
import json
from pathlib import Path
import numpy as np
from .common import (resolve, read, sha, digest, require, atomic_json, atomic_text, atomic_npz,
                     immutable, finish, completed, Progress)
from .inputs import load_inputs, join_manifest, METHODS
from .statistics import PAIRS, BOOT_METRICS

def execute(config, output, run_identity):
    out = Path(output)/"p0"
    identity = digest(dict(run=run_identity, stage="p0"))
    if completed(out,identity):
        print("[P0] COMPLETE verified; reusing committed audit",flush=True)
        return
    cohorts, models, training, protocols, tokens, preserved, runs = load_inputs(config)
    maps, overlaps = join_manifest(cohorts,training,config["registry"],preserved)
    inventory = dict(cohorts={}, source_training=overlaps, protocols=protocols, token_inventory=tokens,
                     source_runs=runs, issues=[], raw_media_decoding="NOT_PERFORMED: P0 is saved-score/metadata validation",
                     token_blob_integrity="NOT_PERFORMED", external_identity_overlap="UNRESOLVED: dataset namespaces are not proof of independence")
    files = []
    for key,c in cohorts.items():
        data=c["data"]
        path = out/"normalized"/(key+".npz")
        atomic_npz(path,**data)
        files.append(path.relative_to(out).as_posix())
        valid=np.isfinite(data["raw"][:,2:]).all(axis=1)
        n=len(valid)
        inventory["cohorts"][key] = {k:v for k,v in c.items() if k!="data"}
        inventory["cohorts"][key].update(n=n, real=int((data["labels"][:,0]==0).sum()), fake=int((data["labels"][:,0]==1).sum()),
                  complete_emotion=int(valid.sum()), missing_emotion=int((~valid).sum()), clusters=len(set(data["cluster"])),
                  unknown_video_labels=int((data["labels"][:,1]<0).sum()), unknown_audio_labels=int((data["labels"][:,2]<0).sum()),
                  normalized_file=path.relative_to(out).as_posix(), normalized_sha256=sha(path),
                  reuse_status="REUSABLE_WITHIN_HISTORICAL_CONDITION_P2_NEEDED" if c["condition"]=="historical" else "REUSABLE_B_C_SAVED_RECEIPTS")
    inventory["issues"] = [
        "FAV Artifact BN/precision equivalence to B/C is not established; historical FAV rows are exploratory, not B/C.",
        "Cross-dataset identity/original overlap remains unresolved without a shared identity/original mapping.",
        "FAV grouping is source-side identity and may omit related target identities.",
        "Prior test observation and source-dev reuse remain; this is not an untouched holdout.",
        "Raw media decoding and token blob verification are outside score-only P0.",
    ]
    # No inference about cross-dataset overlaps from identical/different namespace strings.
    cluster_lines=[]
    for dataset,rows in maps.items():
        for sid,r in sorted(rows.items()):
            cluster_lines.append(json.dumps(dict(dataset=dataset,**r,cluster_id=r["group_id"],
                       cluster_basis="frozen group; FAV source identity / AVDF identity / PGF source video"),ensure_ascii=False))
    atomic_text(out/"CLUSTERS.jsonl","\n".join(cluster_lines)+"\n")
    spec=dict(format="reinforcement_p0_p1_v1", scope="existing scores only; source-dev M5/M6 fitting",
              methods=METHODS, threshold=.5, regularization=.01, epsilon=1e-6,
              primary="cross-dataset M4-M1 ROC-AUC, full cohort",
              primary_family=dict(planned=["FAV__B__AVDF1M","FAV__C__AVDF1M","AVDF1M__B__FAV","AVDF1M__C__FAV"],
                                  available=["AVDF1M__B__FAV","AVDF1M__C__FAV"],complete=False),
              pairs=[list(p) for p in PAIRS],bootstrap_metrics=list(BOOT_METRICS),
              bootstrap=dict(replicates=config["bootstrap"],max_attempts=config["max_attempts"],seed=config["seed"],
                             checkpoint_every=config["checkpoint_every"],unit="verified frozen group",method="paired percentile",
                             confidence=[.95,.9875],invalid_draw="redraw single-class replicates; report count"),
              cohorts=["full","common_emotion"],
              fallback="M2/M3/M4/M5/M6 use source artifact-affine on ANY missing emotion in full cohort; M2/M3 full names include +fallback; M0/M1 unchanged",
              common_cohort="Both emotions finite; identical mask for every method",
              M5_M6_fit="source dev complete-emotion rows only; unweighted logistic BCE, L2 toward [1,0,0], lambda=.01",
              prior_test_observed=True,independent_calibration=False,target_tuning=False,
              conclusions="Historical FAV separate from B/C; P2 pending; bootstrap does not cover training variability")
    atomic_json(out/"INVENTORY.json",inventory)
    atomic_json(out/"MODELS.json",models)
    atomic_json(out/"PRESERVED_INPUTS.json",preserved)
    atomic_json(out/"ANALYSIS_SPEC.json",spec)
    text = """# P0 protocol audit

| Item | FAV source | AVDF1M source |
|---|---|---|
| Artifact inference condition | historical; B/C and FP32 equivalence not established | recorded B/C, FP32 full-audio v2 |
| Artifact audio minimum | historical recipe 284,672 samples; verify source records for exact preprocessing | recorded recipe 510,976 samples; full long waveform |
| Emotion head positive weighting | historical fixed 0.25 | source/modality negative-to-positive ratio |
| Calibration | source dev reused after head selection | source dev reused after head selection |
| Permitted comparison | methods within this historical condition | methods within B, and within C; do not pool |
| New inference needed | P2 B/C audit/re-evaluation before common-condition bidirectional claims | score-only reuse after saved receipt validation |

B uses saved X3D BN buffers. C is the recorded per-video normalization ablation, not a reproduction of historical 4GPU training.
AASIST uses saved BN for both. Exact saved protocol dictionaries are in INVENTORY.json.
P0 verifies completed score files/receipts and referenced weights against their own recorded hashes; it does not request missing workstation media hashes.
Manifest IDs/splits/labels/groups are checked. This is not a new raw-media decode audit.
Cluster mapping uses frozen groups: FAV source-side identity, AVDF1M identity, PGF source-video.
FAV groups do not guarantee grouping of every related target identity.
Cross-dataset identity/original overlap remains UNRESOLVED; namespaces are not an overlap audit.
Token receipts provide a length/availability inventory only. No new token-blob integrity verification or temporal inference occurred.
Existing tests have been observed. 95% intervals are exploratory; planned four-direction/BN family is incomplete until P2.
"""
    atomic_text(out/"PROTOCOL_DIFF.md",text)
    files += ["CLUSTERS.jsonl","INVENTORY.json","MODELS.json","PRESERVED_INPUTS.json","ANALYSIS_SPEC.json","PROTOCOL_DIFF.md"]
    finish(out,identity,files)
    print("[P0] COMPLETE_WITH_DOCUMENTED_LIMITATIONS; FAV B/C and external identity audit remain pending",flush=True)

def verify_preserved(output):
    out=Path(output)/"p0"
    records=read(out/"PRESERVED_INPUTS.json")
    bar=Progress("VERIFY preserved inputs",len(records))
    for i,(path,h) in enumerate(records.items()):
        require(Path(path).is_file() and sha(path)==h,"Preserved input changed/missing: "+path)
        bar.update(i+1)
