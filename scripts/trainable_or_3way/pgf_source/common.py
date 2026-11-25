"""PGF-only run isolation and reviewed Windows atomic writers."""
from pathlib import Path
import gc, json
import numpy as np
import torch
from scripts.trainable_or_3way.common import (labels, check_classes, stamp, status, Progress)
from scripts.trainable_or_3way.reinforcement.common import (
    ROOT, require, read, sha, digest, resolve, atomic_json, atomic_npz, atomic_text,
    load_npz, immutable, finish, completed, verify_files, writer_lock, safe_output, duration)
from scripts.trainable_or_3way.reinforcement_temporal.common import (
    save_torch, cpu_state, seed_all, setup, TRAIN)
from scripts.trainable_or_3way_matched.protocol import ARTIFACT_PROTOCOL, NUMERICS_POLICY
from scripts.trainable_or_3way_matched.memory import MEMORY_POLICY
from scripts.trainable_or_3way.models import VIDEO, AUDIO, TOKEN_RECIPE, DIMS, CLASSES

BASE=Path(__file__).resolve().parent
FORMAT="pgf_source_trainable_final_v1"
TARGETS=("PGF","FAV","AVDF1M")
OUTPUT=ROOT/"reports/trainable_or_pgf_v1/source_PGF/seed_42"
CACHE=Path("D:/datasets/experiment_cache/pgf_source_v1")
VIDEO_POLICY=dict(VIDEO,max_duration=600.)
AUDIO_POLICY=dict(AUDIO,train_batch_size=4,train_padding="right zeros to batch maximum; at least 284672",
                  evaluation_batch_size=1,long="entire waveform",normalization=False)
BN_POLICY=dict(training="ordinary running BN; physical batches video1/audio4",
               B="X3D saved BN, FP32",C="X3D per-video BN, batch1, no running-stat updates, FP32",
               audio="saved BN FP32 shared across B/C",tf32=False)

def implementation():
    from scripts.trainable_or_3way.common import file_set
    files=file_set()
    for folder in (BASE,ROOT/"scripts/trainable_or_3way_matched",
                   ROOT/"scripts/trainable_or_3way/reinforcement",
                   ROOT/"scripts/trainable_or_3way/reinforcement_p2",
                   ROOT/"scripts/trainable_or_3way/reinforcement_temporal"):
        for p in folder.glob("*.py"):
            if not p.name.startswith("test_"): files[p.relative_to(ROOT).as_posix()]=sha(p)
    return {k:v for k,v in files.items() if not Path(k).name.startswith("test_")}

def check_destination(output,cache_root):
    output,cache_root=resolve(output),resolve(cache_root)
    require(output.is_relative_to(ROOT/"reports") and output!=ROOT/"reports","Use a dedicated reports subfolder")
    require(cache_root.is_relative_to(CACHE) and cache_root!=Path("D:/datasets"),"New tokens must stay under "+str(CACHE))
    for parent in output.parents:
        if parent==ROOT/"reports":break
        require(not (parent/"RUN.json").exists() and not (parent/"COMPLETED.json").exists(),"Output is inside another run")
    if output.exists():
        contents=[p for p in output.iterdir() if p.name!=".writer.lock"]
        if (output/"RUN.json").exists():
            require(read(output/"RUN.json").get("format")==FORMAT,"Output belongs to a different experiment")
        else: require(not contents,"Output is not empty; choose a new output")
    # Reject ancestor/descendant collisions with historical runs and immutable inputs.
    for p in (ROOT/"checkpoints",ROOT/"configs",ROOT/"src",ROOT/"scripts",
              ROOT/"reports/hierarchical_score_fusion_gru_temporal_attention_fav_v1",
              ROOT/"reports/fav_avdf1m_reinforcement_v1",ROOT/"reports/fav_avdf1m_reinforcement_p2_v1",
              ROOT/"reports/fav_avdf1m_reinforcement_temporal_v1",ROOT/"reports/fav_avdf1m_reinforcement_p3a_v2",
              ROOT/"reports/trainable_or_3way_v1",ROOT/"reports/trainable_or_3way_fav_size_matched_v1"):
        safe_output(output,[p])
    return output,cache_root

def cohort(rows):
    return dict(sample_id=np.array([r.sample_id for r in rows]),
        labels=np.array([[-1 if x is None else x for x in (r.clip_label,r.video_label,r.audio_label)] for r in rows],dtype=np.int64),
        cluster=np.array([r.group_id for r in rows]))

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def step(output,name,index,total,detail=None):
    atomic_json(Path(output)/"STATUS.json",dict(status="RUNNING",stage=name,stage_index=index,total_stages=total,
                detail_progress=str(detail) if detail else None))
    status("STAGE",f"{index}/{total}: {name}; ETA is reported per measured stage, not for unmeasured stages")
