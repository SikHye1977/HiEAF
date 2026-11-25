"""Separate job identities and the reviewed Windows-safe atomic writer."""
import os, random
from pathlib import Path
import numpy as np
import torch
from scripts.trainable_or_3way.reinforcement.common import (
    ROOT, require, read, digest, sha, resolve, child, atomic_json, atomic_npz,
    atomic_text, load_npz, immutable, finish, completed, verify_files,
    writer_lock, Progress, duration, safe_output, _write_atomic)
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import DIMS, CLASSES

BASE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT/"reports/fav_avdf1m_reinforcement_temporal_v1"
TRAIN = dict(epochs=20, batch_size=8, lr=.0005, weight_decay=.0001,
             gradient_clip=5., checkpoint_steps=100, pos_weight="valid train negatives/positives",
             selection="source dev modality AUC; earliest tie", projection=128,
             precision="FP32", dropout=.2)
VARIANTS = ("T0", "T1", "T2")
TRANSFORMS = ("ordered", "reversed", "shuffle1701", "shuffle1702", "shuffle1703")

def code_identity():
    files = list(BASE.glob("*.py"))
    files += [ROOT/p for p in (
        "scripts/trainable_or/fusion.py",
        "scripts/trainable_or_3way/reinforcement/common.py",
        "scripts/trainable_or_3way/reinforcement/statistics.py",
        "scripts/hierarchical_score_fusion_GRU_Temproal_Attention/model.py",
        "scripts/hierarchical_score_fusion_GRU_Temproal_Attention/common.py",
        "src/dataloader/metadata.py","src/dataloader/records.py",
        "src/utils/local_paths.py")]
    return {str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted(set(files))
            if not p.name.startswith("test_")}

def training_code_identity():
    files=code_identity()
    keep=("reinforcement_temporal/common.py","reinforcement_temporal/model.py",
          "reinforcement_temporal/training.py","Temproal_Attention/model.py",
          "Temproal_Attention/common.py","reinforcement/common.py","reinforcement/statistics.py")
    return {k:v for k,v in files.items() if k.endswith(keep)}

def evaluation_code_identity():
    return {k:v for k,v in code_identity().items() if k.endswith((
        "reinforcement_temporal/common.py","reinforcement_temporal/analysis.py",
        "reinforcement_temporal/statistics.py","reinforcement/common.py",
        "reinforcement/statistics.py","trainable_or/fusion.py"))}

def setup(device, threads):
    require(threads > 0, "threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    if str(device).startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s["python"]); n=s["numpy"]
    np.random.set_state((n[0],np.array(n[1],np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s["torch"])
    if s["cuda"]: torch.cuda.set_rng_state_all(s["cuda"])

def save_torch(path, obj):
    _write_atomic(path, lambda f: torch.save(obj, f))

def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def model_identity(source, mod, variant, seed, train_proof, dev_proof, config, implementation, runtime):
    # Target list, target scores and analysis families must NEVER enter training identity.
    return digest(dict(source=source,modality=mod,variant=variant,seed=seed,
                       train=train_proof,dev=dev_proof,config=config,code=implementation,runtime=runtime))

def checked_receipt(folder):
    folder=Path(folder)
    receipt=read(folder/"COMPLETED.json")
    if "files" in receipt: verify_files(folder, receipt["files"])
    return receipt

def score_or(a,b):
    return 1-(1-a)*(1-b)
