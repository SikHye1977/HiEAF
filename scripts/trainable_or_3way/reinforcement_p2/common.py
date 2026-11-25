"""P2 run identities and preservation helpers; uses the reviewed Windows atomic writer."""
from pathlib import Path
import json
import numpy as np
from scripts.trainable_or_3way.reinforcement.common import (
    ROOT, read, sha, digest, resolve, require, child, atomic_json, atomic_text, atomic_npz,
    load_npz, immutable, finish, completed, verify_files, writer_lock, Progress, duration, safe_output)
BASE = Path(__file__).resolve().parent
P1_DEFAULT = ROOT/"reports/fav_avdf1m_reinforcement_v1"
OUTPUT_DEFAULT = ROOT/"reports/fav_avdf1m_reinforcement_p2_v1"
COHORTS = ("FAV/dev","FAV/test","AVDF1M/test")
VIDEO = dict(frames=128,crop_size=256,resize_short=256,mean=[.45]*3,std=[.225]*3,max_duration=600.)
AUDIO = dict(sample_rate=16000,channels="mono",minimum_samples=284672,
             short="right zero pad",long="whole decoded waveform",batch_size=1,
             truncation=False,normalization=False)
BN = dict(B="X3D eval saved running statistics, FP32",
          C="X3D batch=1 per-video BN, track_running_stats=False, FP32; other layers eval",
          audio="AASIST saved BN eval FP32, same score for B and C",tf32=False)

def stamp(path):
    p=Path(path)
    require(p.is_file(),"Missing current Windows media: "+str(p))
    s=p.stat()
    require(s.st_size>0,"Empty current Windows media: "+str(p))
    return [s.st_size,s.st_mtime_ns]

def file_set():
    paths=list(BASE.glob("*.py"))
    paths += list((ROOT/"scripts/trainable_or_3way/reinforcement").glob("*.py"))
    paths += [ROOT/p for p in (
        "scripts/artifact_emotion_gated_attetnion_optimized/local_inputs.py",
        "scripts/artifact_emotion_gated_attetnion_optimized/common.py",
        "scripts/trainable_or/fusion.py",
        "src/models/x3d.py","src/models/aasist.py","src/models/interfaces.py",
        "src/datasets/media.py","src/datasets/transforms.py","src/dataloader/metadata.py",
        "src/dataloader/records.py","src/utils/local_paths.py",
        "third_party/aasist/models/AASIST.py")]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}

def package_versions():
    from importlib.metadata import version
    import platform,cv2
    versions={k:version(k) for k in ("torch","torchvision","pytorchvideo","numpy","scipy",
                                   "scikit-learn","av","matplotlib","PyYAML")}
    versions["opencv"]=cv2.__version__  # Works with the installed opencv-python-headless distribution.
    return dict(python=platform.python_version(),packages=versions)

def verify_parent(parent):
    """Read only. Original P0/P1 identities stay unchanged after the reviewed I/O patch."""
    parent=Path(parent)
    run=read(parent/"RUN.json")
    p0id=digest(dict(run=digest(run),stage="p0"))
    require(completed(parent/"p0",p0id),"P0 is incomplete")
    p1id=digest(dict(run=digest(run),stage="p1",p0=sha(parent/"p0/COMPLETED.json")))
    require(completed(parent/"p1",p1id),"P1 is incomplete")
    preserved=read(parent/"p0/PRESERVED_INPUTS.json")
    bar=Progress("P2 verify historical inputs",len(preserved))
    for i,(name,h) in enumerate(preserved.items()):
        require(Path(name).is_file() and sha(name)==h,"Historical input changed: "+name)
        bar.update(i+1)
    return dict(run_sha256=sha(parent/"RUN.json"),p0_completed_sha256=sha(parent/"p0/COMPLETED.json"),
                p1_completed_sha256=sha(parent/"p1/COMPLETED.json"),verified_historical_files=len(preserved))

def check_stop(output):
    if (Path(output)/"STOP_REQUESTED").exists():
        raise KeyboardInterrupt("STOP_REQUESTED; committed clips retained")
