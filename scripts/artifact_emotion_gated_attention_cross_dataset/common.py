"""Run identity, immutable protocol and atomic outputs for new local experiments."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from scripts.artifact_emotion_gated_attetnion_optimized.progress import Progress, status
from scripts.artifact_emotion_gated_attetnion_optimized.storage import atomic_json, WriterLock

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
DATASETS = ('FAV', 'AVDF1M', 'PGF')
VIDEO_DIM, AUDIO_DIM = 2048, 160
EMOTION_DIMS = {'video': 1280, 'audio': 768}
PROTOCOL = dict(version=1, input_policy='new_local_experiment', video_frames=128,
                video_size=256, mean=.45, std=.225, audio_rate=16000,
                audio_min_samples=284672, audio_length='whole waveform; minimum right zero pad; batch1',
                emotion_pool='mean valid', emotion_window=48000, emotion_hop=16000,
                gate_dim=128, fusion='1-(1-Vs)*(1-As)', threshold=.5,
                cohort='paired eligibility; unknown modality labels masked')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def implementation():
    paths = list(BASE.glob('*.py')) + list((ROOT/'src/dataloader').glob('*.py'))
    paths += list((ROOT/'scripts/artifact_emotion_gated_attetnion_optimized').glob('*.py'))
    paths += [ROOT/p for p in ('src/datasets/media.py', 'src/utils/local_paths.py',
              'src/models/x3d.py', 'src/models/aasist.py', 'src/models/hsemotion.py',
              'src/models/hierarchical_score_fusion_variation/emotion2vec_seed.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def save_torch(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    with temp.open('wb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def bind_directory(path, identity, resume):
    path = Path(path)
    if path.exists() and not resume:
        raise FileExistsError(f'Use a new output or --resume: {path}')
    path.mkdir(parents=True, exist_ok=True)
    meta = path/'RUN.json'
    if meta.exists():
        if read(meta) != identity:
            raise ValueError('Resume configuration/protocol/source changed')
    elif any(p.name != '.writer.lock' for p in path.iterdir()):
        raise ValueError('Output contains files without RUN.json')
    else:
        atomic_json(meta, identity)


def metrics(labels, probabilities):
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    y = np.asarray(labels); p = np.asarray(probabilities)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid labels/probabilities')
    if not len(y):
        return dict(n=0, roc_auc=None, average_precision=None, balanced_accuracy=None, real_fpr=None)
    cm = confusion_matrix(y, p >= .5, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    two = len(set(y.tolist())) == 2
    return dict(n=len(y), real=int(tn+fp), fake=int(tp+fn),
                roc_auc=float(roc_auc_score(y, p)) if two else None,
                average_precision=float(average_precision_score(y, p)) if two else None,
                balanced_accuracy=float((tn/(tn+fp)+tp/(tp+fn))/2) if two else None,
                real_fpr=float(fp/(tn+fp)) if tn+fp else None,
                confusion_matrix=cm.tolist())
