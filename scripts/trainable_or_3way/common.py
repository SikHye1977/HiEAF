"""Atomic state, deterministic run identities and honest resumed-work progress."""
import json, os, time, hashlib
from pathlib import Path
import numpy as np
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import (
    ROOT, DIMS, CLASSES, read, sha, digest, require, atomic_json, save_torch,
    cpu_state, seed_all, score_or, status, WriterLock)
from scripts.artifact_emotion_gated_attetnion_optimized.progress import duration

DATASETS = ('FAV', 'AVDF1M', 'PGF')
SOURCES = ('AVDF1M', 'PGF')

def immutable(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Identity/config changed; use a new output: '+str(path))
    else:
        atomic_json(path, value)

def file_set():
    paths = list(Path(__file__).parent.glob('*.py'))
    for folder in (
        'scripts/trainable_or',
        'scripts/hierarchical_score_fusion_GRU_Temproal_Attention',
        'scripts/hierarchical_score_fusion_GRU_Tempooral_Attention_cross_dataset',
        'scripts/artifact_emotion_gated_attention_cross_dataset',
        'scripts/artifact_emotion_gated_attetnion_optimized',
        'src/dataloader', 'src/models', 'src/datasets', 'src/utils'):
        paths += list((ROOT/folder).glob('*.py'))
    paths += list((ROOT/'src/models/hierarchical_score_fusion_variation').glob('*.py'))
    paths += list((ROOT/'third_party/aasist').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}

class Progress:
    def __init__(self, name, total, folder=None, resumed=0, unit='samples'):
        self.name, self.total, self.resumed = name, total, resumed
        self.folder, self.unit = Path(folder) if folder else None, unit
        self.started = time.monotonic()
        self.last = -float('inf')
        self.update(resumed, force=True)

    def update(self, done, detail='', force=False):
        now = time.monotonic()
        if not (force or done == self.total or now-self.last >= 5):
            return
        new = done-self.resumed
        elapsed = now-self.started
        rate = new/elapsed if elapsed > 0 else 0
        eta = (self.total-done)/rate if rate > 0 else None
        value = dict(stage=self.name, completed=done, total=self.total,
                     resumed=self.resumed, new=new, elapsed_seconds=elapsed,
                     rate=rate, eta_seconds=eta, detail=detail)
        status(self.name, f'{done}/{self.total} {self.unit} '
               f'({100*done/self.total if self.total else 100:.1f}%) '
               f'elapsed={duration(elapsed)} ETA={duration(eta)} '
               f'rate={rate:.2f} {self.unit}/s resumed={self.resumed} new={new} | {detail}')
        if self.folder:
            atomic_json(self.folder/'progress.json', value)
        self.last = now

def stamp(record):
    s = record.media_path.stat()
    require(s.st_size > 0, 'Empty media: '+record.sample_id)
    return [s.st_size, s.st_mtime_ns]

def labels(records, mod=None):
    key = (mod+'_label') if mod else 'clip_label'
    return np.asarray([-1 if getattr(r,key) is None else getattr(r,key) for r in records], np.int64)

def check_classes(records, mod=None):
    y = labels(records, mod)
    require(set(y[y >= 0].tolist()) == {0,1}, 'Need both known classes: '+str(mod))
    return {'real': int((y==0).sum()), 'fake': int((y==1).sum()), 'unknown': int((y<0).sum())}
