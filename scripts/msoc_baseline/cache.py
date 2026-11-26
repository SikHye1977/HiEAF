import hashlib
import json
import sqlite3
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .common import HERE, atomic_json, bind, digest, exclusive, require, runtime_versions, sha, Progress
from .data import stamp
from .media import POLICY, decode, pack, unpack


def db_open(path, readonly=False):
    path = Path(path).resolve()
    if readonly:
        connection = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=60)
        connection.execute('PRAGMA query_only=ON')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=60)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=FULL')
    connection.execute('PRAGMA cache_size=-8192')
    return connection


def cache_path(args, ds, split):
    base = args.train_cache if split == 'train' else args.eval_cache
    return Path(base) / ds / split / 'inputs.sqlite'


def cache_identity(rows, split):
    versions = runtime_versions()
    return dict(policy=POLICY, media_code=sha(HERE/'media.py'), cohort=digest(rows), train=split == 'train',
                decode_versions={k: versions[k] for k in ('av', 'numpy', 'scipy', 'opencv-python-headless', 'python_speech_features')})


def prepare(path, rows, split, workers, stop, status=None):
    path = Path(path)
    with exclusive(path.with_suffix('.lock')):
        identity = cache_identity(rows, split)
        bind(path.parent / 'IDENTITY.json', identity)
        db = db_open(path)
        try:
            db.execute('CREATE TABLE IF NOT EXISTS inputs (sid TEXT PRIMARY KEY, blob BLOB NOT NULL, sha TEXT NOT NULL, info TEXT NOT NULL)')
            db.commit()
            existing = {x[0] for x in db.execute('SELECT sid FROM inputs')}
            require(existing <= {r['sample_id'] for r in rows}, 'Unexpected IDs in input cache')
            todo = [r for r in rows if r['sample_id'] not in existing]
            side = 128 if split == 'train' else 100
            conservative_bytes = len(todo)*(30*side*side*3+120*13*4)*1.05
            require(shutil.disk_usage(path.parent).free > conservative_bytes,
                    f'Insufficient conservative disk allowance for {path}: need {conservative_bytes/1024**3:.1f} GiB. Choose a larger cache drive.')
            progress = Progress(f'CACHE {path.parent.parent.name}/{split}', len(rows), len(existing), status)
            if todo:
                # Only a bounded queue of decoded clips; never hold an epoch in RAM.
                def work(row):
                    require(stamp(row['path']) == row['stamp'], 'Media changed after planning: '+row['path'])
                    video, audio, info = decode(row['path'], train=split == 'train')
                    require(stamp(row['path']) == row['stamp'], 'Media changed during decode')
                    blob = pack(video, audio)
                    return row['sample_id'], blob, hashlib.sha256(blob).hexdigest(), json.dumps(info)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pending, it = deque(), iter(todo)
                    for _ in range(min(2*workers, len(todo))):
                        pending.append(pool.submit(work, next(it)))
                    n = len(existing)
                    while pending:
                        stop.check()
                        result = pending.popleft().result()
                        db.execute('INSERT INTO inputs VALUES (?,?,?,?)', result)
                        db.commit()  # one sample is the restart boundary
                        n += 1
                        progress.update(n)
                        row = next(it, None)
                        if row is not None:
                            pending.append(pool.submit(work, row))
            require(db.execute('SELECT COUNT(*) FROM inputs').fetchone()[0] == len(rows), 'Incomplete cache')
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            atomic_json(path.parent / 'COMPLETE.json', dict(identity=identity, n=len(rows)))
            progress.update(len(rows), force=True)
        finally:
            db.close()


class CachedDataset(Dataset):
    def __init__(self, path, rows, split, order=None, seed=42, epoch=0, start=0):
        self.path, self.rows, self.split = str(path), rows, split
        self.train = split == 'train'
        self.order = list(range(len(rows))) if order is None else order
        self.seed, self.epoch, self.start = seed, epoch, start
        self.db = None

    def __len__(self):
        return len(self.order)-self.start

    def __getstate__(self):
        state = self.__dict__.copy()
        state['db'] = None
        return state

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def __getitem__(self, index):
        position = index+self.start
        row = self.rows[self.order[position]]
        if self.db is None:
            self.db = db_open(self.path, readonly=True)
        item = self.db.execute('SELECT blob,sha FROM inputs WHERE sid=?', (row['sample_id'],)).fetchone()
        require(item is not None, 'Missing cached sample: '+row['sample_id'])
        blob, expected = item
        require(hashlib.sha256(blob).hexdigest() == expected, 'Corrupt cached sample: '+row['sample_id'])
        video, audio = unpack(blob, 128 if self.train else 100)
        if self.train:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, position]))
            y, x = rng.integers(0, 29, 2)
            video = video[:, y:y+100, x:x+100]
            if rng.random() < .5:
                video = video[:, :, ::-1]
        video = np.ascontiguousarray(video.transpose(3, 0, 1, 2))
        labels = [row['clip_label'], row['video_label'], row['audio_label']]
        labels = [-1 if x is None else x for x in labels]
        return torch.from_numpy(video), torch.from_numpy(audio.copy()), torch.tensor(labels), row['sample_id']


def loader(dataset, batch, workers):
    # A private generator prevents DataLoader startup from consuming the model RNG on resume.
    return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False,
                      generator=torch.Generator().manual_seed(817),
                      **({'prefetch_factor': 2} if workers else {}))


def tensors(batch, device):
    video, audio, labels, ids = batch
    video = video.to(device, non_blocking=True).float().div_(255).sub_(.421).div_(.165)
    return video, audio.to(device, non_blocking=True), labels.to(device, non_blocking=True), ids
