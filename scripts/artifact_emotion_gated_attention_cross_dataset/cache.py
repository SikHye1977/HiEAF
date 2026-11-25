"""Disk-backed pooled features: transactional per-sample resume, no giant RAM stack."""
import io
import sqlite3
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .common import digest


class FeatureCache(Dataset):
    def __init__(self, path, identity, records):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = records
        self.db = sqlite3.connect(str(self.path))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS features (id TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, checksum TEXT, payload BLOB)')
        identity_hash = digest(identity)
        old = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if old and old[0] != identity_hash:
            self.db.close(); raise ValueError('Feature cache belongs to a different source/recipe/cohort')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (identity_hash,))
        self.db.commit()

    def __len__(self):
        return len(self.records)

    def get(self, record):
        row = self.db.execute('SELECT size,mtime,checksum,payload FROM features WHERE id=?', (record.sample_id,)).fetchone()
        if row is None:
            return None
        stat = record.media_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != row[:2]:
            raise ValueError('Media changed since caching: '+record.sample_id)
        import hashlib
        if hashlib.sha256(row[3]).hexdigest() != row[2]:
            raise ValueError('Corrupt feature cache: '+record.sample_id)
        with np.load(io.BytesIO(row[3]), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def put(self, record, value, before):
        import hashlib
        stat = record.media_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != before:
            raise ValueError('Media changed during extraction')
        stream = io.BytesIO(); np.savez(stream, **value); payload = stream.getvalue()
        self.db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?,?)',
                        (record.sample_id, *before, hashlib.sha256(payload).hexdigest(), payload))
        self.db.commit()

    def __getitem__(self, index):
        r = self.records[index]; data = self.get(r)
        if data is None:
            raise ValueError('Incomplete cache: '+r.sample_id)
        return dict(data, sample_id=r.sample_id, group_id=r.group_id,
                    labels=np.array([-1 if v is None else v for v in
                                     (r.clip_label, r.video_label, r.audio_label)], dtype=np.int64))

    def close(self):
        self.db.close()
