"""Streaming frozen inference and atomic per-clip, per-modality resume."""
import hashlib
import io
import sqlite3
import time
from pathlib import Path
import numpy as np
import torch
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import (
    DIMS, CLASSES, require, digest, atomic_json, read, Progress, status)
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.extract import Encoders, validate_tokens
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.train import load_head
from scripts.artifact_emotion_gated_attetnion_optimized.features import audio_chunks
from .inputs import media_stamp


def validate_trace(value, mod):
    valid = value['valid']; t = value['timestamp']; alpha = value['attention']; p = value['probabilities']
    n = len(valid)
    require(n > 0 and valid.dtype == np.bool_ and t.shape == alpha.shape == (n,)
            and p.shape == (n, len(CLASSES[mod])), 'Invalid saved timeline shape')
    require(np.isfinite(t).all() and (np.diff(t) >= 0).all(), 'Invalid saved timestamps')
    require(np.isfinite(alpha).all() and (alpha >= 0).all() and (alpha[~valid] == 0).all(), 'Invalid attention')
    require(np.isfinite(p[valid]).all() and ((p[valid] >= 0) & (p[valid] <= 1)).all()
            and np.allclose(p[valid].sum(1), 1., atol=1e-5), 'Invalid saved emotion probabilities')
    score = float(value['score'])
    if valid.any():
        require(np.isfinite(score) and 0 <= score <= 1 and np.isclose(alpha.sum(), 1., atol=1e-5), 'Invalid saved score')
    else:
        require(np.isnan(score) and not alpha.any(), 'Missing emotion must remain missing')
    if mod == 'video': require(n == 128 and valid.all(), 'Video must retain all 128 center-crop frames')


class ScoreStore:
    def __init__(self, path, identity):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        try:
            self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS samples(sid TEXT PRIMARY KEY,size INTEGER NOT NULL,mtime INTEGER NOT NULL,checksum TEXT NOT NULL,data BLOB NOT NULL)')
            old = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            expected = digest(identity)
            require(old is None or old == (expected,), 'Saved scores belong to a different model/recipe/cohort')
            if old is None:
                require(self.db.execute('SELECT COUNT(*) FROM samples').fetchone()[0] == 0, 'Scores without identity')
                self.db.execute("INSERT INTO metadata VALUES ('identity',?)", (expected,))
            self.db.commit()
        except BaseException:
            self.db.close(); raise

    def ids(self):
        return {r[0] for r in self.db.execute('SELECT sid FROM samples')}

    def get(self, sid, stamp):
        row = self.db.execute('SELECT size,mtime,checksum,data FROM samples WHERE sid=?', (sid,)).fetchone()
        if row is None: return None
        require(row[:2] == tuple(stamp), 'Media stamp changed for saved score: '+sid)
        require(hashlib.sha256(row[3]).hexdigest() == row[2], 'Corrupt saved score: '+sid)
        with np.load(io.BytesIO(row[3]), allow_pickle=False) as z:
            value = {k: z[k].copy() for k in z.files}
        return value

    def put(self, sid, stamp, value):
        buf = io.BytesIO(); np.savez_compressed(buf, **value); payload = buf.getvalue()
        with self.db:
            self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',
                            (sid, *stamp, hashlib.sha256(payload).hexdigest(), payload))

    def checksum(self):
        return digest(list(self.db.execute('SELECT sid,size,mtime,checksum FROM samples ORDER BY sid')))

    def close(self):
        self.db.close()


@torch.inference_mode()
def score_tokens(head, tokens, mod, device):
    validate_tokens(tokens, mod)
    valid = tokens['valid']; attention = np.zeros(len(valid), np.float32)
    score = np.nan
    if valid.any():
        e = torch.from_numpy(np.asarray(tokens['embedding'][valid], np.float32))[None].to(device)
        p = torch.from_numpy(np.asarray(tokens['probabilities'][valid], np.float32))[None].to(device)
        mask = torch.ones(e.shape[:2], dtype=torch.bool, device=device)
        logits, alpha = head(e, p, mask)
        score = float(logits.sigmoid()[0].cpu()); attention[valid] = alpha[0].cpu().numpy()
    # Keep trajectories and attention for analysis, not huge 128x1280 embeddings.
    result = {k: np.asarray(tokens[k]) for k in ('probabilities', 'timestamp', 'valid')}
    for k in ('frame_index', 'start_sample', 'end_sample'):
        if k in tokens: result[k] = np.asarray(tokens[k])
    result.update(score=np.asarray(score, np.float64), attention=attention)
    validate_trace(result, mod)
    return result


class FrozenPredictor:
    def __init__(self, source, mod, device, frame_batch):
        self.source, self.mod, self.device = source, mod, device
        self.core = Encoders(source['assets'], device, frame_batch)
        self.head = None; self.seed = None

    def load(self):
        if self.head is None:
            self.head = load_head(Path(self.source['folder'])/'heads'/self.mod, self.mod,
                                  self.source['lock']['heads'][self.mod]['identity'], self.device)
        if self.mod == 'audio' and self.seed is None:
            from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import FrozenEmotion2VecSeed
            self.core.load_audio()
            self.seed = FrozenEmotion2VecSeed(self.source['assets']['seed_path'], self.device)
            self.seed.model.kwargs['disable_pbar'] = True

    @torch.inference_mode()
    def audio_tokens(self, path):
        from src.datasets.media import read_audio, MediaConfig
        wave, has_audio = read_audio(path, MediaConfig(sample_rate=16000))
        require(has_audio and len(wave) > 0, 'Paired-eligible sample has no decodable audio')
        embeddings, starts, ends, times, valid = [], [], [], [], []
        for start, chunk in audio_chunks(wave.numpy()):
            good = len(chunk) >= 400 and bool(np.any(chunk != 0))
            feature = np.zeros(DIMS['audio'], np.float32)
            if good:
                result = self.seed.model.generate(input=chunk, granularity='utterance',
                    extract_embedding=True, disable_pbar=True, disable_log=True)
                require(len(result) == 1 and 'feats' in result[0], 'Seed embedding missing')
                feature = np.asarray(result[0]['feats'], np.float32).reshape(-1)
                require(feature.shape == (768,) and np.isfinite(feature).all(), 'Invalid seed embedding')
            embeddings.append(feature); starts.append(start); ends.append(start+len(chunk)); valid.append(good)
            times.append((start+len(chunk)/2)/16000)
        e = np.asarray(embeddings, np.float32)
        w, b = self.core.audio_projection
        p = torch.softmax(torch.nn.functional.linear(torch.from_numpy(e), w, b), dim=-1).numpy()[:, self.core.audio_order]
        return dict(embedding=e, probabilities=p, timestamp=np.asarray(times, np.float64),
                    valid=np.asarray(valid, bool), start_sample=np.asarray(starts), end_sample=np.asarray(ends))

    def __call__(self, record):
        self.load()
        tokens = self.core.video(record.media_path) if self.mod == 'video' else self.audio_tokens(record.media_path)
        return score_tokens(self.head, tokens, self.mod, self.device)


def infer_stage(data, mod, folder, identity, predictor):
    """A killed process loses only its currently uncommitted clip, not earlier clips."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    store = ScoreStore(folder/'scores.sqlite', identity)
    records = data['records']; stamps = data['stamps']; total = len(records)
    try:
        existing = store.ids()
        require(existing <= {r.sample_id for r in records}, 'Unexpected sample in saved scores')
        verify = Progress(f'VERIFY {data["target"]} {mod}', len(existing)); checked = 0
        for r in records:
            require(media_stamp(r) == stamps[r.sample_id], 'Media changed before inference: '+r.sample_id)
            if r.sample_id in existing:
                validate_trace(store.get(r.sample_id, stamps[r.sample_id]), mod)
                checked += 1; verify.update(checked)
        receipt = folder/'COMPLETED.json'
        if receipt.exists():
            old = read(receipt)
            require(old['identity'] == identity and len(existing) == total
                    and old['payload_index_sha256'] == store.checksum(), 'Completed score cache changed')
            status('RESUME', f'{data["target"]} {mod}: {total}/{total} verified; inference skipped')
            return old
        bar = Progress(f'INFER {data["target"]} {mod} new', total-len(existing))
        started = time.perf_counter(); new = 0
        for r in records:
            if r.sample_id in existing: continue
            before = media_stamp(r)
            require(before == stamps[r.sample_id], 'Media changed before extraction: '+r.sample_id)
            try:
                value = predictor(r); validate_trace(value, mod)
                require(media_stamp(r) == before, 'Media changed during extraction: '+r.sample_id)
                store.put(r.sample_id, before, value)
            except Exception:
                status('FAILED SAMPLE', f'{data["target"]} {mod} {r.sample_id}')
                raise
            new += 1
            bar.update(new, f'resumed={len(existing)} completed={len(existing)+new}/{total} valid_tokens={int(value["valid"].sum())}/{len(value["valid"])}')
            if new == 1 or new % 25 == 0 or new == total-len(existing):
                atomic_json(folder/'progress.json', dict(completed=len(existing)+new, planned=total,
                    resumed=len(existing), new=new, elapsed_seconds=time.perf_counter()-started))
        result = dict(identity=identity, n=total, new=new, resumed=len(existing),
                      seconds_this_invocation=time.perf_counter()-started, payload_index_sha256=store.checksum())
        atomic_json(receipt, result)
        status('INFER', f'{data["target"]} {mod} COMPLETE {total}/{total}')
        return result
    finally:
        store.close()
