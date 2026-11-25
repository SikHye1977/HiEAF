"""Small score-analysis utilities. Never import torch or any inference module."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else ROOT / p).resolve()

def child(folder, name):
    folder = Path(folder).resolve()
    p = (folder / name).resolve()
    require(p != folder and folder in p.parents, "Path escapes receipt folder: " + str(name))
    return p

def replace_with_retry(temp, path):
    """Retry Windows sharing/access denials; never delete the previous checkpoint."""
    delays = (.05, .1, .2, .4, .8, 1., 1., 1., 1., 1.)
    for attempt in range(len(delays)+1):
        try:
            os.replace(temp, path)
            return
        except OSError as exc:
            # Windows AV/indexers/readers can briefly deny rename/delete sharing.
            # Permanent ACL/read-only errors still fail after a bounded wait.
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            if attempt == 0:
                print(f"[SAVE RETRY] Windows temporarily denied replacing {path}; "
                      "keeping previous checkpoint and retrying", flush=True)
            time.sleep(delays[attempt])

def _write_atomic(path, writer):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="."+path.name+".", suffix=".partial")
    temp = Path(name)
    # Unique sibling temp avoids reuse of a filename still scanned by another process.
    # On permanent failure retain both the old destination and the pending temp.
    with os.fdopen(fd, "wb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    replace_with_retry(temp, path)

def atomic_bytes(path, content):
    _write_atomic(path, lambda f: f.write(content))

def atomic_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))

def atomic_text(path, value):
    atomic_bytes(path, value.encode("utf-8"))

def atomic_npz(path, **arrays):
    _write_atomic(path, lambda f: np.savez_compressed(f, **arrays))

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def immutable(path, obj):
    if Path(path).exists():
        require(read(path) == obj, "Inputs/code/options changed. Preserve this run and use a new --output: " + str(path))
    else:
        atomic_json(path, obj)

def runtime():
    import platform
    from importlib.metadata import version
    return dict(python=platform.python_version(), packages={k: version(k) for k in
                ("numpy", "scipy", "scikit-learn", "matplotlib")})

def implementation():
    return {p.name: sha(p) for p in sorted(BASE.glob("*.py"))}

def implementation_patch(recorded, current=None, manifest_path=None):
    """Allow only the exact reviewed storage-only patch, never arbitrary code changes."""
    current = implementation() if current is None else current
    if recorded == current:
        return None
    manifest_path = BASE/"storage_compatibility.json" if manifest_path is None else Path(manifest_path)
    require(manifest_path.is_file(), "Analysis code changed; use a new output and run p0")
    patch = read(manifest_path)
    require(patch.get("format") == "reinforcement_storage_compat_v1"
            and patch.get("previous") == recorded and patch.get("updated") == current,
            "Analysis code changed outside the reviewed storage-only patch; use a new output")
    changed = {name for name in set(recorded)|set(current) if recorded.get(name) != current.get(name)}
    require(changed == {"common.py", "experiment.py", "test_reinforcement.py"},
            "Storage compatibility must not change statistical/model/data code")
    return dict(patch_id=patch["patch_id"], previous=recorded, updated=current,
                compatibility_sha256=sha(manifest_path),
                scope="Atomic file replacement retry only; statistical code, data, configuration and RNG unchanged")

def safe_output(output, inputs):
    output = Path(output).resolve()
    for p in inputs:
        p = Path(p).resolve()
        require(output != p and output not in p.parents and p not in output.parents,
                "Output overlaps an input folder: " + str(p))
    require(output != ROOT, "Do not use repository root as output")
    return output

@contextmanager
def writer_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    f = (folder / ".writer.lock").open("a+b")
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b"0")
        f.flush()
    f.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError("Another writer is using " + str(folder))
    try:
        yield
    finally:
        f.seek(0)
        if os.name == "nt":
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()

def duration(seconds):
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"

class Progress:
    def __init__(self, name, total, folder=None, resumed=0):
        self.name, self.total, self.folder, self.resumed = name, total, folder, resumed
        self.start, self.last = time.monotonic(), 0.
        self.update(resumed, force=True)

    def update(self, done, detail="", force=False):
        now = time.monotonic()
        if not force and done != self.total and now-self.last < 5:
            return
        elapsed = now-self.start
        rate = (done-self.resumed)/elapsed if elapsed else 0
        eta = (self.total-done)/rate if rate else None
        print(f"[{self.name}] {done}/{self.total} elapsed={duration(elapsed)} ETA={duration(eta)} "
              f"rate={rate:.2f}/s resumed={self.resumed} {detail}", flush=True)
        if self.folder:
            atomic_json(Path(self.folder)/"progress.json", dict(stage=self.name, done=done, total=self.total,
                        elapsed_seconds=elapsed, eta_seconds=eta, resumed=self.resumed, detail=detail))
        self.last = now

def verify_files(folder, mapping):
    for name, expected in mapping.items():
        path = child(folder, name)
        require(path.is_file() and sha(path) == expected, "Missing/changed committed input: " + str(path))

def finish(folder, identity, files):
    folder = Path(folder)
    atomic_json(folder/"COMPLETED.json", dict(identity=identity, files={str(p): sha(folder/p) for p in files}))

def completed(folder, identity):
    p = Path(folder)/"COMPLETED.json"
    if not p.exists():
        return False
    receipt = read(p)
    require(receipt["identity"] == identity, "Completed job identity changed: " + str(folder))
    verify_files(folder, receipt["files"])
    return True
