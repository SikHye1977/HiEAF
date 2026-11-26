import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid
import importlib.metadata

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASETS = ('FAV', 'AVDF1M', 'PGF')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextlib.contextmanager
def atomic_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with tmp.open('wb') as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        # All readers in this runner close before replace, including npz/checkpoint readers.
        for attempt in range(12):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(path, value):
    with atomic_file(path) as f:
        f.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def bind(path, value):
    if Path(path).exists():
        require(read_json(path) == value, f'Configuration/cohort mismatch: {path}. Use a NEW output/cache directory.')
    else:
        atomic_json(path, value)


@contextlib.contextmanager
def exclusive(path):
    import msvcrt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    try:
        f.seek(0)
        if not f.read(1):
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError(f'Another process owns {path}; do not run two writers.') from e
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        f.close()


def hms(seconds):
    if seconds is None:
        return '--:--:--'
    n = max(0, int(seconds))
    return f'{n//3600:02}:{n//60%60:02}:{n%60:02}'


class Progress:
    def __init__(self, name, total, resumed=0, status=None):
        self.name, self.total, self.resumed = name, total, resumed
        self.start = time.perf_counter()
        self.last = 0
        self.status = status
        self.update(resumed, force=True)

    def update(self, done, extra='', force=False):
        now = time.perf_counter()
        if not force and done != self.total and now - self.last < 10:
            return
        elapsed = now - self.start
        rate = (done - self.resumed) / max(elapsed, 1e-9)
        eta = (self.total - done) / rate if rate > 0 else None
        print(f'[{self.name}] {done}/{self.total} ({100*done/max(self.total,1):.1f}%) '
              f'elapsed={hms(elapsed)} ETA={hms(eta)} rate={rate:.2f}/s '
              f'resumed={self.resumed} {extra}', flush=True)
        self.last = now
        if self.status:
            atomic_json(self.status, dict(stage=self.name, done=done, total=self.total,
                        resumed=self.resumed, elapsed_seconds=elapsed, eta_seconds=eta,
                        rate=rate, detail=extra, updated=time.time()))


class Stopped(Exception):
    pass


class Stop:
    def __init__(self, folder):
        self.path = Path(folder) / 'STOP_REQUESTED'
        self.requested = False

    def install(self):
        def handler(*_):
            if self.requested:
                raise KeyboardInterrupt
            self.requested = True
            print('[STOP] Finishing this safe save boundary. Press Ctrl+C again only for immediate abort.', flush=True)
        signal.signal(signal.SIGINT, handler)

    def pending(self):
        return self.requested or self.path.exists()

    def check(self):
        if self.pending():
            raise Stopped('Safely stopped; rerun the same command to resume. Remove STOP_REQUESTED if used.')


def implementation_id():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}


def runtime_versions():
    names = ('torch', 'numpy', 'scipy', 'av', 'opencv-python-headless', 'python_speech_features', 'scikit-learn')
    return {n: importlib.metadata.version(n) for n in names}
