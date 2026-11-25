"""Lossless CPU RGB cache. No models, random sampling, or CUDA in worker threads."""
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import av
import cv2
import numpy as np
import torch
from src.datasets.media import MediaConfig, uniform_indices
from scripts.trainable_or_3way.pgf_source.common import VIDEO_POLICY, require, read, sha, digest, immutable
from scripts.trainable_or_3way.reinforcement.common import atomic_json, _write_atomic

RECIPE = dict(version="pgf_uint8_exact_v1", video=VIDEO_POLICY, dtype="uint8", layout="THWC",
              scope="PGF train real and all PGF dev; index metadata for other PGF train")


def signature(path):
    s = Path(path).stat()
    require(s.st_size > 0, "Empty input/cache")
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def normalized(rgb):
    """Same NumPy division and PyTorch FP32 operation order as read_video."""
    array = rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    mean = torch.tensor(VIDEO_POLICY["mean"], dtype=torch.float32)[:, None, None, None]
    std = torch.tensor(VIDEO_POLICY["std"], dtype=torch.float32)[:, None, None, None]
    return (x-mean)/std


class FrameCache:
    def __init__(self, folder, stamps, reserve_gib=20):
        self.folder = Path(folder)
        self.stamps = stamps
        self.reserve = int(reserve_gib*1024**3)
        self._locks, self._verified = {}, {}
        self._mutex = threading.Lock()
        self.hits = self.decodes = self.index_hits = 0
        immutable(self.folder/"RECIPE.json", RECIPE)

    def _lock(self, sid):
        with self._mutex:
            return self._locks.setdefault(sid, threading.Lock())

    def _source(self, row):
        path = Path(row.media_path)
        sig = signature(path)
        require(sig[:2] == list(self.stamps[row.sample_id]), "Source changed: "+row.sample_id)
        prior = self._verified.get(row.sample_id)
        if prior is not None:
            require(prior[0] == sig, "Source changed during cache use")
            return prior[1]
        value = sha(path)
        require(signature(path) == sig, "Source changed during hashing")
        self._verified[row.sample_id] = (sig, value)
        return value

    def get(self, row):
        require(row.dataset == "PGF" and row.split in ("train", "dev"), "Cache is source train/dev only")
        with self._lock(row.sample_id):
            source_sha = self._source(row)
            binding = dict(sid=row.sample_id, path=str(Path(row.media_path).resolve()),
                           source_sha256=source_sha, stamp=list(self.stamps[row.sample_id]), recipe=RECIPE)
            key = digest(binding)
            base = self.folder/key[:2]/key
            index_path = base.with_suffix(".json")
            data_path = base.with_suffix(".npy")
            cached = row.split == "dev" or row.video_label == 0
            if index_path.exists():
                meta = read(index_path)
                require(meta["binding"] == binding, "Input cache binding changed")
                index = meta["index"]
                require(meta["index_sha256"] == digest(index), "Frame index checksum mismatch")
                self.index_hits += 1
                if cached and meta.get("data_sha256"):
                    require(data_path.is_file(), "Missing committed RGB cache")
                    require(sha(data_path) == meta["data_sha256"], "RGB cache checksum mismatch")
                    rgb = np.load(data_path, allow_pickle=False)
                    require(rgb.dtype == np.uint8 and rgb.shape == (128,256,256,3), "Invalid RGB cache shape/dtype")
                    self._source(row)
                    self.hits += 1
                    return rgb
            else:
                index = self._index(row.media_path)
            rgb = self._decode(row.media_path, index)
            self._source(row)
            meta = dict(binding=binding, index=index, index_sha256=digest(index))
            if cached:
                require(shutil.disk_usage(self.folder).free > self.reserve+rgb.nbytes,
                        "RGB cache disk reserve reached; no sample skipped")
                _write_atomic(data_path, lambda f: np.save(f, rgb, allow_pickle=False))
                meta["data_sha256"] = sha(data_path)
            atomic_json(index_path, meta)  # commit marker last; interrupted uncommitted .npy can be rebuilt
            self.decodes += 1
            return rgb

    @staticmethod
    def _index(path):
        cfg = MediaConfig(**VIDEO_POLICY)
        times = []
        with av.open(str(path)) as container:
            require(bool(container.streams.video), "No video stream")
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 1
            for frame in container.decode(stream):
                require(frame.pts is not None and frame.time_base is not None, "Missing frame timestamp")
                times.append(float(frame.pts*frame.time_base))
                require(len(times) <= cfg.max_frames and times[-1]-times[0] <= cfg.max_duration,
                        "Media resource guard exceeded")
        indices = uniform_indices(times, 128)
        return dict(decoded_frames=len(times), indices=indices.tolist(),
                    timestamps=np.asarray(times)[indices].tolist())

    @staticmethod
    def _decode(path, index):
        indices = np.asarray(index["indices"], dtype=np.int64)
        require(indices.shape == (128,) and indices.min() >= 0 and indices.max() < index["decoded_frames"],
                "Invalid cached frame indices")
        wanted = set(indices.tolist())
        images = {}
        count = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for i, frame in enumerate(container.decode(stream)):
                count += 1
                if i in wanted:
                    rgb = frame.to_ndarray(format="rgb24")
                    h,w = rgb.shape[:2]
                    scale = 256/min(h,w)
                    nw,nh = max(256,round(w*scale)),max(256,round(h*scale))
                    resized = cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
                    x,y = (nw-256)//2,(nh-256)//2
                    images[i] = resized[y:y+256,x:x+256]
        require(count == index["decoded_frames"] and len(images) == len(wanted), "Decoded frame count changed")
        return np.stack([images[int(i)] for i in indices])


class OrderedFrames:
    """Bounded FIFO; completion order never changes sample order or consumes RNG."""
    def __init__(self, records, indices, cache, workers=2, depth=2):
        self.records, self.indices, self.cache = records, list(indices), cache
        self.depth, self.position = depth, 0
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pgf-input")
        self.futures = {}
        for i in range(min(depth, len(self.indices))): self._submit(i)

    def _submit(self, position):
        if position < len(self.indices):
            self.futures[position] = self.pool.submit(self.cache.get, self.records[self.indices[position]])

    def pop(self, index):
        require(self.position < len(self.indices) and self.indices[self.position] == index,
                "Prefetch sample order mismatch")
        position = self.position
        value = self.futures.pop(position).result()
        self.position += 1
        self._submit(position+self.depth)
        return value

    def close(self):
        for future in self.futures.values(): future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.futures.clear()


class TaskInputs:
    """Retain uint8 inputs for one accumulation group, including AMP replay."""
    def __init__(self, records, stamps, config):
        self.records, self.config = records, config
        self.cache = FrameCache(config["frame_cache"], stamps, config["reserve_gib"])
        self.queue = None
        self.group = None

    def begin(self, order):
        self.close()
        self.queue = OrderedFrames(self.records, order, self.cache,
                                   self.config["workers"], self.config["depth"])

    def begin_group(self, indices):
        self.group = list(indices)
        self.saved, self.offset = [], 0

    def tensor(self, indices, device):
        values = []
        if self.group is not None and self.offset == len(self.group): self.offset = 0  # AMP replay
        for ix in indices:
            if self.group is None:
                rgb = self.queue.pop(int(ix))
            else:
                require(self.group[self.offset] == ix, "AMP/input group order mismatch")
                if self.offset == len(self.saved): self.saved.append(self.queue.pop(int(ix)))
                rgb = self.saved[self.offset]
                self.offset += 1
            values.append(normalized(rgb).unsqueeze(0).to(device))
        return torch.cat(values, dim=0)

    def close(self):
        if self.queue is not None: self.queue.close()
        self.queue = None
        self.group = None
        self.saved = []
