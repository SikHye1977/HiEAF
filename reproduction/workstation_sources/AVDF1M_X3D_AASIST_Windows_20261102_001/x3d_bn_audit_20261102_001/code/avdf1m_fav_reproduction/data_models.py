"""Portable AVDF1M adapter and recovered FAV model/media recipes."""
import json
from pathlib import Path
from dataclasses import dataclass

import av
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .prepare import rows


@dataclass
class ModelOutput:
    logits: torch.Tensor
    fake_probability: torch.Tensor


def read_video(path):
    # Recovered src/datasets/media.py: two-pass full-video uniform frame indices.
    times = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f'Missing timestamps: {path}')
            times.append(float(frame.pts * frame.time_base))
            if len(times) > 100000 or times[-1] - times[0] > 600:
                raise ValueError('Media resource guard exceeded')
    ts = np.asarray(times)
    if not len(ts) or not np.isfinite(ts).all() or (np.diff(ts) < 0).any():
        raise ValueError(f'Invalid timestamps: {path}')
    indices = np.linspace(0, len(ts)-1, 128).astype(np.int64)
    wanted = set(indices.tolist())
    images = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                rgb = frame.to_ndarray(format='rgb24')
                h, w = rgb.shape[:2]
                scale = 256 / min(h, w)
                nw, nh = max(256, round(w * scale)), max(256, round(h * scale))
                resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                x, y = (nw-256)//2, (nh-256)//2
                images[i] = resized[y:y+256, x:x+256]
    if len(images) != len(wanted):
        raise ValueError('Frame count changed between decode passes')
    array = np.stack([images[int(i)] for i in indices]).astype(np.float32) / 255.0
    video = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    return (video - .45) / .225


class PortableDataset(Dataset):
    def __init__(self, preflight, split, task):
        self.root = Path(preflight['dataset_root'])
        self.cache = Path(preflight['cache_root'])
        self.target = preflight['audio']['target_num_samples']
        self.task = task
        self.rows = rows(self.root, split)
        for r in self.rows:
            r['labels'] = [r['clip_label'], r['video_label'], r['audio_label']]
        self.audio = {r['sample_id']: r for r in json.loads((self.cache/'audio_index.json').read_text())}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        result = dict(sample_id=r['sample_id'], split=r['split'], labels=torch.tensor(r['labels']),
                      label_mask=torch.ones(3, dtype=torch.bool))
        if self.task == 'video':
            result['video'] = read_video(self.root/r['archive_path'])
        else:
            entry = self.audio[r['sample_id']]
            wave = torch.from_numpy(np.load(self.cache/entry['cache_file'], allow_pickle=False))
            n = len(wave)
            if n != entry['samples'] or n > self.target or not torch.isfinite(wave).all():
                raise ValueError(f'Invalid audio length/content: {r["sample_id"]}')
            result['audio'] = torch.nn.functional.pad(wave, (0, self.target-n))
            result['audio_lengths'] = n
        return result


class X3D(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from pytorchvideo.models.hub import x3d_m
        self.backbone = x3d_m(pretrained=False)
        self.backbone.blocks[-1].proj = nn.Linear(self.backbone.blocks[-1].proj.in_features, 1)
        self.backbone.blocks[-1].activation = nn.Identity()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state']
        weights = {k:v for k,v in state.items() if not k.startswith('blocks.5.proj.')}
        result = self.backbone.load_state_dict(weights, strict=False)
        if set(result.missing_keys) != {'blocks.5.proj.weight','blocks.5.proj.bias'} or result.unexpected_keys:
            raise ValueError(f'Pretrained architecture mismatch: {result}')

    def forward(self, video):
        logits = self.backbone(video).flatten(start_dim=1).squeeze(1)
        return ModelOutput(logits=logits, fake_probability=torch.sigmoid(logits))


class AASIST(nn.Module):
    fake_class_index = 1

    def __init__(self, official_config, offload=True):
        super().__init__()
        from aasist.models.AASIST import Model
        self.backbone = Model(official_config)
        self.offload = offload

    def forward(self, wave):
        from contextlib import nullcontext
        # Copies saved tensors, without recomputation or a second BN update.
        context = torch.autograd.graph.save_on_cpu(pin_memory=False) if self.training and self.offload else nullcontext()
        with context:
            _, logits = self.backbone(wave)
            return ModelOutput(logits=logits, fake_probability=torch.softmax(logits, dim=-1)[:,1])
