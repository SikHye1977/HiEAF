"""Windows-safe DataLoader construction; no GPU operations inside workers."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .collate import collate_samples


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)


def build_loader(dataset, *, split=None, collate_fn=collate_samples,
                 batch_size=1, num_workers=0, seed=42, pin_memory=False):
    split = split or getattr(dataset, 'split', None)
    if split not in {'train', 'dev', 'test'}:
        raise ValueError('split must be train/dev/test')
    if getattr(dataset, 'split', split) != split:
        raise ValueError('Loader split differs from Dataset split')
    if batch_size < 1 or num_workers < 0 or len(dataset) == 0:
        raise ValueError('Invalid batch size, worker count or empty Dataset')
    options = dict(dataset=dataset, batch_size=batch_size, shuffle=split == 'train',
                   num_workers=num_workers, collate_fn=collate_fn,
                   worker_init_fn=seed_worker,
                   generator=torch.Generator().manual_seed(seed),
                   pin_memory=pin_memory, drop_last=False)
    if num_workers > 0:
        options.update(prefetch_factor=1, persistent_workers=True)
    return DataLoader(**options)
