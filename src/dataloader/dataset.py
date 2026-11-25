"""Lazy CPU decoding for new local experiments, with modality-specific output."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.datasets.media import MediaConfig, read_video, read_audio
from .metadata import load_records


class DeepfakeDataset(Dataset):
    def __init__(self, dataset_name, split, *, task='paired', config=None, **metadata_options):
        self.split = split
        self.task = task
        self.config = config or MediaConfig(frames=128, crop_size=256, resize_short=256,
                                           mean=(.45,)*3, std=(.225,)*3)
        self.records = load_records(dataset_name, split, task=task, **metadata_options)
        self.input_policy = 'new_local_media; historical content identity not asserted'

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        labels = [r.clip_label, r.video_label, r.audio_label]
        item = dict(sample_id=r.sample_id, group_id=r.group_id, dataset=r.dataset,
                    split=r.split, path=str(r.media_path), task=self.task,
                    labels=torch.tensor([-1 if v is None else v for v in labels], dtype=torch.long),
                    label_mask=torch.tensor([v is not None for v in labels], dtype=torch.bool))
        try:
            if not r.media_path.is_file() or r.media_path.stat().st_size == 0:
                raise FileNotFoundError(r.media_path)
            if self.task != 'audio':
                item.update(read_video(r.media_path, self.config))
            if self.task != 'video':
                item['audio'], present = read_audio(r.media_path, self.config)
                if not present:
                    raise ValueError('Requested audio stream is absent')
                item['sample_rate'] = self.config.sample_rate
            return item
        except Exception as error:
            raise RuntimeError(f'{r.dataset} {r.sample_id} [{r.media_path}]: {error}') from error
