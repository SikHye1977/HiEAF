"""Stack video; right-zero-pad full waveforms and preserve their valid lengths."""
import torch


def collate_samples(items):
    if not items:
        raise ValueError('Empty batch')
    if len({r['task'] for r in items}) != 1:
        raise ValueError('Cannot mix video/audio/paired tasks in a batch')
    task = items[0]['task']
    batch = {k: [r[k] for r in items] for k in
             ('sample_id', 'group_id', 'dataset', 'split', 'path')}
    batch['task'] = task
    for key in ('labels', 'label_mask'):
        batch[key] = torch.stack([r[key] for r in items])
    if task != 'audio':
        for key in ('video', 'video_mask', 'frame_unique_mask', 'frame_indices', 'timestamps'):
            batch[key] = torch.stack([r[key] for r in items])
        batch['decoded_frames'] = torch.tensor([r['decoded_frames'] for r in items])
    if task != 'video':
        rates = {r['sample_rate'] for r in items}
        if len(rates) != 1:
            raise ValueError('Mixed sample rates')
        lengths = torch.tensor([r['audio'].numel() for r in items], dtype=torch.long)
        audio = torch.zeros(len(items), int(lengths.max()), dtype=torch.float32)
        for i, row in enumerate(items):
            audio[i, :lengths[i]] = row['audio']
        batch.update(audio=audio, audio_lengths=lengths,
                     audio_mask=torch.arange(audio.shape[1])[None, :] < lengths[:, None],
                     sample_rate=next(iter(rates)))
    return batch
