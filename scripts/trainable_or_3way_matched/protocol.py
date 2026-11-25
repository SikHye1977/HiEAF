"""Explicit single-GPU approximation of historical per-GPU/effective batches."""
import torch
from scripts.trainable_or_3way.common import require

ARTIFACT_PROTOCOL = {
    'video': dict(batch_size=1, accumulation=8, effective_batch=8,
                  optimizer='AdamW', loss='unweighted BCEWithLogits', amp=True,
                  amp_initial_scale=16384., gradient_clip_norm=None),
    'audio': dict(batch_size=4, accumulation=4, effective_batch=16,
                  optimizer='Adam', loss='unweighted CE', amp=False,
                  gradient_clip_norm=None),
}
NUMERICS_POLICY = dict(version='matched_amp_retry_v1', max_amp_retries=8,
    recovery='lower AMP scale; restore all buffers/RNG; retry the same accumulation group',
    invalid_loss='stop with sample IDs', fp32_nonfinite_gradient='stop',
    clipping=None, partial_accumulation='weight each microbatch by its sample count')

def balanced_draws(labels, count, seed, epoch):
    pools=[torch.tensor([i for i,y in enumerate(labels) if y==c]) for c in (0,1)]
    require(count>0 and all(len(p) for p in pools),'Both source classes and positive draw count required')
    generator=torch.Generator().manual_seed(seed+epoch)
    sizes=(count//2,count-count//2)
    draws=torch.cat([pool[torch.randint(len(pool),(n,),generator=generator)]
                     for pool,n in zip(pools,sizes)])
    return draws[torch.randperm(count,generator=generator)].tolist()
