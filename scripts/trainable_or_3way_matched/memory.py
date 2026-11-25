"""AASIST activation checkpointing without a second BatchNorm buffer update.

State-dict names and forward math are unchanged. Only training activations inside
six residual blocks are recomputed; eval forwards use the original implementation.
"""
from contextlib import contextmanager, nullcontext
import types
import torch
from torch.utils.checkpoint import checkpoint
from scripts.trainable_or_3way.common import require

MEMORY_POLICY = dict(version='aasist_residual_checkpoint_v1',
    modules='six encoder residual blocks',use_reentrant=False,preserve_rng_state=True,
    batchnorm='ordinary train/eval; restore buffers after recomputation',
    state_dict_keys_unchanged=True,inference_unchanged=True,cuda_allocator_fraction=.70)

@contextmanager
def restore_buffers_after_recompute(module):
    saved={name:value.detach().clone() for name,value in module.named_buffers()}
    try:yield
    finally:
        with torch.no_grad():
            for name,value in module.named_buffers():value.copy_(saved[name])


def checkpoint_block(block):
    if getattr(block,'_matched_checkpoint_enabled',False):return
    forward=block.forward
    def wrapped(this,*args,**kwargs):
        if not (this.training and torch.is_grad_enabled()):return forward(*args,**kwargs)
        return checkpoint(forward,*args,use_reentrant=False,preserve_rng_state=True,
            context_fn=lambda:(nullcontext(),restore_buffers_after_recompute(this)),**kwargs)
    block.forward=types.MethodType(wrapped,block)
    block._matched_checkpoint_enabled=True


def enable_audio_checkpointing(model):
    encoder=model.backbone.encoder
    require(len(encoder)==6,'Unexpected AASIST residual encoder')
    for block in encoder:checkpoint_block(block)
    return model


def configure_cuda_budget(device):
    device=torch.device(device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(MEMORY_POLICY['cuda_allocator_fraction'],device)
