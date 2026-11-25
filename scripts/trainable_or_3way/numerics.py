"""Finite-gradient checks and an overflow-safe clipping fallback."""
import torch
from .common import require

NUMERICS_POLICY = dict(version='amp_retry_v1', max_amp_retries=8,
    recovery='lower GradScaler scale; restore buffers/RNG; retry the same accumulation group',
    invalid_loss='stop with sample IDs', fp32_nonfinite_gradient='stop',
    clipping='max norm 5; FP64 norm fallback if finite FP32 gradients overflow norm reduction')

def gradients_finite(model):
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    require(grads, 'No gradients were produced')
    return bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())

def clip_finite_gradients(parameters, max_norm=5.):
    params=[p for p in parameters if p.grad is not None]
    try:
        return torch.nn.utils.clip_grad_norm_(params,max_norm,error_if_nonfinite=True)
    except RuntimeError as error:
        # Do not turn genuine invalid elements into zero/NaN updates.
        if not all(bool(torch.isfinite(p.grad).all()) for p in params):
            raise
        if 'non-finite' not in str(error):
            raise
        norms=torch.stack([torch.linalg.vector_norm(p.grad.double()) for p in params])
        norm=torch.linalg.vector_norm(norms)
        require(bool(torch.isfinite(norm)), 'Nonfinite FP64 gradient norm')
        coefficient=(max_norm/(norm+1e-6)).clamp(max=1.)
        with torch.no_grad():
            for p in params: p.grad.mul_(coefficient.to(device=p.grad.device,dtype=p.grad.dtype))
        return norm

def finite_state(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite_state(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite_state(v) for v in value)
    return True
