"""NEW transfer helper, not historical evaluation code. No automatic inference.

Only CPU construction/strict state loading was checked during packaging.
Windows execution and media decoding have NOT been tested by the packager.
"""
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
X3D_ROOT = ROOT / 'x3d_bn_audit_20261102_001'
# AASIST is a namespace package in the original repository. Its parent must
# precede the delivery root's unrelated `aasist` materials directory.
for folder in (X3D_ROOT/'dependencies', X3D_ROOT/'code', ROOT/'aasist/dependencies'):
    sys.path.insert(0, str(folder))


def load_x3d(condition='B', device='cpu'):
    """A=running BN/AMP; B=running BN/FP32; C=single-video BN/FP32.

    Forward precision is chosen by the caller; this function only sets BN mode.
    Do not call model.eval() again after configuring C: that resets its BN mode.
    """
    import torch
    from x3d_bn_audit.core import X3D, configure_condition
    if condition not in ('A', 'B', 'C'):
        raise ValueError('condition must be A, B, or C')
    model = X3D()
    state = torch.load(X3D_ROOT/'weights/best_state_dict.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    configure_condition(model, condition)
    return model


def load_aasist(device='cpu'):
    """Load the original wrapper, whose keys include the `backbone.` prefix."""
    import torch
    from avdf1m_fav_reproduction.data_models import AASIST
    config = json.loads((ROOT/'aasist/metadata/effective_config.json').read_text(encoding='utf-8'))
    model = AASIST(config['model']['official_config'], offload=False)
    state = torch.load(ROOT/'aasist/weights/best_state_dict.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def resolve_media(row, avdf1m_root):
    """avdf1m_root is the AV-Deepfake1M folder itself, e.g. D:\\datasets\\AV-Deepfake1M."""
    rel = row.get('relative_path')
    if rel is None:
        archive = row['archive_path']
        if not archive.startswith('AV-Deepfake1M/'):
            raise ValueError('Expected AV-Deepfake1M/ archive prefix')
        rel = archive[len('AV-Deepfake1M/'):]
    p = PurePosixPath(rel)
    if p.is_absolute() or '..' in p.parts or '\\' in rel or ':' in rel:
        raise ValueError('Unsafe relative path')
    return Path(avdf1m_root).joinpath(*p.parts)


def read_video(path):
    """Original, unmodified X3D preprocessing. Returns float32 [3,128,256,256]."""
    from avdf1m_fav_reproduction.data_models import read_video as original_read_video
    return original_read_video(path)


def read_audio(path, ffmpeg='ffmpeg'):
    """New convenience wrapper for the original decode + right-zero-pad recipe.

    Original implementations are prepare.py:main/decode and
    data_models.py:PortableDataset.__getitem__. Returns float32 [1,510976].
    No peak/RMS/z-score normalization, cropping, repeat padding or length mask.
    """
    import numpy as np
    import torch
    target = 510976
    result = subprocess.run([str(ffmpeg), '-v', 'error', '-xerror', '-threads', '1',
                             '-i', str(path), '-map', '0:a:0', '-vn', '-ac', '1',
                             '-ar', '16000', '-f', 'f32le', '-'],
                            capture_output=True, check=True, timeout=120)
    wave = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not len(wave) or len(wave) > target or not np.isfinite(wave).all():
        raise ValueError(f'Expected finite nonempty audio <= {target} samples; got {len(wave)}')
    return torch.nn.functional.pad(torch.from_numpy(wave), (0,target-len(wave))).unsqueeze(0)
