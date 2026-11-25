"""Source-specific artifact models; frozen encoders and unchanged vector gate."""
from pathlib import Path
import torch
import yaml
from src.models.x3d import X3DDeepfakeClassifier
from src.models.aasist import build_aasist
from scripts.artifact_emotion_gated_attetnion_optimized.model import GatedBranch, score_or
from scripts.artifact_emotion_gated_attetnion_optimized.local_inputs import load_local_artifact
from .common import ROOT, read, sha

FAV_ARTIFACT = ROOT/'checkpoints/experiments/x3a_reproduction/fav_indomain_v1/unified_4gpu/x3a_4gpu_001'
FAV_RUN = ROOT/'reports/artifact_emotion_gated_optimized_v1'


def model_config(modality):
    name = 'x3d' if modality == 'video' else 'aasist'
    return yaml.safe_load((ROOT/f'configs/experiments/x3a_reproduction/fav_indomain_v1/{name}_single_gpu.yaml').read_text(encoding='utf-8'))


def fresh_artifact(modality, *, pretrained=True):
    if modality == 'video':
        return X3DDeepfakeClassifier('x3d_m', pretrained=pretrained)
    cfg = model_config(modality)
    return build_aasist(ROOT/cfg['third_party']['path'], cfg['model']['official_config'], fake_class_index=1)


def artifact_input(batch, modality, device):
    if modality == 'video':
        return batch['video'].to(device)
    if batch['audio'].shape[0] != 1:
        raise ValueError('Whole-waveform AASIST requires batch1 for invariant padding')
    n = int(batch['audio_lengths'][0]); wave = batch['audio'][0, :n]
    return torch.nn.functional.pad(wave, (0, max(0, 284672-n)))[None].to(device)


def load_artifact(path, source, modality, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'cross_artifact_v1' or state['source'] != source or state['modality'] != modality:
        raise ValueError('Artifact source/modality mismatch; target-trained weights forbidden')
    model = fresh_artifact(modality, pretrained=False)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), state


def load_head(path, source, modality, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'cross_head_v1' or state['source'] != source or state['modality'] != modality:
        raise ValueError('Head source/modality mismatch')
    model = GatedBranch(state['artifact_dim'], state['emotion_dim'])
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), state


def import_fav(modality):
    """Only the user's recorded FAV checkpoints are allowed by this explicit option."""
    from scripts.artifact_emotion_gated_attetnion_optimized.common import implementation
    from scripts.artifact_emotion_gated_attetnion_optimized.experiment import load_head as old_head
    lock_path = FAV_RUN/'LOCKED_BEFORE_TEST.json'; lock = read(lock_path)
    if lock['implementation'] != implementation():
        raise ValueError('Existing FAV implementation no longer matches its lock')
    part = lock['branches'][modality]; hp = Path(part['checkpoint'])
    if sha(hp) != part['sha256']:
        raise ValueError('FAV head checksum mismatch')
    head, hs = old_head(hp)
    if hs['recipe']['dataset'] != 'FAV':
        raise ValueError('Imported head was not trained on FAV')
    ap = FAV_ARTIFACT/('x3d' if modality == 'video' else 'aasist')/'best.pt'
    if sha(ap) != hs['recipe']['artifact_checkpoint_sha256']:
        raise ValueError('FAV artifact/head pair mismatch')
    artifact, cfg = load_local_artifact(ap, modality)
    if cfg['data']['dataset'] != 'FAV':
        raise ValueError('Imported artifact was not trained on FAV')
    return artifact, head, dict(artifact_path=str(ap), artifact_sha256=sha(ap),
                                 head_path=str(hp), head_sha256=sha(hp))
