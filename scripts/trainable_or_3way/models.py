"""Existing architecture, explicit foundation initialization, frozen temporal tokens."""
from pathlib import Path
import numpy as np
import torch
from src.datasets.media import MediaConfig, read_video, read_audio
from src.models.x3d import X3DDeepfakeClassifier
from scripts.artifact_emotion_gated_attention_cross_dataset.encoders import fresh_artifact
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.model import EmotionFlowHead
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.extract import Encoders, assets, validate_tokens
from scripts.hierarchical_score_fusion_GRU_Tempooral_Attention_cross_dataset.inference import FrozenPredictor
from .common import *

X3D = ROOT/'checkpoints/pretrained/X3D_M.pyth'
VIDEO = dict(frames=128,resize_short=256,crop_size=256,mean=[.45]*3,std=[.225]*3)
AUDIO = dict(sample_rate=16000,channels='mono',minimum_samples=284672,truncation=False,batch_size=1)
TOKEN_RECIPE = dict(video='128 uniform full-video RGB center crops 256; ImageNet normalization',
                    audio='16k mono original unpadded waveform; full 3s windows/hop1s; short whole; trailing partial omitted',
                    dtype='float32',embedding_dims=DIMS,classes=CLASSES)

def local_assets():
    from src.utils.local_paths import load_local_paths
    require(load_local_paths().get('emotion2vec_snapshot'),'Pinned local emotion2vec snapshot required; no downloads')
    require(X3D.is_file(),'Public Kinetics X3D_M.pyth is missing')
    value=assets()
    value.update(x3d_path=str(X3D),x3d_sha256=sha(X3D))
    return value

def artifact(mod,pretrained=True):
    if mod=='audio':
        return fresh_artifact('audio',pretrained=False)
    model=X3DDeepfakeClassifier('x3d_m',pretrained=False)
    if pretrained:
        state=torch.load(X3D,map_location='cpu',weights_only=True)['model_state']
        require(state['blocks.5.proj.weight'].shape[0]==400,'Expected public 400-class Kinetics weights, not source-trained weights')
        state={k:v for k,v in state.items() if k not in ('blocks.5.proj.weight','blocks.5.proj.bias')}
        mismatch=model.backbone.load_state_dict(state,strict=False)
        require(set(mismatch.missing_keys)=={'blocks.5.proj.weight','blocks.5.proj.bias'}
                and not mismatch.unexpected_keys,'Unexpected public X3D state')
    return model

def artifact_tensor(record,mod,device):
    if mod=='video':
        value=read_video(record.media_path,MediaConfig(**VIDEO))['video'][None]
    else:
        wave,present=read_audio(record.media_path,MediaConfig(sample_rate=16000))
        require(present and len(wave)>0,'No audio: '+record.sample_id)
        value=torch.nn.functional.pad(wave,(0,max(0,284672-len(wave))))[None]
    return value.to(device)

class Tokens:
    def __init__(self,asset,mod,device,frame_batch):
        self.asset,self.mod,self.device=asset,mod,device
        self.core=Encoders(asset,device,frame_batch)
        self.audio=None
    def __call__(self,record):
        if self.mod=='video':
            return self.core.video(record.media_path)
        if self.audio is None:
            from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import FrozenEmotion2VecSeed
            self.core.load_audio()
            self.audio=FrozenPredictor.__new__(FrozenPredictor)
            self.audio.core=self.core
            self.audio.seed=FrozenEmotion2VecSeed(self.asset['seed_path'],self.device)
            self.audio.seed.model.kwargs['disable_pbar']=True
        # Reuse the exact earlier 3s/1s waveform/window/seed-class implementation.
        return FrozenPredictor.audio_tokens(self.audio,record.media_path)

@torch.inference_mode()
def emotion_score(model,tokens,device):
    mask=tokens['valid']
    if not mask.any():
        return float('nan')
    e=torch.from_numpy(tokens['embedding'][mask]).float()[None].to(device)
    p=torch.from_numpy(tokens['probabilities'][mask]).float()[None].to(device)
    m=torch.ones(e.shape[:2],dtype=torch.bool,device=device)
    return float(model(e,p,m)[0].sigmoid()[0].cpu())

def load_best(path,kind,mod,identity,device):
    state=torch.load(path,map_location='cpu',weights_only=True)
    require(state['identity']==identity and state['kind']==kind and state['modality']==mod,'Wrong selected model')
    model=artifact(mod,False) if kind=='artifact' else EmotionFlowHead(mod)
    model.load_state_dict(state['model'],strict=True)
    return model.to(device).eval().requires_grad_(False)
