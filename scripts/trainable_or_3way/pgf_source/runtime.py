"""Whole-waveform audio, shared B/C video decode, fixed inference state."""
import subprocess
import numpy as np
import torch
from .common import *
from src.datasets.media import MediaConfig,read_video
from src.utils.local_paths import ffmpeg_executable
from scripts.trainable_or_3way.models import artifact
from scripts.trainable_or_3way.reinforcement_p2.runtime import (
    video_pair,buffer_vector,assert_state,validate_score)

def pad_wave(wave,minimum=AUDIO["minimum_samples"]):
    wave=np.asarray(wave,dtype=np.float32)
    require(wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all(),"Empty/nonfinite audio")
    return np.pad(wave,(0,max(0,minimum-len(wave))))

def decode(row,mod,ffmpeg=None):
    path=Path(row["media_path"])
    require(stamp_path(path)==row["stamp"],"Media changed before decode: "+row["sample_id"])
    media_hash=sha(path)
    if mod=="video":
        d=read_video(path,MediaConfig(**VIDEO_POLICY));x=d["video"]
        require(tuple(x.shape)==(3,128,256,256),"Unexpected video input")
        meta=dict(decoded_frames=d["decoded_frames"])
    else:
        ffmpeg=ffmpeg or ffmpeg_executable()
        p=subprocess.run([str(ffmpeg),"-v","error","-xerror","-threads","1","-i",str(path),"-map","0:a:0",
             "-vn","-ac","1","-ar","16000","-f","f32le","-"],capture_output=True,check=True,timeout=120)
        wave=np.frombuffer(p.stdout,dtype="<f4").copy()
        x=torch.from_numpy(pad_wave(wave))
        meta=dict(decoded_samples=len(wave),input_samples=len(x),long_audio=len(wave)>AUDIO["minimum_samples"])
    require(bool(torch.isfinite(x).all()),"Nonfinite media input")
    require(stamp_path(path)==row["stamp"],"Media changed during decode")
    import hashlib
    return x,dict(meta,media_sha256=media_hash,input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())

def stamp_path(path):
    p=Path(path);require(p.is_file(),"Missing media: "+str(p));s=p.stat()
    require(s.st_size>0,"Empty media");return [s.st_size,s.st_mtime_ns]

def row_for(record,stamps):
    return dict(sample_id=record.sample_id,media_path=str(record.media_path),stamp=stamps[record.sample_id])

def artifact_tensor(record,mod,device):
    x,_=decode(dict(sample_id=record.sample_id,media_path=str(record.media_path),stamp=stamp(record)),mod)
    return x.unsqueeze(0).to(device)

class Engine:
    def __init__(self,path,mod,device,expected):
        require(sha(path)==expected,"Selected artifact checkpoint changed")
        state=torch.load(path,map_location="cpu",weights_only=True)
        require(state["source"]=="PGF" and state["modality"]==mod and state["kind"]=="artifact","Wrong artifact checkpoint")
        self.model=artifact(mod,pretrained=False).to(device).float().eval().requires_grad_(False)
        self.model.load_state_dict(state["model"],strict=True)
        self.before=cpu_state(self.model);self.buffers=buffer_vector(self.model).clone()
        self.mod,self.device=mod,device
    @torch.inference_mode()
    def predict(self,decoded):
        x,meta=decoded;x=x[None].to(self.device)
        if self.mod=="video":value=dict(meta,**video_pair(self.model,x,self.device))
        else:
            require(not any(m.training for m in self.model.modules()),"AASIST must stay eval")
            with torch.autocast("cuda" if self.device.startswith("cuda") else "cpu",enabled=False):
                out=self.model(x)
            value=dict(meta,logits=out.logits[0].float().cpu().tolist(),probability=float(out.fake_probability.item()))
            require(torch.equal(buffer_vector(self.model),self.buffers),"AASIST buffers changed")
        validate_score(value,self.mod);return value
    def close(self):
        assert_state(self.model,self.before)
        del self.model,self.before,self.buffers
        free_gpu()
