"""FP32 artifact inference with immutable parameters/BN buffers and shared B/C decoding."""
import hashlib
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from .common import require,stamp,VIDEO,AUDIO

def configure(device,threads):
    require(str(device).startswith("cuda") and torch.cuda.is_available(),"Use an available CUDA device; no silent CPU fallback")
    torch.cuda.set_device(device)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    return dict(device=device,gpu=torch.cuda.get_device_name(device),torch=torch.__version__,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=threads,
                cudnn_benchmark=False,cudnn_deterministic=False,matmul_tf32=False,cudnn_tf32=False)

def configure_condition(model,condition):
    require(condition in ("B","C"),"Only B/C are supported")
    model.eval().requires_grad_(False)
    count=0
    for m in model.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            require(m.running_mean is not None and m.running_var is not None,"Expected saved BN buffers")
            m.training=(condition=="C")
            m.track_running_stats=(condition=="B")
            count+=1
    require(count>0,"X3D has no BatchNorm layers")
    check_modes(model,condition)

def check_modes(model,condition):
    require(not model.training,"Top-level model must be eval")
    for m in model.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            require(m.training==(condition=="C") and m.track_running_stats==(condition=="B"),"Wrong BN mode")
        else:
            require(not m.training,"Dropout/non-BN layer left in training mode")
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()),"Inference must not create gradients")

def buffer_vector(model):
    values=[b.detach().reshape(-1).to(torch.float64) for b in model.buffers()]
    return torch.cat(values) if values else torch.empty(0)

def assert_state(model,before):
    current=model.state_dict()
    require(set(current)==set(before),"Model state keys changed")
    require(all(torch.equal(v.detach().cpu(),before[k]) for k,v in current.items()),"Model weights or BN buffers changed")

def pad_audio(wave):
    wave=np.asarray(wave,dtype=np.float32)
    require(wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all(),"Invalid decoded audio")
    return np.pad(wave,(0,max(0,AUDIO["minimum_samples"]-len(wave))))

class Decoder:
    def __init__(self,run):
        self.ffmpeg=run["ffmpeg"]["path"]
    def decode(self,row,modality):
        from src.datasets.media import MediaConfig,read_video
        from .common import sha
        import subprocess
        path=row["media_path"]
        require(stamp(path)==row["stamp"],"Media changed before decoding: "+row["sample_id"])
        media_hash=sha(path)
        if modality=="video":
            value=read_video(path,MediaConfig(**VIDEO))
            x=value["video"]
            require(tuple(x.shape)==(3,128,256,256) and x.dtype==torch.float32,"Unexpected X3D input")
            meta=dict(decoded_frames=value["decoded_frames"])
        else:
            require(modality=="audio","Unknown modality")
            p=subprocess.run([self.ffmpeg,"-v","error","-xerror","-threads","1","-i",path,"-map","0:a:0",
                              "-vn","-ac","1","-ar","16000","-f","f32le","-"],capture_output=True,check=True,timeout=120)
            wave=np.frombuffer(p.stdout,dtype="<f4").copy()
            n=len(wave)
            x=torch.from_numpy(pad_audio(wave))
            meta=dict(decoded_samples=n,input_samples=len(x),long_audio=n>AUDIO["minimum_samples"])
        require(torch.isfinite(x).all().item(),"Non-finite decoded input")
        require(stamp(path)==row["stamp"],"Media changed during decoding: "+row["sample_id"])
        meta.update(media_sha256=media_hash,input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
        return x,meta

@torch.inference_mode()
def video_pair(model,x,device):
    require(x.ndim==5 and x.shape[0]==1,"Per-video BN requires batch size 1")
    reference=buffer_vector(model).clone()
    result={}
    context=torch.autocast("cuda",enabled=False) if str(device).startswith("cuda") else nullcontext()
    with context:
        for c in ("B","C"):
            configure_condition(model,c)
            output=model(x)
            z=output.logits.float().reshape(-1)
            p=output.fake_probability.float().reshape(-1)
            require(z.numel()==p.numel()==1,"Unexpected X3D output")
            result[c+"_logit"]=float(z.item())
            result[c+"_probability"]=float(p.item())
    require(torch.equal(reference,buffer_vector(model)),"B/C modified saved BN buffers")
    configure_condition(model,"B")
    return result

class Engine:
    def __init__(self,run,modality):
        from scripts.artifact_emotion_gated_attetnion_optimized.local_inputs import load_local_artifact
        from .common import sha
        asset=run["assets"][modality]
        require(sha(asset["path"])==asset["sha256"],"Artifact checkpoint changed")
        self.model,cfg=load_local_artifact(asset["path"],modality)
        require(cfg==asset["config"],"Embedded model configuration changed")
        self.modality=modality
        self.device=run["runtime"]["device"]
        self.model=self.model.to(self.device).float().eval().requires_grad_(False)
        self.before={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        self.buffers=buffer_vector(self.model).clone()
    @torch.inference_mode()
    def predict(self,decoded):
        x,meta=decoded
        x=x.unsqueeze(0).to(self.device)
        if self.modality=="video":
            result=dict(meta,**video_pair(self.model,x,self.device))
        else:
            require(not any(m.training for m in self.model.modules()),"AASIST must remain eval")
            with torch.autocast("cuda",enabled=False):
                out=self.model(x)
            require(out.logits.shape==(1,2),"Unexpected AASIST output")
            result=dict(meta,logits=out.logits[0].float().cpu().tolist(),probability=float(out.fake_probability.item()))
            require(torch.equal(self.buffers,buffer_vector(self.model)),"AASIST BN buffers changed")
        validate_score(result,self.modality)
        return result
    def close(self):
        assert_state(self.model,self.before)
        del self.model,self.before,self.buffers
        torch.cuda.empty_cache()

def validate_score(value,modality):
    require(all(isinstance(value.get(k),str) and len(value[k])==64 for k in ("media_sha256","input_sha256")),"Missing score input identity")
    if modality=="video":
        pairs=[(value[c+"_logit"],value[c+"_probability"]) for c in ("B","C")]
        from scipy.special import expit
        for z,p in pairs:
            require(np.isfinite(z) and np.isfinite(p) and 0<=p<=1 and abs(float(expit(z))-p)<2e-7,"Invalid video output")
    else:
        z=np.asarray(value["logits"])
        p=value["probability"]
        from scipy.special import softmax
        require(z.shape==(2,) and np.isfinite(z).all() and np.isfinite(p) and 0<=p<=1
                and abs(float(softmax(z)[1])-p)<2e-7,"Invalid audio output")
