"""Load verified bundle assets and preserve the workstation inference recipes."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scripts.trainable_or_3way.common import ROOT, require, sha, cpu_state

DEFAULT_BUNDLE = ROOT.parent/'AVDF1M_X3D_AASIST_Windows_20261102_001'

def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def by_id(rows):
    result = {r['sample_id']:r for r in rows}
    require(len(result)==len(rows), 'Duplicate sample ID')
    return result

def verify_bundle(bundle):
    bundle=Path(bundle).resolve(); entries={}
    for line in (bundle/'SHA256SUMS.txt').read_text().splitlines():
        expected,rel=line.split(maxsplit=1); rel=rel.lstrip('*');path=(bundle/rel).resolve()
        require(path.is_relative_to(bundle) and rel not in entries, 'Invalid checksum entry')
        require(path.is_file() and sha(path)==expected, 'Bundle integrity failure: '+rel)
        entries[rel]=expected
    return dict(path=str(bundle),manifest_sha256=sha(bundle/'SHA256SUMS.txt'),files=len(entries),
                x3d=entries['x3d_bn_audit_20261102_001/weights/best_state_dict.pth'],
                aasist=entries['aasist/weights/best_state_dict.pth'])

def helper(bundle):
    # Suppress __pycache__ in the read-only imported source tree.
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('workstation_delivery_helper',Path(bundle)/'windows_loaders.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def configure(device,threads,tf32):
    require(device.startswith('cuda') and torch.cuda.is_available(), 'A CUDA GPU is required')
    torch.cuda.set_device(device);torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=(tf32=='workstation')
    torch.use_deterministic_algorithms(False)
    return dict(device=device,gpu=torch.cuda.get_device_name(device),torch=torch.__version__,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=threads,
                cudnn_benchmark=False,cudnn_deterministic=False,matmul_tf32=False,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,tf32_policy=tf32)

class Engines:
    def __init__(self,bundle,device,ffmpeg,conditions):
        self.helper=helper(bundle); self.device=device;self.ffmpeg=ffmpeg;self.conditions=conditions
        self.video=None;self.audio=None;self.before=None
        self.audio_index=by_id(json.loads((Path(bundle)/'aasist/reference/audio_index.json').read_text()))
        self.dev_reference=by_id(jsonl(Path(bundle)/'x3d_bn_audit_20261102_001/results/B/predictions.jsonl'))
    def load(self,modality):
        self.release()
        self.active=(self.helper.load_x3d('B',self.device) if modality=='video' else self.helper.load_aasist(self.device))
        self.before=cpu_state(self.active);self.modality=modality
    def release(self):
        if hasattr(self,'active'):
            from x3d_bn_audit.core import assert_state_preserved
            assert_state_preserved(self.active,self.before)
            del self.active,self.before
            torch.cuda.empty_cache()
    def decode(self,records):
        import hashlib,io
        result=[]
        for r in records:
            media_hash=sha(r.media_path)
            expected=self.dev_reference.get(r.sample_id)
            if expected is not None:require(media_hash==expected['media_sha256'],'Dev media hash changed: '+r.sample_id)
            if self.modality=='video':
                x=self.helper.read_video(r.media_path)
                require(tuple(x.shape)==(3,128,256,256),'Unexpected video shape')
            else:
                x=self.helper.read_audio(r.media_path,ffmpeg=self.ffmpeg).squeeze(0)
                require(tuple(x.shape)==(510976,),'Unexpected audio shape')
                # Dev/cache identity can be verified even though no .npy files were transferred.
                old=self.audio_index.get(r.sample_id)
                if old is not None:
                    mem=io.BytesIO();np.save(mem,x[:old['samples']].numpy(),allow_pickle=False)
                    require(hashlib.sha256(mem.getvalue()).hexdigest()==old['sha256']
                            and not bool(torch.count_nonzero(x[old['samples']:]).item()),
                            'Decoded audio differs from workstation cache: '+r.sample_id)
            require(torch.isfinite(x).all().item(),'Nonfinite decoded input')
            if expected is not None and self.modality=='video':
                require(hashlib.sha256(x.numpy().tobytes()).hexdigest()==expected['input_tensor_sha256'],
                        'Video preprocessing differs from workstation: '+r.sample_id)
            result.append((x,dict(media_sha256=media_hash,input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())))
        return result
    @torch.no_grad()
    def predict(self,decoded):
        from x3d_bn_audit.core import configure_condition,check_modes
        if self.modality=='video':
            require(len(decoded)==1,'X3D per-video BN requires batch size 1')
            x,meta=decoded[0];x=x.unsqueeze(0).to(self.device)
            value=dict(meta)
            for condition in self.conditions:
                configure_condition(self.active,condition);check_modes(self.active,condition)
                with torch.autocast('cuda',dtype=torch.float16,enabled=(condition=='A')):
                    logit,prob=self.active(x)
                value[condition+'_logit']=float(logit.item())
                value[condition+'_probability']=float(prob.item())
            return [value]
        require(not any(m.training for m in self.active.modules()),'AASIST must remain in eval mode')
        x=torch.stack([v[0] for v in decoded]).to(self.device)
        with torch.autocast('cuda',enabled=False):output=self.active(x)
        return [dict(meta,logits=z.detach().cpu().numpy(),probability=float(p))
                for (_,meta),z,p in zip(decoded,output.logits,output.fake_probability)]
