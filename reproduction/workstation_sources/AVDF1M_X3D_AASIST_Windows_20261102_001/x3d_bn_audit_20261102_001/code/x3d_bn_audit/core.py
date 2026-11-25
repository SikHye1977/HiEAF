"""Model construction, isolated BN modes, integrity and binary diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)

CONDITIONS = {
    'A': {'name': 'running_bn_amp_fp16', 'amp': True, 'per_video_bn': False},
    'B': {'name': 'running_bn_fp32', 'amp': False, 'per_video_bn': False},
    'C': {'name': 'per_video_bn_fp32', 'amp': False, 'per_video_bn': True},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


class X3D(nn.Module):
    """Same backbone/head/forward as the training snapshot; load the FULL state."""
    def __init__(self):
        super().__init__()
        from pytorchvideo.models.hub import x3d_m
        self.backbone = x3d_m(pretrained=False)
        self.backbone.blocks[-1].proj = nn.Linear(self.backbone.blocks[-1].proj.in_features, 1)
        self.backbone.blocks[-1].activation = nn.Identity()

    def forward(self, video):
        logits = self.backbone(video).flatten(start_dim=1).squeeze(1)
        return logits, torch.sigmoid(logits)


def configure_condition(model, condition):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            # Keep original running buffers registered and unchanged. In C,
            # _BatchNorm.forward passes None as running_mean/var, so only this
            # single clip's N,T,H,W statistics enter each layer's normalization.
            module.training = CONDITIONS[condition]['per_video_bn']
            module.track_running_stats = not CONDITIONS[condition]['per_video_bn']
    check_modes(model, condition)


def check_modes(model, condition):
    assert not model.training
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert module.training == CONDITIONS[condition]['per_video_bn'], name
            assert module.track_running_stats != CONDITIONS[condition]['per_video_bn'], name
        else:
            assert not module.training, name
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def state_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(key.encode()+str(t.dtype).encode()+str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_state_preserved(model, reference):
    state = model.state_dict()
    if set(state) != set(reference):
        raise AssertionError('State keys changed')
    changed = [k for k in state if not torch.equal(state[k].detach().cpu(), reference[k])]
    if changed:
        raise AssertionError(f'Parameters or buffers changed: {changed}')
    return {'all_tensors_equal': True, 'tensor_count': len(state), 'state_sha256': state_digest(state),
            'parameter_tensor_count': len(list(model.named_parameters())),
            'buffer_tensor_count': len(list(model.named_buffers())),
            'bn_layers': sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())}


def distribution(values):
    return dict(zip(('min','p05','median','p95','max'),
                    [float(x) for x in np.quantile(values,[0,.05,.5,.95,1])]))


def metrics(rows):
    y = np.array([r['video_label'] for r in rows], dtype=int)
    z = np.array([r['logit'] for r in rows], dtype=np.float64)
    p = np.array([r['fake_probability'] for r in rows], dtype=np.float64)
    p64 = np.array([r['fake_probability_from_logit_float64'] for r in rows])
    pred = (p >= .5).astype(int)
    cm = confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    real_both = np.array([r['video_label']==0 and r['audio_label']==0 for r in rows])
    return dict(samples=len(rows),video_real=int(sum(y==0)),video_fake=int(sum(y==1)),
                roc_auc_logit=float(roc_auc_score(y,z)),roc_auc_probability=float(roc_auc_score(y,p)),
                roc_auc_probability_from_logit_float64=float(roc_auc_score(y,p64)),
                precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred)),
                f1=float(f1_score(y,pred)),accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),real_fpr=fp/(tn+fp),
                confusion_matrix=cm.tolist(),confusion_counts=dict(TN=tn,FP=fp,FN=fn,TP=tp),
                both_real=dict(count=int(sum(real_both)),FP=int(sum(pred[real_both])),
                               FPR=float(np.mean(pred[real_both]))),
                distribution={label:dict(logit=distribution(z[y==value]),fake_probability=distribution(p[y==value]))
                              for label,value in [('real',0),('fake',1)]},
                saturation={label:dict(probability_zero=int(sum(p[y==value]==0)),
                                      probability_one=int(sum(p[y==value]==1)),
                                      unique_probabilities=len(np.unique(p[y==value])),
                                      unique_logits=len(np.unique(z[y==value])))
                            for label,value in [('real',0),('fake',1)]},
                prediction_disagreements_probability_vs_logit=int(sum(pred!=(z>=0))),threshold=.5)


def baseline_comparison(rows, stored):
    old = {r['sample_id']:r for r in stored}
    assert set(old)=={r['sample_id'] for r in rows}
    probs=[];logits=[];pred_mismatches=[]
    for r in rows:
        o=old[r['sample_id']]
        assert o['label']==r['video_label'] and o['split']=='dev'
        probs.append(abs(r['fake_probability']-o['fake_probability']))
        logits.append(abs(r['logit']-o['logit']))
        if r['predicted_label']!=int(o['fake_probability']>=.5):pred_mismatches.append(r['sample_id'])
    auc_old=float(roc_auc_score([o['label'] for o in stored],[o['fake_probability'] for o in stored]))
    stats=metrics(rows)
    passed=(stats['confusion_counts']['FP']==1184 and stats['confusion_counts']['TP']==1568
            and not pred_mismatches and abs(stats['roc_auc_probability']-auc_old)<1e-6)
    return dict(passed=passed,rule='same sample predictions, FP=1184, TP=1568, probability AUC absolute difference <1e-6',
                stored_probability_auc=auc_old,new_probability_auc=stats['roc_auc_probability'],
                exact_logit_matches=sum(x==0 for x in logits),exact_probability_matches=sum(x==0 for x in probs),
                max_abs_logit_difference=max(logits),max_abs_probability_difference=max(probs),
                prediction_mismatches=pred_mismatches)
