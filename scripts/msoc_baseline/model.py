"""Pure PyTorch adapter of official MSOC; upstream snapshot remains unchanged."""
import importlib.util
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import HERE, read_json, require, sha


def verify_upstream():
    lock = read_json(HERE / 'upstream.lock.json')
    for name, expected in lock['files'].items():
        require(sha(HERE / 'upstream' / name) == expected, f'Changed upstream source: {name}')
    return lock['commit']


def upstream(name):
    path = HERE / 'upstream/src' / name
    lock = read_json(HERE/'upstream.lock.json')
    require(sha(path) == lock['files']['src/'+name], f'Upstream import integrity error: {name}')
    spec = importlib.util.spec_from_file_location('_msoc_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResEncoder(upstream('models/talkNet/resnet.py').ResEncoder):
    def forward(self, x):
        # Official no_grad around max-pool accidentally detaches the 3D frontend.
        # Preserve all original layers/parameters, restore normal gradient flow.
        b = x.shape[0]
        x = self.frontend3D(x)
        t = x.shape[2]
        x = self.trunk(self.threeD_to_2D_tensor(x))
        return x.reshape(b, t, -1)


class OCSoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.center = nn.Parameter(torch.randn(1, dim))
        nn.init.kaiming_uniform_(self.center, 0.25)

    def score(self, x):
        return (F.normalize(x.float(), dim=1) @ F.normalize(self.center.float(), dim=1).T).squeeze(1)

    def forward(self, x, real_label):
        require(bool(torch.all((real_label == 0) | (real_label == 1))), 'OC loss needs known modality labels')
        score = self.score(x)
        distance = torch.where(real_label == 1, .9 - score, score - .2)
        return F.softplus(20 * distance).mean()


class MSOC(nn.Module):
    def __init__(self, backbone='resnet'):
        super().__init__()
        self.backbone = backbone
        ae = upstream('models/talkNet/audioEncoder.py').audioEncoder
        self.a_audioEncoder = ae(layers=[2, 3, 4, 2], num_filters=[16, 32, 64, 128])
        self.av_audioEncoder = ae(layers=[2, 3, 4, 2], num_filters=[16, 32, 64, 128])
        if backbone == 'resnet':
            vc = upstream('models/talkNet/visualEncoder.py').visualConv1D
            self.v_visualFrontend = ResEncoder(relu_type='relu', light=True)
            self.av_visualFrontend = ResEncoder(relu_type='relu', light=True)
            self.v_visualConv1D, self.av_visualConv1D = vc(), vc()
            vdim = 128
        elif backbone == 'scnet':
            sc = upstream('models/SCNet.py').scnet50_v1d
            self.v_visualFrontend, self.av_visualFrontend = sc(30), sc(30)
            vdim = 512
        else:
            raise ValueError(backbone)
        self.loss_audio, self.av_loss_audio = OCSoftmax(128), OCSoftmax(128)
        self.loss_video, self.av_loss_video = OCSoftmax(vdim), OCSoftmax(vdim)
        self.av_classifier = nn.Sequential(nn.Linear(vdim + 128, vdim + 128),
                                          nn.ReLU(inplace=True), nn.Linear(vdim + 128, 2))

    def visual(self, x, av=False):
        front = self.av_visualFrontend if av else self.v_visualFrontend
        if self.backbone == 'scnet':
            b, c, t, h, w = x.shape
            # Correct [B,C,T,H,W] -> [B*T,C,H,W]; never reinterpret RGB/time axes.
            return front(x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)).reshape(b, t, -1)
        x = front(x).transpose(1, 2)
        return (self.av_visualConv1D if av else self.v_visualConv1D)(x).transpose(1, 2)

    def forward(self, video, audio):
        x = audio.unsqueeze(1).transpose(2, 3)
        a, av_a = self.a_audioEncoder(x), self.av_audioEncoder(x)
        v, av_v = self.visual(video), self.visual(video, av=True)
        logits = self.av_classifier(torch.cat((av_a, av_v), dim=2).mean(dim=1))
        return logits, av_v.mean(1), av_a.mean(1), v.mean(1), a.mean(1)

    def objective(self, outputs, labels):
        # Current labels: fake=1. Official MSOC labels: real=1.
        clip, video, audio = (1-labels[:, i] for i in range(3))
        logit, av_v, av_a, v, a = outputs
        with torch.autocast(device_type=logit.device.type, enabled=False):
            return (F.cross_entropy(logit.float(), clip) + self.loss_video(v, video)
                    + self.loss_audio(a, audio) + self.av_loss_video(av_v, video)
                    + self.av_loss_audio(av_a, audio))

    def scores(self, outputs):
        logit, _, _, v, a = outputs
        with torch.autocast(device_type=logit.device.type, enabled=False):
            vs, aus = self.loss_video.score(v), self.loss_audio.score(a)
            av = torch.softmax(logit.float(), dim=1)[:, 1]
            real = ((vs > .5).float() + (aus > .5).float() + av) / 3
            return torch.stack((1-real, vs, aus, av), dim=1)
