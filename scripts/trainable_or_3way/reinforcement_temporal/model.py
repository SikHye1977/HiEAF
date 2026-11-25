"""Controlled T0/T1/T2 heads. T2 exactly matches the historical architecture."""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.model import EmotionFlowHead
from .common import DIMS, CLASSES, require, digest

class TemporalHead(EmotionFlowHead):
    def __init__(self, modality, variant):
        require(variant in ("T0","T1","T2"), "Unknown temporal variant")
        super().__init__(modality)
        self.variant=variant
        hidden=64 if modality=="video" else 128
        if variant!="T2": del self.attention
        if variant=="T0":
            del self.gru
            self.token_mlp=nn.Sequential(nn.Linear(128+len(CLASSES[modality]),hidden),
                                         nn.ReLU(),nn.Dropout(.2))

    def forward(self, embedding, probabilities, mask):
        if self.variant=="T2": return super().forward(embedding,probabilities,mask)
        require(embedding.ndim==3 and mask.dtype==torch.bool and mask.shape==embedding.shape[:2],
                "Expected embeddings and prefix boolean mask")
        lengths=mask.sum(1)
        expected=torch.arange(mask.shape[1],device=mask.device)[None]<lengths[:,None]
        require(bool((lengths>0).all()) and torch.equal(mask,expected),"Compact valid tokens in time order")
        require(probabilities.shape==(*mask.shape,len(CLASSES[self.modality])),"Invalid native classes")
        e,p=embedding[mask],probabilities[mask]
        require(bool(torch.isfinite(e).all()) and bool(torch.isfinite(p).all())
                and bool(((p>=0)&(p<=1)).all())
                and torch.allclose(p.sum(-1),torch.ones_like(p[:,0]),atol=1e-5),"Invalid tokens")
        joined=torch.cat([self.projection(e),p],-1)
        x=joined.new_zeros((*mask.shape,joined.shape[-1])); x[mask]=joined
        if self.variant=="T0":
            h=self.token_mlp(x)
        else:
            packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
            packed_h,_=self.gru(packed)
            h,_=pad_packed_sequence(packed_h,batch_first=True,total_length=mask.shape[1])
        alpha=mask.to(h.dtype)/lengths[:,None]
        context=(h*alpha.unsqueeze(-1)).sum(1)
        return self.classifier(context).squeeze(-1),alpha

def permutation(sid, mod, transform, n):
    order=np.arange(n,dtype=np.int64)
    if transform=="reversed": return order[::-1].copy()
    if transform.startswith("shuffle"):
        seed=int(transform.removeprefix("shuffle"))
        rng=np.random.default_rng(int(digest([seed,str(sid),mod])[:16],16))
        return rng.permutation(order)
    require(transform=="ordered","Unknown sequence transform")
    return order

def compact(tokens, sid, mod, transform="ordered"):
    positions=np.flatnonzero(tokens["valid"])
    order=permutation(sid,mod,transform,len(positions))
    selected=positions[order]
    # Each native probability vector travels with its embedding. Cache remains read-only.
    return dict(embedding=np.array(tokens["embedding"][selected],copy=True),
                probabilities=np.array(tokens["probabilities"][selected],copy=True),
                original_positions=selected)

def batch(items, device):
    require(items and all(len(r["embedding"]) for r in items),"Empty sequence in batch")
    n=max(len(r["embedding"]) for r in items)
    e=torch.zeros(len(items),n,items[0]["embedding"].shape[1])
    p=torch.zeros(len(items),n,items[0]["probabilities"].shape[1])
    m=torch.zeros(len(items),n,dtype=torch.bool)
    for i,r in enumerate(items):
        t=len(r["embedding"])
        e[i,:t]=torch.from_numpy(r["embedding"]).float()
        p[i,:t]=torch.from_numpy(r["probabilities"]).float()
        m[i,:t]=True
    return e.to(device),p.to(device),m.to(device)
