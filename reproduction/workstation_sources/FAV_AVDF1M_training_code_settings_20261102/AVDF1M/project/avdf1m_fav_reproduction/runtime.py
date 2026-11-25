"""Preparation/provenance adapted from the recovered historical FAV trainer."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import default_collate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from .data_models import AASIST, X3D, PortableDataset
from .prepare import digest

PREFLIGHT = None
PRETRAINED = None
SMOKE = False


def binary_metrics(labels, probabilities):
    pred = np.asarray(probabilities) >= .5
    return dict(roc_auc=float(roc_auc_score(labels, probabilities)),
                accuracy=float(accuracy_score(labels, pred)),
                balanced_accuracy=float(balanced_accuracy_score(labels, pred)),
                f1=float(f1_score(labels, pred, zero_division=0)))


def validate_predictions(records, expected):
    ids = [r['sample_id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Dev coverage mismatch or duplicate predictions')
    if any(r['split'] != 'dev' for r in records):
        raise ValueError('Only dev predictions permitted')


def prepare(config_path, project_root, task, device):
    root = Path(project_root)
    config = json.loads(Path(config_path).read_text())
    preflight = json.loads(Path(PREFLIGHT).read_text())
    dataset_root = Path(preflight['dataset_root'])
    for split, sha in preflight['manifest_sha256'].items():
        if digest(dataset_root/f'metadata/portable/{split}.jsonl') != sha:
            raise ValueError('Frozen manifest changed')
    if digest(Path(preflight['cache_root'])/'audio_index.json') != preflight['audio_index_sha256']:
        raise ValueError('Audio index changed')
    config['audio']['target_num_samples'] = preflight['audio']['target_num_samples']
    config['smoke_only'] = SMOKE
    if SMOKE:
        config['training']['epochs'] = 1
        config['training']['sampler']['global_epoch_samples'] = 32
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(2)
    source_hashes = {str(p.relative_to(root)): digest(p) for p in sorted((root/'avdf1m_fav_reproduction').rglob('*.py'))}
    source_hashes['environment-workstation.yml'] = digest(root/'environment-workstation.yml')
    if task == 'video':
        import pytorchvideo
        if digest(PRETRAINED) != 'eb9583d5f5a988a5c5f06f75760b070a5c9715794bd3b0536b78bd30595dcf67':
            raise ValueError('Transferred Kinetics X3D-M checksum mismatch')
        base = Path(pytorchvideo.__file__).parent
        source_hashes.update({'pytorchvideo/'+str(p.relative_to(base)):digest(p) for p in sorted(base.rglob('*.py'))})
        factory = lambda: X3D(PRETRAINED)
        weight = dict(initialization='Kinetics-400', sha256=digest(PRETRAINED))
    else:
        vendor = root/'aasist'
        commit = subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
        if commit != 'a04c9863f63d44471dde8a6abcb3b082b07cd1d1' or subprocess.check_output(['git','-C',str(vendor),'status','--porcelain'],text=True).strip():
            raise ValueError('AASIST vendor commit/clean tree mismatch')
        source_hashes['aasist/models/AASIST.py'] = digest(vendor/'models/AASIST.py')
        factory = lambda: AASIST(config['model']['official_config'], config['cpu_saved_tensor_offload'])
        weight = dict(initialization='random', sha256=None)
    train, dev = [PortableDataset(preflight, s, task) for s in ('train','dev')]
    if SMOKE:
        # Fixed balanced small cohort, exercising all four joint label classes.
        def small(ds):
            groups = [[r for r in ds.rows if (r['video_label'],r['audio_label']) == pair][:4]
                      for pair in ((0,0),(0,1),(1,0),(1,1))]
            ds.rows = [r for group in groups for r in group]
        small(train); small(dev)
    provenance = dict(format='avdf1m_fav_recipe_v1', config=config, config_sha256=digest(config_path),
                      preflight_sha256=digest(PREFLIGHT), manifest_sha256=preflight['manifest_sha256'],
                      audio=preflight['audio'], model_source_sha256=source_hashes, weight=weight,
                      runtime=dict(torch=torch.__version__,cuda=torch.version.cuda),
                      unified_source_sha256=digest(root/'avdf1m_fav_reproduction/train.py'),
                      historical_trainer_sha256='78372929c8e34c335c7230efd1a779d39e452da97d4643aa9ccb6b67beb7cbf1')
    return config, train, dev, factory, default_collate, provenance
