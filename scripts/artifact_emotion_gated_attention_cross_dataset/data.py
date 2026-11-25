"""Source/target separation and evidence-backed optional PGF label sidecar."""
import json
from dataclasses import replace
from pathlib import Path
from src.dataloader import DeepfakeDataset
from .common import ROOT, DATASETS, sha, digest


def read_overlay(path):
    if path is None:
        return {}
    result = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line); sid = r['sample_id']
            if not sid.startswith('PGF:') or sid in result:
                raise ValueError('Overlay IDs must be unique PGF sample IDs')
            if not all(r.get(k) for k in ('evidence_file', 'rule_version')) or 'evidence_record' not in r:
                raise ValueError('Overlay requires evidence_file, evidence_record, rule_version')
            evidence = Path(r['evidence_file'])
            if not evidence.is_absolute():
                evidence = ROOT/evidence
            if not evidence.is_file():
                raise FileNotFoundError(evidence)
            for key in ('video_label', 'audio_label'):
                value = r.get(key)
                if value is not None and (type(value) is not int or value not in (0, 1)):
                    raise ValueError('Overlay labels must be 0/1/null')
            result[sid] = r
    return result


def apply_overlay(records, overlay):
    output = []
    for r in records:
        o = overlay.get(r.sample_id)
        if o:
            v, a = o.get('video_label'), o.get('audio_label')
            for old, new in ((r.video_label, v), (r.audio_label, a)):
                if old is not None and new != old:
                    raise ValueError('Overlay conflicts with an existing label')
            if v is not None and a is not None and int(v or a) != r.clip_label:
                raise ValueError('Overlay modality OR conflicts with clip label')
            if r.clip_label == 0 and (v == 1 or a == 1):
                raise ValueError('Real clip cannot have a fake modality')
            r = replace(r, video_label=v, audio_label=a)
        output.append(r)
    return output


def make_dataset(name, split, modality, overlay=None, limit=None):
    if name not in DATASETS or modality not in ('video', 'audio'):
        raise ValueError('Invalid dataset/modality')
    # Select paired eligibility consistently, then decode only the requested modality.
    ds = DeepfakeDataset(name, split, task='paired')
    ds.records = apply_overlay(ds.records, overlay or {})
    if limit is not None:
        # Deterministic ID order across all branches, explicitly marked smoke-only.
        ordered = sorted(ds.records, key=lambda r: digest(r.sample_id))
        buckets = {}
        for r in ordered:
            buckets.setdefault((r.clip_label, r.video_label, r.audio_label), r)
        chosen = list(buckets.values())[:limit]
        ids = {r.sample_id for r in chosen}
        ds.records = chosen + [r for r in ordered if r.sample_id not in ids][:max(0, limit-len(chosen))]
    ds.task = modality
    if not ds.records:
        raise ValueError('Empty eligible cohort')
    return ds


def label_of(record, modality):
    return getattr(record, modality+'_label')


def require_train_labels(ds, modality):
    values = [label_of(r, modality) for r in ds.records]
    if set(v for v in values if v is not None) != {0, 1}:
        raise ValueError(f'{ds.records[0].dataset} {ds.split} {modality}: both known classes required; '
                         'PGF needs a verified --pgf-labels sidecar')
    return dict(total=len(values), known=sum(v is not None for v in values), unknown=values.count(None))


def check_disjoint(train, dev):
    for key in ('sample_id', 'group_id'):
        if {getattr(r, key) for r in train.records} & {getattr(r, key) for r in dev.records}:
            raise ValueError('Source train/dev overlap: '+key)


def cohort_id(ds):
    return digest([(r.sample_id, r.group_id, r.split, str(r.media_path),
                    r.clip_label, r.video_label, r.audio_label) for r in ds.records])
