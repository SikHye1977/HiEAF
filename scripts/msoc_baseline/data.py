"""Read-only frozen cohorts; current Windows media validation for a NEW local comparison."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .common import ROOT, HERE, DATASETS, atomic_json, digest, read_json, require, sha, Progress
from src.utils.local_paths import load_local_paths, resolve_media_path
from .paths import configuration, pgf_metadata_path

SIZES = {'FAV': (14724, 3147, 3126), 'AVDF1M': (14724, 3147, 120327), 'PGF': (9948, 2092, 2082)}
REGISTRY = ROOT / 'configs/fixed_manifests.json'
ELIGIBILITY = ROOT / 'reports/phase0_v1/eligibility_v4.jsonl'
SUBSETS = ROOT / 'scripts/trainable_or_3way/subsets/AVDF1M_fav_size_seed42'


def stamp(path):
    s = Path(path).stat()
    require(s.st_size > 0, f'Empty media: {path}')
    return [s.st_size, s.st_mtime_ns]


def load_cohorts(output, data_config=None):
    """One pass over each manifest and eligibility, not nine full reads."""
    pins = read_json(HERE / 'data_pins.json')
    machine = configuration(data_config)
    metadata = machine['metadata_root']
    registry_path = metadata / REGISTRY.relative_to(ROOT)
    eligibility_path = metadata / ELIGIBILITY.relative_to(ROOT)
    subsets_path = metadata / SUBSETS.relative_to(ROOT)
    print('[DATA] Verifying fixed manifests, eligibility, selected AVDF1M IDs and PGF labels...', flush=True)
    require(sha(registry_path) == pins['registry'], 'Registry changed')
    require(sha(eligibility_path) == pins['eligibility'], 'Eligibility changed')
    registry = read_json(registry_path)['datasets']
    selected = {}
    for split in ('train', 'dev'):
        p = subsets_path / f'{split}.jsonl'
        require(sha(p) == pins['subsets'][split], f'Subset changed: {split}')
        selected[split] = {r['sample_id']: r for r in map(json.loads, p.read_text(encoding='utf-8').splitlines())}
    rows = {}
    for ds in DATASETS:
        entry = registry[ds]
        require(entry['sha256'] == pins['manifests'][ds], f'Manifest pin changed: {ds}')
        seen, groups = set(), {}
        h = hashlib.sha256()
        with (metadata / entry['manifest']).open('rb') as f:
            for line in f:
                h.update(line)
                r = json.loads(line)
                sid, group, part = r['sample_id'], r['group_id'], r['split']
                require(r['dataset'] == ds and part in ('train', 'dev', 'test'), 'Invalid dataset/split')
                require(sid not in seen and sid and group, 'Duplicate/missing ID/group')
                seen.add(sid)
                require(group not in groups or groups[group] == part, 'Group crosses splits')
                groups[group] = part
                for k in ('clip_label', 'video_label', 'audio_label'):
                    require(r[k] is None or (type(r[k]) is int and r[k] in (0, 1)), 'Invalid label')
                if ds == 'AVDF1M' and part in selected and sid not in selected[part]:
                    continue
                rows[sid] = r
        require(h.hexdigest() == entry['sha256'], f'Manifest hash mismatch: {ds}')
    eligible, found = set(), set()
    with eligibility_path.open(encoding='utf-8') as f:
        for line in f:
            e = json.loads(line)
            sid = e['sample_id']
            if sid not in rows:
                continue
            r = rows[sid]
            require(sid not in found and e['dataset'] == r['dataset'] and e['split'] == r['split']
                    and e['manifest_sha256'] == pins['manifests'][r['dataset']], 'Eligibility mismatch')
            found.add(sid)
            if e['eligible']['paired']:
                eligible.add(sid)
    require(found == set(rows), 'Missing eligibility')
    for s in selected.values():
        require(set(s) <= eligible, 'Selected IDs not eligible')
    pgf = {}
    for kind, entry in pins['pgf'].items():
        p = pgf_metadata_path(entry, machine)
        require(sha(p) == entry['sha256'], 'PGF metadata changed')
        ix = defaultdict(list)
        for r in read_json(p)[entry['key']]:
            ix[r['filename']].append(r)
        pgf[kind] = ix
    config = machine['local_paths']
    cohorts = {ds: {s: [] for s in ('train', 'dev', 'test')} for ds in DATASETS}
    progress = Progress('DATA current Windows file checks', len(eligible))
    for i, sid in enumerate(sorted(eligible)):
        r = rows[sid]
        path = resolve_media_path(r['video_path'], config)
        v, a, c = r['video_label'], r['audio_label'], r['clip_label']
        if r['dataset'] == 'PGF':
            kind = 'real' if sid.startswith('PGF:real/') else 'fake'
            evidence = pgf[kind][path.name]
            require(bool(evidence), 'Missing PGF modality evidence')
            if kind == 'real':
                v, a = 0, 0
            else:
                signatures = {(e.get('raw_video_name'), e.get('target_lang'), e.get('tts_technique'), e.get('sync_tech')) for e in evidence}
                require(len(signatures) == 1, 'Ambiguous PGF label evidence')
                e = evidence[0]
                require(r['group_id'] == 'PGF:source-video:' + e['raw_video_name'], 'PGF group mismatch')
                a = 1 if e.get('tts_technique') in {'MicroTts', 'Xtts', 'Bark', 'Tacotron', 'Vall'} else None
                v = 1 if e.get('sync_tech') in {'video_retalking', 'Wav2Lip'} else None
        require(c in (0, 1), 'Missing clip label')
        if v is not None and a is not None:
            require(c == max(v, a), 'Clip/modality OR mismatch')
        if r['split'] != 'test':
            require(v is not None and a is not None, 'Training/dev modality labels required')
        if r['dataset'] == 'AVDF1M' and r['split'] in selected:
            old = selected[r['split']][sid]
            require([v, a, c] == [old['video_label'], old['audio_label'], old['clip_label']], 'Subset label mismatch')
        cohorts[r['dataset']][r['split']].append(dict(sample_id=sid, path=str(path),
            stamp=stamp(path), video_label=v, audio_label=a, clip_label=c, group_id=r['group_id']))
        progress.update(i + 1)
    summary = {}
    for ds, parts in cohorts.items():
        for j, (split, records) in enumerate(parts.items()):
            require(len(records) == SIZES[ds][j], f'Unexpected cohort size: {ds}/{split}')
            summary[f'{ds}/{split}'] = dict(n=len(records), identity=digest(records),
                clip_counts=dict(Counter(str(r['clip_label']) for r in records)),
                joint_counts=dict(Counter(f"{r['video_label']},{r['audio_label']}" for r in records)))
    # Do not replace a prior run's media stamps or cohort identity.
    from .common import bind
    bind(Path(output) / 'COHORTS.json', summary)
    return cohorts, summary


def epoch_order(rows, seed, epoch):
    """Joint-class balanced replacement; fixed eligible IDs, deterministic draw positions."""
    import numpy as np
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r['video_label'], r['audio_label'])].append(i)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = []
    for j, key in enumerate(sorted(groups)):
        n = len(rows) // len(groups) + (j < len(rows) % len(groups))
        order.extend(rng.choice(groups[key], n, replace=True).tolist())
    rng.shuffle(order)
    return order
