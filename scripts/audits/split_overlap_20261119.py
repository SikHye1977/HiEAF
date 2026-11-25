"""Read-only dataset audit. Writes new evidence only; never edits experiment inputs.

Run with the workspace venv. Metadata associations are not biometric matches.
URL equality establishes a shared source recording, not overlapping time spans.
"""
import csv
import hashlib
import itertools
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/diagnostics/split_overlap_audit_20261119'
SPLITS = ('train', 'dev', 'test')
INPUTS = {}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def track(p):
    p = Path(p)
    INPUTS[str(p)] = {'sha256': sha(p), 'size': p.stat().st_size}
    return p


def read(p):
    return json.loads(track(p).read_text(encoding='utf-8-sig'))


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def csvout(name, rows):
    if not rows:
        (OUT / name).write_text('NO_MATCHES\n', encoding='utf-8')
        return
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def urls(text):
    # Only provider-specific video IDs. Generic channel/news/film labels excluded.
    yt = set(re.findall(r'(?:youtube\.com/(?:watch\?[^\s]*?v=|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])', text))
    bv = set(re.findall(r'bilibili\.com/video/(BV[A-Za-z0-9]+)', text))
    av = set(re.findall(r'bilibili\.com/video/(av\d+)', text))
    return {f'youtube:{x}' for x in yt} | {f'bilibili:{x}' for x in bv | av}


def overlaps(rows, key):
    sets = {s: set().union(*(r[key] if isinstance(r[key], set) else {r[key]} for r in rows if r['split'] == s)) for s in SPLITS}
    return {f'{a}__{b}': len(sets[a] & sets[b]) for a, b in itertools.combinations(SPLITS, 2)}


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=False)
    registry = read(ROOT / 'configs/fixed_manifests.json')
    subset = ROOT / 'scripts/trainable_or_3way/subsets/AVDF1M_fav_size_seed42'
    selection = read(subset / 'selection.json')
    chosen = {}
    for s in ('train', 'dev'):
        p = track(subset / f'{s}.jsonl')
        assert INPUTS[str(p)]['sha256'] == selection['files'][p.name]
        chosen[s] = {json.loads(line)['sample_id'] for line in p.open(encoding='utf8')}
    eligible = set()
    ep = track(ROOT / 'reports/phase0_v1/eligibility_v4.jsonl')
    for line in ep.open(encoding='utf8'):
        r = json.loads(line)
        if r['eligible']['paired']:
            eligible.add(r['sample_id'])
    print(f'[AUDIT] paired eligible IDs {len(eligible):,}', flush=True)

    pgf_root = Path('D:/datasets/PolyGlotFake/json_file')
    pgf = read(pgf_root / 'real_json_file/all_real_video.json')['videos']
    read(pgf_root / 'fake_Json_file/all_fake_video.json')
    # Cross-check aggregates against all language-specific original source records.
    pgf_sources = defaultdict(set)
    pgf_source_text = defaultdict(set)
    pgf_meta_records = defaultdict(list)
    for p in sorted((pgf_root / 'real_json_file').glob('*.json')):
        data = read(p)
        for r in data['videos']:
            pgf_sources[r['filename']] |= urls(r.get('source', ''))
            pgf_source_text[r['filename']].add(r.get('source', ''))
            pgf_meta_records[r['filename']].append({'file': str(p), 'record': r})

    datasets, summaries = {}, {}
    av_official = defaultdict(set)
    av_official_source = defaultdict(set)
    av_official_counts = Counter()
    av_matrix_all = Counter()
    av_matrix_used = Counter()
    fav_real_paths = []
    for ds in ('FAV', 'AVDF1M', 'PGF'):
        entry = registry['datasets'][ds]
        p = track(ROOT / entry['manifest'])
        assert INPUTS[str(p)]['sha256'] == entry['sha256']
        historical_summary = read(ROOT / entry['summary'])
        rows, raw_counts, pair_counts, all_groups = [], Counter(), Counter(), defaultdict(set)
        seen = set()
        for i, line in enumerate(p.open(encoding='utf8'), 1):
            r = json.loads(line)
            sid, split = r['sample_id'], r['split']
            assert sid not in seen
            seen.add(sid)
            raw_counts[split] += 1
            all_groups[r['group_id']].add(split)
            if ds == 'AVDF1M':
                official = r['official_split']
                assert r['metadata']['split'] == official
                av_official_counts[official] += 1
                av_matrix_all[(split, official)] += 1
                av_official[official].add(r['identity_id'])
                av_official_source[official].add(r['source_id'])
            if sid not in eligible:
                continue
            pair_counts[split] += 1
            if ds == 'AVDF1M' and split in chosen and sid not in chosen[split]:
                continue
            c = dict(dataset=ds, sample_id=sid, split=split, group_id=r['group_id'],
                     original=r['original_group_id'], label=r['clip_label'],
                     persons=set(), candidates=set(), origins=set(), path=r['video_path'])
            if ds == 'FAV':
                c['persons'] = {r['source_id']}
                c['source_person'] = r['source_id']
                c['candidates'] = set(r.get('filename_identity_candidates', []))
                for m in r['metadata_records']:
                    c['persons'] |= {m[k] for k in ('source', 'target1', 'target2') if re.fullmatch(r'id\d{5}', m.get(k, ''))}
                c['explicit_targets'] = c['persons'] - {r['source_id']}
                c['persons'] |= set(r.get('related_identity_ids', []))
                if r['clip_label'] == 0:
                    fav_real_paths.append(c)
            elif ds == 'AVDF1M':
                c['source_person'] = r['identity_id']
                c['persons'] = {r['identity_id']}
                parts = r['source_id'].split('/')
                assert len(parts) == 3 and re.fullmatch(r'[A-Za-z0-9_-]{11}', parts[1])
                c['origins'] = {'youtube:' + parts[1]}
                c['official_split'] = r['official_split']
                c['raw_original'] = r['metadata'].get('original')
                av_matrix_used[(split, r['official_split'])] += 1
            else:
                c['raw_clip'] = r['source_id']
                c['origins'] = pgf_sources[r['source_id']].copy()
            rows.append(c)
            if i % 200000 == 0:
                print(f'[AUDIT] {ds} scanned {i:,}', flush=True)
        assert all(len(s) == 1 for s in all_groups.values())
        datasets[ds] = rows
        summaries[ds] = dict(frozen_counts=dict(raw_counts), paired_counts=dict(pair_counts),
            actual_counts=dict(Counter(r['split'] for r in rows)),
            actual_groups={s: len({r['group_id'] for r in rows if r['split'] == s}) for s in SPLITS},
            actual_classes={s: dict(Counter(r['label'] for r in rows if r['split'] == s)) for s in SPLITS},
            group_overlap=overlaps(rows, 'group_id'), original_overlap=overlaps(rows, 'original'),
            sample_overlap=overlaps(rows, 'sample_id'), person_metadata_overlap=overlaps(rows, 'persons'),
            source_video_url_overlap=overlaps(rows, 'origins'),
            policy=historical_summary['policy']['group_definitions'][ds])
        print('[AUDIT]', ds, summaries[ds]['actual_counts'], flush=True)
    csvout('avdf_official_to_study.csv', [dict(scope=scope, study_split=s, official_split=o, n=n) for scope, mat in [('frozen_all', av_matrix_all), ('actual_used', av_matrix_used)] for (s, o), n in sorted(mat.items())])
    write('official_split_relationship.json', dict(counts=dict(av_official_counts),
        official_train_val_identity_overlap=len(av_official['train'] & av_official['val']),
        official_train_val_source_clip_overlap=len(av_official_source['train'] & av_official_source['val']),
        note='Computed from frozen records preserving official metadata; official test excluded.'))

    # FAV: confirmed metadata ID associations vs filename-only hints are separate.
    fr = datasets['FAV']
    fav_details = []
    for s, t in itertools.permutations(SPLITS, 2):
        target_sources = {r['source_person'] for r in fr if r['split'] == t}
        for r in fr:
            if r['split'] != s:
                continue
            for kind, ids in [('explicit_target_id', r['explicit_targets']), ('filename_candidate_only', r['candidates'] - r['persons'])]:
                for person in sorted(ids & target_sources):
                    fav_details.append(dict(sample_split=s, other_source_split=t, sample_id=r['sample_id'],
                                            source_person=r['source_person'], related_person=person, evidence=kind))
    csvout('fav_related_identity_cross_split.csv', fav_details)
    summaries['FAV']['related_identity_cross_split_counts'] = dict(Counter(f"{r['evidence']}:{r['sample_split']}->{r['other_source_split']}" for r in fav_details))

    # Cross-dataset FAV/AVDF person match is valid only within the documented shared VoxCeleb2 namespace.
    cross_people, cross_summary = [], []
    for a, b in [('FAV', 'AVDF1M'), ('AVDF1M', 'FAV')]:
        for mode, source_splits in [('train', {'train'}), ('train_and_dev', {'train', 'dev'})]:
            source = [r for r in datasets[a] if r['split'] in source_splits]
            target = [r for r in datasets[b] if r['split'] == 'test']
            for view, key in [('source_identity', 'source_person'), ('all_explicit_metadata_identities', 'persons')]:
                get = lambda r: r[key] if isinstance(r[key], set) else {r[key]}
                sp = set().union(*(get(r) for r in source))
                tp = set().union(*(get(r) for r in target))
                inter = sp & tp
                hits = [r for r in target if get(r) & inter]
                cross_summary.append(dict(source=a, target=b, source_scope=mode, identity_scope=view,
                    shared_identities=len(inter), source_identities=len(sp), target_identities=len(tp),
                    target_samples=len(target), affected_target_samples=len(hits),
                    affected_real=sum(r['label'] == 0 for r in hits), affected_fake=sum(r['label'] == 1 for r in hits)))
                for person in sorted(inter):
                    cross_people.append(dict(source=a,target=b,source_scope=mode,identity_scope=view,identity=person,
                        source_samples=sum(person in get(r) for r in source),target_samples=sum(person in get(r) for r in target)))
    csvout('fav_avdf_shared_identities.csv', cross_people)
    csvout('cross_dataset_identity_summary.csv', cross_summary)

    # URL-level origin families, including PGF shorter clips of a common online recording.
    origins = defaultdict(list)
    for ds, rows in datasets.items():
        for r in rows:
            for key in r['origins']:
                origins[key].append(r)
    collisions, pgf_origin_details = [], []
    for key, rr in origins.items():
        ds = {r['dataset'] for r in rr}
        if len(ds) > 1:
            for r in rr:
                collisions.append(dict(origin=key, dataset=r['dataset'], split=r['split'], sample_id=r['sample_id']))
        pp = [r for r in rr if r['dataset'] == 'PGF']
        if len({r['split'] for r in pp}) > 1:
            for r in pp:
                pgf_origin_details.append(dict(origin=key, split=r['split'], raw_clip=r['raw_clip'], sample_id=r['sample_id'], label=r['label']))
    csvout('cross_dataset_shared_video_urls.csv', collisions)
    csvout('pgf_shared_recording_cross_split.csv', pgf_origin_details)
    pgf_summary = dict(metadata_real_clips=len(pgf),
        clips_with_canonical_url=sum(bool(pgf_sources[r['filename']]) for r in pgf),
        distinct_canonical_urls=len(set().union(*pgf_sources.values())),
        actual_clips={s: len({r['raw_clip'] for r in datasets['PGF'] if r['split'] == s}) for s in SPLITS},
        actual_samples_with_url=sum(bool(r['origins']) for r in datasets['PGF']),
        within_split_url_overlap=summaries['PGF']['source_video_url_overlap'],
        shared_urls_across_splits=len({r['origin'] for r in pgf_origin_details}),
        affected_samples=len({r['sample_id'] for r in pgf_origin_details}),
        origin_details=[dict(origin=k, clips=sorted({r['raw_clip'] for r in rr if r['dataset']=='PGF'}),
                              splits=sorted({r['split'] for r in rr if r['dataset']=='PGF'}))
                        for k, rr in origins.items() if len({r['split'] for r in rr if r['dataset']=='PGF'})>1],
        identity_status='UNRESOLVED: source strings and age/sex are not unique person identifiers',
        interpretation='Same URL means same online recording; timestamp overlap and person identity not determined.')
    write('pgf_recording_origins.json', pgf_summary)
    write('summary.json', dict(datasets=summaries, cross_dataset_identity=cross_summary,
        cross_dataset_url_shared=len({r['origin'] for r in collisions}),
        cross_dataset_url_rows=len(collisions), pgf=pgf_summary,
        limitations=['No biometric identification', 'No exhaustive byte or audiovisual content fingerprint scan',
                     'FAV has no original YouTube mapping in the local release metadata',
                     'PGF age/sex/source strings do not establish person identities',
                     'Shared VoxCeleb2 ID is person overlap, not necessarily the same recording'],
        elapsed_seconds=time.time()-start))
    # Full compact per-sample provenance for independent reproduction of counts.
    with (OUT/'used_sample_provenance.jsonl').open('w',encoding='utf8') as f:
        for rows in datasets.values():
            for r in rows:
                f.write(json.dumps({k:sorted(v) if isinstance(v,set) else v for k,v in r.items()},ensure_ascii=False)+'\n')
    changed = [p for p, info in INPUTS.items() if sha(Path(p)) != info['sha256']]
    write('PRESERVED_INPUTS.json', dict(files=INPUTS, changed=changed, status='PASS' if not changed else 'FAIL'))
    assert not changed
    print('[AUDIT COMPLETE]', str(OUT), 'elapsed', round(time.time()-start,1), flush=True)
    print(json.dumps({'cross_dataset_identity':cross_summary,'pgf_url_cross_split':pgf_summary['within_split_url_overlap'],'cross_dataset_url_shared':len({r['origin'] for r in collisions})},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
