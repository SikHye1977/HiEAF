"""Fixed, label/identity-stratified source subsets; never change frozen splits."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT/'configs/fixed_manifests.json'
ELIGIBILITY = ROOT/'reports/phase0_v1/eligibility_v4.jsonl'
DEFAULT_SUBSET = Path(__file__).parent/'subsets/AVDF1M_fav_size_seed42/selection.json'
FIELDS = ('sample_id','media_path','dataset','split','group_id',
          'clip_label','video_label','audio_label')
FORMAT = 'avdf_fav_size_subset_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rank(seed, *parts):
    return hashlib.sha256(json.dumps([seed,*parts],separators=(',',':')).encode()).hexdigest()


def apportion(counts, n, tie=None):
    """Exact integer largest-remainder allocation; no replacement capacity overflow."""
    total = sum(counts.values())
    require(type(n) is int and 0 <= n <= total and total > 0, 'Insufficient candidates or invalid count')
    require(all(type(v) is int and v > 0 for v in counts.values()), 'Invalid capacities')
    qr = {k: divmod(n*v,total) for k,v in counts.items()}
    out = {k:q for k,(q,_) in qr.items()}
    order = sorted(counts,key=lambda k:(-qr[k][1],tie(k) if tie else k))
    for k in order[:n-sum(out.values())]:
        out[k] += 1
    return out


def joint(record):
    v,a = record.video_label,record.audio_label
    require(all(type(x) is int and x in (0,1) for x in (v,a)), 'Unknown modality label')
    require(record.clip_label == int(bool(v or a)), 'Clip/modality label conflict: '+record.sample_id)
    return str(v)+str(a)


def select(records, n, seed, split):
    require(type(n) is int and n > 0, 'Positive subset size required')
    cells = defaultdict(lambda:defaultdict(list))
    seen = set()
    for r in records:
        require(r.dataset=='AVDF1M' and r.split==split and r.group_id,
                'Wrong source/split/group for sampling')
        require(r.sample_id not in seen, 'Duplicate sampling candidate')
        seen.add(r.sample_id)
        if r.video_label is None or r.audio_label is None:
            continue
        cells[joint(r)][r.group_id].append(r)
    counts = {k:sum(map(len,groups.values())) for k,groups in cells.items()}
    require(set(counts)=={'00','01','10','11'}, 'All four joint classes required')
    quotas = apportion(counts,n)
    chosen = []
    for k in sorted(cells):
        groups = cells[k]
        allocation = apportion({g:len(v) for g,v in groups.items()},quotas[k],
                               tie=lambda g:rank(seed,split,'group-quota',k,g))
        for g in sorted(groups):
            ordered = sorted(groups[g],key=lambda r:(rank(seed,split,'clip',r.sample_id),r.sample_id))
            chosen.extend(ordered[:allocation[g]])
    chosen.sort(key=lambda r:r.sample_id)
    require(len(chosen)==n and len({r.sample_id for r in chosen})==n, 'Sampling count mismatch')
    return chosen, dict(paired_candidates=len(records),unknown_excluded=len(records)-sum(counts.values()),
        known_candidates=sum(counts.values()),candidate_joint_counts=dict(sorted(counts.items())),
        selected_joint_counts=dict(sorted(quotas.items())),
        candidate_identities=len({r.group_id for r in records if r.video_label is not None and r.audio_label is not None}),
        selected_identities=len({r.group_id for r in chosen}))


def as_row(r):
    return dict(sample_id=r.sample_id,media_path=r.media_path.resolve().as_posix(),
                dataset=r.dataset,split=r.split,group_id=r.group_id,
                clip_label=r.clip_label,video_label=r.video_label,audio_label=r.audio_label)


def encode_rows(records):
    rows = [as_row(r) for r in records]
    jsonl = ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows).encode('utf-8')
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text,fieldnames=FIELDS,lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return jsonl, text.getvalue().encode('utf-8-sig')


def write_once(path, content):
    path = Path(path)
    if path.exists():
        require(path.read_bytes()==content, 'Subset output changed; use a new folder: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_bytes(content)
    temporary.replace(path)


def subset_path(args):
    custom = getattr(args,'source_subset',None)
    full = getattr(args,'full_source_data',False)
    require(not (custom and full), 'Use --source-subset or --full-source-data, not both')
    if args.source != 'AVDF1M':
        require(custom is None, '--source-subset currently supports AVDF1M only')
        return None
    return None if full else Path(custom or DEFAULT_SUBSET).resolve()


def read_bundle(path, registry=REGISTRY, eligibility=ELIGIBILITY):
    path = Path(path).resolve()
    require(path.is_file(), 'Fixed subset missing; run prepare_subset.py first: '+str(path))
    d = json.loads(path.read_text(encoding='utf-8'))
    require(d.get('format')==FORMAT and d.get('dataset')=='AVDF1M', 'Invalid subset descriptor')
    require(set(d['sizes'])=={'train','dev'} and all(type(n) is int and n>0 for n in d['sizes'].values()),
            'Subset must contain only train/dev with positive counts')
    require(d['registry_sha256']==sha(registry), 'Subset registry changed')
    require(d['eligibility_sha256']==sha(eligibility), 'Subset eligibility changed')
    required = {'train.jsonl','train.csv','dev.jsonl','dev.csv','summary.json'}
    require(set(d['files'])==required, 'Unexpected subset files')
    for name,checksum in d['files'].items():
        require(sha(path.parent/name)==checksum, 'Subset file changed: '+name)
    rows = {}
    for split in ('train','dev'):
        rows[split] = [json.loads(line) for line in (path.parent/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        rr = rows[split]
        require(len(rr)==d['sizes'][split] and len({r['sample_id'] for r in rr})==len(rr),
                'Subset count or duplicate ID: '+split)
        for r in rr:
            require(set(r)==set(FIELDS) and r['dataset']=='AVDF1M' and r['split']==split,
                    'Subset record schema/dataset/split mismatch')
            require(r['sample_id'].startswith('AVDF1M:') and r['group_id'] and Path(r['media_path']).is_absolute(),
                    'Invalid subset ID/group/path')
            require(all(type(r[k]) is int and r[k] in (0,1) for k in ('clip_label','video_label','audio_label')),
                    'Subset labels must be known')
            require(r['clip_label']==int(bool(r['video_label'] or r['audio_label'])), 'Subset clip label conflict')
        with (path.parent/(split+'.csv')).open(encoding='utf-8-sig',newline='') as f:
            csv_rows = list(csv.DictReader(f))
        require(csv_rows==[{k:str(r[k]) for k in FIELDS} for r in rr], 'CSV/JSONL subset disagreement')
    for key in ('sample_id','media_path','group_id'):
        require(not ({r[key] for r in rows['train']} & {r[key] for r in rows['dev']}),
                'Subset train/dev overlap: '+key)
    return dict(path=path,descriptor=d,rows=rows,
        receipt=dict(path=path.as_posix(),sha256=sha(path),files=d['files'],
                     sizes=d['sizes'],seed=d['seed'],policy=d['policy']))


def verify_selected(records, expected):
    """The canonical loader owns paths, labels, eligibility and frozen split checks."""
    actual = {r.sample_id:as_row(r) for r in records}
    require(len(actual)==len(records)==len(expected), 'Loaded subset count mismatch')
    require(actual=={r['sample_id']:r for r in expected}, 'Subset disagrees with canonical metadata/Windows paths')
