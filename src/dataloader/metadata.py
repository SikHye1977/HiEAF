"""Read immutable manifests; map paths without editing historical records."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from .records import SampleRecord
from src.utils.local_paths import load_local_paths

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {'fakeavceleb': 'FAV', 'fav': 'FAV', 'avdf1m': 'AVDF1M',
           'av-deepfake1m': 'AVDF1M', 'polyglotfake': 'PGF', 'pgf': 'PGF'}


def load_records(dataset_name, split, *, task='paired', registry=None,
                 eligibility_path=ROOT/'reports/phase0_v1/eligibility_v4.jsonl',
                 local_paths=None, sample_ids=None, limit=None):
    """Return records for a NEW LOCAL run; eligibility=None explicitly disables quality filtering.

    limit is a deterministic first-N smoke subset, never a new split.
    Current manifest integrity is checked; historical media identity is not asserted.
    """
    dataset = ALIASES.get(dataset_name.lower())
    if dataset is None or split not in ('train', 'dev', 'test'):
        raise ValueError('Invalid dataset or split')
    if task not in ('video', 'audio', 'paired'):
        raise ValueError('Invalid task')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if limit is not None and sample_ids is not None:
        raise ValueError('Use limit or sample_ids, not both')
    registry = Path(registry) if registry else ROOT/'configs/fixed_manifests.json'
    entry = json.loads(registry.read_text(encoding='utf-8'))['datasets'][dataset]
    manifest = Path(entry['manifest'])
    if not manifest.is_absolute():
        manifest = ROOT/manifest
    config = load_local_paths(local_paths)
    mappings = sorted([(PurePosixPath(m['source']), Path(m['target']).resolve())
                       for m in config.get('media_roots', [])],
                      key=lambda pair: len(pair[0].parts), reverse=True)
    selected = set(sample_ids) if sample_ids is not None else None
    records, seen, groups = [], set(), {}
    digest = hashlib.sha256()
    with manifest.open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            sid, group, part = row['sample_id'], row['group_id'], row['split']
            if row['dataset'] != dataset or part not in ('train', 'dev', 'test'):
                raise ValueError('Manifest dataset/split mismatch')
            if not sid or sid in seen or not group:
                raise ValueError('Duplicate/missing sample or group ID')
            seen.add(sid)
            if group in groups and groups[group] != part:
                raise ValueError('Group crosses splits: '+group)
            groups[group] = part
            labels = [row[k] for k in ('clip_label', 'video_label', 'audio_label')]
            if any(v is not None and (type(v) is not int or v not in (0, 1)) for v in labels):
                raise ValueError('Labels must be 0, 1 or null')
            if part != split or (selected is not None and sid not in selected):
                continue
            source = PurePosixPath(row['video_path'])
            if '..' in source.parts or '\\' in str(source):
                raise ValueError('Unsafe manifest path')
            for old_root, new_root in mappings:
                try:
                    relative = source.relative_to(old_root)
                except ValueError:
                    continue
                path = new_root.joinpath(*relative.parts)
                break
            else:
                raise ValueError('Unmapped source path: '+str(source))
            records.append(SampleRecord(dataset, sid, path, split, *labels, group))
    if digest.hexdigest() != entry['sha256']:
        raise ValueError('Manifest digest differs from registry')
    del seen, groups
    if selected is not None and {r.sample_id for r in records} != selected:
        raise ValueError('Requested IDs are not all in this split')
    if eligibility_path is not None:
        wanted = {r.sample_id for r in records}
        found, eligible = set(), set()
        with Path(eligibility_path).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['dataset'] != dataset or row['sample_id'] not in wanted:
                    continue
                sid = row['sample_id']
                if sid in found or row['manifest_sha256'] != entry['sha256'] or row['split'] != split:
                    raise ValueError('Duplicate or mismatched eligibility record')
                found.add(sid)
                if row['eligible'][task]:
                    eligible.add(sid)
        if found != wanted:
            raise ValueError('Missing eligibility records')
        if selected is not None and not selected <= eligible:
            raise ValueError('Requested IDs include ineligible media')
        records = [r for r in records if r.sample_id in eligible]
    return records if limit is None else records[:limit]
