"""Verify the portable package, copy to a new HDD directory, and decode full audio.

No cohort filtering/repartitioning or silent media replacement is allowed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(root, split):
    result = [json.loads(s) for s in (root / f'metadata/portable/{split}.jsonl').read_text().splitlines()]
    for r in result:
        path = (root / r['archive_path']).resolve()
        if not path.is_relative_to(root.resolve()) or r['split'] != split:
            raise ValueError('Invalid path/split')
        if r['clip_label'] != (r['video_label'] | r['audio_label']):
            raise ValueError('Invalid modality labels')
        if any(r[k] not in (0, 1) for k in ('video_label', 'audio_label', 'clip_label')):
            raise ValueError('Unknown label')
    if len({r['sample_id'] for r in result}) != len(result):
        raise ValueError('Duplicate sample IDs')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--ffmpeg', default='/home/sikhye/miniforge3/envs/deepfake/bin/ffmpeg')
    args = p.parse_args()
    source = args.source.resolve()
    data = {s: rows(source, s) for s in ('train', 'dev')}
    for key in ('sample_id', 'group_id', 'archive_path'):
        if {r[key] for r in data['train']} & {r[key] for r in data['dev']}:
            raise ValueError(f'Cross-split overlap: {key}')
    if {s: len(v) for s, v in data.items()} != {'train': 14724, 'dev': 3147}:
        raise ValueError('Unexpected package size')
    if args.destination.exists() or args.cache.exists():
        raise FileExistsError('Use new destination and cache directories; originals are never overwritten')
    checks = []
    for line in (source / 'CHECKSUMS.sha256').read_text(encoding='utf-8-sig').splitlines():
        sha, rel = line.split(maxsplit=1)
        rel = rel.lstrip('*')
        path = (source / rel).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Unsafe checksum path')
        checks.append((path, sha))
    print(f'Verifying {len(checks)} exported files', flush=True)
    def verify(item):
        path, sha = item
        if digest(path) != sha:
            raise ValueError(f'Checksum mismatch: {path}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, checks))
    print('Source checksums passed; copying to new HDD destination', flush=True)
    shutil.copytree(source, args.destination)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, [(args.destination / path.relative_to(source), sha) for path, sha in checks]))
    args.cache.mkdir(parents=True)
    print('Destination checksums passed; decoding all train/dev audio without truncation', flush=True)
    flat = data['train'] + data['dev']
    def decode(r):
        key = hashlib.sha256(r['sample_id'].encode()).hexdigest()
        path = args.destination / r['archive_path']
        result = subprocess.run([args.ffmpeg, '-v', 'error', '-xerror', '-threads', '1', '-i', str(path),
                                 '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
                                capture_output=True, check=True, timeout=120)
        wave = np.frombuffer(result.stdout, dtype='<f4')
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError(f'Empty/nonfinite audio: {path}')
        output = args.cache / f'{key}.npy'
        np.save(output, wave)
        return dict(sample_id=r['sample_id'], split=r['split'], samples=len(wave),
                    cache_file=output.name, sha256=digest(output))
    measured = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, r in enumerate(pool.map(decode, flat), 1):
            measured.append(r)
            if i % 500 == 0:
                print(f'Audio {i}/{len(flat)}, {time.monotonic()-started:.1f}s', flush=True)
    maximum = max(r['samples'] for r in measured if r['split'] == 'train')
    dev_max = max(r['samples'] for r in measured if r['split'] == 'dev')
    report = dict(format='avdf1m_fav_recipe_preflight_v1', package_sha256=digest(source/'PACKAGE.json'),
                  source=str(source), dataset_root=str(args.destination.resolve()),
                  cache_root=str(args.cache.resolve()), source_and_copy_checksums_verified=len(checks),
                  manifest_sha256={s:digest(args.destination/f'metadata/portable/{s}.jsonl') for s in data},
                  counts={s:len(v) for s,v in data.items()},
                  joint_counts={s:dict(Counter(str(r['video_label'])+str(r['audio_label']) for r in v)) for s,v in data.items()},
                  audio=dict(sample_rate=16000, target_num_samples=maximum, dev_max_samples=dev_max,
                             length_policy='pad_to_eligible_train_max', truncation='prohibited',
                             historical_fav_target_num_samples=284672),
                  audio_index_sha256=None, test_accessed=False)
    index = args.cache/'audio_index.json'
    index.write_text(json.dumps(measured, indent=2)+'\n')
    report['audio_index_sha256'] = digest(index)
    (args.cache/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    if dev_max > maximum:
        raise ValueError('Dev audio exceeds train max; explicit policy decision required')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
