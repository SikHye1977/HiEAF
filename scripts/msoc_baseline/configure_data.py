"""Create machine-local paths from the transferred metadata and media; never train."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import uuid
import zipfile

from .common import ROOT, HERE, DATASETS, bind, read_json, require, sha


def safe_relative(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.parts and '..' not in path.parts
            and '\\' not in name and ':' not in name, f'Invalid relative path: {name}')
    return Path(*path.parts)


def unpack_metadata(archive, destination):
    """Preserve any existing metadata directory; new extractions use a staging dir."""
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if destination.exists():
        require(destination.is_dir(), f'Not a metadata directory: {destination}')
        return destination
    checksum = archive.with_suffix(archive.suffix + '.sha256')
    require(archive.is_file() and checksum.is_file(), f'Metadata ZIP and .zip.sha256 required: {archive}')
    tokens = checksum.read_text(encoding='utf-8-sig').split()
    require(tokens, f'Empty ZIP checksum: {checksum}')
    expected = tokens[0].lower()
    require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'Invalid ZIP checksum')
    print(f'[CONFIG] Checking metadata ZIP: {archive}', flush=True)
    require(sha(archive) == expected, 'Metadata ZIP checksum mismatch; existing files retained')
    stage = destination.with_name(destination.name + '.partial-' + uuid.uuid4().hex)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            safe_relative(member.filename)
        require(z.testzip() is None, 'Metadata ZIP CRC check failed')
        stage.mkdir(parents=True, exist_ok=False)
        z.extractall(stage)
    verify_metadata(stage)
    require(not destination.exists(), f'Metadata destination appeared during extraction: {destination}')
    # Both paths are siblings in the explicitly requested metadata parent.
    require(stage.parent == destination.parent, 'Metadata staging must share its destination parent')
    os.rename(stage, destination)
    print(f'[CONFIG] Metadata extracted: {destination}', flush=True)
    return destination


def verify_metadata(metadata):
    metadata = Path(metadata)
    pins = read_json(HERE/'data_pins.json')
    checks = {'configs/fixed_manifests.json': pins['registry'],
              'reports/phase0_v1/eligibility_v4.jsonl': pins['eligibility']}
    registry_path = metadata/'configs/fixed_manifests.json'
    require(registry_path.is_file() and sha(registry_path) == pins['registry'],
            f'Frozen registry missing or changed under metadata root: {metadata}')
    registry = read_json(registry_path)['datasets']
    checks.update({registry[ds]['manifest']: pins['manifests'][ds] for ds in DATASETS})
    checks.update({f'scripts/trainable_or_3way/subsets/AVDF1M_fav_size_seed42/{s}.jsonl': v
                   for s, v in pins['subsets'].items()})
    for entry in pins['pgf'].values():
        rel = PureWindowsPath(entry['path']).relative_to('D:/datasets/PolyGlotFake').as_posix()
        checks['datasets/PolyGlotFake/'+rel] = entry['sha256']
    reference = read_json(HERE/'reference_portable.lock.json')
    checks[reference['path']] = reference['sha256']
    for i, (relative, expected) in enumerate(checks.items(), 1):
        path = metadata/safe_relative(relative)
        print(f'[CONFIG metadata] {i}/{len(checks)} {relative}', flush=True)
        require(path.is_file() and sha(path) == expected, f'Metadata missing/changed: {path}')


def media_probes(metadata):
    """Use paired-eligible IDs, with AVDF1M train/dev restricted to the actual subset."""
    probes = {ds: {} for ds in DATASETS}
    subset = metadata/'scripts/trainable_or_3way/subsets/AVDF1M_fav_size_seed42'
    for split in ('train', 'dev'):
        with (subset/f'{split}.jsonl').open(encoding='utf-8') as f:
            row = json.loads(next(f))
        probes['AVDF1M'][split] = safe_relative(row['sample_id'].split(':', 1)[1])
    print('[CONFIG] Selecting existing paired-eligible train/dev/test sample paths...', flush=True)
    with (metadata/'reports/phase0_v1/eligibility_v4.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            ds, split = row['dataset'], row['split']
            if ds in probes and split not in probes[ds] and row['eligible']['paired']:
                probes[ds][split] = safe_relative(row['sample_id'].split(':', 1)[1])
            if all(len(parts) == 3 for parts in probes.values()):
                break
    require(all(set(parts) == {'train', 'dev', 'test'} for parts in probes.values()), 'Missing eligible root probes')
    return {ds: list(parts.values()) for ds, parts in probes.items()}


def discover_roots(parent, probes, overrides=None, max_depth=3):
    """Inspect wrapper directories only; do not recurse through video collections."""
    parent = Path(parent).resolve()
    require(parent.is_dir(), f'Dataset parent does not exist: {parent}')
    matches = {ds: set() for ds in DATASETS}
    overrides = overrides or {}
    def contains(folder, ds):
        return all((folder/p).is_file() and (folder/p).stat().st_size > 0 for p in probes[ds])
    for ds, folder in overrides.items():
        folder = Path(folder).resolve()
        require(contains(folder, ds), f'{ds} root does not contain the selected sample paths: {folder}')
        matches[ds].add(folder)
    frontier, seen = [parent], set()
    for depth in range(max_depth + 1):
        following = []
        for folder in frontier:
            folder = folder.resolve()
            if folder in seen:
                continue
            seen.add(folder)
            require(len(seen) <= 20000, 'Too many directories; specify --fav-root/--avdf1m-root/--pgf-root explicitly')
            hit = False
            for ds in DATASETS:
                if contains(folder, ds):
                    hit = True
                    if ds not in overrides:
                        matches[ds].add(folder)
            if not hit and depth < max_depth:
                with os.scandir(folder) as entries:
                    following.extend(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False)
                                     and not e.name.startswith('.') and e.name not in {'reports', 'configs', 'scripts', 'checkpoints'})
        frontier = following
    for ds, candidates in matches.items():
        require(len(candidates) == 1,
                f'{ds}: found {len(candidates)} possible roots: {sorted(map(str, candidates))}. '
                f'Provide --{ds.lower()}-root with the directory containing {probes[ds][0]}')
    return {ds: str(next(iter(candidates))) for ds, candidates in matches.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets-dir', type=Path, required=True)
    p.add_argument('--metadata-archive', type=Path)
    p.add_argument('--metadata-root', type=Path)
    p.add_argument('--output', type=Path, default=HERE/'data.local.json')
    for ds in DATASETS:
        p.add_argument(f'--{ds.lower()}-root', type=Path)
    args = p.parse_args()
    require(Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve(), 'Use workspace .venv')
    parent = args.datasets_dir.resolve()
    metadata = args.metadata_root or parent/'MSOC_metadata_20261126'
    archive = args.metadata_archive or parent/'MSOC_metadata_20261126.zip'
    already_extracted = metadata.exists()
    metadata = unpack_metadata(archive, metadata)
    if already_extracted:
        verify_metadata(metadata)
    overrides = {ds: getattr(args, ds.lower()+'_root') for ds in DATASETS if getattr(args, ds.lower()+'_root')}
    roots = discover_roots(parent, media_probes(metadata), overrides)
    config = dict(metadata_root=str(metadata), dataset_roots=roots)
    bind(args.output.resolve(), config)  # Existing different configuration is never overwritten.
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    print(f'[CONFIG] COMPLETE: {args.output.resolve()}; representative paths only, no training.', flush=True)
    print('[CONFIG] Next: experiment.py preflight with this --data-config for complete media checks.', flush=True)


if __name__ == '__main__':
    main()
