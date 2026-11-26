"""Create the small external metadata handoff; no media, weights or feature caches."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import zipfile
from .common import ROOT, HERE, read_json, require, sha


def package(output):
    output = Path(output)
    require(not output.exists(), 'Preserve the existing archive; choose a new filename')
    pins = read_json(HERE/'data_pins.json')
    items = {'configs/fixed_manifests.json': ROOT/'configs/fixed_manifests.json',
             'reports/phase0_v1/eligibility_v4.jsonl': ROOT/'reports/phase0_v1/eligibility_v4.jsonl'}
    require(sha(items['configs/fixed_manifests.json']) == pins['registry'], 'Registry changed')
    require(sha(items['reports/phase0_v1/eligibility_v4.jsonl']) == pins['eligibility'], 'Eligibility changed')
    registry = read_json(items['configs/fixed_manifests.json'])['datasets']
    for ds, entry in registry.items():
        require(sha(ROOT/entry['manifest']) == pins['manifests'][ds], 'Manifest changed: '+ds)
        for key in ('manifest', 'summary'):
            items[entry[key]] = ROOT/entry[key]
    subset = ROOT/'scripts/trainable_or_3way/subsets/AVDF1M_fav_size_seed42'
    for path in sorted(subset.iterdir()):
        if path.suffix in ('.json', '.jsonl', '.csv'):
            items[path.relative_to(ROOT).as_posix()] = path
    for split in ('train', 'dev'):
        require(sha(subset/f'{split}.jsonl') == pins['subsets'][split], 'Subset changed: '+split)
    for entry in pins['pgf'].values():
        path = Path(entry['path'])
        require(sha(path) == entry['sha256'], 'PGF labels changed')
        rel = PureWindowsPath(entry['path']).relative_to('D:/datasets/PolyGlotFake')
        items['datasets/PolyGlotFake/'+rel.as_posix()] = path
    pin = read_json(HERE/'reference_portable.lock.json')
    require(sha(ROOT/pin['path']) == pin['sha256'], 'Reference changed')
    items[pin['path']] = ROOT/pin['path']
    checks = {name: dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in items.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for i, (name, path) in enumerate(items.items(), 1):
            print(f'[METADATA] {i}/{len(items)} {name}', flush=True)
            z.write(path, name)
        z.writestr('CONTENTS_SHA256.json', json.dumps(checks, indent=2)+'\n')
    with zipfile.ZipFile(output) as z:
        for name, info in checks.items():
            h = hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda: f.read(4*1024*1024), b''):
                    h.update(chunk)
            require(h.hexdigest() == info['sha256'], 'Archive verification failed: '+name)
    signature = sha(output)
    output.with_suffix('.zip.sha256').write_text(signature+'  '+output.name+'\n', encoding='ascii')
    print(json.dumps(dict(archive=str(output), bytes=output.stat().st_size, sha256=signature,
                         payload_files=len(items), verified=True, contains_media=False), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    package(p.parse_args().output)
