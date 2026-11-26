"""Machine-local roots only; frozen manifests/IDs/labels are never rewritten."""
from pathlib import Path, PureWindowsPath
from .common import ROOT, DATASETS, read_json, require
from src.utils.local_paths import load_local_paths

ORIGINAL_ROOTS = {
    'FAV': '/home/hdd1/sikhye/deepfake-research/datasets/FakeAVCeleb_v1.2',
    'AVDF1M': '/home/hdd1/sikhye/deepfake-research/datasets/AV-Deepfake1M',
    'PGF': '/home/hdd1/sikhye/deepfake-research/datasets/PolyGlotFake/PolyGlotFake',
}


def configuration(path=None):
    if path is None:
        return {'metadata_root': ROOT, 'local_paths': load_local_paths(), 'dataset_roots': {}}
    raw = read_json(Path(path))
    require(set(raw) <= {'metadata_root', 'dataset_roots'}, 'Unknown data configuration field')
    roots = raw['dataset_roots']
    require(set(roots) == set(DATASETS), 'Specify FAV, AVDF1M and PGF dataset roots')
    metadata = Path(raw['metadata_root'])
    require(metadata.is_absolute(), 'metadata_root must be absolute')
    for ds, root in roots.items():
        require(Path(root).is_absolute(), f'{ds} root must be absolute')
    return {'metadata_root': metadata, 'dataset_roots': roots,
            'local_paths': {'media_roots': [{'source': ORIGINAL_ROOTS[ds], 'target': roots[ds]}
                                          for ds in DATASETS]}}


def pgf_metadata_path(entry, config):
    relative = PureWindowsPath(entry['path']).relative_to(PureWindowsPath('D:/datasets/PolyGlotFake'))
    require('..' not in relative.parts, 'Invalid PGF metadata path')
    packaged = config['metadata_root'] / 'datasets/PolyGlotFake' / Path(*relative.parts)
    if packaged.is_file():
        return packaged
    if not config['dataset_roots']:
        return Path(entry['path'])
    return Path(config['dataset_roots']['PGF']).joinpath(*relative.parts)
