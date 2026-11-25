"""Read-only joins to completed FAV heads and same-cohort artifact predictions."""
import csv
import sqlite3
from pathlib import Path
import numpy as np
import torch
from src.dataloader.metadata import load_records
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import (
    ROOT, OUTPUT as FAV_RUN, require, read, sha, digest, atomic_json, status, Progress,
    implementation as source_implementation)
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.extract import assets
from scripts.artifact_emotion_gated_attention_cross_dataset.common import digest as producer_digest

BASE = Path(__file__).resolve().parent
ARTIFACT_RUN = ROOT/'reports/three_way_gated_v1/source_FAV/seed_42'
OUTPUT = ROOT/'reports/hierarchical_score_fusion_gru_temporal_attention_cross_dataset_v1/source_FAV/seed_42'
MODALITIES = ('video', 'audio')


def implementation():
    paths = list(BASE.glob('*.py'))
    paths += [ROOT/p for p in ('src/dataloader/metadata.py', 'src/dataloader/records.py',
        'scripts/artifact_emotion_gated_attetnion_optimized/features.py',
        'scripts/artifact_emotion_gated_attention_cross_dataset/common.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths) if not p.name.startswith('test_')}


def verify_files(folder):
    folder = Path(folder)
    done = read(folder/'COMPLETED.json')
    require(bool(done.get('files')), 'Missing completion file checksums: '+str(folder))
    for name, expected in done['files'].items():
        require(sha(folder/name) == expected, 'Changed completed file: '+str(folder/name))
    return done


def source_model(folder):
    folder = Path(folder).resolve()
    done = verify_files(folder)
    run, lock = read(folder/'RUN.json'), read(folder/'LOCKED_BEFORE_TEST.json')
    identity = digest(run)
    require(done['identity'] == lock['identity'] == identity, 'Source run identity mismatch')
    require(done['source'] == done['target'] == 'FAV' and not done['smoke_only'], 'Need full completed FAV experiment')
    cfg = run['config']
    require(not cfg['smoke_only'] and cfg['threshold'] == lock['threshold'] == .5, 'Wrong source protocol')
    require(cfg['fusion'] == lock['fusion'] == 'artifact_OR_then_emotion_OR_then_final_OR', 'Source fusion differs')
    require(run['implementation'] == source_implementation(), 'Source temporal implementation changed')
    require(cfg['video_input']['frames'] == 128 and cfg['video_input']['crop'] == 256
            and cfg['video_input']['resize_short'] == 256
            and cfg['video_input']['region'] == 'X3D-identical center crop', 'Unsupported HSE input')
    from src.utils.local_paths import load_local_paths
    require(bool(load_local_paths().get('emotion2vec_snapshot')), 'Configure the existing local seed snapshot; no automatic download')
    installed = assets()
    require(installed == run['assets'], 'Installed emotion encoders differ from the trained source')
    for mod in MODALITIES:
        head = read(folder/'heads'/mod/'COMPLETED.json')
        require(head['identity'] == lock['heads'][mod]['identity']
                and head['checkpoint_sha256'] == lock['heads'][mod]['checkpoint_sha256']
                and sha(folder/'heads'/mod/'best.pt') == head['checkpoint_sha256'], 'Source head changed')
        require(head['epochs_completed'] == cfg['epochs'], 'Source training did not complete')
    return dict(folder=str(folder), run=run, lock=lock, assets=installed,
                receipt=dict(run_sha256=sha(folder/'RUN.json'), lock_sha256=sha(folder/'LOCKED_BEFORE_TEST.json'),
                             completion_sha256=sha(folder/'COMPLETED.json'), heads=lock['heads']))


def verify_producer_lock(run, lock):
    # Old artifact identities use their producer's JSON serialization (spaces and
    # ensure_ascii=False). New GRU identities retain the compact source digest.
    require(lock['run_identity'] == producer_digest(run), 'Artifact producer run identity mismatch')
    require(lock['protocol'] == run['protocol'], 'Artifact producer protocol mismatch')


def artifact_model(folder, source, targets):
    folder = Path(folder).resolve()
    run, lock, done = [read(folder/n) for n in ('RUN.json', 'LOCKED_BEFORE_TEST.json', 'COMPLETED.json')]
    require(run['source'] == lock['source'] == done['source'] == 'FAV'
            and run['options']['reuse_fav'] and not run['smoke_only'] and not done['smoke_only'], 'Need full FAV artifact producer')
    require(set(targets) <= set(done['targets']), 'Artifact producer target is incomplete')
    verify_producer_lock(run, lock)
    require(run['label_overlay_sha256'] is None, 'Label overlays are not part of this fixed comparison')
    protocol = run['protocol']
    for key, value in dict(video_frames=128, video_size=256, mean=.45, std=.225,
                           audio_rate=16000, audio_min_samples=284672,
                           audio_length='whole waveform; minimum right zero pad; batch1').items():
        require(protocol[key] == value, 'Artifact input recipe differs: '+key)
    require(sha(ROOT/'configs/fixed_manifests.json') == run['registry_sha256']
            and sha(ROOT/'reports/phase0_v1/eligibility_v4.jsonl') == run['eligibility_sha256'], 'Frozen data metadata changed')
    for mod in MODALITIES:
        path = folder/'artifact'/mod/'best.pt'
        require(sha(path) == lock['artifact'][mod], 'Artifact export changed')
        state = torch.load(path, map_location='cpu', weights_only=True)
        require(state['format'] == 'cross_artifact_v1' and state['source'] == 'FAV'
                and state['modality'] == mod, 'Wrong artifact export')
        original = state['imported']['artifact_sha256']
        for split in ('train', 'dev', 'test'):
            require(original == source['lock']['artifact_sources'][mod][split]['artifact_checkpoint_sha256'],
                    'Target artifact weights differ from fixed FAV weights')
        del state
    seals = {}
    for target in targets:
        result = folder/'evaluation'/target
        seal = read(result/'COMPLETED.json')
        for key, path in [('lock_sha256', folder/'LOCKED_BEFORE_TEST.json'),
                          ('predictions_sha256', result/'predictions.csv'), ('metrics_sha256', result/'metrics.json')]:
            require(sha(path) == seal[key], 'Artifact producer evaluation changed: '+str(path))
        seals[target] = seal
    return dict(folder=folder, run=run, lock=lock, receipt=dict(run_sha256=sha(folder/'RUN.json'),
                lock_sha256=sha(folder/'LOCKED_BEFORE_TEST.json'), targets=seals))


def cohort_id(records):
    return producer_digest([(r.sample_id, r.group_id, r.split, str(r.media_path),
                    r.clip_label, r.video_label, r.audio_label) for r in records])


def labels_of(record):
    return [-1 if v is None else v for v in (record.clip_label, record.video_label, record.audio_label)]


def join_predictions(path, records):
    lookup = {r.sample_id: i for i, r in enumerate(records)}
    require(len(lookup) == len(records) > 0, 'Duplicate/empty frozen cohort')
    values = np.empty((len(records), 3), np.float64)
    seen = set()
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            sid = row['sample_id']
            require(sid in lookup and sid not in seen, 'Unexpected/duplicate prediction ID: '+sid)
            seen.add(sid); i = lookup[sid]; r = records[i]
            require(row['group_id'] == r.group_id and
                    [int(row[k]) for k in ('clip_label', 'video_label', 'audio_label')] == labels_of(r),
                    'Producer group/label mismatch: '+sid)
            values[i] = [float(row[k]) for k in ('x3d_score', 'aasist_score', 'gated_or')]
    require(len(seen) == len(records), 'Incomplete artifact prediction cohort')
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'Invalid producer scores')
    return values


def media_stamp(record):
    stat = record.media_path.stat()
    require(stat.st_size > 0, 'Empty media: '+str(record.media_path))
    return (stat.st_size, stat.st_mtime_ns)


def target_data(producer, target):
    status('INPUTS', target+': reading frozen test cohort and completed artifact predictions')
    records = load_records(target, 'test', task='paired')
    scores = join_predictions(producer['folder']/'evaluation'/target/'predictions.csv', records)
    cohort = cohort_id(records)
    stamps = {}
    for mod in MODALITIES:
        path = producer['folder']/'cache'/f'{mod}_{target}_test.sqlite'
        expected = dict(run=producer_digest(producer['run']), source='FAV', target=target, modality=mod,
                        artifact=producer['lock']['artifact'][mod], lock=producer_digest(producer['lock']), cohort=cohort)
        db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
        try:
            db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
            require(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone() == (producer_digest(expected),),
                    'Artifact cache cohort/recipe identity mismatch')
            current = {sid: (size, mtime) for sid, size, mtime in db.execute('SELECT id,size,mtime FROM features')}
            require(set(current) == {r.sample_id for r in records}, 'Artifact cache IDs differ')
            if stamps: require(current == stamps, 'Artifact modality media versions differ')
            stamps = current
        finally:
            db.close()
    bar = Progress('CHECK MEDIA '+target, len(records))
    for i, r in enumerate(records, 1):
        require(media_stamp(r) == stamps[r.sample_id], 'Media changed since artifact prediction: '+r.sample_id)
        bar.update(i)
    return dict(records=records, scores=scores, stamps=stamps, cohort=cohort, target=target)
