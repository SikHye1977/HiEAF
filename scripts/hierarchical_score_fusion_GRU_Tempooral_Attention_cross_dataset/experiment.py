"""FAV -> AVDF1M / PGF with fixed GRU heads and hierarchical score fusion."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.hierarchical_score_fusion_GRU_Tempooral_Attention_cross_dataset'
import argparse
import gc
import importlib.metadata
import time
import traceback
import torch
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import (
    require, read, sha, digest, atomic_json, seed_all, WriterLock, status)
from .inputs import (FAV_RUN, ARTIFACT_RUN, OUTPUT, MODALITIES, source_model,
                     artifact_model, target_data, implementation, verify_files)
from .inference import FrozenPredictor, infer_stage
from .evaluate import evaluate


def run(args):
    require(args.frame_batch_size > 0 and args.threads > 0, 'Batch/threads must be positive')
    require(len(set(args.targets)) == len(args.targets), 'Duplicate target')
    if args.device.startswith('cuda'): require(torch.cuda.is_available(), 'CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    seed_all(42)
    status('PREFLIGHT', 'Verifying completed FAV heads, pretrained assets and original artifact score receipts')
    source = source_model(args.source_run)
    producer = artifact_model(args.artifact_run, source, args.targets)
    descriptor = dict(format='frozen_fav_temporal_cross_v1', source='FAV', targets=args.targets,
        source_folder=source['folder'], source_receipt=source['receipt'], artifact_folder=str(producer['folder']),
        artifact_receipt=producer['receipt'], implementation=implementation(),
        source_implementation=source['run']['implementation'],
        packages={n:importlib.metadata.version(n) for n in ('torch','numpy','av','opencv-python-headless','funasr','timm')},
        config=dict(device=args.device, frame_batch_size=args.frame_batch_size, threads=args.threads,
            prediction_batch_size=1, precision='FP32', threshold=.5, seed=42, zero_shot=True,
            fusion='artifact_OR_then_emotion_OR_then_final_OR', temporal_head='unchanged source heads',
            video_input=source['run']['config']['video_input'],
            audio_input='16k mono; unpadded 3s complete windows hop1s; short clips whole; native seed embedding+projection',
            missing_policy='complete-emotion primary; omit missing terms for full-cohort secondary',
            cache='per-clip score + native emotion probabilities + timestamps + attention; embeddings not retained'))
    identity = digest(descriptor)
    if args.command == 'preflight':
        for target in args.targets:
            data = target_data(producer, target)
            status('PREFLIGHT', f'{target}: {len(data["records"])} media/IDs/labels and cached artifact scores verified')
            del data; gc.collect()
        status('PREFLIGHT', 'PASS; no media decoded, no encoder or head inference, no training')
        return
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    with WriterLock(output):
        plan = output/'RUN.json'
        if plan.exists():
            require(args.resume, 'Output exists; use the same command with --resume')
            require(read(plan) == descriptor, 'Inputs/model/code/settings changed; choose a new output')
        else:
            require(not [p for p in output.iterdir() if p.name != '.writer.lock'], 'Nonempty output without RUN.json')
            atomic_json(plan, descriptor)
        lock = output/'LOCKED_BEFORE_TARGET.json'
        locked = dict(identity=identity, source_receipt=source['receipt'], artifact_receipt=producer['receipt'],
                      threshold=.5, fusion=descriptor['config']['fusion'], target_training=False)
        if lock.exists(): require(read(lock) == locked, 'Target evaluation lock changed')
        else: atomic_json(lock, locked)
        reports = {}; begin = time.perf_counter()
        for target_no, target in enumerate(args.targets, 1):
            status('TARGET', f'{target_no}/{len(args.targets)} FAV -> {target}; weights fixed, no training')
            data = target_data(producer, target); caches = {}
            for mod in MODALITIES:
                cache_id = dict(run=identity, target=target, modality=mod, cohort=data['cohort'])
                folder = output/'cache'/target/mod
                predictor = FrozenPredictor(source, mod, args.device, args.frame_batch_size)
                infer_stage(data, mod, folder, cache_id, predictor)
                caches[mod] = (folder, cache_id)
                del predictor; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            result = output/'evaluation'/target
            if (result/'COMPLETED.json').exists():
                done = verify_files(result)
                require(done['lock_sha256'] == sha(lock), 'Completed target result lock changed')
                reports[target] = read(result/'metrics.json')
                status('EVALUATE', target+': completed result verified')
            else:
                reports[target] = evaluate(data, caches, result, lock)
            primary = reports[target]['primary_complete_emotion']
            status('RESULT', f'{target} hierarchical={primary["hierarchical_or"]} artifact={primary["artifact_or"]}')
            del data; gc.collect()
        atomic_json(output/'SUMMARY.json', dict(source='FAV', targets=args.targets, metrics=reports, zero_shot=True))
        atomic_json(output/'last_invocation_timing.json', dict(seconds=time.perf_counter()-begin))
        files = ['RUN.json','LOCKED_BEFORE_TARGET.json','SUMMARY.json']
        for target in args.targets:
            files += [f'evaluation/{target}/{name}' for name in ('COMPLETED.json','metrics.json','predictions.csv')]
            files += [f'cache/{target}/{mod}/COMPLETED.json' for mod in MODALITIES]
        atomic_json(output/'COMPLETED.json', dict(identity=identity, source='FAV', targets=args.targets,
            zero_shot=True, smoke_only=False, files={name:sha(output/name) for name in files}))
        status('PIPELINE', f'COMPLETE FAV -> {args.targets}; output={output}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight','run'])
    p.add_argument('--source-run', type=Path, default=FAV_RUN)
    p.add_argument('--artifact-run', type=Path, default=ARTIFACT_RUN)
    p.add_argument('--targets', nargs='+', choices=['AVDF1M','PGF'], default=['AVDF1M','PGF'])
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--frame-batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        status('INTERRUPTED', 'Committed clips retained. Rerun the SAME command with --resume.'); sys.exit(130)
    except Exception:
        traceback.print_exc(); status('FAILED', 'Stopped; earlier committed scores retained; no successful completion for the failed stage.'); sys.exit(1)
