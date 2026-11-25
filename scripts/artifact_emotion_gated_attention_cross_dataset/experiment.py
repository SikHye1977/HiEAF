"""CLI: preflight and source-train -> two-target evaluation. No target adaptation."""
import argparse
import gc
import importlib.metadata
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.artifact_emotion_gated_attention_cross_dataset.common import (
    DATASETS, PROTOCOL, ROOT, read, sha, digest, implementation, atomic_json,
    bind_directory, WriterLock, seed_all, save_torch, status)
from scripts.artifact_emotion_gated_attention_cross_dataset.data import (
    make_dataset, read_overlay, require_train_labels, check_disjoint, cohort_id)
from scripts.artifact_emotion_gated_attention_cross_dataset.encoders import (
    import_fav, fresh_artifact, load_artifact, GatedBranch, model_config)
from scripts.artifact_emotion_gated_attention_cross_dataset.train import fit, cpu_state
from scripts.artifact_emotion_gated_attention_cross_dataset.features import Extractor
from scripts.artifact_emotion_gated_attention_cross_dataset.evaluate import evaluate


def assets():
    from src.models.hsemotion import hsemotion_checkpoint, HSE_SHA256
    from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import download_seed
    from src.utils.local_paths import load_local_paths
    import cv2
    hse = hsemotion_checkpoint()
    if sha(hse) != HSE_SHA256:
        raise ValueError('Official HSEmotion weight integrity failed')
    if not load_local_paths().get('emotion2vec_snapshot'):
        raise ValueError('Pinned local emotion2vec+ seed installation required')
    snapshot, revision = download_seed()
    return dict(hsemotion_sha256=HSE_SHA256, emotion2vec_revision=revision,
                emotion2vec_sha256=sha(Path(snapshot)/'model.pt'),
                face_detector_sha256=sha(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'),
                aasist_source_sha256=sha(ROOT/'third_party/aasist/models/AASIST.py'))


def preflight(args):
    result = {'assets': {}, 'sources': {}}
    try:
        result['assets'] = dict(status='pass', **assets())
    except Exception as e:
        result['assets'] = dict(status='blocked', reason=str(e))
    overlay = read_overlay(args.pgf_labels)
    for source in args.sources:
        status('PREFLIGHT', f'{source}: checking source labels and fixed cohorts')
        try:
            if source == 'PGF' and not overlay:
                raise ValueError('PGF modality labels are unknown; supply an evidence-backed --pgf-labels JSONL')
            train = make_dataset(source, 'train', 'video', overlay)
            dev = make_dataset(source, 'dev', 'video', overlay)
            check_disjoint(train, dev)
            counts = {mod: {'train': require_train_labels(train, mod), 'dev': require_train_labels(dev, mod)}
                      for mod in ('video', 'audio')}
            result['sources'][source] = dict(status='pass', counts=counts,
                 note='Metadata/assets ready; not a full decode or GPU training test')
            del train, dev
        except Exception as e:
            result['sources'][source] = dict(status='blocked', reason=str(e))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.output, result)
    return 0 if result['assets']['status']=='pass' and all(v['status']=='pass' for v in result['sources'].values()) else 2


def run(args):
    if args.reuse_fav and args.source != 'FAV':
        raise ValueError('--reuse-fav is only valid with --source FAV')
    if args.source in args.targets or len(set(args.targets)) != len(args.targets):
        raise ValueError('Targets must be unique and different from the source')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    overlay = read_overlay(args.pgf_labels)
    if args.source == 'PGF' and not overlay:
        raise ValueError('PGF source requires --pgf-labels; no fabricated modality labels')
    seed_all(args.seed)
    root = args.output.resolve()
    # Recipe identity includes source data and software, never source-workstation media hashes.
    opts = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
            if k not in ('command', 'output', 'resume')}
    identity = dict(options=opts, protocol=PROTOCOL, implementation=implementation(), assets=assets(),
        registry_sha256=sha(ROOT/'configs/fixed_manifests.json'),
        eligibility_sha256=sha(ROOT/'reports/phase0_v1/eligibility_v4.jsonl'),
        label_overlay_sha256=sha(args.pgf_labels) if args.pgf_labels else None,
        packages={n:importlib.metadata.version(n) for n in ('torch','pytorchvideo','numpy','av','opencv-python-headless','hsemotion','funasr','timm')},
        artifact_configs={m:model_config(m) for m in ('video','audio')},
        source=args.source, smoke_only=args.limit is not None)
    # Bind run before expensive work; OS lock prevents concurrent writes into this source run.
    if root.exists() and not args.resume:
        raise FileExistsError('Output already exists; use --resume or choose a new directory')
    root.mkdir(parents=True, exist_ok=True)
    with WriterLock(root):
        bind_directory(root, identity, True)
        outputs = {}
        if not args.reuse_fav:
            # Validate both branch labels before any artifact training starts.
            train = make_dataset(args.source, 'train', 'video', overlay, args.limit)
            dev = make_dataset(args.source, 'dev', 'video', overlay, args.limit)
            check_disjoint(train, dev)
            for modality in ('video','audio'):
                require_train_labels(train, modality); require_train_labels(dev, modality)
            del train, dev
        for modality in ('video','audio'):
            seed_all(args.seed)
            ap = root/'artifact'/modality/'best.pt'; hp = root/'heads'/modality/'best.pt'
            base_meta = dict(source=args.source, modality=modality, run_identity=digest(identity))
            if args.reuse_fav:
                artifact, head, provenance = import_fav(modality)
                if not ap.exists():
                    save_torch(ap, dict(base_meta, format='cross_artifact_v1', model=cpu_state(artifact), imported=provenance))
                    save_torch(hp, dict(base_meta, format='cross_head_v1', model=cpu_state(head),
                                       artifact_dim=2048 if modality=='video' else 160,
                                       emotion_dim=1280 if modality=='video' else 768,
                                       artifact_sha256=sha(ap), imported=provenance))
                elif not hp.exists():
                    save_torch(hp, dict(base_meta, format='cross_head_v1', model=cpu_state(head),
                                       artifact_dim=2048 if modality=='video' else 160,
                                       emotion_dim=1280 if modality=='video' else 768,
                                       artifact_sha256=sha(ap), imported=provenance))
                del artifact, head
            else:
                train = make_dataset(args.source, 'train', modality, overlay, args.limit)
                dev = make_dataset(args.source, 'dev', modality, overlay, args.limit)
                check_disjoint(train, dev)
                meta = dict(base_meta, format='cross_artifact_v1', train_cohort=cohort_id(train), dev_cohort=cohort_id(dev))
                # Completed epochs are recovered by fit; no pretrained download when resuming.
                artifact = fresh_artifact(modality, pretrained=not (ap.parent/'last.pt').exists())
                artifact = fit(artifact, train, dev, ap.parent, meta, args)
                extractor = Extractor(artifact, modality, args.device, args.face_batch_size)
                cache_identity = dict(run=digest(identity), source=args.source, modality=modality, artifact=sha(ap))
                tc = extractor.extract(train, root/'cache'/f'{modality}_train.sqlite', cache_identity)
                dc = extractor.extract(dev, root/'cache'/f'{modality}_dev.sqlite', cache_identity)
                del extractor, artifact
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                meta = dict(base_meta, format='cross_head_v1', artifact_dim=2048 if modality=='video' else 160,
                            emotion_dim=1280 if modality=='video' else 768, artifact_sha256=sha(ap),
                            train_cohort=cohort_id(train), dev_cohort=cohort_id(dev))
                seed_all(args.seed)
                head = GatedBranch(meta['artifact_dim'], meta['emotion_dim'])
                fit(head, tc, dc, hp.parent, meta, args, head=True)
                tc.close(); dc.close(); del tc,dc,head,train,dev
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        lock = dict(source=args.source, protocol=PROTOCOL, run_identity=digest(identity), smoke_only=args.limit is not None,
                    artifact={m:sha(root/'artifact'/m/'best.pt') for m in ('video','audio')},
                    heads={m:sha(root/'heads'/m/'best.pt') for m in ('video','audio')})
        lp = root/'LOCKED_BEFORE_TEST.json'
        if lp.exists() and read(lp) != lock:
            raise ValueError('Locked source weights changed')
        atomic_json(lp, lock)
        for target in args.targets:
            caches = {}
            try:
                for modality in ('video','audio'):
                    ap = root/'artifact'/modality/'best.pt'
                    if sha(ap) != lock['artifact'][modality]:
                        raise ValueError('Locked artifact changed')
                    ds = make_dataset(target, 'test', modality, overlay, args.limit)
                    artifact, _ = load_artifact(ap, args.source, modality, args.device)
                    extractor = Extractor(artifact, modality, args.device, args.face_batch_size)
                    cache_identity = dict(run=digest(identity), source=args.source, target=target,
                                          modality=modality, artifact=sha(ap), lock=digest(lock))
                    caches[modality] = extractor.extract(ds, root/'cache'/f'{modality}_{target}_test.sqlite', cache_identity)
                    del extractor,artifact,ds
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                outputs[target] = evaluate(caches['video'], caches['audio'], args.source, target, root, args.device, lock)
                print(json.dumps(outputs[target], ensure_ascii=False, indent=2), flush=True)
            finally:
                for cache in caches.values(): cache.close()
        atomic_json(root/'SUMMARY.json', outputs)
        atomic_json(root/'COMPLETED.json', dict(source=args.source, targets=args.targets, smoke_only=args.limit is not None))
        status('PIPELINE', f'COMPLETE {args.source} -> {args.targets}; output={root}')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('preflight')
    q.add_argument('--sources', nargs='+', choices=DATASETS, default=list(DATASETS))
    q.add_argument('--pgf-labels', type=Path); q.add_argument('--output', type=Path)
    q = sub.add_parser('summarize')
    q.add_argument('--runs', nargs='+', required=True, type=Path)
    q.add_argument('--output', required=True, type=Path)
    q = sub.add_parser('run')
    q.add_argument('--source', required=True, choices=DATASETS)
    q.add_argument('--targets', nargs='+', choices=DATASETS)
    q.add_argument('--output', required=True, type=Path)
    q.add_argument('--pgf-labels', type=Path)
    q.add_argument('--reuse-fav', action='store_true', help='Import the recorded FAV artifact/head pairs; do not retrain FAV')
    q.add_argument('--resume', action='store_true')
    q.add_argument('--device', default='cuda:0'); q.add_argument('--seed', type=int, default=42)
    q.add_argument('--artifact-epochs', type=int, default=20); q.add_argument('--head-epochs', type=int, default=30)
    q.add_argument('--artifact-lr', type=float, default=1e-4); q.add_argument('--head-lr', type=float, default=1e-3)
    q.add_argument('--head-batch-size', type=int, default=64); q.add_argument('--accumulation', type=int, default=8)
    q.add_argument('--workers', type=int, default=0); q.add_argument('--face-batch-size', type=int, default=1)
    q.add_argument('--limit', type=int, help='Smoke-only stratified first-N subset; never a full experiment result')
    args = p.parse_args(argv)
    try:
        if args.command == 'preflight':
            return preflight(args)
        if args.command == 'summarize':
            from scripts.artifact_emotion_gated_attention_cross_dataset.summarize import summarize
            print(json.dumps(summarize(args.runs,args.output),ensure_ascii=False,indent=2)); return 0
        args.targets = args.targets or [d for d in DATASETS if d != args.source]
        for name in ('artifact_epochs','head_epochs','head_batch_size','accumulation','face_batch_size'):
            if getattr(args,name) < 1: raise ValueError(name+' must be positive')
        if args.workers < 0 or args.artifact_lr<=0 or args.head_lr<=0 or (args.limit is not None and args.limit<1):
            raise ValueError('Invalid worker/learning rate/limit')
        return run(args)
    except KeyboardInterrupt:
        status('STOP', 'Interrupted; repeat the identical command with --resume'); return 130
    except Exception as error:
        status('BLOCKED', str(error))
        if args.command == 'run' and args.output.is_dir():
            import traceback
            (args.output/'last_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
