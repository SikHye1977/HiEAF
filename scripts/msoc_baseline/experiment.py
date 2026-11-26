"""New local MSOC comparison. Run with the workspace .venv or scripts/run.ps1."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.msoc_baseline.common import (HERE, DATASETS, atomic_json, bind, digest, exclusive,
    implementation_id, read_json, require, runtime_versions, sha, Stop, Stopped, hms)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'prepare', 'benchmark', 'run', 'status', 'compare'])
    p.add_argument('--sources', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--targets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--output', type=Path, default=ROOT/'reports/msoc_resnet_v1')
    p.add_argument('--train-cache', type=Path, default=ROOT/'reports/msoc_inputs_v1')
    p.add_argument('--eval-cache', type=Path, default=ROOT/'reports/msoc_eval_inputs_v1')
    p.add_argument('--data-config', type=Path, help='Local JSON with metadata_root and dataset_roots; does not change frozen metadata')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--backbone', choices=['resnet', 'scnet'], default='resnet')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--eval-batch', type=int, default=64)
    p.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--decode-workers', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--save-seconds', type=int, default=180)
    p.add_argument('--benchmark-batches', type=int, default=8)
    p.add_argument('--benchmark-samples', type=int, default=12)
    return p


def benchmark(args, cohorts, stop):
    """Throwaway synthetic optimizer test + bounded SOURCE TRAIN decode timing, no reported accuracy."""
    import torch
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from scripts.msoc_baseline.model import MSOC
    from scripts.msoc_baseline.media import decode, pack
    from scripts.msoc_baseline.train import setup_device, amp_dtype, seed_all
    device = setup_device(args)
    dtype = amp_dtype(args, device)
    seed_all(args.seed)
    model = MSOC(args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.5, .9), weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    video = torch.randn(args.batch_size, 3, 30, 100, 100, device=device)
    audio = torch.randn(args.batch_size, 120, 13, device=device)
    labels = torch.randint(0, 2, (args.batch_size, 3), device=device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    model.train()
    for i in range(args.benchmark_batches+2):
        stop.check()
        sync()
        t = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            loss = model.objective(model(video, audio), labels)
        require(bool(torch.isfinite(loss)), 'Benchmark nonfinite loss')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sync()
        dt = time.perf_counter()-t
        if i >= 2:
            times.append(dt)
        print(f'[BENCHMARK synthetic train] {i+1}/{args.benchmark_batches+2} seconds={dt:.3f} loss={float(loss):.4f}', flush=True)
    peak = torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else None
    evaltimes = []
    model.eval()
    # Reallocate eval batch to measure the actual FP32 setting.
    evideo = torch.randn(args.eval_batch, 3, 30, 100, 100, device=device)
    eaudio = torch.randn(args.eval_batch, 120, 13, device=device)
    with torch.inference_mode():
        for i in range(args.benchmark_batches+2):
            sync()
            t = time.perf_counter()
            score = model.scores(model(evideo, eaudio))
            require(bool(torch.isfinite(score).all()), 'Nonfinite benchmark prediction')
            sync()
            if i >= 2:
                evaltimes.append(time.perf_counter()-t)
    # Distributed source-train IDs, no target/test information used for runtime choices.
    records = []
    for ds in args.sources:
        rows = cohorts[ds]['train']
        records.extend(rows[int(i)] for i in np.linspace(0, len(rows)-1, args.benchmark_samples, dtype=int))
    lengths = []
    def work(r):
        v, a, info = decode(r['path'], train=True)
        return dict(sample_id=r['sample_id'], bytes=len(pack(v, a)), **info)
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        for r in pool.map(work, records):
            lengths.append(r)
    decode_seconds = time.perf_counter()-t
    train_rate = args.batch_size/float(np.mean(times))
    eval_rate = args.eval_batch/float(np.mean(evaltimes))
    ntrain = sum(len(cohorts[s]['train']) for s in args.sources)
    ndev = sum(len(cohorts[s]['dev']) for s in args.sources)
    ntest = sum(len(cohorts[t]['test']) for t in args.targets)*len(args.sources)
    unique = ntrain+ndev+sum(len(cohorts[t]['test']) for t in args.targets)
    estimate = ntrain*args.epochs/train_rate+(ndev*args.epochs+ntest)/eval_rate+unique/(len(records)/decode_seconds)
    result = dict(scope='Throwaway random-input training + FP32 inference + source-train-only bounded media decode; NOT a scientific experiment.',
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  synthetic_train_samples_per_second=train_rate, synthetic_eval_samples_per_second=eval_rate,
                  peak_training_allocated_GiB=peak, source_decode_samples_per_second=len(records)/decode_seconds,
                  decoded=lengths, theoretical_compute_decode_hours=estimate/3600,
                  planning_hours_with_IO_allowance=[estimate*1.3/3600, estimate*2/3600],
                  caveats='Warm synthetic GPU timing excludes input-cache disk/CPU contention, checkpoint writes, dev worker startup. Full-cache prep can differ by target codec/length. Not a completion guarantee.')
    dest = args.output/f'BENCHMARK_{time.time_ns()}.json'
    atomic_json(dest, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def run(args):
    require(Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve(), 'Use workspace .venv/Scripts/python.exe')
    require(args.epochs > 0 and args.batch_size > 0 and args.accumulation > 0 and args.eval_batch > 0, 'Positive training settings required')
    require(args.workers >= 0 and args.decode_workers > 0 and args.cpu_threads > 0 and args.save_seconds > 0, 'Invalid worker/save settings')
    require(args.benchmark_batches > 0 and args.benchmark_samples > 0, 'Positive benchmark settings required')
    require(len(set(args.sources)) == len(args.sources) and len(set(args.targets)) == len(args.targets), 'Duplicate sources/targets')
    args.output = args.output.resolve()
    if args.command == 'compare':
        from scripts.msoc_baseline.compare import compare
        with exclusive(args.output/'RUN.lock'):
            compare(args.output, args.data_config)
        return
    if args.command == 'status':
        for p in sorted(args.output.rglob('STATUS.json')):
            print(p, json.dumps(read_json(p), ensure_ascii=False), flush=True)
        for p in sorted(args.output.rglob('*COMPLETE.json')):
            print('[COMPLETE]', p)
        return
    from scripts.msoc_baseline.model import verify_upstream
    from scripts.msoc_baseline.data import load_cohorts
    from scripts.msoc_baseline.media import POLICY, decode
    from scripts.msoc_baseline.cache import cache_path, prepare
    from scripts.msoc_baseline.train import fit, setup_device, amp_dtype
    from scripts.msoc_baseline.evaluate import evaluate, summarize
    commit = verify_upstream()
    device = setup_device(args)
    amp_dtype(args, device)
    stop = Stop(args.output)
    stop.install()
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive(args.output/'RUN.lock'):
        stop.check()
        cohorts, summary = load_cohorts(args.output, args.data_config)
        from scripts.msoc_baseline.compare import verify_reference
        reference_check = verify_reference(cohorts, args.data_config)
        if args.command == 'preflight':
            checked = []
            for ds in args.sources:
                # One example from every present source-training modality class.
                groups = {}
                for r in cohorts[ds]['train']:
                    groups.setdefault((r['video_label'], r['audio_label']), r)
                for row in groups.values():
                    v, a, info = decode(row['path'], train=True)
                    checked.append(dict(sample_id=row['sample_id'], video_shape=list(v.shape), audio_shape=list(a.shape), **info))
            atomic_json(args.output/'PREFLIGHT.json', dict(commit=commit, cohorts=summary, bounded_decode=checked,
                        reference_check=reference_check, device=str(device), trained=False))
            print('[PREFLIGHT] PASS; training has not started.', flush=True)
            return
        if args.command == 'benchmark':
            benchmark(args, cohorts, stop)
            return
        config = {k: getattr(args, k) for k in ('seed', 'backbone', 'epochs', 'batch_size', 'accumulation', 'precision', 'eval_batch', 'device')}
        run_identity = dict(format='msoc_local_v1', official_commit=commit, code=implementation_id(),
                            input_policy=POLICY, config=config, data_pins=read_json(HERE/'data_pins.json'),
                            runtime=runtime_versions(), upstream_lock_sha256=sha(HERE/'upstream.lock.json'))
        run_identity['comparison_reference'] = reference_check['reference']
        bind(args.output/'RUN.json', run_identity)
        # prepare-only may be interrupted and reused. During run prepare each source
        # just in time; tests reuse a single target cache across all source models.
        if args.command == 'prepare':
            jobs = [(s, p) for s in args.sources for p in ('train', 'dev')]+[(t, 'test') for t in args.targets]
            for ds, split in jobs:
                prepare(cache_path(args, ds, split), cohorts[ds][split], split, args.decode_workers, stop, args.output/'STATUS.json')
            print('[PREPARE] COMPLETE; no training/evaluation executed.', flush=True)
            return
        for source in args.sources:
            print(f'[STAGE] source training {source}; sources={args.sources}; then targets={args.targets}', flush=True)
            folder = args.output/f'source_{source}'
            paths = {s: cache_path(args, source, s) for s in ('train', 'dev')}
            for split in paths:
                prepare(paths[split], cohorts[source][split], split, args.decode_workers, stop, args.output/'STATUS.json')
            train_identity = dict(run=run_identity, source=source, train=summary[f'{source}/train']['identity'],
                                  dev=summary[f'{source}/dev']['identity'])
            fit(folder, paths, cohorts[source], args, train_identity, stop)
        for target in args.targets:
            path = cache_path(args, target, 'test')
            prepare(path, cohorts[target]['test'], 'test', args.decode_workers, stop, args.output/'STATUS.json')
            for source in args.sources:
                folder = args.output/f'source_{source}'
                evaluate(folder/'test'/target, folder/'best.pt', path, cohorts[target]['test'], args, run_identity, stop)
                summarize(args.output)
        results = summarize(args.output)
        from scripts.msoc_baseline.compare import compare
        compare(args.output, args.data_config)
        atomic_json(args.output/'PIPELINE_COMPLETE.json', dict(sources=args.sources, targets=args.targets,
                     completed_pairs=[f"{r['source']}->{r['target']}" for r in results], timestamp=time.time()))
        print('[PIPELINE] COMPLETE. Results:', args.output/'SUMMARY.csv', flush=True)


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except Stopped as e:
        print('[PAUSED]', e, flush=True)
        sys.exit(130)
    except (Exception, KeyboardInterrupt):
        print('[FAILED/INTERRUPTED] Last committed cache/checkpoint/scores retained; no completion for failed stage.', flush=True)
        raise
