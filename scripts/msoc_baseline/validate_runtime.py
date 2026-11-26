"""Bounded cache + Windows worker + real-media GPU inference check, never a research run."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.msoc_baseline.common import atomic_json, require, Stop, read_json


def main():
    import torch
    from scripts.msoc_baseline.cache import prepare, CachedDataset, loader, tensors
    from scripts.msoc_baseline.model import MSOC
    from src.utils.local_paths import resolve_media_path, load_local_paths
    from scripts.msoc_baseline.data import stamp
    out = ROOT/'reports/msoc_implementation_validation_20261125/runtime'
    require(not (out/'COMPLETE.json').exists(), 'Validation already complete; preserve its result')
    # Only the source TRAIN examples already examined by the bounded benchmark.
    benchmarks = sorted(out.parent.glob('BENCHMARK_*.json'))
    evidence = read_json(benchmarks[-1])
    wanted = {r['sample_id'] for r in evidence['decoded']}
    rows, mapping = [], load_local_paths()
    registry = read_json(ROOT/'configs/fixed_manifests.json')['datasets']
    for ds, entry in registry.items():
        with (ROOT/entry['manifest']).open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r['sample_id'] not in wanted:
                    continue
                require(r['split'] == 'train', 'Runtime check must use source train only')
                path = resolve_media_path(r['video_path'], mapping)
                rows.append(dict(sample_id=r['sample_id'], path=str(path), stamp=stamp(path),
                                 clip_label=r['clip_label'], video_label=r['video_label'], audio_label=r['audio_label'], group_id=r['group_id']))
    require(len(rows) == len(wanted), 'Missing runtime examples')
    stop = Stop(out)
    path = out/'inputs.sqlite'
    prepare(path, rows, 'train', 4, stop)
    # Full training class already parity-tested; this checks input transfer and Windows workers.
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MSOC().to('cuda:0').eval()
    order = [i % len(rows) for i in range(384)]
    ds = CachedDataset(path, rows, 'train', order=order)
    times, seen, start = [], 0, time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in loader(ds, 64, 2):
                v, a, labels, ids = tensors(batch, 'cuda:0')
                s = model.scores(model(v, a))
                require(bool(torch.isfinite(s).all()), 'Nonfinite real-media output')
                require(s.shape == (len(v), 4), 'Wrong score shape')
                torch.cuda.synchronize()
                times.append(time.perf_counter())
                seen += len(ids)
                print(f'[RUNTIME INPUT CHECK] {seen}/{len(order)}', flush=True)
    finally:
        ds.close()
    result = dict(status='PASS', scope='36 source-train clips; 384 repeated draws; no real-data optimization or accuracy measurement',
                  samples=len(rows), repeated_draws=seen, total_seconds=times[-1]-start,
                  warm_cached_FP32_samples_per_second=64*(len(times)-1)/(times[-1]-times[0]),
                  note='Small cache is OS-warm; excludes cold large-cache disk performance.')
    atomic_json(out/'COMPLETE.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
