"""Frozen-checkpoint A/B/C evaluation; torchrun workers never share BN statistics."""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.special import expit
import torch
import torch.distributed as dist

from avdf1m_fav_reproduction.data_models import read_video
from .core import (CONDITIONS, X3D, assert_state_preserved, baseline_comparison,
                   check_modes, configure_condition, metrics, sha256, write_json)


def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def emit(event, **fields):
    print(json.dumps(dict(event=event, time=time.time(), **fields)), flush=True)


def fresh_model(weight, condition, device):
    reference = torch.load(weight, map_location='cpu', weights_only=True)
    model = X3D().to(device)
    model.load_state_dict(reference, strict=True)
    configure_condition(model, condition)
    before = assert_state_preserved(model, reference)
    return model, reference, before


def infer(model, video, condition, device):
    assert video.shape == (3, 128, 256, 256) and video.dtype == torch.float32
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16,
                                        enabled=CONDITIONS[condition]['amp']):
        z, p = model(video.unsqueeze(0).to(device))
    assert z.numel() == p.numel() == 1
    assert torch.isfinite(z).all() and torch.isfinite(p).all()
    expected = torch.float16 if CONDITIONS[condition]['amp'] else torch.float32
    assert z.dtype == p.dtype == expected, (condition, z.dtype, p.dtype)
    logit, prob = float(z.float().item()), float(p.float().item())
    return dict(logit=logit, fake_probability=prob, predicted_label=int(prob >= .5),
                fake_probability_from_logit_float64=float(expit(logit)),
                logit_dtype=str(z.dtype), probability_dtype=str(p.dtype))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    args = p.parse_args(); bundle = args.bundle.resolve()
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert world == 4
    torch.set_num_threads(2)
    random.seed(42+rank); np.random.seed(42+rank); torch.manual_seed(42+rank)
    torch.cuda.set_device(local_rank); device = torch.device('cuda', local_rank)
    dist.init_process_group('gloo', timeout=timedelta(minutes=120))
    config = json.loads((bundle/'metadata/config.json').read_text())
    dataset = Path(config['dataset_root'])
    rows = load_rows(bundle/'manifests/dev_portable.jsonl'); local_rows = rows[rank::world]
    assert len(rows) == 3147
    weight = bundle/'weights/best_state_dict.pth'; weight_hash = sha256(weight)
    audit = json.loads((bundle/'metadata/checkpoint_audit.json').read_text())
    assert weight_hash == audit['original_export_sha256']
    runtime = dict(rank=rank, local_rank=local_rank, device=torch.cuda.get_device_name(device),
                   argv=sys.argv, torch=torch.__version__, cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                   cudnn_deterministic=torch.backends.cudnn.deterministic,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   grad_enabled_during_forward=False, optimizer_created=False)
    write_json(bundle/f'metadata/runtime_rank{rank}.json', runtime)

    # Check one real input on every GPU and all modes before committing a long run.
    first = local_rows[0]; first_video = read_video(dataset/first['archive_path'])
    smoke = []
    for condition in CONDITIONS:
        model, reference, before = fresh_model(weight, condition, device)
        value = infer(model, first_video, condition, device)
        after = assert_state_preserved(model, reference); check_modes(model, condition)
        smoke.append(dict(condition=condition, sample_id=first['sample_id'], **value,
                          state_before=before, state_after=after))
        del model, reference
        gc.collect(); torch.cuda.empty_cache()
    del first_video
    write_json(bundle/f'metadata/smoke_rank{rank}.json', smoke)
    dist.barrier()
    if rank == 0: emit('SMOKE_PASSED', ranks=world, conditions=list(CONDITIONS))

    summary = {}
    for condition in CONDITIONS:
        dest = bundle/'results'/condition
        if rank == 0: dest.mkdir(exist_ok=False)
        dist.barrier()
        assert sha256(weight) == weight_hash
        model, reference, before = fresh_model(weight, condition, device)
        torch.cuda.synchronize(); dist.barrier()
        started = time.time(); clock = time.monotonic(); last_update = clock
        decode_seconds = 0.; inference_seconds = 0.; integrity_seconds = 0.
        torch.cuda.reset_peak_memory_stats(device)
        with (dest/f'predictions_rank{rank}.jsonl').open('x') as stream:
            for i, row in enumerate(local_rows):
                path = dataset/row['archive_path']
                t = time.monotonic()
                assert path.stat().st_size == row['media_bytes']
                media_hash = sha256(path)
                assert media_hash == row['media_sha256'], row['sample_id']
                integrity_seconds += time.monotonic()-t
                t = time.monotonic(); video = read_video(path)
                decode_seconds += time.monotonic()-t
                t = time.monotonic()
                input_hash = hashlib.sha256(video.numpy().tobytes()).hexdigest()
                integrity_seconds += time.monotonic()-t
                t = time.monotonic(); value = infer(model, video, condition, device)
                inference_seconds += time.monotonic()-t
                result = dict(sample_id=row['sample_id'], relative_path=row['relative_path'],
                              archive_path=row['archive_path'], video_label=row['video_label'],
                              audio_label=row['audio_label'], split='dev', condition=condition,
                              condition_name=CONDITIONS[condition]['name'], threshold=.5,
                              media_sha256=media_hash, input_tensor_sha256=input_hash, rank=rank, **value)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
                del video
                now = time.monotonic()
                if i == 0 or now-last_update >= 30 or i+1 == len(local_rows):
                    progress = dict(condition=condition, rank=rank, completed=i+1, total=len(local_rows),
                                    elapsed_seconds=now-clock, timestamp=time.time())
                    temp = dest/f'.progress_rank{rank}.tmp'
                    write_json(temp, progress); temp.replace(dest/f'progress_rank{rank}.json')
                    emit('PROGRESS', **progress); last_update = now
        torch.cuda.synchronize()
        elapsed = time.monotonic()-clock
        after = assert_state_preserved(model, reference); check_modes(model, condition)
        assert before == after and sha256(weight) == weight_hash
        write_json(dest/f'state_preservation_rank{rank}.json', dict(condition=condition, before=before, after=after,
                   all_non_bn_modules_eval=True, bn_affine_unchanged=True, no_parameter_gradients=True,
                   checkpoint_file_sha256=weight_hash, checkpoint_file_unchanged=True))
        timing = dict(rank=rank, samples=len(local_rows), started=started, ended=time.time(),
                      elapsed_seconds=elapsed, decoding_seconds=decode_seconds,
                      inference_seconds=inference_seconds, integrity_seconds=integrity_seconds,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
        write_json(dest/f'timing_rank{rank}.json', timing)
        del model, reference; gc.collect(); torch.cuda.empty_cache()
        dist.barrier()
        approved = [True]
        if rank == 0:
            merged = [r for worker in range(world) for r in load_rows(dest/f'predictions_rank{worker}.jsonl')]
            indexed = {r['sample_id']:r for r in merged}
            assert len(merged) == len(indexed) == len(rows)
            merged = [indexed[r['sample_id']] for r in rows]
            with (dest/'predictions.jsonl').open('x') as f:
                f.writelines(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in merged)
            stats = metrics(merged)
            timings = [json.loads((dest/f'timing_rank{w}.json').read_text()) for w in range(world)]
            stats['wall_seconds'] = max(t['elapsed_seconds'] for t in timings)
            stats['wall_time_definition'] = 'max worker elapsed; includes media hashing, decoding, input hashing, GPU inference, JSONL writing; excludes model load/smoke/metrics'
            stats['condition'] = condition; stats['condition_name'] = CONDITIONS[condition]['name']
            write_json(dest/'metrics.json', stats); summary[condition] = stats
            if condition == 'A':
                comparison = baseline_comparison(merged, load_rows(bundle/'reference/best_dev.jsonl'))
                write_json(dest/'baseline_reproduction.json', comparison)
                approved[0] = comparison['passed']
                emit('BASELINE_REPRODUCTION', **comparison)
            else:
                baseline = load_rows(bundle/'results/A/predictions.jsonl')
                assert all(a['sample_id'] == b['sample_id'] and a['input_tensor_sha256'] == b['input_tensor_sha256']
                           and a['media_sha256'] == b['media_sha256'] for a,b in zip(baseline,merged))
            emit('CONDITION_COMPLETE', condition=condition, metrics=stats)
        dist.broadcast_object_list(approved, src=0)
        if not approved[0]:
            raise RuntimeError('A did not reproduce the stored baseline. B/C halted pending explanation.')
        dist.barrier()

    if rank == 0:
        originals = json.loads((bundle/'metadata/original_files_sha256.json').read_text())
        actual = {path:sha256(path) for path in originals}
        assert actual == originals, 'An original protected file changed during evaluation'
        write_json(bundle/'metadata/original_integrity_after.json', dict(all_unchanged=True, files=actual))
        write_json(bundle/'results/summary.json', summary)
        write_json(bundle/'metadata/EVALUATION_COMPLETED.json', dict(completed=time.time(),
                   all_conditions_completed=True, original_files_unchanged=True,
                   identical_preprocessed_inputs_across_conditions=True, samples_per_condition=3147))
        emit('EVALUATION_COMPLETE', bundle=str(bundle))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
