import copy
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from .common import atomic_file, atomic_json, bind, digest, hms, read_json, require, sha, Progress
from .cache import CachedDataset, loader, tensors
from .data import epoch_order
from .model import MSOC


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state_all([r.cpu() for r in rng['cuda']])


def save_checkpoint(path, value):
    with atomic_file(path) as f:
        torch.save(value, f)


def load_checkpoint(path):
    # Only this run's locally generated trusted checkpoint, verified by run identity below.
    return torch.load(path, map_location='cpu', weights_only=False)


def metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    require(len(y) > 0 and np.isfinite(score).all() and np.isin(y, (0, 1)).all(), 'Invalid predictions')
    pred = (score >= .5).astype(int)  # official real > .5, therefore fake >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), real=int(tn+fp), fake=int(fn+tp),
                roc_auc=float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
                fake_f1=float(f1_score(y, pred, zero_division=0)),
                real_fpr=float(fp/(tn+fp)) if tn+fp else None,
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]], threshold=.5)


def setup_device(args, inference=False):
    device = torch.device(args.device)
    require(device.type == 'cpu' or torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not inference
    return device


def amp_dtype(args, device):
    if args.precision == 'fp32' or device.type != 'cuda':
        return None
    if args.precision == 'bf16':
        require(torch.cuda.is_bf16_supported(), 'BF16 unavailable; choose --precision fp32 in a NEW run')
        return torch.bfloat16
    return torch.float16


@torch.inference_mode()
def validate(model, path, rows, args, stop, status):
    model.eval()
    dataset = CachedDataset(path, rows, 'dev')
    p = Progress('DEV FP32', len(rows), status=status)
    ys, scores, total, done = [], [], 0., 0
    try:
        for batch in loader(dataset, args.eval_batch, args.workers):
            stop.check()
            video, audio, labels, _ = tensors(batch, args.device)
            outputs = model(video, audio)
            loss = model.objective(outputs, labels)
            require(bool(torch.isfinite(loss)), 'Nonfinite validation loss')
            score = model.scores(outputs)[:, 0]
            n = len(video)
            total += float(loss)*n
            done += n
            ys.extend(labels[:, 0].tolist())
            scores.extend(score.tolist())
            p.update(done)
    finally:
        dataset.close()
    return dict(metrics(ys, scores), loss=total/len(rows), elapsed_seconds=time.perf_counter()-p.start)


def fit(folder, paths, rows, args, identity, stop):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    bind(folder/'IDENTITY.json', identity)
    completed = folder/'TRAIN_COMPLETE.json'
    if completed.exists():
        receipt = read_json(completed)
        require(receipt['identity'] == identity and sha(folder/'best.pt') == receipt['best_sha256'], 'Completed training changed')
        print(f'[TRAIN] Reusing completed {folder}', flush=True)
        return
    seed_all(args.seed)
    device = setup_device(args)
    dtype = amp_dtype(args, device)
    model = MSOC(args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.5, .9), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=3, min_lr=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    state = dict(epoch=0, cursor=0, updates=0, loss_sum=0., best_auc=-1., best_epoch=0,
                 train_seconds=0., epoch_seconds=0., history=[], overflow_skips=0)
    latest = folder/'latest.pt'
    if latest.exists():
        ck = load_checkpoint(latest)
        require(ck['identity'] == identity, 'Resume identity mismatch')
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        state = ck['state']
        restore_rng(ck['rng'])
        del ck
        print(f"[RESUME] epoch={state['epoch']+1} completed_draws={state['cursor']} updates={state['updates']}", flush=True)

    def save():
        save_checkpoint(latest, dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), state=state, rng=rng_state()))

    if not latest.exists():
        save()  # recoverable zero-update initialization
    last_save = time.perf_counter()
    for epoch in range(state['epoch'], args.epochs):
        model.train()
        order = epoch_order(rows['train'], args.seed, epoch)
        dataset = CachedDataset(paths['train'], rows['train'], 'train', order, args.seed, epoch, state['cursor'])
        p = Progress(f'TRAIN {folder.name} epoch {epoch+1}/{args.epochs}', len(order), state['cursor'], folder/'STATUS.json')
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        group_size = min(args.batch_size*args.accumulation, len(order)-state['cursor'])
        try:
            for batch in loader(dataset, args.batch_size, args.workers):
                video, audio, labels, _ = tensors(batch, device)
                with torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                    outputs = model(video, audio)
                    loss = model.objective(outputs, labels)
                require(bool(torch.isfinite(loss)), 'Nonfinite loss: previous committed checkpoint retained')
                n = len(video)
                scaler.scale(loss * (n/group_size)).backward()
                state['loss_sum'] += float(loss.detach())*n
                state['cursor'] += n
                accumulated += n
                if accumulated == group_size:
                    scaler.unscale_(optimizer)
                    finite = bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
                    if dtype != torch.float16:
                        require(finite, 'Nonfinite gradients: previous checkpoint retained; no corrupt update saved')
                    # Official optimizer does not clip gradients. FP16 overflow skips the step,
                    # retaining already-updated BN as standard GradScaler behaviour.
                    if not finite:
                        state['overflow_skips'] += 1
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    state['updates'] += 1
                    now = time.perf_counter()
                    state['train_seconds'] += now-tick
                    state['epoch_seconds'] += now-tick
                    tick = now
                    remaining = args.epochs*len(order) - epoch*len(order)-state['cursor']
                    avg = (epoch*len(order)+state['cursor'])/max(state['train_seconds'], 1e-9)
                    p.update(state['cursor'], f"loss={state['loss_sum']/state['cursor']:.5f} updates={state['updates']} "
                             f"remaining_train~{hms(remaining/avg)} (dev/cache/test excluded)")
                    if time.perf_counter()-last_save >= args.save_seconds or stop.pending() or state['cursor'] == len(order):
                        save()
                        last_save = time.perf_counter()
                    stop.check()
                    accumulated = 0
                    group_size = min(args.batch_size*args.accumulation, len(order)-state['cursor'])
        finally:
            dataset.close()
        result = validate(model, paths['dev'], rows['dev'], args, stop, folder/'STATUS.json')
        require(result['roc_auc'] is not None, 'Dev must contain both clip classes')
        scheduler.step(result['loss'])
        if result['roc_auc'] > state['best_auc']:
            state['best_auc'], state['best_epoch'] = result['roc_auc'], epoch+1
            save_checkpoint(folder/'best.pt', dict(identity=identity, model=model.state_dict(), epoch=epoch+1, dev=result))
        row = dict(epoch=epoch+1, train_loss=state['loss_sum']/len(order), train_seconds=state['epoch_seconds'],
                   dev=result, best_epoch=state['best_epoch'], lr=optimizer.param_groups[0]['lr'])
        state['history'].append(row)
        state.update(epoch=epoch+1, cursor=0, loss_sum=0., epoch_seconds=0.)
        save()
        atomic_json(folder/'HISTORY.json', state['history'])
        print(f"[EPOCH COMPLETE] {epoch+1}/{args.epochs} dev_auc={result['roc_auc']:.6f} best_epoch={state['best_epoch']}", flush=True)
        stop.check()
    atomic_json(completed, dict(identity=identity, best_sha256=sha(folder/'best.pt'),
                               best_epoch=state['best_epoch'], history=state['history']))
