"""Independent source-only artifact/head training with epoch checkpoint recovery."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.dataloader import build_loader
from .common import seed_all, save_torch, Progress, status, metrics
from .data import require_train_labels, check_disjoint, label_of
from .encoders import fresh_artifact, artifact_input, GatedBranch


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(model, train_data, dev_data, path, meta, args, *, head=False):
    """Only source train/dev are accepted by the caller. Resume at completed epoch boundary."""
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    epochs = args.head_epochs if head else args.artifact_epochs
    batch_size = args.head_batch_size if head else 1
    lr = args.head_lr if head else args.artifact_lr
    index = 1 if meta['modality'] == 'video' else 2
    labels = [label_of(r, meta['modality']) for r in train_data.records]
    keep = [i for i, v in enumerate(labels) if v is not None]
    values = np.array([labels[i] for i in keep])
    if set(values.tolist()) != {0, 1}:
        raise ValueError('Both source training classes required')
    class_weight = torch.tensor([len(values)/(2*int((values==k).sum())) for k in (0,1)], device=args.device)
    model = model.to(args.device).requires_grad_(True)
    optimizer_type = torch.optim.Adam if not head and meta['modality']=='audio' else torch.optim.AdamW
    opt = optimizer_type(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if not head else None
    use_amp = not head and meta['modality']=='video' and args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    last = path/'last.pt'; best = -1.; start_epoch = 0; history = []
    if last.exists():
        s = torch.load(last, map_location='cpu', weights_only=True)
        if s['meta'] != meta:
            raise ValueError('Training checkpoint identity changed')
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
        if scheduler:
            scheduler.load_state_dict(s['scheduler'])
        scaler.load_state_dict(s['scaler'])
        start_epoch, best, history = s['next_epoch'], s['best_auc'], s['history']
        torch.set_rng_state(s['torch_rng'])
        if torch.cuda.is_available() and s['cuda_rng']:
            torch.cuda.set_rng_state_all(s['cuda_rng'])
    for epoch in range(start_epoch, epochs):
        # Epoch-specific order means an interrupted epoch restarts from last.pt exactly.
        subset = Subset(train_data, keep)
        if head:
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(args.seed+epoch))
        else:
            loader = build_loader(subset, split='train', batch_size=1, num_workers=args.workers, seed=args.seed+epoch)
        accumulation = 1 if head else args.accumulation
        model.train(); opt.zero_grad(set_to_none=True); total_loss = 0.; count = 0
        progress = Progress(f'TRAIN {meta["source"]} {meta["modality"]} {"head" if head else "artifact"} epoch {epoch+1}/{epochs}', len(loader), unit='batches')
        for step, b in enumerate(loader):
            y = b['labels'][:, index].to(args.device)
            with torch.autocast(device_type='cuda' if args.device.startswith('cuda') else 'cpu', enabled=use_amp):
                if head:
                    logits, _ = model(b['artifact'].float().to(args.device), b['emotion'].float().to(args.device), b['emotion_valid'].to(args.device))
                else:
                    logits = model(artifact_input(b, meta['modality'], args.device)).logits
                if not head and meta['modality']=='audio':
                    losses = torch.nn.functional.cross_entropy(logits, y, reduction='none')
                else:
                    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float(), reduction='none')
                loss = (losses*class_weight[y]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            # Final partial accumulation group uses its actual number of batches.
            group_start = (step//accumulation)*accumulation
            divisor = min(accumulation, len(loader)-group_start)
            scaler.scale(loss/divisor).backward()
            if (step+1)%accumulation == 0 or step+1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())*len(y); count += len(y)
            progress.update(step+1, f'loss={total_loss/count:.6f}')
        scores, truth = predict(model, dev_data, meta['modality'], args.device, head=head)
        score = metrics(truth[:, index][truth[:, index]>=0], scores[truth[:, index]>=0])['roc_auc']
        if score is None:
            raise ValueError('Both source dev classes required')
        history.append(dict(epoch=epoch+1, loss=total_loss/count, dev_auc=score))
        if score > best:
            best = score
            save_torch(path/'best.pt', dict(meta, model=cpu_state(model), best_epoch=epoch+1, dev_auc=score))
        if scheduler:
            scheduler.step()
        save_torch(last, dict(meta=meta, model=cpu_state(model), optimizer=opt.state_dict(),
                   scheduler=scheduler.state_dict() if scheduler else None, scaler=scaler.state_dict(),
                   next_epoch=epoch+1, best_auc=best, history=history,
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        from .common import atomic_json
        atomic_json(path/'history.json', history)
        status('TRAIN', f'epoch={epoch+1}/{epochs} source_dev_auc={score:.6f} best={best:.6f}')
    state = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def predict(model, data, modality, device, *, head=False):
    if head:
        loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=0)
    else:
        loader = build_loader(data, split=data.split, num_workers=0)
    model.eval(); probabilities, labels = [], []
    progress = Progress('PREDICT '+modality, len(data))
    count = 0
    for b in loader:
        if head:
            logits, _ = model(b['artifact'].float().to(device), b['emotion'].float().to(device), b['emotion_valid'].to(device))
            p = logits.sigmoid()
        else:
            p = model(artifact_input(b, modality, device)).fake_probability
        probabilities.append(p.cpu().numpy()); labels.append(b['labels'].numpy())
        count += len(p); progress.update(count)
    return np.concatenate(probabilities), np.concatenate(labels)
