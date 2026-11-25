"""Recovered FAV 4GPU loop, adapted to portable AVDF1M manifests.

See historical_functions.json for original function hashes. Core sampling,
optimizer, AMP overflow, BN broadcast and checkpoint loop are retained.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from avdf1m_fav_reproduction.runtime import prepare, binary_metrics, validate_predictions

AUDIO_TARGET = None


def validate_ddp_provenance(state, provenance, checkpoint_path):
    if state['provenance'] != provenance:
        raise ValueError('Resume immutable provenance mismatch')
    return None

def emit(stage, **fields):
    print(time.strftime('%H:%M:%S'), stage,
          ' '.join(f'{key}={value}' for key, value in fields.items()), flush=True)

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def epoch_draws(labels, count, seed, epoch):
    """Exact 1:1 replacement draws; independent epoch generator, no rank/sharding."""
    pools = [torch.tensor([i for i, y in enumerate(labels) if y == cls]) for cls in (0, 1)]
    if not all(len(pool) for pool in pools):
        raise ValueError('Both training classes are required')
    generator = torch.Generator().manual_seed(seed + epoch)
    sizes = (count // 2, count - count // 2)
    draws = torch.cat([pool[torch.randint(len(pool), (n,), generator=generator)]
                       for pool, n in zip(pools, sizes)])
    return draws[torch.randperm(count, generator=generator)].tolist()

def batch_forward(model, batch, device, task):
    if set(batch['split']) - {'train', 'dev'}:
        raise ValueError('Test/unknown split forbidden')
    column = 1 if task == 'video' else 2
    mask = batch['label_mask'][:, column].bool()
    if not mask.any():
        raise ValueError('Batch has no supervised labels')
    y = batch['labels'][mask, column].to(device)
    x = batch[task][mask].to(device)
    expected = (3, 128, 256, 256) if task == 'video' else (AUDIO_TARGET,)
    if tuple(x.shape[1:]) != expected:
        raise ValueError(f'Invalid {task} shape: {tuple(x.shape)}')
    if task == 'audio':
        base_model = getattr(model, 'module', model)
        if base_model.fake_class_index != 1 or (batch['audio_lengths'] > AUDIO_TARGET).any():
            raise ValueError('AASIST mapping/truncation mismatch')
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite input')
    out = model(x)
    if not torch.isfinite(out.logits).all() or not torch.isfinite(out.fake_probability).all():
        raise FloatingPointError('Nonfinite model output')
    ids = [sid for sid, keep in zip(batch['sample_id'], mask.tolist()) if keep]
    return out, y, ids

def make_optimizer(model, config, device):
    t = config['training']; o = t['optimizer']
    cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[o['name']]
    optimizer = cls(model.parameters(), lr=o['learning_rate'], weight_decay=o['weight_decay'])
    if t['scheduler']['name'] != 'cosine':
        raise ValueError('Only configured cosine scheduler is supported')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'])
    scaler = torch.amp.GradScaler(device.type, enabled=t['mixed_precision'],
                                 init_scale=t.get('amp_initial_scale', 16384))
    return optimizer, scheduler, scaler

def optimizer_update(model, optimizer, scaler, config, consecutive, parallel=None):
    old = float(scaler.get_scale())
    scaler.unscale_(optimizer)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing = [n for n, p in parameters.items() if p.grad is None]
    nonfinite = [n for n, p in parameters.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
    required = set(config.get('required_gradient_parameters', []))
    invalid = bool(required - parameters.keys() or required.intersection(missing) or len(missing) == len(parameters))
    if parallel:
        parallel.check('Required/no gradient: '+str(missing) if invalid else None)
    elif invalid:
        raise FloatingPointError(f'Required/no gradient: {missing}')
    overflow = bool(nonfinite)
    if parallel:
        overflow = any(parallel.collect(overflow))
    if overflow and not scaler.is_enabled():
        raise FloatingPointError(f'FP32 nonfinite gradients: {nonfinite}')
    if not overflow and config.get('gradient_clip_norm') is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
    if overflow and parallel:
        # All ranks skip together, including ranks with locally finite gradients.
        # Public explicit scale update resets GradScaler optimizer bookkeeping.
        scaler.update(new_scale=old * scaler.get_backoff_factor())
    else:
        scaler.step(optimizer); scaler.update()
    bad = not overflow and any(not torch.isfinite(p).all() for p in parameters.values())
    if parallel:
        parallel.check('Nonfinite parameter after optimizer update' if bad else None)
    elif bad:
        raise FloatingPointError('Nonfinite parameter after optimizer update')
    consecutive = consecutive + 1 if overflow else 0
    record = dict(overflow=overflow, step_performed=not overflow,
                  previous_scale=old, new_scale=float(scaler.get_scale()),
                  missing_gradient=missing, nonfinite_gradient=nonfinite, consecutive=consecutive)
    if consecutive >= config.get('max_consecutive_overflows', 5) or record['new_scale'] < config.get('amp_min_scale', 1):
        raise FloatingPointError(f'Repeated overflow or minimum scale: {record}')
    return record

def capture_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)

def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)

@contextmanager
def stop_requests():
    """Finish the current update and atomically save on Ctrl+C; do not start dev."""
    request = {'signal': None}
    previous = {}
    def handler(number, frame):
        request['signal'] = number
        emit('STOP_REQUESTED', signal=number, action='save at next optimizer boundary')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[sig] = signal.signal(sig, handler)
    try:
        yield request
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)

def metrics(labels, probabilities):
    from sklearn.metrics import roc_curve, log_loss
    result = binary_metrics(labels, probabilities)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    index = np.argmin(np.abs(fpr - (1 - tpr)))
    result.update(eer=float((fpr[index] + 1 - tpr[index]) / 2),
                  log_loss=float(log_loss(labels, probabilities, labels=[0, 1])))
    return result

def better(candidate, best):
    return best is None or candidate['roc_auc'] > best['roc_auc'] or (
        candidate['roc_auc'] == best['roc_auc'] and candidate['eer'] < best['eer'])

class Parallel:
    """Small explicit collective boundary; object/error traffic uses CPU Gloo."""
    def __init__(self, control):
        import torch.distributed as dist
        self.dist = dist; self.control = control
        self.rank = dist.get_rank(); self.world = dist.get_world_size()

    def collect(self, value):
        values = [None] * self.world
        self.dist.all_gather_object(values, value, group=self.control)
        return values

    def check(self, error):
        errors = self.collect(error)
        if any(x is not None for x in errors):
            raise RuntimeError(f'Coordinated rank error: {errors}')

    def call(self, function):
        result = None; error = None
        try:
            result = function()
        except Exception:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
        self.check(error)
        return result

    def root(self, function):
        result = self.call(lambda: function() if self.rank == 0 else None)
        return self.collect(result)[0]

    def buffers(self, model):
        for buffer in model.buffers():
            self.dist.broadcast(buffer, src=0)

def draw_shard(draws, rank, world, offset=0):
    if len(draws) % world or not 0 <= rank < world:
        raise ValueError('Training draws must divide evenly across devices')
    local = draws[rank::world]
    if not 0 <= offset <= len(local):
        raise ValueError('Invalid per-rank sampler offset')
    return local[offset:]

def save_ddp(path, model, optimizer, scheduler, scaler, position, provenance, device, parallel):
    parallel.buffers(model)
    states = parallel.collect(dict(rng=capture_rng(device), position=dict(position)))
    def write():
        payload = dict(format='x3a_ddp_v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), provenance=provenance,
            rank_states=states, world_size=parallel.world)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        emit('CHECKPOINT', path=path, epoch=position['epoch'], offset=position['offset'], successful=position['successful'])
    parallel.root(write)

def load_ddp(path, model, optimizer, scheduler, scaler, provenance, device, parallel):
    def read():
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('format') != 'x3a_ddp_v1' or state.get('world_size') != parallel.world:
            raise ValueError('Only matching unified 4-GPU checkpoints can resume')
        migration = validate_ddp_provenance(state, provenance, path)
        if migration:
            emit('SOURCE_MIGRATION', rank=parallel.rank, **migration)
        positions=[r['position'] for r in state['rank_states']]
        for key in ('epoch','offset','epoch_complete','successful','attempts'):
            if len({p[key] for p in positions}) != 1:
                raise ValueError('Checkpoint rank cursor mismatch')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        local=state['rank_states'][parallel.rank]
        restore_rng(local['rng'],device)
        return local['position']
    return parallel.call(read)

def train_distributed(name, args, run_dir, checkpoint_dir, device, parallel):
    from torch.nn.parallel import DistributedDataParallel
    config_path=ROOT/f'avdf1m_fav_reproduction/configs/{name}.json'
    task='video' if name=='x3d' else 'audio'
    report=run_dir/name; outdir=checkpoint_dir/name
    parallel.root(lambda:(report.mkdir(exist_ok=True),outdir.mkdir(exist_ok=True)))
    emit('PREPARE',rank=parallel.rank,model=name)
    config,train,dev,factory,collate,provenance=parallel.call(lambda:prepare(config_path,ROOT,task,device))
    t=config['training'];batch_size=t['batch_size'];accum=t['gradient_accumulation_steps']
    provenance['execution']=dict(mode='ddp',world_size=parallel.world,physical_devices=[0,1,2,3],
        per_device_batch=batch_size,accumulation=accum,effective_batch=batch_size*accum*parallel.world,
        learning_rate=t['optimizer']['learning_rate'])
    raw=parallel.call(lambda:factory().to(device))
    # Explicit buffer sync at dev/save boundaries; dev forward never uses DDP.
    model=DistributedDataParallel(raw,device_ids=[device.index] if device.type == 'cuda' else None,
        broadcast_buffers=False, find_unused_parameters=(name == 'aasist'))
    optimizer,scheduler,scaler=make_optimizer(raw,config,device)
    # Different dropout RNG streams, but identical initialized model parameters.
    torch.manual_seed(config['experiment']['seed']+parallel.rank)
    position=dict(epoch=0,offset=0,epoch_complete=False,successful=0,attempts=0,
        overflows=0,consecutive=0,best=None,best_epoch=None,loss_sum=0.,loss_count=0,labels={'0':0,'1':0})
    recovery=outdir/'recovery.pt'
    resume_stage=args.resume and not parallel.root(lambda:args.model=='all' and not any(report.iterdir()) and not any(outdir.iterdir()))
    if resume_stage and (report/'COMPLETED.json').exists():
        emit('ALREADY_COMPLETE', model=name, rank=parallel.rank)
        return
    if resume_stage:
        position=load_ddp(recovery,raw,optimizer,scheduler,scaler,provenance,device,parallel)
        emit('RESUME',rank=parallel.rank,**position)
    parallel.root(lambda:atomic_json(report/'run_manifest.json',provenance))
    last_print=[0.]
    def progress(stage,current,total,**fields):
        now=time.monotonic()
        if current not in (0,1,total) and now-last_print[0]<args.log_seconds:return
        last_print[0]=now
        status=dict(stage=stage,rank=parallel.rank,epoch=position['epoch'],batch=current,total=total,
            offset=position['offset'],successful=position['successful'],attempts=position['attempts'],
            scale=scaler.get_scale(),timestamp=time.time(),**fields)
        atomic_json(report/f'heartbeat_rank{parallel.rank}.json',status)
        emit(stage.upper(),**{k:v for k,v in status.items() if k!='stage'})
    def save(path):save_ddp(path,raw,optimizer,scheduler,scaler,position,provenance,device,parallel)
    with stop_requests() as stop:
        try:
            if not resume_stage:save(recovery)
            if position['epoch_complete']:
                position.update(epoch=position['epoch']+1,offset=0,epoch_complete=False,
                    loss_sum=0.,loss_count=0,labels={'0':0,'1':0})
            labels=[r['labels'][1 if task=='video' else 2] for r in train.rows]
            for epoch in range(position['epoch'],t['epochs']):
                position['epoch']=epoch
                draws=epoch_draws(labels,t['sampler']['global_epoch_samples'],config['experiment']['seed'],epoch)
                local=draw_shard(draws,parallel.rank,parallel.world,position['offset'])
                loader=DataLoader(Subset(train,local),batch_size=batch_size,collate_fn=collate,num_workers=0,
                    generator=torch.Generator().manual_seed(42+epoch))
                model.train();optimizer.zero_grad(set_to_none=True);pending=0
                progress('train',0,len(loader))
                iterator=iter(loader)
                for i in range(len(loader)):
                    batch=parallel.call(lambda:next(iterator))
                    sync=(i+1)%accum==0 or i+1==len(loader)
                    with nullcontext() if sync else model.no_sync():
                        def forward():
                            with torch.autocast(device.type,dtype=torch.float16,enabled=t['mixed_precision']):
                                out,y,ids=batch_forward(model,batch,device,task)
                                loss=(torch.nn.functional.binary_cross_entropy_with_logits(out.logits,y.float())
                                      if task=='video' else torch.nn.functional.cross_entropy(out.logits,y.long()))
                            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                            return loss,y
                        loss,y=parallel.call(forward)
                        group_samples=min(accum*batch_size,len(local)-(i//accum)*accum*batch_size)
                        scaler.scale(loss*(len(y)/group_samples)).backward()
                    pending+=len(y);position['loss_sum']+=float(loss.detach())*len(y);position['loss_count']+=len(y)
                    for label in y.tolist():position['labels'][str(label)]+=1
                    if sync:
                        record=optimizer_update(raw,optimizer,scaler,t,position['consecutive'],parallel)
                        decisions=parallel.collect((record['step_performed'],record['new_scale']))
                        parallel.check(None if len(set(decisions))==1 else 'Optimizer/AMP decision divergence')
                        position['attempts']+=1;position['successful']+=int(record['step_performed'])
                        position['overflows']+=int(record['overflow']);position['consecutive']=record['consecutive']
                        position['offset']+=pending;pending=0;optimizer.zero_grad(set_to_none=True)
                        def write_update():
                            with (report/f'updates_rank{parallel.rank}.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(record,epoch=epoch,offset=position['offset'],
                                    successful=position['successful'],attempt=position['attempts'],loss=float(loss.detach())))+'\n')
                        parallel.call(write_update)
                        if record['overflow'] or record['missing_gradient']:emit('GRADIENT',rank=parallel.rank,**record)
                        stopping=any(parallel.collect(stop['signal'] is not None))
                        if stopping or (record['step_performed'] and position['successful']%t.get('recovery_interval_steps',200)==0):save(recovery)
                        if stopping:raise InterruptedError('User requested coordinated stop; recovery saved')
                    parallel.call(lambda:progress('train',i+1,len(loader),loss=float(loss.detach())))
                save(recovery)
                # No per-batch collective: uneven dev shards safely finish independently.
                dev_subset=Subset(dev,list(range(parallel.rank,len(dev),parallel.world)))
                dev_loader=DataLoader(dev_subset,batch_size=batch_size,collate_fn=collate,num_workers=0,
                    generator=torch.Generator().manual_seed(0))
                raw.eval();records=[]
                def local_eval():
                    with torch.no_grad():
                        for i,batch in enumerate(dev_loader):
                            if stop['signal']:raise InterruptedError('Stop requested during dev; train-end recovery retained')
                            with torch.autocast(device.type,dtype=torch.float16,enabled=t['mixed_precision']):
                                output,y,ids=batch_forward(raw,batch,device,task)
                            records.extend(dict(sample_id=s,split='dev',label=l,logit=z,fake_probability=p,checkpoint_epoch=epoch)
                                for s,l,z,p in zip(ids,y.tolist(),output.logits.float().tolist(),output.fake_probability.float().tolist()))
                            progress('dev',i+1,len(dev_loader))
                parallel.call(local_eval)
                parts=parallel.collect(records);stats=parallel.collect(dict(position))
                def score():
                    rows=[r for part in parts for r in part]
                    validate_predictions(rows,[r['sample_id'] for r in dev.rows])
                    result=metrics([r['label'] for r in rows],[r['fake_probability'] for r in rows])
                    improved=better(result,position['best'])
                    pred=''.join(json.dumps(r)+'\n' for r in sorted(rows,key=lambda r:r['sample_id']))
                    (report/f'dev_{epoch:03d}.jsonl').write_text(pred)
                    if improved:(report/'best_dev.jsonl').write_text(pred)
                    count=sum(p['loss_count'] for p in stats)
                    atomic_json(report/f'epoch_{epoch:03d}.json',dict(epoch=epoch,dev=result,
                        train_loss=sum(p['loss_sum'] for p in stats)/count if count else None,
                        sampled_labels={key:sum(p['labels'][key] for p in stats) for key in ('0','1')},
                        rank_draws=[p['offset'] for p in stats],draw_position_overlap=0,
                        successful=position['successful'],attempts=position['attempts'],overflows=position['overflows']))
                    return result,improved
                result,improved=parallel.root(score)
                if improved:position['best']=result;position['best_epoch']=epoch
                scheduler.step();position['epoch_complete']=True
                save(outdir/'last.pt')
                if improved:save(outdir/'best.pt')
                save(recovery)
                if parallel.rank==0:emit('EPOCH_COMPLETE',epoch=epoch,**result)
                position.update(epoch=epoch+1,offset=0,epoch_complete=False,loss_sum=0.,loss_count=0,labels={'0':0,'1':0})
            def complete():
                hashes={p.name:digest(p) for p in (outdir/'last.pt',outdir/'best.pt',recovery)}
                atomic_json(report/'COMPLETED.json',dict(epochs=t['epochs'],best_epoch=position['best_epoch'],sha256=hashes))
                atomic_json(report/'run_status.json',dict(status='completed',completed=True,test_accessed=False))
                emit('COMPLETED',model=name,sha256=hashes)
            parallel.root(complete)
        except BaseException:
            text=traceback.format_exc();print(text,file=sys.stderr,flush=True)
            (report/f'error_rank{parallel.rank}.log').write_text(text)
            atomic_json(report/f'status_rank{parallel.rank}.json',dict(status='stopped_or_failed',position=position,completed=False))
            # Do not enter new NCCL collectives from a lone exception path.
            # Last healthy optimizer-boundary recovery remains untouched.
            raise
    del model,raw,optimizer,scheduler,scaler
    torch.cuda.empty_cache()


def single_gpu_smoke(args, device):
    """Disposable actual-media optimizer/inference check; never initializes a full run."""
    for name in (('x3d','aasist') if args.model == 'all' else (args.model,)):
        task = 'video' if name == 'x3d' else 'audio'
        cfg, train, dev, factory, collate, provenance = prepare(
            ROOT/f'avdf1m_fav_reproduction/configs/{name}.json', ROOT, task, device)
        model = factory().to(device)
        t = cfg['training']
        optimizer, scheduler, scaler = make_optimizer(model, cfg, device)
        loader = DataLoader(train, batch_size=t['batch_size'], collate_fn=collate,
                            generator=torch.Generator().manual_seed(42))
        model.train(); optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        for i, batch in enumerate(loader):
            with torch.autocast(device.type, dtype=torch.float16, enabled=t['mixed_precision']):
                out, y, ids = batch_forward(model, batch, device, task)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(out.logits, y.float()) if task == 'video'
                        else torch.nn.functional.cross_entropy(out.logits, y.long()))
            scaler.scale(loss / t['gradient_accumulation_steps']).backward()
            if i+1 == t['gradient_accumulation_steps']:
                record = optimizer_update(model, optimizer, scaler, t, 0)
                if not record['step_performed']:
                    raise RuntimeError('Single GPU smoke optimizer update overflowed')
                break
        model.eval()
        batch = next(iter(DataLoader(dev, batch_size=1, collate_fn=collate)))
        with torch.no_grad(), torch.autocast(device.type,dtype=torch.float16,enabled=t['mixed_precision']):
            output, _, _ = batch_forward(model,batch,device,task)
        result = dict(model=name, loss=float(loss.detach()), update=record,
                      dev_probability=output.fake_probability.tolist(), seconds=time.monotonic()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        atomic_json(args.output/f'single_gpu_{name}.json', result)
        emit('SINGLE_GPU_SMOKE_PASSED', **result)
        del model, optimizer, scheduler, scaler, loader, out, output, loss, batch
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def main():
    import avdf1m_fav_reproduction.runtime as runtime
    from datetime import timedelta
    import torch.distributed as dist
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['x3d','aasist','all'], default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--single-gpu-smoke', action='store_true')
    parser.add_argument('--log-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.log_seconds <= 0:
        parser.error('Positive log interval required')
    runtime.PREFLIGHT = args.preflight.resolve()
    runtime.PRETRAINED = args.pretrained.resolve()
    runtime.SMOKE = args.smoke or args.single_gpu_smoke
    global AUDIO_TARGET
    AUDIO_TARGET = json.loads(args.preflight.read_text())['audio']['target_num_samples']
    if args.single_gpu_smoke:
        args.output.mkdir(parents=True, exist_ok=False)
        return single_gpu_smoke(args, torch.device('cuda:0'))
    if os.environ.get('WORLD_SIZE') != '4' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('Requires CUDA_VISIBLE_DEVICES=0,1,2,3 and torchrun --nproc_per_node=4')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); device = torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=300))
    control = dist.new_group(backend='gloo',timeout=timedelta(seconds=300))
    parallel = Parallel(control)
    run_dir, checkpoint_dir = args.output/'reports', args.output/'checkpoints'
    try:
        def directories():
            if not args.resume and args.output.exists():
                raise FileExistsError('Use a new output directory or explicit --resume')
            if args.resume and not run_dir.is_dir():
                raise FileNotFoundError('Resume run missing')
            run_dir.mkdir(parents=True,exist_ok=True); checkpoint_dir.mkdir(parents=True,exist_ok=True)
            if not args.resume:
                atomic_json(args.output/'launch.json',dict(argv=sys.argv, command_line=' '.join(sys.argv),
                            visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],started=time.time(),smoke=args.smoke))
        parallel.root(directories)
        for name in (('x3d','aasist') if args.model=='all' else (args.model,)):
            train_distributed(name,args,run_dir,checkpoint_dir,device,parallel)
        def export():
            exports = {}
            for name in (('x3d','aasist') if args.model=='all' else (args.model,)):
                path = checkpoint_dir/name/'best.pt'
                state = torch.load(path,map_location='cpu',weights_only=False)
                target = checkpoint_dir/name/'best_state_dict.pth'
                torch.save(state['model'],target)
                exports[name] = dict(best_checkpoint_sha256=digest(path), state_dict_sha256=digest(target),
                                     best_epoch=state['rank_states'][0]['position']['best_epoch'])
            atomic_json(args.output/'COMPLETED.json',dict(exports=exports,test_accessed=False,smoke=args.smoke))
        parallel.root(export)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
