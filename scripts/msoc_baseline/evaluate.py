import csv
import io
from pathlib import Path
import time
import torch
from .common import atomic_file, atomic_json, bind, digest, read_json, require, sha, Progress
from .cache import CachedDataset, db_open, loader, tensors
from .model import MSOC
from .train import load_checkpoint, metrics, setup_device


@torch.inference_mode()
def evaluate(folder, checkpoint, cache, rows, args, run_identity, stop):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(run=run_identity, checkpoint=sha(checkpoint), cohort=digest(rows), precision='fp32',
                    inference='official mean; real cosine gates >0.5; fake=1-real; threshold >=0.5')
    bind(folder/'IDENTITY.json', identity)
    if (folder/'COMPLETE.json').exists():
        receipt = read_json(folder/'COMPLETE.json')
        require(receipt['identity'] == identity and receipt['predictions_sha256'] == sha(folder/'predictions.csv')
                and receipt['metrics_sha256'] == sha(folder/'metrics.json'), 'Completed evaluation changed')
        print(f'[EVALUATE] Reusing {folder}', flush=True)
        return read_json(folder/'metrics.json')
    device = setup_device(args, inference=True)
    ck = load_checkpoint(checkpoint)
    require(ck['identity']['run'] == run_identity, 'Wrong source checkpoint')
    model = MSOC(args.backbone).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    del ck
    db = db_open(folder/'scores.sqlite')
    try:
        db.execute('CREATE TABLE IF NOT EXISTS scores(sid TEXT PRIMARY KEY, fake REAL, video_cos REAL, audio_cos REAL, av_real REAL)')
        db.commit()
        existing = {x[0] for x in db.execute('SELECT sid FROM scores')}
        require(existing <= {r['sample_id'] for r in rows}, 'Unexpected prediction IDs')
        order = [i for i, r in enumerate(rows) if r['sample_id'] not in existing]
        dataset = CachedDataset(cache, rows, 'test', order=order)
        source_name = folder.parents[1].name.removeprefix('source_')
        progress = Progress(f'TEST {source_name}->{folder.name} FP32', len(rows), len(existing), folder/'STATUS.json')
        done = len(existing)
        try:
            for batch in loader(dataset, args.eval_batch, args.workers):
                stop.check()
                video, audio, labels, ids = tensors(batch, device)
                # Clip evaluation never needs unknown test modality labels.
                scores = model.scores(model(video, audio)).cpu()
                require(bool(torch.isfinite(scores).all()), 'Nonfinite test predictions')
                db.executemany('INSERT INTO scores VALUES (?,?,?,?,?)', [(sid, *s) for sid, s in zip(ids, scores.tolist())])
                db.commit()
                done += len(ids)
                progress.update(done)
        finally:
            dataset.close()
        predictions = {r[0]: r[1:] for r in db.execute('SELECT * FROM scores')}
        require(set(predictions) == {r['sample_id'] for r in rows}, 'Prediction coverage mismatch')
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        db.close()
    result = metrics([r['clip_label'] for r in rows], [predictions[r['sample_id']][0] for r in rows])
    result.update(method=f'MSOC-{args.backbone}', inference='FP32 stored model BN',
                  unknown_modality_labels=sum(r['video_label'] is None or r['audio_label'] is None for r in rows),
                  checkpoint_sha256=identity['checkpoint'], cohort_identity=identity['cohort'])
    atomic_json(folder/'metrics.json', result)
    text = io.StringIO(newline='')
    writer = csv.writer(text)
    writer.writerow(['sample_id', 'group_id', 'clip_label', 'video_label', 'audio_label', 'fake_score',
                     'video_real_cosine', 'audio_real_cosine', 'av_real_probability'])
    for row in rows:
        writer.writerow([row[k] for k in ('sample_id', 'group_id', 'clip_label', 'video_label', 'audio_label')]
                        + list(predictions[row['sample_id']]))
    with atomic_file(folder/'predictions.csv') as f:
        f.write(text.getvalue().encode('utf-8'))
    atomic_json(folder/'COMPLETE.json', dict(identity=identity, predictions_sha256=sha(folder/'predictions.csv'),
                                           metrics_sha256=sha(folder/'metrics.json')))
    print(f'[TEST COMPLETE] {folder} {result}', flush=True)
    return result


def summarize(output):
    rows = []
    for p in sorted(Path(output).glob('source_*/test/*/metrics.json')):
        if not (p.parent/'COMPLETE.json').exists():
            continue
        r = read_json(p)
        source, target = p.parents[2].name.removeprefix('source_'), p.parent.name
        rows.append(dict(source=source, target=target, **r))
    atomic_json(Path(output)/'SUMMARY.json', rows)
    text = io.StringIO(newline='')
    writer = csv.writer(text)
    writer.writerow(['source', 'target', 'method', 'n', 'ROC-AUC (%)', 'Fake F1 (%)', 'Real FPR (%)'])
    for r in rows:
        writer.writerow([r['source'], r['target'], r['method'], r['n']]+[100*r[k] if r[k] is not None else '' for k in ('roc_auc', 'fake_f1', 'real_fpr')])
    with atomic_file(Path(output)/'SUMMARY.csv') as f:
        f.write(text.getvalue().encode('utf-8-sig'))
    return rows
