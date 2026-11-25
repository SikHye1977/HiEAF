"""Aggregate only matching-seed completed directional results; never pool test samples."""
import csv
from pathlib import Path
from .common import read, sha, atomic_json, DATASETS


def summarize(runs, output):
    rows = []; seeds = set(); pairs = set(); smoke = set()
    for folder in runs:
        folder = Path(folder)
        run = read(folder/'RUN.json'); complete = read(folder/'COMPLETED.json')
        seed = run['options']['seed']; seeds.add(seed); smoke.add(run['smoke_only'])
        for target in complete['targets']:
            result = read(folder/'evaluation'/target/'metrics.json')
            marker = read(folder/'evaluation'/target/'COMPLETED.json')
            if marker['metrics_sha256'] != sha(folder/'evaluation'/target/'metrics.json') or marker['lock_sha256'] != sha(folder/'LOCKED_BEFORE_TEST.json') or marker['predictions_sha256'] != sha(folder/'evaluation'/target/'predictions.csv'):
                raise ValueError('Evaluation files changed')
            pair = (result['source'], target)
            if pair in pairs or pair[0] == pair[1]:
                raise ValueError('Duplicate/non-cross direction')
            pairs.add(pair)
            rows.append(dict(source=pair[0], target=target, seed=seed, n=result['gated_or']['n'],
                             gated_auc=result['gated_or']['roc_auc'], x3a_auc=result['x3a_or']['roc_auc'],
                             delta_auc_pp=result['delta_auc_pp'],
                             gated_ba=result['gated_or']['balanced_accuracy'], x3a_ba=result['x3a_or']['balanced_accuracy']))
    if len(seeds) != 1 or len(smoke) != 1:
        raise ValueError('Aggregate a single seed and a single smoke/full protocol at a time')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    expected = {(s,t) for s in DATASETS for t in DATASETS if s!=t}
    complete_six = pairs == expected and not next(iter(smoke))
    valid = [r for r in rows if r['delta_auc_pp'] is not None]
    summary = dict(complete_six_directions=complete_six, smoke_only=next(iter(smoke)),
                   directions=rows, missing_directions=sorted(expected-pairs),
                   observed_direction_macro_delta_auc_pp=sum(r['delta_auc_pp'] for r in valid)/len(valid) if valid else None)
    atomic_json(output/'matrix.json', summary)
    with (output/'matrix.csv').open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return summary
