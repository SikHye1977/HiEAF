"""Join new MSOC results to preserved manuscript results, checking exact IDs/labels."""
import csv
import io
from pathlib import Path
import numpy as np
from .common import ROOT, HERE, atomic_file, atomic_json, digest, read_json, require, sha
from .paths import configuration


def cohort_pairs(path):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            pairs = [(str(sid), int(label)) for sid, label in zip(z['sample_id'], z['labels'][:, 0])]
    else:
        with path.open(encoding='utf-8-sig', newline='') as f:
            pairs = [(r['sample_id'], int(r['clip_label'])) for r in csv.DictReader(f)]
    require(len({sid for sid, _ in pairs}) == len(pairs), 'Duplicate comparison prediction IDs')
    return digest(sorted(pairs))


def load_reference(data_config=None):
    pin = read_json(HERE/'reference_portable.lock.json')
    path = configuration(data_config)['metadata_root'] / pin['path']
    require(sha(path) == pin['sha256'], 'Portable historical reference changed')
    reference = read_json(path)
    require(reference['format'] == 'msoc_hieaf_reference_v1', 'Unknown historical reference format')
    require(len(reference['rows']) == 54, 'Expected preserved 54-row reference')
    return pin, reference


def verify_reference(cohorts, data_config=None):
    pin, reference = load_reference(data_config)
    expected = {ds: digest(sorted((r['sample_id'], r['clip_label']) for r in parts['test']))
                for ds, parts in cohorts.items()}
    for ds, parts in cohorts.items():
        require(reference['cohorts'][ds] == dict(n=len(parts['test']), sha256=expected[ds]),
                'Historical and current test IDs/labels differ: '+ds)
    print('[REFERENCE] Preserved 54 rows: all nine test cohorts match exact IDs/clip labels.', flush=True)
    return dict(reference=pin, metric_files=len(reference['evidence']), exact_test_cohorts=expected)


def compare(output, data_config=None):
    output = Path(output)
    pin, reference = load_reference(data_config)
    historical = reference['rows']
    require(len(historical) == 54, 'Expected preserved 54-row reference')
    proof = reference['evidence']
    method_names = dict(baseline='Baseline', artifact_affine='an artifact-only baseline with learned score adjustment', proposed='HiEAF')
    old_hashes = {k: v['metrics_sha256'] for k, v in proof.items()}
    table, deltas = [], []
    for result_path in sorted(output.glob('source_*/test/*/metrics.json')):
        folder = result_path.parent
        if not (folder/'COMPLETE.json').exists():
            continue
        receipt = read_json(folder/'COMPLETE.json')
        require(sha(result_path) == receipt['metrics_sha256'] and sha(folder/'predictions.csv') == receipt['predictions_sha256'], 'New result changed')
        current = read_json(result_path)
        source, target = folder.parents[1].name.removeprefix('source_'), folder.name
        now_cohort = cohort_pairs(folder/'predictions.csv')
        old = [r for r in historical if r['source'] == source and r['target'] == target]
        require(len(old) == 6, 'Expected two conditions times three historical methods')
        for row in old:
            require(reference['cohorts'][target] == dict(n=current['n'], sha256=now_cohort)
                    and row['n'] == current['n'], 'Comparison IDs/clip labels differ')
            display = {'B': 'A (stored X3D BN)', 'C': 'B (per-video X3D BN)'}[row['condition']]
            table.append(dict(source=source, target=target, method=method_names[row['method']], condition=display,
                              n=row['n'], **{k: row[k] for k in ('roc_auc', 'fake_f1', 'real_fpr')}))
            if row['method'] == 'proposed':
                deltas.append(dict(source=source, target=target, hieaf_condition=display,
                                  comparison='HiEAF minus MSOC (percentage points; lower FPR is better)',
                                  **{k: 100*(row[k]-current[k]) for k in ('roc_auc', 'fake_f1', 'real_fpr')}))
        # MSOC has native stored BN, not HiEAF's X3D-specific A/B interventions.
        table.append(dict(source=source, target=target, method=current['method'], condition='native stored BN / FP32',
                          n=current['n'], **{k: current[k] for k in ('roc_auc', 'fake_f1', 'real_fpr')}))
    require(bool(table), 'No completed MSOC evaluations available yet')
    atomic_json(output/'COMPARISON_WITH_HIEAF.json', dict(rows=table, deltas=deltas,
                historical_reference=pin, historical_metric_hashes=old_hashes,
                exact_test_ids_and_clip_labels_verified=True,
                note='Point estimates only. Same fixed cohorts; model-specific inputs differ. MSOC ResNet uses prefix 30 frames; no new statistical superiority test.'))
    text = io.StringIO(newline='')
    writer = csv.writer(text)
    writer.writerow(['Source', 'Target', 'Method', 'Condition', 'n', 'ROC-AUC (%)', 'Fake F1 (%)', 'Real FPR (%)'])
    for r in table:
        writer.writerow([r[k] for k in ('source', 'target', 'method', 'condition', 'n')]
                        +[100*r[k] for k in ('roc_auc', 'fake_f1', 'real_fpr')])
    with atomic_file(output/'COMPARISON_WITH_HIEAF.csv') as f:
        f.write(text.getvalue().encode('utf-8-sig'))
    print(f'[COMPARE] Wrote {len(table)} rows; exact test IDs/labels matched; historical results unchanged.', flush=True)
