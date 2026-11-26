"""Export verified historical metrics and exact test-cohort digests, no model inference.

Run on the evidence-holding PC only. The resulting sealed reference can be moved
without copying HiEAF weights, feature caches, or absolute-path result trees.
"""
import argparse
from pathlib import Path
from .common import ROOT, HERE, atomic_json, read_json, require, sha
from .compare import cohort_pairs


def export(destination):
    destination = Path(destination)
    pin = read_json(HERE/'reference.lock.json')
    require(sha(ROOT/pin['path']) == pin['sha256'], 'Historical reference changed')
    old = read_json(ROOT/pin['path'])
    rows = old['previous']['common_bc']['rows']
    require(len(rows) == 54, 'Expected 54 historical rows')
    evidence, cohorts = {}, {}
    for row in rows:
        p = Path(row['metrics_path'])
        p = p if p.is_absolute() else ROOT/p
        key = str(p)
        if key not in evidence:
            require(key in old['evidence'] and sha(p) == old['evidence'][key], 'Metric evidence changed: '+key)
            pp = p.parent/'predictions.csv'
            if not pp.is_file():
                pp = p.parent/'predictions.npz'
            signature = cohort_pairs(pp)
            evidence[key] = dict(metrics_sha256=sha(p), predictions_sha256=sha(pp),
                                 cohort_sha256=signature)
        info = dict(n=row['n'], sha256=evidence[key]['cohort_sha256'])
        require(row['target'] not in cohorts or cohorts[row['target']] == info, 'Historical test cohorts differ')
        cohorts[row['target']] = info
    portable = dict(format='msoc_hieaf_reference_v1', source_reference=pin,
                    rows=rows, cohorts=cohorts, evidence=evidence,
                    note='Export of preserved metrics with SHA256-verified evidence and test ID/clip-label digests; no predictions recomputed.')
    if destination.exists():
        require(read_json(destination) == portable, 'Existing reference differs; preserve it and choose another output')
    else:
        atomic_json(destination, portable)
    lock = dict(path='reports/msoc_reference/REFERENCE.json', sha256=sha(destination))
    lock_path = HERE/'reference_portable.lock.json'
    if lock_path.exists():
        require(read_json(lock_path) == lock, 'Existing portable reference pin differs')
    else:
        atomic_json(lock_path, lock)
    print(f'[REFERENCE EXPORT] PASS rows={len(rows)} metric_files={len(evidence)} path={destination}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'reports/msoc_reference/REFERENCE.json')
    export(p.parse_args().output)
