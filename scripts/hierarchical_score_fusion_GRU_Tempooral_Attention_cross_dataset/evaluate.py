"""Same-ID zero-shot comparisons, with unknown labels and missing emotions explicit."""
import csv
import os
from pathlib import Path
import numpy as np
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import (
    require, metrics, score_or, atomic_json, sha, Progress, status)
from .inputs import labels_of
from .inference import ScoreStore, validate_trace


def metric_or_empty(y, p):
    if not len(y):
        return dict(n=0, real=0, fake=0, roc_auc=None, average_precision=None,
                    balanced_accuracy=None, real_fpr=None, confusion_matrix=[[0,0],[0,0]], threshold=.5)
    return metrics(y, p)


def summarize(labels, av, aa, ev, ea, gated):
    labels = np.asarray(labels, np.int64)
    validv, valida = np.isfinite(ev), np.isfinite(ea)
    complete = validv & valida
    artifact = score_or(av, aa)
    emotion = score_or(np.where(validv, ev, 0.), np.where(valida, ea, 0.))
    final = score_or(artifact, emotion)
    require(np.all(final+1e-12 >= artifact), 'OR fusion unexpectedly lowered artifact scores')
    scores = dict(artifact_or=artifact, emotion_or=emotion, hierarchical_or=final, previous_gated_or=gated)
    known = labels[:,0] >= 0
    def cohort(mask):
        result = {name: metric_or_empty(labels[mask,0], p[mask]) for name, p in scores.items()}
        new, old = result['hierarchical_or'], result['artifact_or']
        result['delta_vs_artifact_pp'] = {k: 100*(new[k]-old[k]) if new[k] is not None and old[k] is not None else None
                                         for k in ('roc_auc', 'balanced_accuracy', 'real_fpr')}
        result['paired_changes_vs_artifact'] = dict(
            rescued_fake=int(np.sum(mask & (labels[:,0] == 1) & (artifact < .5) & (final >= .5))),
            added_real_false_positives=int(np.sum(mask & (labels[:,0] == 0) & (artifact < .5) & (final >= .5))))
        return result
    branches = {}
    for name, ix, p, valid in [('video_emotion',1,ev,validv), ('audio_emotion',2,ea,valida),
                                ('x3d',1,av,np.ones(len(av),bool)), ('aasist',2,aa,np.ones(len(aa),bool))]:
        mask = (labels[:,ix] >= 0) & valid
        branches[name] = dict(metric_or_empty(labels[mask,ix], p[mask]),
                             unknown_labels=int(np.sum(labels[:,ix] < 0)), missing_score=int(np.sum(~valid)))
    report = dict(primary_complete_emotion=cohort(known & complete), secondary_full_cohort=cohort(known),
                  branches=branches, coverage=dict(planned=len(labels), evaluated=int(known.sum()),
                    complete_emotion=int(complete.sum()), video_emotion_missing=int((~validv).sum()),
                    audio_emotion_missing=int((~valida).sum())))
    return report, scores, complete


def evaluate(data, caches, output, lock):
    records = data['records']; n = len(records)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    emotion = {}; token_counts = {}
    for mod, (folder, identity) in caches.items():
        store = ScoreStore(Path(folder)/'scores.sqlite', identity)
        try:
            require(store.ids() == {r.sample_id for r in records}, 'Incomplete inference scores')
            values = []; counts = []; bar = Progress('READ SCORES '+data['target']+' '+mod, n)
            for i, r in enumerate(records, 1):
                trace = store.get(r.sample_id, data['stamps'][r.sample_id]); validate_trace(trace, mod)
                values.append(float(trace['score'])); counts.append(int(trace['valid'].sum())); bar.update(i)
            emotion[mod] = np.asarray(values); token_counts[mod] = np.asarray(counts)
        finally:
            store.close()
    labels = np.array([labels_of(r) for r in records], np.int64)
    av, aa, gated = data['scores'].T; ev, ea = emotion['video'], emotion['audio']
    report, scores, complete = summarize(labels, av, aa, ev, ea, gated)
    report.update(source='FAV', target=data['target'], split='test', zero_shot=True, smoke_only=False, threshold=.5,
        fixed_artifact_scores=True, lock_sha256=sha(lock),
        checkpoint_selection='completed FAV dev modality AUC; no target fitting or tuning',
        target_status='previously evaluated by gated experiment; not an untouched holdout',
        fusion='A=OR(Av,Aa); E=OR(Ev,Ea); Final=OR(A,E)',
        missing_policy='primary: both emotion branches valid; secondary: omit missing terms; CSV score blank',
        token_coverage={mod: dict(valid_tokens=int(c.sum()), one_valid_token=int(np.sum(c==1)),
                                min=int(c.min()), median=float(np.median(c)), max=int(c.max())) for mod,c in token_counts.items()})
    names = ['sample_id','group','clip_label','video_label','audio_label','Av','Aa','Ev','Ea',
             'artifact_or','emotion_or','hierarchical_or','previous_gated_or','included_primary','video_valid_tokens','audio_valid_tokens']
    temp = output/'predictions.csv.partial'
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(names)
        for i, r in enumerate(records):
            writer.writerow([r.sample_id,r.group_id,*labels[i].tolist(),float(av[i]),float(aa[i]),
                float(ev[i]) if np.isfinite(ev[i]) else '',float(ea[i]) if np.isfinite(ea[i]) else '',
                *[float(scores[k][i]) for k in ('artifact_or','emotion_or','hierarchical_or','previous_gated_or')],
                bool(complete[i] and labels[i,0] >= 0),int(token_counts['video'][i]),int(token_counts['audio'][i])])
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, output/'predictions.csv')
    atomic_json(output/'metrics.json', report)
    atomic_json(output/'COMPLETED.json', dict(lock_sha256=sha(lock), files={n:sha(output/n) for n in ('metrics.json','predictions.csv')}))
    status('EVALUATE', f'{data["target"]} COMPLETE n={n} primary_n={int(complete.sum())}')
    return report
