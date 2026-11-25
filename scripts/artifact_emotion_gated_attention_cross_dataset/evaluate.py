"""Paired same-ID X3A-versus-gated evaluation, with unknown-label masking."""
import csv
from pathlib import Path
import numpy as np
from .common import metrics, atomic_json, sha, read
from .encoders import load_head, score_or
from .train import predict


def paired_metrics(video, audio, v, a):
    vi = [r.sample_id for r in video.records]; ai = [r.sample_id for r in audio.records]
    if vi != ai or len(set(vi)) != len(vi):
        raise ValueError('Video/audio IDs or ordering differ')
    truth = np.array([[r.clip_label, -1 if r.video_label is None else r.video_label,
                       -1 if r.audio_label is None else r.audio_label] for r in video.records])
    other = np.array([[r.clip_label, -1 if r.video_label is None else r.video_label,
                       -1 if r.audio_label is None else r.audio_label] for r in audio.records])
    if not np.array_equal(truth, other):
        raise ValueError('Video/audio labels differ')
    va = np.array([float(video[i]['artifact_probability']) for i in range(len(video))])
    aa = np.array([float(audio[i]['artifact_probability']) for i in range(len(audio))])
    gated, baseline = score_or(v, a), score_or(va, aa)
    result = {'gated_or': metrics(truth[:,0], gated), 'x3a_or': metrics(truth[:,0], baseline)}
    for name, probability, ix in [('video',v,1),('audio',a,2),('x3d',va,1),('aasist',aa,2)]:
        known = truth[:,ix]>=0; result[name] = metrics(truth[known,ix], probability[known])
        result[name]['unknown_labels'] = int((~known).sum())
    result['delta_auc_pp'] = (result['gated_or']['roc_auc']-result['x3a_or']['roc_auc'])*100 if result['gated_or']['roc_auc'] is not None else None
    result['coverage'] = dict(planned=len(vi), evaluated=len(vi), missing_video=0, missing_audio=0,
                               video_emotion_missing=sum(not bool(video[i]['emotion_valid']) for i in range(len(video))),
                               audio_emotion_missing=sum(not bool(audio[i]['emotion_valid']) for i in range(len(audio))))
    return result, truth, va, aa, gated, baseline


def evaluate(video, audio, source, target, root, device, lock):
    if source == target:
        raise ValueError('Cross-dataset evaluation requires distinct source and target')
    probabilities = {}
    for modality, cache in [('video',video), ('audio',audio)]:
        path = root/'heads'/modality/'best.pt'
        if sha(path) != lock['heads'][modality]:
            raise ValueError('Locked head changed')
        model, state = load_head(path, source, modality, device)
        if state['artifact_sha256'] != lock['artifact'][modality]:
            raise ValueError('Head was trained with a different artifact encoder')
        probabilities[modality], _ = predict(model, cache, modality, device, head=True)
    result, truth, va, aa, gated, baseline = paired_metrics(video, audio, probabilities['video'], probabilities['audio'])
    result.update(source=source, target=target, split='test', threshold=.5, smoke_only=lock['smoke_only'])
    folder = root/'evaluation'/target; folder.mkdir(parents=True, exist_ok=True)
    temp = folder/'predictions.csv.partial'
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id','group_id','clip_label','video_label','audio_label','video_score','audio_score','x3d_score','aasist_score','gated_or','x3a_or'])
        for i, r in enumerate(video.records):
            writer.writerow([r.sample_id,r.group_id,*truth[i],probabilities['video'][i],probabilities['audio'][i],va[i],aa[i],gated[i],baseline[i]])
    temp.replace(folder/'predictions.csv')
    atomic_json(folder/'metrics.json', result)
    atomic_json(folder/'COMPLETED.json', dict(lock_sha256=sha(root/'LOCKED_BEFORE_TEST.json'), predictions_sha256=sha(folder/'predictions.csv'), metrics_sha256=sha(folder/'metrics.json')))
    return result
