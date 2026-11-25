"""Frozen source artifact + pretrained emotion features; long audio is never cut."""
import numpy as np
import torch
from .cache import FeatureCache
from .data import cohort_id
from .common import EMOTION_DIMS, PROTOCOL, Progress, sha
from scripts.artifact_emotion_gated_attetnion_optimized.fast_video import shared_video, batched_tokens
from scripts.artifact_emotion_gated_attetnion_optimized.features import auditory_tokens, pool_tokens


class Extractor:
    def __init__(self, artifact, modality, device, face_batch_size=1):
        self.artifact = artifact.to(device).eval().requires_grad_(False)
        self.modality, self.device, self.face_batch_size = modality, device, face_batch_size
        if modality == 'video':
            import cv2
            from pathlib import Path
            from src.models.hsemotion import EmotionExtractor
            self.emotion = EmotionExtractor(device)
            self.emotion.recognizer.model.requires_grad_(False)
            self.detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'))
            if self.detector.empty():
                raise FileNotFoundError('Haar face detector missing')
            cv2.setNumThreads(1)
        else:
            from src.utils.local_paths import load_local_paths
            from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import download_seed, FrozenEmotion2VecSeed
            if not load_local_paths().get('emotion2vec_snapshot'):
                raise ValueError('Install the pinned local emotion2vec seed first')
            snapshot, _ = download_seed()
            self.emotion = FrozenEmotion2VecSeed(snapshot, device)

    @torch.inference_mode()
    def sample(self, ds, index):
        r = ds.records[index]; dim = EMOTION_DIMS[self.modality]
        if self.modality == 'video':
            data = shared_video(r.media_path, ds.config, self.detector)
            pred = self.artifact(data['video'][None].to(self.device))
            tokens = batched_tokens(data, self.emotion, dim, self.face_batch_size)
        else:
            item = ds[index]; wave = item['audio']
            padded = torch.nn.functional.pad(wave, (0, max(0, 284672-len(wave))))
            pred = self.artifact(padded[None].to(self.device))
            tokens = auditory_tokens(wave.numpy(), self.emotion, dim)
        a = pred.features[0].cpu().numpy().astype(np.float32)
        e = pool_tokens(tokens['embedding'], tokens['valid'], dim)
        p = float(pred.fake_probability[0].cpu())
        if not np.isfinite(a).all() or not np.isfinite(e).all() or not 0 <= p <= 1:
            raise ValueError('Invalid extracted feature/probability: '+r.sample_id)
        return dict(artifact=a, emotion=e, emotion_valid=np.asarray(tokens['valid'].any()),
                    artifact_probability=np.asarray(p, dtype=np.float32),
                    valid_tokens=np.asarray(tokens['valid'].sum()), total_tokens=np.asarray(len(tokens['valid'])))

    def extract(self, ds, path, identity):
        cache = FeatureCache(path, dict(identity, cohort=cohort_id(ds)), ds.records)
        progress = Progress(f'EXTRACT {ds.records[0].dataset} {ds.split} {self.modality}', len(ds))
        reused = 0
        try:
            for i, r in enumerate(ds.records):
                if cache.get(r) is not None:
                    reused += 1
                else:
                    stat = r.media_path.stat(); before = (stat.st_size, stat.st_mtime_ns)
                    cache.put(r, self.sample(ds, i), before)
                progress.update(i+1, f'resumed={reused}')
            return cache
        except BaseException:
            cache.close()
            raise
