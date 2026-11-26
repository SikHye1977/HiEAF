"""Bounded synchronized prefix decoding; no full video/audio extraction to disk."""
import io
import json
import zlib
import av
import cv2
import numpy as np
from python_speech_features import mfcc
from .common import require

POLICY = dict(version='msoc_prefix_v1', seconds=1.2, video_frames=30, video_fps=25,
              train_cache_side=128, eval_side=100, rgb=True, crop=100,
              sample_rate=16000, channel='first', mfcc_frames=120, mfcc_dim=13,
              mfcc_winlen=.025, mfcc_winstep=.010, mfcc_pcm_scale=32768,
              image_mean=.421, image_std=.165,
              spatial='resize short side 128, center square, train random 100 crop/flip',
              short_video='zero pad beyond decoded video duration', short_audio='wrap MFCC',
              timestamps='video first PTS origin; nearest frame at j/25; audio aligned to video origin')


def spatial(frame, side=128):
    h, w = frame.shape[:2]
    scale = 128 / min(h, w)
    frame = cv2.resize(frame, (max(128, round(w*scale)), max(128, round(h*scale))), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    y, x = (h-side)//2, (w-side)//2
    return np.ascontiguousarray(frame[y:y+side, x:x+side])


def decode(path, train=False):
    cv2.setNumThreads(1)
    side = 128 if train else 100
    frames, times, end = [], [], 0.
    with av.open(str(path)) as container:
        require(bool(container.streams.video) and bool(container.streams.audio), f'Missing video/audio: {path}')
        stream = container.streams.video[0]
        stream.thread_count = 1
        rate = float(stream.average_rate or 25)
        origin = None
        for f in container.decode(stream):
            require(f.pts is not None, f'Missing video PTS: {path}')
            absolute = float(f.pts * f.time_base)
            if origin is None:
                origin = absolute
            t = absolute - origin
            require(not times or t >= times[-1], f'Nonmonotone video PTS: {path}')
            frames.append(spatial(f.to_ndarray(format='rgb24'), side))
            times.append(t)
            end = t + (float(f.duration * f.time_base) if getattr(f, 'duration', 0) else 1 / rate)
            if t >= 1.24:
                break
    require(bool(frames), f'No decoded video: {path}')
    video = np.zeros((30, side, side, 3), dtype=np.uint8)
    for j in range(30):
        target = j / 25
        if target < end:
            video[j] = frames[int(np.argmin(np.abs(np.asarray(times)-target)))]
    # 120 MFCC frames require 1.215 s, including the last 25 ms analysis window.
    wave = np.zeros(19440, dtype=np.float32)
    covered_end, decoded_audio = 0, 0
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        stream.thread_count = 1
        resampler = av.AudioResampler(format='fltp', layout=stream.layout, rate=16000)
        finished = False
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                require(f.pts is not None, f'Missing audio PTS: {path}')
                offset = round((float(f.pts * f.time_base) - origin) * 16000)
                samples = f.to_ndarray()[0]  # first channel, matching official loader
                left, right = max(0, offset), min(len(wave), offset+len(samples))
                if right > left:
                    wave[left:right] = samples[left-offset:right-offset]
                    covered_end = max(covered_end, right)
                    decoded_audio += right-left
                if offset + len(samples) >= len(wave):
                    finished = True
                    break
            if finished:
                break
        if not finished:
            for f in resampler.resample(None):
                require(f.pts is not None, 'Missing flushed audio PTS')
                offset = round((float(f.pts * f.time_base)-origin)*16000)
                samples = f.to_ndarray()[0]
                left, right = max(0, offset), min(len(wave), offset+len(samples))
                if right > left:
                    wave[left:right] = samples[left-offset:right-offset]
                    covered_end = max(covered_end, right)
                    decoded_audio += right-left
    require(decoded_audio > 0, f'No audio overlapping video prefix: {path}')
    pcm = np.clip(np.rint(wave[:covered_end]*32768), -32768, 32767).astype(np.int16)
    audio = mfcc(pcm, 16000, numcep=13, winlen=.025, winstep=.010).astype(np.float32)
    if len(audio) < 120:
        audio = np.pad(audio, ((0, 120-len(audio)), (0, 0)), mode='wrap')
    audio = np.ascontiguousarray(audio[:120])
    require(np.isfinite(audio).all(), f'Nonfinite MFCC: {path}')
    info = dict(video_origin_seconds=origin, decoded_frames=len(frames),
                valid_video_tokens=int(sum(j/25 < end for j in range(30))),
                audio_samples=covered_end, source_fps=rate)
    return video, audio, info


def pack(video, audio):
    require(video.dtype == np.uint8 and audio.shape == (120, 13), 'Invalid cache arrays')
    return zlib.compress(video.tobytes()+audio.astype('<f4').tobytes(), level=1)


def unpack(blob, side):
    raw = zlib.decompress(blob)
    n = 30*side*side*3
    require(len(raw) == n + 120*13*4, 'Truncated/invalid cached arrays')
    return (np.frombuffer(raw[:n], dtype=np.uint8).reshape(30, side, side, 3),
            np.frombuffer(raw[n:], dtype='<f4').reshape(120, 13))
