"""Immutable manifests and explicit, evidence-derived PGF modality-label overlays."""
from dataclasses import replace
from pathlib import Path
from collections import defaultdict
from src.dataloader.metadata import load_records
from src.datasets.media import MediaConfig, read_video, read_audio
from .common import *
from .sampling import verify_selected

REGISTRY = ROOT/'configs/fixed_manifests.json'
ELIGIBILITY = ROOT/'reports/phase0_v1/eligibility_v4.jsonl'
PGF_ROOT = Path('D:/datasets/PolyGlotFake/json_file')
PGF_FILES = {'fake': PGF_ROOT/'fake_Json_file/all_fake_video.json',
             'real': PGF_ROOT/'real_json_file/all_real_video.json'}
TTS = {'MicroTts','Xtts','Bark','Tacotron','Vall'}
SYNC = {'video_retalking','Wav2Lip'}

def pgf_index():
    indices = {}
    for kind,path in PGF_FILES.items():
        data = read(path)
        entries = data['video' if kind=='fake' else 'videos']
        by_name = defaultdict(list)
        for i,row in enumerate(entries):
            by_name[row['filename']].append((i,row))
        indices[kind] = (path, sha(path), by_name)
    return indices

def pgf_labels(record, index):
    """Evidence is distributed list membership + explicit TTS/lip-sync methods."""
    kind = 'real' if record.sample_id.startswith('PGF:real/') else 'fake'
    require(record.sample_id.startswith('PGF:'+kind+'/'), 'Unrecognized PGF ID')
    path, file_hash, names = index[kind]
    found = names.get(record.media_path.name, [])
    require(found, 'PGF metadata missing: '+record.sample_id)
    require(record.clip_label == (kind=='fake'), 'PGF list/clip label conflict')
    evidence = []
    if kind == 'real':
        v,a = 0,0
    else:
        # Conflicting duplicate records cannot be silently selected.
        signatures = {(x.get('raw_video_name'),x.get('target_lang'),
                       x.get('tts_technique'),x.get('sync_tech')) for _,x in found}
        require(len(signatures)==1, 'Conflicting PGF metadata: '+record.sample_id)
        row = found[0][1]
        require(record.group_id == 'PGF:source-video:'+row['raw_video_name'], 'PGF source group mismatch')
        require('/to_'+row['target_lang']+'/' in record.sample_id, 'PGF target language mismatch')
        a = 1 if row.get('tts_technique') in TTS else None
        v = 1 if row.get('sync_tech') in SYNC else None
    for old,new in ((record.video_label,v),(record.audio_label,a)):
        require(old is None or old == new, 'PGF evidence conflicts with existing label')
    for i,row in found:
        evidence.append(dict(index=i, record_sha256=digest(row)))
    proof = dict(sample_id=record.sample_id, video_label=v, audio_label=a,
                 evidence_file=str(path), evidence_sha256=file_hash,
                 evidence_records=evidence, rule_version='pgf_tts_sync_membership_v1')
    return replace(record,video_label=v,audio_label=a), proof

def source_metadata(source, subset=None):
    require(source in SOURCES, 'Only AVDF1M and PGF sources are supported; FAV is already complete')
    records, proofs = {}, []
    require(subset is None or source=='AVDF1M', 'Fixed subset supports AVDF1M only')
    index = pgf_index() if source=='PGF' else None
    for split in ('train','dev','test'):
        status('METADATA', source+' '+split)
        expected = subset['rows'][split] if subset is not None and split in ('train','dev') else None
        options = {'sample_ids': [r['sample_id'] for r in expected]} if expected is not None else {}
        rows = load_records(source,split,task='paired',**options)
        if expected is not None:
            verify_selected(rows,expected)
        if index:
            pairs = [pgf_labels(r,index) for r in rows]
            rows = [r for r,_ in pairs]; proofs += [p for _,p in pairs]
        records[split] = rows
        for mod in ('video','audio'):
            if split != 'test': check_classes(rows,mod)
        check_classes(rows)
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        for key in ('sample_id','group_id'):
            require(not ({getattr(r,key) for r in records[a]} &
                         {getattr(r,key) for r in records[b]}), 'Source split overlap: '+key)
    return records,proofs

def targets(source):
    return [source]+[d for d in DATASETS if d != source]

def record_identity(records):
    return digest([(r.sample_id,r.group_id,r.split,str(r.media_path),
                    r.clip_label,r.video_label,r.audio_label) for r in records])

def media_identity(records, stamps):
    return digest([(r.sample_id,str(r.media_path),stamps[r.sample_id]) for r in records])

def decode_probe(record):
    """Decode one full clip on CPU with the production readers; no encoders."""
    # Three resized frames suffice for this smoke check. Training still uses 128.
    cfg = MediaConfig(frames=3, resize_short=256, crop_size=256, sample_rate=16000)
    video = read_video(record.media_path, cfg)
    pixels = video['video']
    require(tuple(pixels.shape) == (3,3,256,256) and np.isfinite(pixels.numpy()).all(),
            'Invalid decoded video')
    require(video['decoded_frames'] > 0, 'No decoded video frames')
    wave,present = read_audio(record.media_path, cfg)
    require(present and wave.ndim == 1 and len(wave) > 0 and np.isfinite(wave.numpy()).all(),
            'Missing, empty or invalid decoded audio')
    return dict(sample_id=record.sample_id, decoded_frames=int(video['decoded_frames']),
                checked_video_shape=list(pixels.shape), audio_samples=len(wave), sample_rate=16000)


def verify_media(cohorts, decode_samples=2):
    """Validate current local media, without asserting source-workstation identity.

    All files receive availability/stat checks. A deterministic bounded subset
    is decoded here; every remaining file is decoded when the experiment uses it.
    """
    require(type(decode_samples) is int and decode_samples > 0, 'decode_samples must be positive')
    total = sum(len(rows) for rows in cohorts.values())
    require(total > 0, 'No media selected')
    stamps = {}
    bar = Progress('VERIFY LOCAL FILES',total)
    for name,rows in cohorts.items():
        require(rows, 'Empty cohort: '+name)
        for r in rows:
            require(name == r.dataset+'/'+r.split, 'Cohort dataset/split mismatch: '+r.sample_id)
            require(r.sample_id and r.group_id, 'Missing sample/group ID')
            require(r.sample_id not in stamps, 'Duplicate sample ID: '+r.sample_id)
            require(all(v is None or (type(v) is int and v in (0,1))
                        for v in (r.clip_label,r.video_label,r.audio_label)),
                    'Invalid label: '+r.sample_id)
            require(r.media_path.is_absolute() and r.media_path.is_file(),
                    'Missing local media file: '+r.sample_id+' | '+str(r.media_path))
            stamps[r.sample_id] = stamp(r)
            bar.update(len(stamps), name)
    planned = {name:[rows[int(i)] for i in
                np.linspace(0,len(rows)-1,min(decode_samples,len(rows)),dtype=np.int64)]
               for name,rows in cohorts.items()}
    decoded = {}
    bar = Progress('VERIFY LOCAL DECODE',sum(len(rows) for rows in planned.values()))
    done = 0
    for name,rows in planned.items():
        decoded[name] = []
        for r in rows:
            try:
                require(stamp(r)==stamps[r.sample_id], 'Media changed before decode')
                result = decode_probe(r)
                require(stamp(r)==stamps[r.sample_id], 'Media changed during decode')
            except Exception as error:
                raise ValueError('Local decode check failed: '+r.sample_id+' | '
                                 +str(r.media_path)+' | '+str(error)) from error
            decoded[name].append(result)
            done += 1
            bar.update(done, r.sample_id)
    validation = dict(policy='current_windows_v1', source_workstation_hashes_required=False,
        file_count=len(stamps), file_checks='absolute path, regular nonempty file, size/mtime_ns',
        decode_samples_per_cohort=decode_samples, decoded_count=done,
        decode_selection='evenly spaced indices in frozen cohort order v1',
        decoded=decoded, all_media_decoded=(done==total),
        remaining_decode='production readers during training/extraction; errors stop the run')
    status('LOCAL MEDIA PASS',f'{len(stamps)} files checked; {done}/{total} clips decoded; no model inference')
    return stamps, validation


def counts(records):
    return {str(m or "clip"): {str(k):int((labels(records,m)==k).sum()) for k in (-1,0,1)}
            for m in (None,'video','audio')}
