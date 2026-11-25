"""Additional read-only evidence: file equality, origin families, provenance binding."""
import csv
import hashlib
import itertools
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_overlap_20261119 import ROOT, OUT, csvout, sha, write


def separated(rows):
    return len({(r['dataset'],r['split']) for r in rows}) > 1


def local(row):
    ds=row['dataset'];marker={'FAV':'/FakeAVCeleb_v1.2/','AVDF1M':'/AV-Deepfake1M/','PGF':'/PolyGlotFake/PolyGlotFake/'}[ds]
    directory={'FAV':'FakeAVCeleb_v1.2','AVDF1M':'AV-Deepfake1M','PGF':'PolyGlotFake'}[ds]
    return Path('D:/datasets')/directory/row['path'].split(marker,1)[1]


def main():
    started=time.time()
    rows=[json.loads(l) for l in (OUT/'used_sample_provenance.jsonl').open(encoding='utf8')]
    inputs={str(OUT/'used_sample_provenance.jsonl'):sha(OUT/'used_sample_provenance.jsonl')}
    evidence=[]
    # Actual imported AVDF training/dev manifest membership must equal sampled selection.
    bundle=Path('C:/Users/uxmlab/hsh/AVDF1M_X3D_AASIST_Windows_20261102_001')
    for sub in ['aasist/manifests/train_original.jsonl','aasist/manifests/dev_original.jsonl',
                'x3d_bn_audit_20261102_001/manifests/dev_original.jsonl']:
        p=bundle/sub;inputs[str(p)]=sha(p)
        imported=[json.loads(l) for l in p.open(encoding='utf8')]
        split=imported[0]['split'];actual={r['sample_id'] for r in rows if r['dataset']=='AVDF1M' and r['split']==split}
        got={r['sample_id'] for r in imported}
        evidence.append(dict(file=str(p),n=len(imported),split=split,missing=len(actual-got),extra=len(got-actual),same_ids=actual==got))
        assert got==actual
    # Source recording coverage and sample counts affected by source train/dev origin reuse.
    within=[];details=[]
    for ds in ('AVDF1M','PGF'):
        rr=[r for r in rows if r['dataset']==ds]
        for mode,parts in [('train',{'train'}),('train_and_dev',{'train','dev'})]:
            known=set().union(*(set(r['origins']) for r in rr if r['split'] in parts))
            target=[r for r in rr if r['split']=='test']
            hits=[r for r in target if set(r['origins'])&known]
            within.append(dict(dataset=ds,source_scope=mode,shared_urls=len(set().union(*(set(r['origins'])&known for r in target))),
                target_samples=len(target),affected_samples=len(hits),affected_real=sum(r['label']==0 for r in hits),
                affected_fake=sum(r['label']==1 for r in hits),target_with_url=sum(bool(r['origins']) for r in target)))
            for r in hits:
                details.append(dict(dataset=ds,source_scope=mode,sample_id=r['sample_id'],label=r['label'],origins=';'.join(sorted(set(r['origins'])&known))))
    csvout('within_dataset_recording_exposure.csv',within)
    csvout('within_dataset_recording_exposed_test_samples.csv',details)

    # FAV original-video filename fields are not fully documented. Keep them as candidates.
    candidate_index=defaultdict(list)
    candidate_rows=[]
    for r in rows:
        if r['dataset']!='FAV':continue
        name=r['sample_id'].split('/')[-1]
        for person,yt in re.findall(r'(id\d{5})_([A-Za-z0-9_-]{11})_(?:faceswap|fsgan)',name):
            candidate_index['youtube:'+yt].append(r)
            candidate_rows.append(dict(sample_id=r['sample_id'],split=r['split'],filename_person=person,youtube_candidate=yt,
                evidence='filename pattern; source/donor role and exact original clip not verified'))
    csvout('fav_filename_video_candidates.csv',candidate_rows)
    candidate_matches=[]
    for r in rows:
        if r['dataset']=='FAV':continue
        for origin in r['origins']:
            if origin in candidate_index:
                favs=candidate_index[origin]
                candidate_matches.append(dict(origin=origin,target_dataset=r['dataset'],target_split=r['split'],
                    target_sample_id=r['sample_id'],fav_candidate_splits=';'.join(sorted({x['split'] for x in favs})),
                    fav_candidate_samples=len(favs)))
    csvout('fav_filename_video_cross_dataset_candidates.csv',candidate_matches)

    # Byte identity: same size -> same sampled bytes -> full SHA256. A negative
    # size/sample comparison is sufficient to rule out whole-file byte equality.
    bysize=defaultdict(list);missing=[];stat_before={}
    for i,r in enumerate(rows,1):
        p=local(r);r['local_path']=str(p)
        try:
            st=p.stat();assert st.st_size>0
        except (OSError,AssertionError) as e:
            missing.append(dict(sample_id=r['sample_id'],path=str(p),error=str(e)));continue
        stat_before[str(p)]=(st.st_size,st.st_mtime_ns)
        bysize[st.st_size].append(r)
        if i%20000==0:print(f'[FILE AUDIT] stat {i}/{len(rows)} elapsed={time.time()-started:.1f}s',flush=True)
    candidates=[v for v in bysize.values() if separated(v)]
    candidate_files=sum(len(v) for v in candidates)
    print('[FILE AUDIT] candidate files',candidate_files,flush=True)
    sampled=defaultdict(list)
    for i,r in enumerate(itertools.chain.from_iterable(candidates),1):
        p=Path(r['local_path']);size=stat_before[str(p)][0]
        with p.open('rb') as f:
            h=hashlib.sha256(f.read(32768));f.seek(max(0,size-32768));h.update(f.read(32768))
        sampled[(size,h.hexdigest())].append(r)
        if i%2000==0:print(f'[FILE AUDIT] prefix/suffix {i}/{candidate_files} elapsed={time.time()-started:.1f}s',flush=True)
    full=defaultdict(list);full_n=0
    for rr in sampled.values():
        if separated(rr):
            for r in rr:
                full[sha(Path(r['local_path']))].append(r);full_n+=1
    matches=[];match_groups=[]
    for h,rr in full.items():
        if not separated(rr):continue
        match_groups.append(dict(sha256=h,datasets=sorted({r['dataset'] for r in rr}),
            splits=sorted({r['split'] for r in rr}),sample_ids=[r['sample_id'] for r in rr]))
        for r in rr:matches.append(dict(sha256=h,dataset=r['dataset'],split=r['split'],sample_id=r['sample_id'],path=r['local_path']))
    csvout('exact_file_duplicates.csv',matches)
    changed=[]
    for p,old in stat_before.items():
        st=Path(p).stat()
        if (st.st_size,st.st_mtime_ns)!=old:changed.append(p)
    write('exact_file_check.json',dict(scope='All current paired/selected samples, across any dataset/split boundary',
        planned=len(rows),existing=len(stat_before),missing=missing,size_candidate_files=candidate_files,
        full_hash_files=full_n,exact_duplicate_groups=match_groups,changed_media_stat=changed,
        method='size -> first/last32KiB SHA256 -> complete SHA256; no decoded/perceptual comparison',
        interpretation='Zero byte duplicates does not exclude re-encoded, cropped, temporal or identity overlap'))
    write('additional_evidence.json',dict(imported_manifest_binding=evidence,within_dataset_recording_exposure=within,
        fav_filename_candidate_rows=len(candidate_rows),fav_filename_candidate_videos=len(candidate_index),
        fav_filename_cross_dataset_candidate_rows=len(candidate_matches),
        fav_filename_cross_dataset_candidate_videos=len({r['origin'] for r in candidate_matches}),
        inputs_sha256=inputs,input_changes=[p for p,h in inputs.items() if sha(Path(p))!=h],elapsed_seconds=time.time()-started))
    print('[DETAIL AUDIT COMPLETE]',len(match_groups),'byte duplicate groups; elapsed',round(time.time()-started,1),flush=True)
    assert not missing and not changed


if __name__=='__main__':main()
