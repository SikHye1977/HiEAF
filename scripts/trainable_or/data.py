"""Verified read-only import of full FAV temporal caches and artifact scores."""
from pathlib import Path
import numpy as np
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import ROOT,read,sha,digest,require,status
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.data import load_packed
from scripts.artifact_emotion_gated_attetnion_optimized.common import load_cache

EXPECTED = dict(train=14724, dev=3147, test=3126)
LEGACY = 'scripts/hierarchical_score_fusion_GRU_Temproal_Attention/'

def verify_source(root):
    root=Path(root)
    run,done=read(root/'RUN.json'),read(root/'COMPLETED.json')
    require(digest(run)==done['identity'],'Source identity mismatch')
    require(not done['smoke_only'] and done['source']==done['target']=='FAV','Need completed full FAV source')
    for name,h in done['files'].items():
        require(sha(root/name)==h,'Source output changed: '+name)
    for name in ('model.py','common.py','data.py','train.py'):
        rel=LEGACY+name
        require(sha(ROOT/rel)==run['implementation'][rel],'Source dependency changed: '+rel)
    for mod in ('video','audio'):
        for split,n in EXPECTED.items():
            require(run['source'][mod][split]['n']==n,'Source is not full cohort')
    return run,done

def align(reference,candidate):
    require(len(set(candidate['sample_id']))==len(candidate['sample_id']),'Duplicate artifact IDs')
    lookup={str(s):i for i,s in enumerate(candidate['sample_id'])}
    require(set(lookup)==set(map(str,reference['sample_id'])),'Cohort IDs differ')
    order=np.array([lookup[str(s)] for s in reference['sample_id']])
    for key in ('sample_id','group','labels'):
        require(np.array_equal(reference[key],candidate[key][order]),'Conflicting '+key)
    return order

def load_split(root,run,done,split):
    require(split in EXPECTED,'Unknown split')
    result={}
    for mod in ('video','audio'):
        status('CACHE',f'{mod} {split}: verifying full frozen cache')
        source=run['source'][mod][split]
        identity=dict(run=done['identity'],modality=mod,split=split,source=source)
        d=load_packed(Path(root)/'cache'/mod/split,identity)
        artifact,meta=load_cache(source['source_folder'],mod,split)
        require(meta['sha256']==source['features_sha256'],'Artifact features changed')
        require(sha(Path(source['source_folder'])/'manifest.json')==source['manifest_sha256'],'Artifact manifest changed')
        require(meta['recipe']['artifact_checkpoint_sha256']==source['artifact_checkpoint_sha256'],'Artifact weights changed')
        order=align(d,artifact)
        require(np.array_equal(d['artifact_probability'],artifact['artifact_probability'][order]),'Artifact scores differ')
        require(len(d['sample_id'])==EXPECTED[split] and np.isin(d['labels'],[0,1]).all(),'Invalid full FAV cohort')
        # This specific FAV center-crop cache has valid observations for every clip.
        # Refuse silent exclusions if a different cache is supplied.
        require((d['valid_counts']>0).all(),'Missing emotion sequences: full-cohort run cannot silently exclude clips')
        result[mod]=d
    for key in ('sample_id','group','labels'):
        require(np.array_equal(result['video'][key],result['audio'][key]),'Modality alignment mismatch')
    return result

def disjoint(a,b):
    for key in ('sample_id','group'):
        require(not(set(a[key])&set(b[key])),'Split overlap: '+key)

def cohort(data):
    d=data['video']
    return dict(n=len(d['sample_id']),groups=len(set(d['group'])),
                ordered_ids_sha256=digest(d['sample_id'].tolist()),
                labels_sha256=digest(d['labels'].tolist()),
                counts={k:np.bincount(d['labels'][:,j],minlength=2).tolist()
                        for j,k in enumerate(('clip','video','audio'))},
                complete_emotion={m:int((data[m]['valid_counts']>0).sum()) for m in data})
