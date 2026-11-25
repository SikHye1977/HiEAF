"""Read-only provenance checks and exact frozen PGF cohort alignment."""
from pathlib import Path
import numpy as np
from scripts.trainable_or_3way.reinforcement_p2.common import (
    ROOT, read, sha, digest, resolve, require, child, completed, file_set,
    package_versions, VIDEO, AUDIO, BN, load_npz)
from scripts.trainable_or_3way.reinforcement.inputs import receipt, load_scores, join_manifest
from scripts.trainable_or_3way.reinforcement_p2.inputs import compare_records, checkpoint_config

FORMAT = 'fav_pgf_common_bc_v1'
P2 = ROOT/'reports/fav_avdf1m_reinforcement_p2_v1'
CROSS = ROOT/'reports/trainable_or_cross_dataset_v1/source_FAV/seed_42'
SUMMARY = ROOT/'reports/diagnostics/three_dataset_bc_summary_20261113.json'
OUTPUT = ROOT/'reports/fav_pgf_common_bc_v1'

def preserve(path, proofs, expected=None):
    path=resolve(path)
    actual=sha(path)
    require(expected is None or actual==expected,'Input changed: '+str(path))
    proofs[str(path)]=actual
    return path

def verify_preserved(proofs):
    for p,h in proofs.items():
        require(Path(p).is_file() and sha(p)==h,'Preserved input changed: '+p)

def verify_summary_rows(summary):
    """Check reported values against receipts instead of trusting a copied summary."""
    cache={}
    aliases={'baseline':'M0','artifact_affine':'M1','proposed':'M4'}
    for r in summary['rows']:
        if r['scope']!='common_BC':continue
        path=resolve(r['metrics_path'])
        if path not in cache:cache[path]=read(path)
        d=cache[path];method=r['method']
        if r['source']=='FAV':
            require(d['source']==r['source'] and d['target']==r['target'] and d['condition']==r['condition'], 'FAV summary condition mismatch')
            actual=d['metrics'][aliases[method]]
        elif r['source']=='AVDF1M':
            require(d['source']==r['source'] and d['target']==r['target'] and d['bn_condition']==r['condition'], 'AVDF1M summary condition mismatch')
            actual=d['full_cohort']['trainable_final' if method=='proposed' else method]
        else:
            require(r['source']=='PGF','Unknown summary source')
            panel=d[r['condition']]
            require(panel['source']==r['source'] and panel['target']==r['target'],'PGF summary condition mismatch')
            actual=panel['full']['trainable_final' if method=='proposed' else method]
        for field in ('n','real','fake','roc_auc','fake_f1','real_fpr'):
            key='f1' if field=='fake_f1' and 'fake_f1' not in actual else field
            require(abs(r[field]-actual[key])<1e-12,'Summary metric differs from original: '+str(path)+' '+field)

def verify_source(p2, cross, proofs):
    """Reuse the exact P2 source-dev mapping, not a new target-dependent fit."""
    pr,pdone=receipt(p2,proofs)
    require(pr['format']=='reinforcement_p2_bc_v1' and pr['source']=='FAV'
            and pr['conditions']==['B','C'],'Need the completed FAV P2 B/C run')
    require(pr['video']==VIDEO and pr['audio']==AUDIO and pr['bn']==BN,'P2 inference recipe changed')
    # These imported production functions must be the P2 implementation that made the tables.
    current=file_set()
    require(pr['implementation']==current,'P2 implementation changed; do not silently alter the common protocol')
    lock=read(p2/'fusion/LOCKED_BEFORE_TEST.json')
    models=read(p2/'fusion/FUSION.json')
    require(lock['run']==digest(pr) and lock['source']=='FAV' and lock['conditions']==['B','C']
            and not lock['target_training'] and lock['threshold']==.5
            and lock['models_sha256']==sha(p2/'fusion/FUSION.json'),'P2 fusion lock mismatch')
    for c in ('B','C'):
        preserve(p2/f'raw/FAV/dev/{c}.npz',proofs,lock['dev_inputs'][c])
    cr,_=receipt(cross,proofs)
    require(cr['source']=='FAV' and 'PGF' in cr['targets'] and cr['config']['zero_shot']
            and not cr['config']['target_training'],'Wrong historical target source')
    fav=resolve(cr['source_folder'])
    fr,_=receipt(fav,proofs)
    fl=read(fav/'LOCKED_BEFORE_TEST.json')
    cf=read(cross/'LOCKED_BEFORE_TARGET.json')
    ff=read(fav/'FUSION.json')
    require(fl['identity']==digest(fr) and fl['fusion_sha256']==sha(fav/'FUSION.json'),'FAV source lock mismatch')
    require(cf['identity']==digest(cr) and cf['fusion']==ff,'Historical cross-source lock mismatch')
    for k,n in (('run_sha256','RUN.json'),('completion_sha256','COMPLETED.json'),
                ('lock_sha256','LOCKED_BEFORE_TEST.json'),('fusion_sha256','FUSION.json')):
        require(cr['source_receipt'][k]==cf['source_receipt'][k]==sha(fav/n),'Source receipt mismatch: '+n)
    for mod,h in fl['heads'].items():
        preserve(fav/'heads'/mod/'best.pt',proofs,h)
        require(cr['source_receipt']['heads'][mod]['checkpoint_sha256']==h,'Emotion heads differ')
    # Bind the P2 emotion source to the same FAV source run used for PGF emotion scores.
    p0=resolve(pr['parent'])/'p0'
    inv=read(p0/'INVENTORY.json')
    preserve(p0/'INVENTORY.json',proofs)
    saved=read(p0/'PRESERVED_INPUTS.json')
    preserve(p0/'PRESERVED_INPUTS.json',proofs)
    for n in ('RUN.json','FUSION.json','LOCKED_BEFORE_TEST.json','COMPLETED.json'):
        p=fav/n
        require(saved.get(str(p))==sha(p),'P2 and PGF emotion source differ: '+n)
    dev=inv['cohorts']['FAV__historical__FAV__dev']
    require(resolve(dev['csv'])==fav/'dev/predictions.csv','P2 used another FAV emotion dev cohort')
    require(dev['normalized_sha256']==pr['cohorts']['FAV/dev']['sha256'],'P2 FAV dev source differs')
    preserve(p0/dev['normalized_file'],proofs,dev['normalized_sha256'])
    csv_path=cross/'evaluation/PGF/predictions.csv'
    require(str(csv_path) in proofs,'PGF score CSV not bound by completion receipt')
    data=load_scores(csv_path,ff)
    return pr,models,data

def implementation():
    current=file_set()
    for p in Path(__file__).parent.glob('*.py'):current[p.relative_to(ROOT).as_posix()]=sha(p)
    return current

def prepare(args):
    from src.dataloader.metadata import load_records
    from scripts.trainable_or_3way.reinforcement_p2.runtime import configure
    p2,cross=resolve(args.p2),resolve(args.historical)
    proofs={}
    pr,models,data=verify_source(p2,cross,proofs)
    registry=preserve(pr['metadata']['registry'],proofs,pr['metadata']['registry_sha256'])
    local=preserve(args.local_paths,proofs)
    eligibility=preserve(ROOT/'reports/phase0_v1/eligibility_v4.jsonl',proofs,pr['metadata']['eligibility_sha256'])
    ffmpeg=preserve(args.ffmpeg,proofs,pr['ffmpeg']['sha256'])
    require(args.threads>=1 and args.decode_workers>=1 and 1<=args.prefetch<=4,'Invalid threads/workers/prefetch')
    # Normalize groups and labels against the unchanged manifest, not a newly generated split.
    cohort={'PGF':dict(source='FAV',target='PGF',condition='historical',split='test',data=data)}
    join_manifest(cohort,{},registry,proofs)
    records=load_records('PGF','test',task='paired',registry=registry,eligibility_path=eligibility,
                         local_paths=local,sample_ids=data['sample_id'].tolist())
    rows=compare_records(data,records,'PGF','test')
    require(len(rows)==2082 and int((data['labels'][:,0]==0).sum())==112,'Unexpected frozen PGF test cohort')
    for mod,asset in pr['assets'].items():
        preserve(asset['path'],proofs,asset['sha256'])
        require(checkpoint_config(asset['path'],asset['sha256'],mod)==asset['config'],'Artifact configuration changed')
    summary=preserve(args.summary,proofs)
    old=read(summary)
    require(len([r for r in old['rows'] if r['scope']=='common_BC'])==48,'Expected 16 existing conditions with three methods')
    for ev in old['evidence']:
        preserve(ev['path'],proofs,ev['sha256'])
        cp=preserve(ev['completion_receipt'],proofs)
        done=read(cp)
        relative=resolve(ev['path']).relative_to(cp.parent).as_posix()
        require(done['files'].get(relative)==ev['sha256'],'Summary metric not bound by its completion receipt')
    verify_summary_rows(old)
    run=dict(format=FORMAT,source='FAV',targets=['PGF'],conditions=['B','C'],
        p2=str(p2),historical=str(cross),summary=str(summary),assets=pr['assets'],
        runtime=configure(args.device,args.threads),packages=package_versions(),implementation=implementation(),
        ffmpeg=dict(path=str(ffmpeg),sha256=sha(ffmpeg)),video=VIDEO,audio=AUDIO,bn=BN,
        decode_workers=args.decode_workers,prefetch=args.prefetch,
        cohorts={'PGF/test':dict(path=str(resolve(args.output)/'inputs/PGF_test.npz'),
                  n=len(rows),cohort_sha256=digest(rows))},
        normalized_content_sha256=digest({k:sha_array(v) for k,v in sorted(data.items())}),
        fusion=dict(p2_models_sha256=sha(p2/'fusion/FUSION.json'),threshold=.5,refit=False,
                    source='FAV/dev',target_training=False),
        preserved_inputs=proofs,encoder_training=False,emotion_inference=False,
        bootstrap=False,prior_targets_observed=True,current_windows_data_policy=True)
    return run,{'PGF/test':rows},data,models

def sha_array(a):
    import hashlib
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
