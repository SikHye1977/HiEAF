"""Verify audit cohorts against cached FAV IDs and production PGF run receipts."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from split_overlap_20261119 import ROOT,OUT,sha,csvout,write
from split_overlap_details_20261119 import local


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    rows=[json.loads(l) for l in (OUT/'used_sample_provenance.jsonl').open(encoding='utf8')]
    byds=defaultdict(list)
    for r in rows:byds[r['dataset']].append(r)
    checks=[];inputs={}
    for split in ('train','dev','test'):
        for mod in ('video','audio'):
            p=ROOT/f'reports/hierarchical_score_fusion_gru_temporal_attention_fav_v1/seed_42/cache/{mod}/{split}/index.npz'
            inputs[str(p)]=sha(p)
            with np.load(p,allow_pickle=False) as z:ids=set(z['sample_id'].tolist())
            actual={r['sample_id'] for r in byds['FAV'] if r['split']==split}
            checks.append(dict(kind='FAV actual feature cache IDs',path=str(p),n=len(ids),pass_check=ids==actual))
    p=ROOT/'reports/trainable_or_pgf_optimized_v1/source_PGF/seed_42/RUN.json'
    inputs[str(p)]=sha(p);run=json.loads(p.read_text(encoding='utf8'))
    for split in ('train','dev','test'):
        rr=[r for r in byds['PGF'] if r['split']==split]
        # PGF production uses metadata-verified authentic/authentic and fake/fake.
        candidate=[(r['sample_id'],r['group_id'],split,str(local(r)),r['label'],r['label'],r['label']) for r in rr]
        actual=digest(candidate);expected=run['cohorts']['PGF/'+split]['records_sha256']
        checks.append(dict(kind='PGF production cohort hash',split=split,n=len(rr),actual=actual,expected=expected,pass_check=actual==expected))
    bundle=Path('C:/Users/uxmlab/hsh/AVDF1M_X3D_AASIST_Windows_20261102_001')
    pp=bundle/'x3d_bn_audit_20261102_001/metadata/training_provenance.json'
    inputs[str(pp)]=sha(pp);prov=json.loads(pp.read_text(encoding='utf8'))
    for split in ('train','dev'):
        p=bundle/f'aasist/manifests/{split}_original.jsonl';actual=sha(p);inputs[str(p)]=actual
        checks.append(dict(kind='X3D training manifest hash matches imported AASIST manifest',split=split,
                           actual=actual,expected=prov['manifest_sha256'][split],pass_check=actual==prov['manifest_sha256'][split]))

    # Independent CSV-based FAV person associations: valid id fields only.
    p=Path('D:/datasets/FakeAVCeleb_v1.2/meta_data.csv');inputs[str(p)]=sha(p)
    fav_byid={r['sample_id']:r for r in byds['FAV']};csvpeople={};n=0
    with p.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            # The distributed CSV header calls filename 'path'; final unnamed field is its folder.
            folder=r[''].replace('FakeAVCeleb/','',1);sid='fav:'+folder+'/'+r['path']
            if sid not in fav_byid:continue
            ids={r[k] for k in ('source','target1','target2') if r[k].startswith('id') and len(r[k])==7 and r[k][2:].isdigit()}
            csvpeople.setdefault(sid,set()).update(ids);n+=1
    mismatch=[sid for sid,r in fav_byid.items() if csvpeople.get(sid)!=set(r['persons'])]
    checks.append(dict(kind='FAV identity associations independently match distributed CSV',n=len(csvpeople),
                       mismatches=len(mismatch),pass_check=not mismatch and len(csvpeople)==len(fav_byid)))
    # Audit-only exclusion eligibility, not altered splits or a new evaluated result.
    eligibility=[];stats=[]
    for source,target in [('FAV','AVDF1M'),('AVDF1M','FAV')]:
        known=set().union(*(set(r['persons']) for r in byds[source] if r['split'] in ('train','dev')))
        tt=[r for r in byds[target] if r['split']=='test']
        kept=[]
        for r in tt:
            ids=set(r['persons'])&known;keep=not ids
            eligibility.append(dict(source=source,target=target,sample_id=r['sample_id'],label=r['label'],
                                    keep_by_known_person_metadata=int(keep),overlapping_ids=';'.join(sorted(ids))))
            if keep:kept.append(r)
        stats.append(dict(source=source,target=target,original_n=len(tt),retained_n=len(kept),
                          retained_real=sum(r['label']==0 for r in kept),retained_fake=sum(r['label']==1 for r in kept),
                          rule='exclude any explicit related person in source train OR dev; no performance used'))
    csvout('candidate_person_disjoint_test_eligibility_NOT_APPLIED.csv',eligibility)
    write('candidate_person_disjoint_test_counts.json',stats)
    # Explicitly represent unmeasured PGF identities, rather than an empty intersection.
    p=OUT/'summary.json';summary=json.loads(p.read_text(encoding='utf8'))
    summary['datasets']['PGF']['person_metadata_overlap']=None
    summary['datasets']['PGF']['person_identity_status']='NOT_EVALUABLE: no unique person identifier'
    write('summary.json',summary)
    changed=[p for p,h in inputs.items() if sha(Path(p))!=h]
    write('COHORT_VERIFICATION.json',dict(checks=checks,inputs=inputs,changed=changed,
        status='PASS' if all(c['pass_check'] for c in checks) and not changed else 'FAIL'))
    print(json.dumps(dict(checks=checks,retained=stats),ensure_ascii=False),flush=True)
    assert all(c['pass_check'] for c in checks) and not changed


if __name__=='__main__':main()
