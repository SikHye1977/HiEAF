"""Resumable single-GPU baseline evaluation; no training or emotion-head fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__='scripts.trainable_or_3way.workstation_evaluation'
import argparse,csv,json,time,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score
from src.dataloader.records import SampleRecord
from src.dataloader.metadata import load_records
from src.utils.local_paths import load_local_paths
from scripts.trainable_or_3way.common import (ROOT,require,sha,digest,read,atomic_json,immutable,
    WriterLock,Progress,stamp,status)
from scripts.trainable_or_3way.storage import Store
from .runtime import DEFAULT_BUNDLE,verify_bundle,helper,configure,Engines,jsonl,by_id

# Fixed before observing GPU comparisons; no automatic tolerance relaxation.
TOLERANCE={'A':dict(logit_atol=.02,logit_rtol=.002,probability_atol=.002),
           'B':dict(logit_atol=.002,logit_rtol=.0002,probability_atol=.0005),
           'C':dict(logit_atol=.002,logit_rtol=.0002,probability_atol=.0005),
           'audio':dict(logit_atol=.002,logit_rtol=.0002,probability_atol=.0005)}

def implementation():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/p for p in ('scripts/trainable_or_3way/common.py','scripts/trainable_or_3way/storage.py',
        'scripts/hierarchical_score_fusion_GRU_Temproal_Attention/common.py',
        'scripts/artifact_emotion_gated_attetnion_optimized/storage.py',
        'scripts/artifact_emotion_gated_attetnion_optimized/progress.py','src/dataloader/metadata.py',
        'src/dataloader/records.py','src/utils/local_paths.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths) if not p.name.startswith('test_')}

def select_smoke(rows,per_group):
    result=[]
    # Metadata-only deterministic selection, balanced across four modality-label pairs.
    for pair in ((0,0),(0,1),(1,0),(1,1)):
        group=sorted([r for r in rows if (r.video_label,r.audio_label)==pair],key=lambda r:digest([42,r.sample_id]))
        require(len(group)>=per_group,'Insufficient smoke stratum')
        result += group[:per_group]
    return result

def cohort(args,hlp):
    refs={}
    if args.split=='dev':
        require(args.dataset=='AVDF1M','Only the fixed source AVDF1M dev subset is supported')
        rs=jsonl(args.bundle/'x3d_bn_audit_20261102_001/manifests/dev_portable.jsonl')
        refs=by_id(rs)
        local=by_id(jsonl(ROOT/'scripts/trainable_or_3way/subsets/AVDF1M_fav_size_seed42/dev.jsonl'))
        require(set(refs)==set(local) and len(rs)==3147,'Source dev IDs differ')
        data_root=Path(load_local_paths()['avdf1m_root'])
        for r in rs:
            require(all(r[k]==local[r['sample_id']][k] for k in
                ('split','group_id','video_label','audio_label','clip_label')),'Source dev labels/split changed')
        rows=[SampleRecord('AVDF1M',r['sample_id'],hlp.resolve_media(r,data_root),'dev',
               r['clip_label'],r['video_label'],r['audio_label'],r['group_id']) for r in rs]
    else:
        rows=load_records(args.dataset,'test',task='paired')
    require(rows and len({r.sample_id for r in rows})==len(rows),'Invalid cohort')
    if args.command=='smoke': rows=select_smoke(rows,args.per_group)
    return rows,refs

def prepare(args):
    require(args.audio_batch_size>0 and args.threads>0 and args.per_group>0,'Positive sizes required')
    require(len(set(args.conditions))==len(args.conditions),'Duplicate BN condition')
    require(args.command!='smoke' or (args.dataset,args.split)==('AVDF1M','dev'),'Smoke uses source dev only')
    bundle=verify_bundle(args.bundle);hlp=helper(args.bundle)
    env=configure(args.device,args.threads,args.tf32)
    status('METADATA',args.dataset+'/'+args.split)
    rows,refs=cohort(args,hlp);stamps={}
    bar=Progress('VERIFY FILES',len(rows))
    for i,r in enumerate(rows):
        require(r.media_path.is_file(),'Missing media: '+str(r.media_path))
        stamps[r.sample_id]=stamp(r)
        if r.sample_id in refs:require(stamps[r.sample_id][0]==refs[r.sample_id]['media_bytes'],'Dev size changed')
        bar.update(i+1)
    metadata=[(r.sample_id,r.group_id,r.split,str(r.media_path),r.clip_label,r.video_label,r.audio_label,stamps[r.sample_id]) for r in rows]
    compatibility=dict(bundle=bundle,implementation=implementation(),runtime=env,audio_batch_size=args.audio_batch_size,
        ffmpeg_sha256=sha(args.ffmpeg),video_recipe='128 uniform frames, 256 center crop; native bundle reader',
        audio_recipe='16k mono; right zero pad 510976; overflow errors; saved BN; FP32')
    run=dict(format='workstation_baseline_evaluation_v1',source='AVDF1M',dataset=args.dataset,split=args.split,
        smoke_only=args.command=='smoke',n=len(rows),cohort_sha256=digest(metadata),conditions=args.conditions,
        compatibility=compatibility,tolerance=TOLERANCE,threshold=.5,
        current_windows_test_data_policy=True,unknown_modality_labels_preserved=True,
        training=False,prior_targets_observed=True,source_dev_used_for_checkpoint_selection=True)
    if args.command=='evaluate':
        require(args.smoke_report is not None,'Provide the passed --smoke-report before full evaluation')
        smoke=read(args.smoke_report)
        require(smoke['status']=='PASS' and smoke['compatibility']==compatibility,'Smoke report incompatible or not passed')
        require(set(args.conditions)<=set(smoke['conditions']),'Requested condition was not smoke tested')
        run['smoke_report_sha256']=sha(args.smoke_report)
    return run,rows,stamps,refs

def validate(value,modality,conditions):
    require(len(str(value['media_sha256']))==64 and len(str(value['input_sha256']))==64,'Missing input identity')
    if modality=='video':
        for c in conditions:
            z=float(value[c+'_logit']);p=float(value[c+'_probability'])
            require(np.isfinite(z) and np.isfinite(p) and 0<=p<=1,'Invalid video prediction')
    else:
        z=np.asarray(value['logits']);p=float(value['probability'])
        require(z.shape==(2,) and np.isfinite(z).all() and np.isfinite(p) and 0<=p<=1,'Invalid audio prediction')

def score_batches(folder,identity,rows,stamps,decode,predict,validate_value,batch_size,output):
    """Preserve original batch boundaries on resume; commit each completed sample."""
    store=Store(folder,identity)
    try:
        existing=store.ids();require(existing<={r.sample_id for r in rows},'Unexpected cache IDs')
        for r in rows:
            require(stamp(r)==stamps[r.sample_id],'Media changed: '+r.sample_id)
            if r.sample_id in existing:validate_value(store.get(r.sample_id,stamps[r.sample_id]))
        done=Path(folder)/'COMPLETED.json'
        if done.exists():
            require(read(done)==dict(identity=identity,n=len(rows),index_sha256=store.index_hash())
                    and len(existing)==len(rows),'Completed cache modified')
            status('RESUME',str(folder)+' already complete');return
        batches=[rows[i:i+batch_size] for i in range(0,len(rows),batch_size)]
        pending=[b for b in batches if any(r.sample_id not in existing for r in b)]
        bar=Progress('SCORE '+identity['modality'],len(rows),output,resumed=len(existing));count=len(existing)
        def checked_decode(batch):
            for r in batch:require(stamp(r)==stamps[r.sample_id],'Media changed before decoding')
            result=decode(batch)
            for r in batch:require(stamp(r)==stamps[r.sample_id],'Media changed during decoding')
            return result
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(checked_decode,pending[0]) if pending else None
            for i,batch in enumerate(pending):
                if (output/'STOP_REQUESTED').exists():raise KeyboardInterrupt('Stop requested; committed samples retained')
                decoded=future.result()
                future=pool.submit(checked_decode,pending[i+1]) if i+1<len(pending) else None
                values=predict(decoded);require(len(values)==len(batch),'Wrong prediction batch size')
                for r,value in zip(batch,values):
                    require(stamp(r)==stamps[r.sample_id],'Media changed during inference')
                    validate_value(value)
                    if r.sample_id in existing:continue
                    store.put(r.sample_id,stamps[r.sample_id],value);count+=1
                bar.update(count)
        store.finish(rows)
    finally:store.close()

def values(out,run,rows,stamps,mod):
    identity=dict(run=digest(run),modality=mod)
    store=Store(out/'scores'/mod,identity,readonly=True)
    try:return {r.sample_id:store.get(r.sample_id,stamps[r.sample_id]) for r in rows}
    finally:store.close()

def metrics(y,p,z=None):
    y=np.array([-1 if v is None else v for v in y]);p=np.asarray(p,dtype=np.float64)
    mask=y>=0;y=y[mask];p=p[mask]
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid metric input')
    cm=confusion_matrix(y,p>=.5,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel());both=tn+fp>0 and fn+tp>0
    return dict(n=len(y),unknown_labels=int((~mask).sum()),real=tn+fp,fake=fn+tp,
        roc_auc_probability=float(roc_auc_score(y,p)) if both else None,
        roc_auc_logit=float(roc_auc_score(y,np.asarray(z)[mask])) if both and z is not None else None,
        fake_f1=float(f1_score(y,p>=.5,zero_division=0)) if len(y) else None,
        real_fpr=fp/(tn+fp) if tn+fp else None,confusion_matrix=cm.tolist(),threshold=.5)

def export(out,run,rows,stamps):
    v=values(out,run,rows,stamps,'video');a=values(out,run,rows,stamps,'audio')
    ap=np.array([float(a[r.sample_id]['probability']) for r in rows]);az=np.array([a[r.sample_id]['logits'] for r in rows])
    report=dict(source='AVDF1M',dataset=run['dataset'],split=run['split'],smoke_only=run['smoke_only'],n=len(rows),
                audio=metrics([r.audio_label for r in rows],ap,az[:,1]-az[:,0]),conditions={})
    for c in run['conditions']:
        vp=np.array([float(v[r.sample_id][c+'_probability']) for r in rows]);vz=np.array([float(v[r.sample_id][c+'_logit']) for r in rows])
        report['conditions'][c]=dict(video=metrics([r.video_label for r in rows],vp,vz),
            baseline_or=metrics([r.clip_label for r in rows],1-(1-vp)*(1-ap)))
    temp=out/'predictions.csv.partial'
    fields=['sample_id','group_id','clip_label','video_label','audio_label','Aa','Aa_logit_real','Aa_logit_fake']
    fields += [key for c in run['conditions'] for key in (c+'_Av',c+'_logit',c+'_baseline_or')]
    with temp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,r in enumerate(rows):
            d=dict(sample_id=r.sample_id,group_id=r.group_id,clip_label=r.clip_label,video_label=r.video_label,audio_label=r.audio_label,
                   Aa=ap[i],Aa_logit_real=az[i,0],Aa_logit_fake=az[i,1])
            for c in run['conditions']:
                p=float(v[r.sample_id][c+'_probability']);d.update({c+'_Av':p,c+'_logit':float(v[r.sample_id][c+'_logit']),c+'_baseline_or':1-(1-p)*(1-ap[i])})
            writer.writerow(d)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,out/'predictions.csv');atomic_json(out/'metrics.json',report)
    return v,a

def compare_smoke(args,run,rows,v,a,refs):
    result=dict(compatibility=run['compatibility'],conditions=run['conditions'],n=len(rows),tolerance=TOLERANCE,comparison={})
    reference={c:by_id(jsonl(args.bundle/f'x3d_bn_audit_20261102_001/results/{c}/predictions.jsonl')) for c in run['conditions']}
    reference['audio']=by_id(jsonl(args.bundle/'aasist/results/best_dev.jsonl'))
    for c in [*run['conditions'],'audio']:
        dz=[];dp=[];bad=[];flips=[];input_bad=[];limits=TOLERANCE[c]
        for r in rows:
            sid=r.sample_id;old=reference[c][sid];new=a[sid] if c=='audio' else v[sid]
            require(str(new['media_sha256'])==refs[sid]['media_sha256'],'Workstation media hash mismatch')
            if c!='audio' and str(new['input_sha256'])!=old['input_tensor_sha256']:input_bad.append(sid)
            z=np.asarray(new['logits'] if c=='audio' else new[c+'_logit'],dtype=float)
            oldz=np.asarray(old['logit'],dtype=float)
            p=float(new['probability'] if c=='audio' else new[c+'_probability']);oldp=old['fake_probability']
            dz.append(float(np.max(np.abs(z-oldz))));dp.append(abs(p-oldp))
            if not np.allclose(z,oldz,atol=limits['logit_atol'],rtol=limits['logit_rtol']) or abs(p-oldp)>limits['probability_atol']:bad.append(sid)
            if (p>=.5)!=(oldp>=.5):flips.append(sid)
        result['comparison'][c]=dict(max_abs_logit_difference=max(dz),max_abs_probability_difference=max(dp),
            outside_tolerance=bad,decision_mismatches=flips,input_mismatches=input_bad)
    result['status']='PASS' if all(not x[k] for x in result['comparison'].values() for k in ('outside_tolerance','decision_mismatches','input_mismatches')) else 'NEEDS_REVIEW'
    atomic_json(args.output/'SMOKE_COMPARISON.json',result)
    return result

def execute(args,run,rows,stamps,refs):
    out=args.output
    immutable(out/'RUN.json',run)
    immutable(out/'COHORT.json',dict(rows=[dict(sample_id=r.sample_id,group_id=r.group_id,split=r.split,media_path=str(r.media_path),
        clip_label=r.clip_label,video_label=r.video_label,audio_label=r.audio_label,stamp=stamps[r.sample_id]) for r in rows]))
    if args.command=='preflight':
        atomic_json(out/'PREFLIGHT.json',dict(status='PASS',run=digest(run),n=len(rows),model_forward=False))
        status('PREFLIGHT','PASS; no inference');return
    engines=Engines(args.bundle,args.device,args.ffmpeg,args.conditions)
    try:
        for mod in ('video','audio'):
            engines.load(mod)
            atomic_json(out/'STATUS.json',dict(status='RUNNING',stage=mod,run=digest(run)))
            def validate_ref(value,mod=mod):validate(value,mod,args.conditions)
            score_batches(out/'scores'/mod,dict(run=digest(run),modality=mod),rows,stamps,
                          engines.decode,engines.predict,validate_ref,1 if mod=='video' else args.audio_batch_size,out)
            engines.release()
    finally:engines.release()
    v,a=export(out,run,rows,stamps)
    if args.command=='smoke':
        report=compare_smoke(args,run,rows,v,a,refs)
        print(json.dumps(report['comparison'],indent=2),flush=True)
        require(report['status']=='PASS','GPU smoke needs review; full evaluation is blocked; see SMOKE_COMPARISON.json')
    atomic_json(out/'COMPLETED.json',dict(run=digest(run),n=len(rows),files={p.name:sha(p) for p in [out/'metrics.json',out/'predictions.csv']}))
    atomic_json(out/'STATUS.json',dict(status='COMPLETE',run=digest(run),n=len(rows)))
    status('COMPLETE',str(out))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','smoke','evaluate'])
    p.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',choices=['AVDF1M','FAV','PGF'],default='AVDF1M')
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--conditions',nargs='+',choices=['A','B','C'])
    p.add_argument('--smoke-report',type=Path)
    p.add_argument('--per-group',type=int,default=2)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--audio-batch-size',type=int,default=4)
    p.add_argument('--tf32',choices=['off','workstation'],default='off')
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'.venv/Scripts/ffmpeg.exe')
    return p

def main():
    args=parser().parse_args();args.bundle=args.bundle.resolve();args.output=args.output.resolve()
    args.conditions=args.conditions or (['A','B','C'] if args.command=='smoke' else ['B','C'])
    require(args.output.is_relative_to(ROOT/'reports') and not args.output.is_relative_to(args.bundle),'Use a separate reports output')
    args.output.mkdir(parents=True,exist_ok=True)
    with WriterLock(args.output):
        owned=False
        try:
            run,rows,stamps,refs=prepare(args)
            if (args.output/'RUN.json').exists():
                require(read(args.output/'RUN.json')==run,'Output belongs to different settings; use a new directory')
            else:
                require(not any(p.name not in ('.writer.lock',) for p in args.output.iterdir()),
                        'Nonempty output without RUN.json; use a fresh directory')
            owned=True
            execute(args,run,rows,stamps,refs)
        except KeyboardInterrupt:
            if owned:atomic_json(args.output/'STATUS.json',dict(status='INTERRUPTED',committed_scores_retained=True))
            status('STOPPED','Re-run the identical command to resume');raise SystemExit(130)
        except Exception as exc:
            if owned:atomic_json(args.output/'STATUS.json',dict(status='FAILED',error=str(exc),committed_scores_retained=True))
            traceback.print_exc();raise SystemExit(2)
if __name__=='__main__':main()
