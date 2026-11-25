"""Extend only the old audio length guard; preserve existing outputs and complete all cohorts."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]));__package__='scripts.trainable_or_3way.workstation_audio_extension'
import argparse,hashlib,sqlite3,json,subprocess,time,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from src.dataloader.records import SampleRecord
from scripts.trainable_or_3way.common import ROOT,read,sha,digest,require,immutable,atomic_json,WriterLock,Progress,stamp,status
from scripts.trainable_or_3way.storage import Store
from scripts.trainable_or_3way.workstation_evaluation.runtime import verify_bundle,configure,Engines,DEFAULT_BUNDLE
from scripts.trainable_or_3way.workstation_evaluation.experiment import export,validate

MIN_SAMPLES=510976
POLICY=dict(version='aasist_full_audio_minpad_v2',sample_rate=16000,channels='mono',minimum_samples=MIN_SAMPLES,
    short='unchanged right-zero-padding to training maximum 510976',long='preserve full decoded waveform; single-clip forward',
    truncation=False,windowing=False,normalization=False,short_batch_size=4,fp32=True,saved_batchnorm=True,tf32=False)
PARENT=ROOT/'reports/workstation_baseline_evaluation_v1'
OUTPUT=ROOT/'reports/workstation_baseline_full_audio_v2'
COHORTS=('dev_AVDF1M','test_FAV','test_PGF','test_AVDF1M')

def code_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in ('scripts/trainable_or_3way/storage.py','scripts/trainable_or_3way/common.py',
        'scripts/trainable_or_3way/workstation_evaluation/experiment.py','scripts/trainable_or_3way/workstation_evaluation/runtime.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths) if not p.name.startswith('test_')}

def records(folder):
    rows=read(folder/'COHORT.json')['rows'];dataset=read(folder/'RUN.json')['dataset']
    return [SampleRecord(dataset,r['sample_id'],Path(r['media_path']),r['split'],r['clip_label'],r['video_label'],r['audio_label'],r['group_id']) for r in rows],{r['sample_id']:r['stamp'] for r in rows}

def pad_wave(wave):
    require(wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all(),'Empty or nonfinite audio')
    return np.pad(wave,(0,max(0,MIN_SAMPLES-len(wave))))

def decode_one(r,expected,ffmpeg):
    require(stamp(r)==expected,'Media changed before decode: '+r.sample_id)
    h=sha(r.media_path)
    p=subprocess.run([str(ffmpeg),'-v','error','-xerror','-threads','1','-i',str(r.media_path),'-map','0:a:0','-vn','-ac','1','-ar','16000','-f','f32le','-'],capture_output=True,check=True,timeout=120)
    wave=np.frombuffer(p.stdout,dtype='<f4').copy();n=len(wave);wave=pad_wave(wave)
    require(stamp(r)==expected,'Media changed while decoding: '+r.sample_id)
    return torch.from_numpy(wave),dict(media_sha256=h,input_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),decoded_samples=n,input_samples=len(wave),long_audio=n>MIN_SAMPLES)

@torch.no_grad()
def infer(model,decoded,device):
    require(not any(m.training for m in model.modules()),'All AASIST modules must remain eval')
    result=[None]*len(decoded);short=[i for i,(x,_) in enumerate(decoded) if len(x)==MIN_SAMPLES]
    groups=([short] if short else [])+[[i] for i,(x,_) in enumerate(decoded) if len(x)>MIN_SAMPLES]
    for ids in groups:
        x=torch.stack([decoded[i][0] for i in ids]).to(device)
        with torch.autocast('cuda',enabled=False):out=model(x)
        for j,i in enumerate(ids):
            result[i]=dict(decoded[i][1],logits=out.logits[j].detach().cpu().numpy(),probability=float(out.fake_probability[j]))
            validate(result[i],'audio',['B','C'])
    return result

def source_cache(path,identity,rows,stamps,complete_required):
    store=Store(path,identity,readonly=True)
    try:
        ids=store.ids();require(ids<={r.sample_id for r in rows},'Unexpected source cache IDs')
        index=store.index_hash();done=Path(path)/'COMPLETED.json'
        if done.exists():require(read(done)==dict(identity=identity,n=len(rows),index_sha256=index) and len(ids)==len(rows),'Source completion differs')
        if complete_required:require(done.exists(),'Required source cache incomplete')
        checked=0
        for sid,s,checksum,blob in store.db.execute('SELECT sid,stamp,checksum,data FROM samples'):
            require(json.loads(s)==stamps[sid] and hashlib.sha256(blob).hexdigest()==checksum,'Corrupt input cache: '+sid);checked+=1
        return dict(n=len(ids),index_sha256=index,complete=done.exists(),verified_blobs=checked)
    finally:store.close()

def clone_cache(src,dst,old_identity,new_identity,rows,stamps,require_complete):
    proof=source_cache(src,old_identity,rows,stamps,require_complete)
    dst.mkdir(parents=True,exist_ok=True);record=dst/'INHERITED.json'
    expected=dict(parent_path=str(src),parent_identity=old_identity,parent_cache=proof,new_identity=new_identity)
    if (dst/'samples.sqlite').exists():
        if not record.exists():
            check=Store(dst,new_identity,readonly=True)
            try:require(check.index_hash()==proof['index_sha256'],'Interrupted import differs')
            finally:check.close()
            immutable(record,expected)
        require(read(record)==expected,'Changed source cache')
        if proof['complete']:immutable(dst/'COMPLETED.json',dict(identity=new_identity,n=len(rows),index_sha256=proof['index_sha256']))
        return proof
    tmp=dst/'samples.sqlite.partial'
    old=sqlite3.connect((src/'samples.sqlite').as_uri()+'?mode=ro',uri=True);new=sqlite3.connect(tmp)
    try:
        old.backup(new)
        with new:new.execute("UPDATE metadata SET value=? WHERE key='identity'",(digest(new_identity),))
    finally:old.close();new.close()
    os.replace(tmp,dst/'samples.sqlite');immutable(record,expected)
    if proof['complete']:immutable(dst/'COMPLETED.json',dict(identity=new_identity,n=len(rows),index_sha256=proof['index_sha256']))
    return proof

def smoke(args,env,bundle):
    folder=args.output/'validation';folder.mkdir(parents=True,exist_ok=True)
    identity=dict(implementation=code_hashes(),runtime=env,bundle=bundle,policy=POLICY,ffmpeg_sha256=sha(args.ffmpeg))
    path=folder/'SMOKE.json'
    if path.exists():
        saved=read(path);require(saved['identity']==identity and saved['status']=='PASS','Smoke settings differ');return saved
    short,st=records(args.parent/'smoke');short=short[:4]
    allrows,allst=records(args.parent/'test_AVDF1M');oldrun=read(args.parent/'test_AVDF1M/RUN.json')
    old=Store(args.parent/'test_AVDF1M/scores/audio',dict(run=digest(oldrun),modality='audio'),readonly=True)
    try:existing=old.ids()
    finally:old.close()
    pending=next(allrows[i:i+4] for i in range(0,len(allrows),4) if any(r.sample_id not in existing for r in allrows[i:i+4]))
    engine=Engines(args.bundle,args.device,args.ffmpeg,['B','C']);engine.load('audio');checks=[]
    try:
        decoded=[decode_one(r,st[r.sample_id],args.ffmpeg) for r in short]
        for r,(x,_) in zip(short,decoded):require(torch.equal(x,engine.helper.read_audio(r.media_path,ffmpeg=args.ffmpeg).squeeze(0)),'Short audio input changed')
        new=infer(engine.active,decoded,args.device)
        run=read(args.parent/'smoke/RUN.json');cache=Store(args.parent/'smoke/scores/audio',dict(run=digest(run),modality='audio'),readonly=True)
        try:
            for r,v in zip(short,new):
                old=cache.get(r.sample_id,st[r.sample_id]);dz=float(np.max(np.abs(np.asarray(old['logits'])-v['logits'])));dp=abs(float(old['probability'])-v['probability'])
                require(np.allclose(old['logits'],v['logits'],atol=.002,rtol=.0002) and dp<=.0005 and (float(old['probability'])>=.5)==(v['probability']>=.5),'Short audio parity failed')
                checks.append(dict(sample_id=r.sample_id,input_equal=True,max_logit_difference=dz,probability_difference=dp,decision_equal=True))
        finally:cache.close()
        longs=[decode_one(r,allst[r.sample_id],args.ffmpeg) for r in pending]
        outputs=infer(engine.active,longs,args.device)
        longchecks=[dict(sample_id=r.sample_id,input_samples=int(v['input_samples']),decoded_samples=int(v['decoded_samples']),finite=True) for r,v in zip(pending,outputs)]
    finally:engine.release()
    result=dict(status='PASS',identity=identity,short_checks=checks,long_checks=longchecks,state_preserved=True)
    atomic_json(path,result);status('SMOKE','PASS: identical short inputs, finite full-length long-audio predictions, model state unchanged')
    return result

def score_audio(args,out,run,rows,stamps):
    ident=dict(run=digest(run),modality='audio');store=Store(out/'scores/audio',ident)
    engine=None
    try:
        existing=store.ids()
        if (out/'scores/audio/COMPLETED.json').exists():
            require(read(out/'scores/audio/COMPLETED.json')==dict(identity=ident,n=len(rows),index_sha256=store.index_hash()) and len(existing)==len(rows),'Completed audio changed');return
        batches=[rows[i:i+4] for i in range(0,len(rows),4)];pending=[b for b in batches if any(r.sample_id not in existing for r in b)]
        bar=Progress('AASIST FULL AUDIO',len(rows),out,resumed=len(existing));count=len(existing);longs=0
        engine=Engines(args.bundle,args.device,args.ffmpeg,['B','C']);engine.load('audio')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            queue={}
            def submit(i):
                if i<len(pending):queue[i]=[pool.submit(decode_one,r,stamps[r.sample_id],args.ffmpeg) for r in pending[i]]
            for i in range(min(3,len(pending))):submit(i)
            for i,batch in enumerate(pending):
                if (out/'STOP_REQUESTED').exists() or (args.output/'STOP_REQUESTED').exists():raise KeyboardInterrupt('Stop requested')
                decoded=[f.result() for f in queue.pop(i)];submit(i+3)
                values=infer(engine.active,decoded,args.device)
                for r,v in zip(batch,values):
                    require(stamp(r)==stamps[r.sample_id],'Media changed during inference: '+r.sample_id)
                    if r.sample_id in existing:continue
                    store.put(r.sample_id,stamps[r.sample_id],v);count+=1;longs+=bool(v['long_audio'])
                bar.update(count,'new_long_audio='+str(longs))
        store.finish(rows)
    finally:
        if engine is not None:engine.release()
        store.close()

def run(args):
    args.parent=args.parent.resolve();args.output=args.output.resolve()
    require(args.output.is_relative_to(ROOT/'reports') and not args.output.is_relative_to(args.parent) and not args.parent.is_relative_to(args.output),'Use a separate output tree')
    require(args.workers>0,'Positive decode worker count required')
    args.output.mkdir(parents=True,exist_ok=True)
    with WriterLock(args.output):
        bundle=verify_bundle(args.bundle);env=configure(args.device,4,'off')
        base=read(args.parent/'dev_AVDF1M/RUN.json')['compatibility']
        require(base['runtime']==env and base['bundle']==bundle and base['ffmpeg_sha256']==sha(args.ffmpeg),'Runtime/assets differ from original')
        for p,h in base['implementation'].items():require(sha(ROOT/p)==h,'Original implementation changed')
        valid=smoke(args,env,bundle)
        if args.command=='smoke':return
        preserved={};summary={}
        for name in COHORTS:
            parent=args.parent/name;out=args.output/name;old=read(parent/'RUN.json')
            require(old['compatibility']==base,'Original runtime differs across cohorts')
            rows,stamps=records(parent)
            meta=read(parent/'COHORT.json')['rows'];require(digest([(r['sample_id'],r['group_id'],r['split'],r['media_path'],r['clip_label'],r['video_label'],r['audio_label'],r['stamp']) for r in meta])==old['cohort_sha256'],'Original cohort changed')
            for r in rows:require(stamp(r)==stamps[r.sample_id],'Media changed: '+r.sample_id)
            proof={f:sha(parent/f) for f in ('RUN.json','COHORT.json','STATUS.json')}
            if (parent/'COMPLETED.json').exists():
                done=read(parent/'COMPLETED.json');require(done['run']==digest(old),'Original run completion changed')
                for f,h in done['files'].items():require(sha(parent/f)==h,'Original output changed');proof[f]=h
                proof['COMPLETED.json']=sha(parent/'COMPLETED.json')
            preserved[name]=proof
            compat=dict(base,audio_recipe=POLICY,audio_extension_implementation=code_hashes())
            desc=dict(old,format='workstation_baseline_full_audio_v2',compatibility=compat,
                parent=dict(path=str(parent),run_sha256=sha(parent/'RUN.json'),files=proof),
                validation_sha256=sha(args.output/'validation/SMOKE.json'),decode_workers=args.workers)
            out.mkdir(parents=True,exist_ok=True);immutable(out/'RUN.json',desc);immutable(out/'COHORT.json',read(parent/'COHORT.json'))
            if (out/'COMPLETED.json').exists():
                done=read(out/'COMPLETED.json');require(done['run']==digest(desc),'New completion identity mismatch')
                for f,h in done['files'].items():require(sha(out/f)==h,'New completed output changed')
                summary[name]=read(out/'metrics.json');status('RESUME',name+' complete');continue
            try:
                atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='inherit_verified_scores'))
                for mod in ('video','audio'):
                    clone_cache(parent/'scores'/mod,out/'scores'/mod,dict(run=digest(old),modality=mod),dict(run=digest(desc),modality=mod),rows,stamps,mod=='video')
                atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='audio'))
                score_audio(args,out,desc,rows,stamps)
                status('EXPORT',name+' all clips, both BN conditions')
                v,a=export(out,desc,rows,stamps)
                for r in rows:require(str(v[r.sample_id]['media_sha256'])==str(a[r.sample_id]['media_sha256']),'Video/audio source file differs: '+r.sample_id)
                long_ids=[r.sample_id for r in rows if bool(a[r.sample_id].get('long_audio',False))]
                verification=dict(status='PASS',n=len(rows),long_audio_ids=long_ids,long_audio_n=len(long_ids),same_media_verified=True,source_files_preserved=True)
                if (parent/'metrics.json').exists():require(read(parent/'metrics.json')==read(out/'metrics.json'),'Already completed baseline scores changed')
                for f,h in proof.items():require(sha(parent/f)==h,'Original output was changed')
                atomic_json(out/'VERIFICATION.json',verification)
                atomic_json(out/'COMPLETED.json',dict(run=digest(desc),n=len(rows),files={f:sha(out/f) for f in ('metrics.json','predictions.csv','VERIFICATION.json')}))
                atomic_json(out/'STATUS.json',dict(status='COMPLETE',n=len(rows)))
                summary[name]=read(out/'metrics.json');status('COMPLETE',name)
                del v,a
            except BaseException as exc:
                atomic_json(out/'STATUS.json',dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc)));raise
        atomic_json(args.output/'PRESERVED_PARENT_FILES.json',dict(parent=str(args.parent),files=preserved))
        atomic_json(args.output/'SUMMARY.json',summary)
        atomic_json(args.output/'COMPLETED.json',dict(policy=POLICY,cohorts=COHORTS,receipts={n:sha(args.output/n/'COMPLETED.json') for n in COHORTS}))
        status('ALL COMPLETE',str(args.output))

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('smoke','run'))
    p.add_argument('--parent',type=Path,default=PARENT);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'.venv/Scripts/ffmpeg.exe');return p
if __name__=='__main__':
    try:run(parser().parse_args())
    except KeyboardInterrupt:status('INTERRUPTED','Committed scores retained; repeat same command');raise SystemExit(130)
    except Exception:traceback.print_exc();raise SystemExit(1)
