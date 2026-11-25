"""FAV -> AVDF1M/PGF, frozen learned final fusion; cached scores by default."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='scripts.trainable_or_cross_dataset'
import argparse,gc,time,traceback,importlib.metadata
import numpy as np
import torch
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import (
    ROOT,read,sha,digest,require,atomic_json,WriterLock,status,seed_all,Progress)
from scripts.hierarchical_score_fusion_GRU_Tempooral_Attention_cross_dataset.inputs import artifact_model,target_data,verify_files
from scripts.hierarchical_score_fusion_GRU_Tempooral_Attention_cross_dataset.inference import (
    FrozenPredictor,infer_stage,ScoreStore,validate_trace)
from .inputs import SOURCE,SCORES,OUTPUT,ARTIFACT,MODALITIES,load_source,reuse_source,join_emotion_csv
from .evaluate import evaluate

def implementation(source):
    paths=set(Path(__file__).parent.glob('*.py'))
    for folder in ('scripts/trainable_or','scripts/hierarchical_score_fusion_GRU_Tempooral_Attention_cross_dataset'):
        paths.update((ROOT/folder).glob('*.py'))
    paths.update(ROOT/n for n in source['run']['implementation'])
    paths.update(ROOT/n for n in ('src/dataloader/metadata.py','src/dataloader/records.py',
                  'scripts/artifact_emotion_gated_attention_cross_dataset/common.py',
                  'scripts/artifact_emotion_gated_attetnion_optimized/features.py'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}

def immutable(path,value):
    if path.exists():require(read(path)==value,'Inputs/model/settings changed; use a new output')
    else:atomic_json(path,value)

def read_inferred(data,caches):
    result={}
    for mod,(folder,identity) in caches.items():
        store=ScoreStore(folder/'scores.sqlite',identity)
        try:
            require(store.ids()=={r.sample_id for r in data['records']},'Incomplete inference')
            values=[];counts=[];bar=Progress('READ INFERENCE '+data['target']+' '+mod,len(data['records']))
            for i,r in enumerate(data['records'],1):
                trace=store.get(r.sample_id,data['stamps'][r.sample_id]);validate_trace(trace,mod)
                values.append(float(trace['score']));counts.append(int(trace['valid'].sum()));bar.update(i)
            result[mod]=np.asarray(values);result[mod+'_counts']=np.asarray(counts)
        finally:store.close()
    return result

def write_summary(output,reports):
    lines=['# FAV 학습형 최종 융합 cross-dataset 결과','',
           'FAV에서 선택한 heads와 최종 융합을 고정한 zero-shot test. 임계값0.5. 수치는 %.',
           '새 target 학습·재보정 없음. PGF 모달리티 라벨이 없으면 해당 branch 지표는 null.','',
           '| Target | Cohort | Method | n | ROC-AUC | Fake F1 | Real FPR |',
           '|---|---|---|---:|---:|---:|---:|']
    def fmt(v):return 'null' if v is None else f'{100*v:.4f}'
    for target,report in reports.items():
        for cohort in ('primary_complete_emotion','secondary_full_cohort'):
            for method in ('artifact_or','hierarchical_or','trainable_final','artifact_affine','previous_gated_or'):
                m=report[cohort][method]
                lines.append(f'| {target} | {cohort} | {method} | {m["n"]} | {fmt(m["roc_auc"])} | {fmt(m["f1"])} | {fmt(m["real_fpr"])} |')
    lines+=['','기존 target들은 이전에 평가한 데이터이며 untouched holdout이 아니다.',
            'Source dev를 head 선택과 최종 융합 학습에 재사용한 한계가 유지된다.',
            'Previous gated는 얼굴 crop 조건이 달라 순수한 융합 방식만의 비교로 해석하지 않는다.','']
    (output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

def run(args):
    output=args.output.resolve()
    if args.command=='verify':
        done=verify_files(output);status('VERIFY','PASS: '+str(done['targets']));return
    require(args.frame_batch_size>0 and args.threads>0 and args.fusion_batch_size>0,'Invalid batch/threads')
    require(len(set(args.targets))==len(args.targets),'Duplicate targets')
    device=args.device or ('cpu' if args.mode=='reuse-scores' else 'cuda:0')
    if args.mode=='infer' and device.startswith('cuda'):
        require(torch.cuda.is_available(),'CUDA requested but unavailable')
    torch.set_num_threads(args.threads);seed_all(42)
    status('PREFLIGHT','Verify FAV heads/fusion; target fitting is disabled')
    source=load_source(args.source_run)
    for protected in (args.source_run,args.artifact_run,args.score_run,Path(source['parent']['folder'])):
        require(output!=protected.resolve() and not output.is_relative_to(protected.resolve()),'Output must not overwrite source runs')
    producer=artifact_model(args.artifact_run,source,args.targets)
    reuse=reuse_source(args.score_run,source,producer,args.targets) if args.mode=='reuse-scores' else None
    descriptor=dict(format='trainable_final_cross_v1',source='FAV',targets=args.targets,source_folder=source['folder'],
                    source_receipt=source['receipt'],artifact_receipt=producer['receipt'],
                    reuse_receipt=reuse['receipt'] if reuse else None,implementation=implementation(source),
                    config=dict(mode=args.mode,device=device,frame_batch_size=args.frame_batch_size,threads=args.threads,
                                fusion_batch_size=args.fusion_batch_size,prediction_batch_size=1,precision='FP32',threshold=.5,
                                zero_shot=True,target_training=False,seed=42,
                                missing_policy='any missing emotion -> source artifact-affine fallback'),
                    packages={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy')})
    identity=digest(descriptor)
    def reused(data,target):
        seal=reuse['receipt']['targets'][target]
        require(data['cohort']==seal['cohort'] and len(data['records'])==seal['n'],'Reuse cohort differs')
        path=Path(reuse['folder'])/'evaluation'/target/'predictions.csv'
        require(sha(path)==seal['predictions_sha256'],'Historical scores changed')
        return join_emotion_csv(path,data)
    if args.command=='preflight':
        for target in args.targets:
            data=target_data(producer,target)
            if reuse:reused(data,target)
            status('PREFLIGHT',f'{target}: {len(data["records"])} IDs/labels/media and available scores verified')
            del data;gc.collect()
        status('PREFLIGHT PASS','No fusion evaluation, encoder inference, training, or output run created')
        return
    require(args.command=='run','Unknown command')
    output.mkdir(parents=True,exist_ok=True)
    with WriterLock(output):
        if (output/'RUN.json').exists():require(args.resume,'Output exists; add --resume')
        else:require(not [p for p in output.iterdir() if p.name!='.writer.lock'],'Nonempty output without RUN.json')
        immutable(output/'RUN.json',descriptor)
        if (output/'COMPLETED.json').exists():
            done=verify_files(output);require(done['identity']==identity,'Completed identity changed')
            status('COMPLETE','Already complete; saved outputs verified');return
        lock=output/'LOCKED_BEFORE_TARGET.json'
        immutable(lock,dict(identity=identity,source_receipt=source['receipt'],
                            fusion=source['fusion'],threshold=.5,target_training=False))
        reports={};begin=time.perf_counter()
        def stage(name,**kw):
            atomic_json(output/'progress.json',dict(stage=name,seconds_this_invocation=time.perf_counter()-begin,**kw))
            status('STAGE',name)
        for target_no,target in enumerate(args.targets,1):
            stage('check_'+target,target=target,target_index=target_no,targets=len(args.targets))
            data=target_data(producer,target);cache_receipts={}
            if reuse:
                stage('reuse_scores_'+target);emotion=reused(data,target)
            else:
                caches={}
                for mod in MODALITIES:
                    stage('infer_'+target+'_'+mod)
                    cid=dict(run=identity,target=target,modality=mod,cohort=data['cohort'])
                    folder=output/'cache'/target/mod
                    predictor=FrozenPredictor(source,mod,device,args.frame_batch_size)
                    infer_stage(data,mod,folder,cid,predictor)
                    caches[mod]=(folder,cid);cache_receipts[mod]=sha(folder/'COMPLETED.json')
                    del predictor;gc.collect()
                    if device.startswith('cuda'):torch.cuda.empty_cache()
                emotion=read_inferred(data,caches)
            stage('fuse_'+target)
            eval_id=dict(run=identity,target=target,cohort=data['cohort'],cache_receipts=cache_receipts)
            reports[target]=evaluate(data,emotion,source['fusion'],output/'evaluation'/target,lock,eval_id,args.fusion_batch_size)
            del data,emotion;gc.collect()
        atomic_json(output/'SUMMARY.json',dict(source='FAV',targets=args.targets,zero_shot=True,mode=args.mode,metrics=reports))
        write_summary(output,reports)
        atomic_json(output/'timing.json',dict(seconds_this_invocation=time.perf_counter()-begin))
        files=['RUN.json','LOCKED_BEFORE_TARGET.json','SUMMARY.json','RESULTS.md','timing.json']
        files += [f'evaluation/{t}/{n}' for t in args.targets for n in ('COMPLETED.json','metrics.json','predictions.csv')]
        if not reuse:files += [f'cache/{t}/{m}/COMPLETED.json' for t in args.targets for m in MODALITIES]
        atomic_json(output/'COMPLETED.json',dict(identity=identity,source='FAV',targets=args.targets,zero_shot=True,
                    files={n:sha(output/n) for n in files}))
        stage('complete')
        status('PIPELINE COMPLETE',str(output))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','run','verify'])
    p.add_argument('--source-run',type=Path,default=SOURCE)
    p.add_argument('--artifact-run',type=Path,default=ARTIFACT)
    p.add_argument('--score-run',type=Path,default=SCORES)
    p.add_argument('--targets',nargs='+',choices=['AVDF1M','PGF'],default=['AVDF1M','PGF'])
    p.add_argument('--mode',choices=['reuse-scores','infer'],default='reuse-scores')
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--device',default=None)
    p.add_argument('--frame-batch-size',type=int,default=32)
    p.add_argument('--fusion-batch-size',type=int,default=512)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    return p
if __name__=='__main__':
    try:run(parser().parse_args())
    except KeyboardInterrupt:
        status('INTERRUPTED','Committed inference clips / fusion batches retained. Repeat the same command with --resume.');sys.exit(130)
    except Exception:
        traceback.print_exc();status('FAILED','No completion for failed stage; previous committed data retained.');sys.exit(1)
