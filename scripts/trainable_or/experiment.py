"""Train full FAV emotion heads and replace final hierarchical OR with learned score fusion."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='scripts.trainable_or'
import argparse,csv,time,traceback
import numpy as np
import scipy,torch
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import (
    ROOT,read,sha,digest,require,atomic_json,score_or,status,WriterLock)
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.train import fit as fit_head,load_head,predict as predict_head
from .data import verify_source,load_split,disjoint,cohort,EXPECTED,LEGACY
from .fusion import fit as fit_score,predict as predict_score
from .evaluation import metrics,verify_outputs,METHODS
from .report import write_report

SOURCE=ROOT/'reports/hierarchical_score_fusion_gru_temporal_attention_fav_v1/seed_42'
OUTPUT=ROOT/'reports/trainable_or_fav_v1/seed_42'

def immutable(path,value):
    if path.exists():require(read(path)==value,'Run settings/inputs changed: '+str(path))
    else:atomic_json(path,value)

def implementation():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/LEGACY/n for n in ('common.py','data.py','model.py','train.py')]
    paths += [ROOT/'scripts/artifact_emotion_gated_attetnion_optimized'/n
              for n in ('common.py','local_inputs.py','storage.py','progress.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def receipt_hashes(source):
    return {f'{m}/{s}':sha(source/'cache'/m/s/'COMPLETED.json')
            for m in ('video','audio') for s in EXPECTED}

def save_predictions(path,data,ev,ea,scores):
    path.parent.mkdir(parents=True,exist_ok=True)
    v,a=data['video'],data['audio']
    temp=path.with_suffix('.csv.partial')
    columns=['sample_id','group','clip_label','video_label','audio_label','Av','Aa','Ev','Ea',*scores]
    with temp.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(columns)
        for i,sid in enumerate(v['sample_id']):
            w.writerow([str(sid),str(v['group'][i]),*v['labels'][i].tolist(),
                        float(v['artifact_probability'][i]),float(a['artifact_probability'][i]),
                        float(ev[i]),float(ea[i]),*[float(p[i]) for p in scores.values()]])
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def predict_pair(heads,data,cfg):
    scores={}
    for mod in ('video','audio'):
        scores[mod],_=predict_head(heads[mod],data[mod],mod,cfg['head_device'],
                                  cfg['batch_size'],progress=True)
        require(np.isfinite(scores[mod]).all(),'Missing/nonfinite emotion score: '+mod)
    return scores['video'],scores['audio']

def run(args):
    started=time.perf_counter();out=args.output.resolve()
    if args.command=='verify':
        result=verify_outputs(out)
        status('VERIFY',str(result));return result
    require(args.epochs>0 and args.batch_size>0 and args.lr>0 and args.pos_weight>0 and args.threads>0,'Invalid training settings')
    require(args.device=='cpu' or (args.device.startswith('cuda') and torch.cuda.is_available()),'Device unavailable')
    torch.set_num_threads(args.threads)
    source=args.source.resolve()
    require(source!=out and not out.is_relative_to(source),'Output must be separate from immutable source')
    status('PREFLIGHT','Verifying completed full FAV source, code and cache receipts')
    source_run,source_done=verify_source(source)
    cfg=dict(seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,
             weight_decay=1e-4,pos_weight=args.pos_weight,head_device=args.device,threads=args.threads,
             full_train=True,threshold=.5,score_regularization=.01,
             artifact='frozen imported FAV X3D_m/AASIST',emotion_heads='fresh GRU + Temporal Attention',
             head_selection='full source dev modality ROC-AUC',calibration='same full source dev',
             input_recipe=source_run['config']['video_input'],audio_input=source_run['config']['audio_input'])
    desc=dict(format='full_fav_trainable_final_v1',source=str(source),source_completion_sha256=sha(source/'COMPLETED.json'),
              cache_receipts=receipt_hashes(source),config=cfg,implementation=implementation(),
              torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
              full_encoder_retraining=False,prior_test_observed=True,independent_calibration=False)
    identity=digest(desc)
    def load(split):
        for m in ('video','audio'):
            require(sha(source/'cache'/m/split/'COMPLETED.json')==desc['cache_receipts'][f'{m}/{split}'],'Cache receipt changed')
        return load_split(source,source_run,source_done,split)
    if args.command=='preflight':
        data={s:load(s) for s in EXPECTED}
        for x,y in (('train','dev'),('train','test'),('dev','test')):
            disjoint(data[x]['video'],data[y]['video'])
        result={s:cohort(d) for s,d in data.items()}
        status('PREFLIGHT PASS',str(result));return result
    require(args.command=='run','Unknown command')
    if (out/'RUN.json').exists():require(args.resume,'Existing output: use --resume')
    out.mkdir(parents=True,exist_ok=True)
    with WriterLock(out):
        immutable(out/'RUN.json',desc)
        if (out/'COMPLETED.json').exists():
            require(read(out/'COMPLETED.json')['identity']==identity,'Completion identity differs')
            verify_outputs(out);status('COMPLETE','Already complete; outputs verified');return
        def stage(name,**extra):
            atomic_json(out/'progress.json',dict(stage=name,seconds_this_invocation=time.perf_counter()-started,**extra))
            status('STAGE',name)
        stage('load_full_train_dev')
        train,dev=load('train'),load('dev')
        disjoint(train['video'],dev['video'])
        cohorts=dict(train=cohort(train),dev=cohort(dev))
        immutable(out/'TRAIN_DEV_COHORTS.json',cohorts)
        status('FULL DATA',f'Train={cohorts["train"]["n"]} Dev={cohorts["dev"]["n"]}; no subset')
        selections={};heads={}
        for mod in ('video','audio'):
            stage('train_'+mod,epochs=cfg['epochs'],batch_size=cfg['batch_size'])
            hid=digest(dict(run=identity,modality=mod,train_cache=train[mod]['completion_sha256'],dev_cache=dev[mod]['completion_sha256']))
            folder=out/'heads'/mod
            selections[mod]=fit_head(train[mod],dev[mod],mod,folder,cfg,hid)
            require(selections[mod]['train_n']==EXPECTED['train'],'Not all train samples used')
            heads[mod]=load_head(folder,mod,hid,args.device)
        # Persist the selected heads before learning the score-level layer.
        selection_lock=dict(identity=identity,selection=selections,
                            head_hashes={m:sha(out/'heads'/m/'best.pt') for m in heads})
        immutable(out/'HEADS_SELECTED.json',selection_lock)
        stage('fit_final_score_on_dev')
        ev,ea=predict_pair(heads,dev,cfg)
        a=score_or(dev['video']['artifact_probability'],dev['audio']['artifact_probability'])
        e=score_or(ev,ea)
        fusion_path=out/'FUSION.json'
        if fusion_path.exists():
            fused=read(fusion_path)
            require(fused['head_selection_sha256']==sha(out/'HEADS_SELECTED.json'),'Fusion belongs to other heads')
        else:
            fused=dict(head_selection_sha256=sha(out/'HEADS_SELECTED.json'),
                       trainable_final=fit_score(a,e,dev['video']['labels'][:,0],split='dev'),
                       artifact_affine=fit_score(a,None,dev['video']['labels'][:,0],split='dev'))
            atomic_json(fusion_path,fused)
        dev_scores=dict(artifact_or=a,emotion_or=e,hierarchical_or=score_or(a,e),
                        trainable_final=predict_score(fused['trainable_final'],a,e),
                        artifact_affine=predict_score(fused['artifact_affine'],a))
        save_predictions(out/'dev/predictions.csv',dev,ev,ea,dev_scores)
        immutable(out/'LOCKED_BEFORE_TEST.json',dict(identity=identity,heads=selection_lock['head_hashes'],
                  fusion_sha256=sha(fusion_path),threshold=.5,test_used_for_selection=False,
                  dev_reused_for_head_selection_and_fusion=True))
        stage('evaluate_full_test')
        test=load('test');cohorts['test']=cohort(test)
        disjoint(train['video'],test['video']);disjoint(dev['video'],test['video'])
        ev,ea=predict_pair(heads,test,cfg)
        a=score_or(test['video']['artifact_probability'],test['audio']['artifact_probability'])
        e=score_or(ev,ea)
        scores=dict(artifact_or=a,emotion_or=e,hierarchical_or=score_or(a,e),
                    trainable_final=predict_score(fused['trainable_final'],a,e),
                    artifact_affine=predict_score(fused['artifact_affine'],a))
        results={k:metrics(test['video']['labels'][:,0],p) for k,p in scores.items()}
        delta={name:{key:100*(results['trainable_final'][key]-results[name][key])
                    for key in ('roc_auc','f1','macro_f1','balanced_accuracy','real_fpr')}
               for name in ('hierarchical_or','artifact_or','artifact_affine')}
        epoch_seconds=sum(sum(h['seconds'] for h in read(out/'heads'/m/'history.json')) for m in heads)
        summary=dict(source='FAV',target='FAV',full_cohort=True,config=cfg,cohorts=cohorts,selection=selections,
                     metrics=results,deltas_pp=delta,fusion_weights={k:fused[k]['weights'] for k in ('trainable_final','artifact_affine')},
                     branch_metrics=dict(video_emotion=metrics(test['video']['labels'][:,1],ev),
                                         audio_emotion=metrics(test['audio']['labels'][:,2],ea)),
                     all_fake_f1=float(2*results['artifact_or']['fake']/(EXPECTED['test']+results['artifact_or']['fake'])),
                     wall_seconds=time.perf_counter()-started,training_epoch_seconds=epoch_seconds)
        save_predictions(out/'test/predictions.csv',test,ev,ea,scores)
        atomic_json(out/'metrics.json',summary)
        verification=verify_outputs(out,check_completion=False)
        atomic_json(out/'VERIFICATION.json',verification)
        write_report(out,summary,verification)
        paths=['RUN.json','TRAIN_DEV_COHORTS.json','HEADS_SELECTED.json','FUSION.json','LOCKED_BEFORE_TEST.json',
               'dev/predictions.csv','test/predictions.csv','metrics.json','VERIFICATION.json','REPORT.md']
        paths += ['heads/'+m+'/'+n for m in heads for n in ('best.pt','history.json','COMPLETED.json')]
        # Completion is committed only after all outputs and arithmetic have been checked.
        atomic_json(out/'COMPLETED.json',dict(identity=identity,source='FAV',target='FAV',full_cohort=True,
                    evaluated=EXPECTED['test'],files={n:sha(out/n) for n in paths}))
        stage('complete',evaluated=EXPECTED['test'])
        status('RESULT',str(results));status('DELTA_PP',str(delta));status('PIPELINE COMPLETE',str(out))
        return summary

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','run','verify'])
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=.0005);p.add_argument('--pos-weight',type=float,default=.25)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',action='store_true')
    return p
if __name__=='__main__':
    try:run(parser().parse_args())
    except KeyboardInterrupt:
        status('INTERRUPTED','Last complete epochs retained. Rerun identical command with --resume.');sys.exit(130)
    except Exception:
        traceback.print_exc();status('FAILED','No success for failed stage; prior committed epochs retained.');sys.exit(1)
