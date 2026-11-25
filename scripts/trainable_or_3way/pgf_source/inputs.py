"""Frozen PGF label evidence, metadata, read-only reusable test tokens."""
import copy, json, shutil
from .common import *
from scripts.trainable_or_3way.data import source_metadata,record_identity,media_identity,counts,REGISTRY,ELIGIBILITY
from scripts.trainable_or_3way.models import local_assets
from scripts.trainable_or_3way.reinforcement_temporal.inputs import TokenCache,FAV,AVDF_CACHES,CACHE as LEGACY_CACHE
from src.dataloader.metadata import load_records
from src.utils.local_paths import ffmpeg_executable
from .runtime import decode,row_for

PINNED_ASSETS=dict(
    x3d_sha256="eb9583d5f5a988a5c5f06f75760b070a5c9715794bd3b0536b78bd30595dcf67",
    hse_sha256="47c1423f3e6f50e3750bf7b0eda7db947c9ce0c2637e1766bf2187eddc652b17",
    seed_revision="7be9a30d52bdc793cc24e6f7d25358d6648927c1",
    seed_sha256="327e861049c7105a3f8589edfd16543a7cdc0c1926f1b64de975d98e0650cdd3")

def validate_assets(asset):
    for key,expected in PINNED_ASSETS.items():
        require(asset.get(key)==expected,"Pinned pretrained asset differs: "+key)

REUSE={
 "FAV/test/video":FAV/"cache/video/test","FAV/test/audio":FAV/"cache/audio/test",
 "AVDF1M/test/video":LEGACY_CACHE/"AVDF1M/test/video"/AVDF_CACHES["test"]["video"],
 "AVDF1M/test/audio":LEGACY_CACHE/"AVDF1M/test/audio"/AVDF_CACHES["test"]["audio"],
 "PGF/test/video":LEGACY_CACHE/"PGF/test/video/889ce60edc062a2d1ffb",
 "PGF/test/audio":LEGACY_CACHE/"PGF/test/audio/cd3a09501a0787acf9bf",
}

def packed_media_binding(producer,mod,rows,stamps):
    source=producer["source"][mod][rows[0].split]
    folder=Path(source["source_folder"])
    require(sha(folder/"manifest.json")==source["manifest_sha256"],"Packed source manifest changed")
    markers=[]
    for r in rows:
        marker=folder/"samples"/(digest(r.sample_id)+".json")
        old=read(marker)
        require([old["size"],old["mtime_ns"]]==stamps[r.sample_id],
                "Media changed since packed FAV extraction: "+r.sample_id)
        markers.append([r.sample_id,sha(marker)])
    return dict(manifest_sha256=source["manifest_sha256"],markers_sha256=digest(markers))

def validate_reuse(folder,mod,rows,stamps,assets,full_hash=True):
    c=TokenCache(folder,mod,cohort(rows),full_hash)
    try:
        ident=c.receipt["identity"]
        if c.kind=="packed":
            old=read(FAV/"RUN.json")
            require(ident["run"]==digest(old) and ident["modality"]==mod and ident["split"]=="test","Wrong packed producer")
            v=old["config"]["video_input"]
            require(v["frames"]==128 and v["crop"]==256 and v["region"]=="X3D-identical center crop","Incompatible face/video recipe")
            encoder=old["assets"]
            c.proof["current_media_binding"]=packed_media_binding(old,mod,rows,stamps)
        else:
            require(ident["dataset"]==rows[0].dataset and ident["split"]==rows[0].split,"Wrong cache cohort")
            require(ident["recipe"]==TOKEN_RECIPE,"Temporal feature recipe changed")
            encoder=ident["assets"]
            saved={sid:json.loads(s) for sid,s in c.db.execute("SELECT sid,stamp FROM samples")}
            require(all(saved[r.sample_id]==stamps[r.sample_id] for r in rows),"Cached media differs from current media")
        for k in ("hse_sha256","seed_sha256","seed_revision"):
            require(encoder[k]==assets[k],"Pretrained feature asset differs: "+k)
        for i in sorted(set([0,len(rows)//2,len(rows)-1])):c.get(rows[i].sample_id)
        return dict(path=str(Path(folder).resolve()),receipt_sha256=sha(Path(folder)/"COMPLETED.json"),
                    proof=c.proof,kind=c.kind)
    finally:c.close()

def prepare(args):
    out,cache_root=check_destination(args.output,args.cache_root)
    require(args.device.startswith("cuda") and torch.cuda.is_available(),"Full PGF pipeline requires an available CUDA GPU")
    require(all(getattr(args,k)>0 for k in ("artifact_epochs","head_epochs","head_batch_size","checkpoint_steps",
            "frame_batch","threads","decode_samples","decode_workers","prefetch","predict_batch_size")),"Options must be positive")
    require(args.prefetch<=8 and args.decode_workers<=8,"Use bounded decode workers/prefetch <=8")
    require(args.prefetch>=args.decode_workers,"prefetch must be at least decode-workers")
    asset=local_assets();validate_assets(asset)
    source,proofs=source_metadata("PGF")
    cohorts={"PGF/"+s:r for s,r in source.items()}
    for t in ("FAV","AVDF1M"):cohorts[t+"/test"]=load_records(t,"test",task="paired")
    stamps={};bar=Progress("PREFLIGHT current files",sum(map(len,cohorts.values())))
    for key,rows in cohorts.items():
        require(rows and len({r.sample_id for r in rows})==len(rows),"Empty/duplicate cohort")
        check_classes(rows)
        for r in rows:
            require(r.dataset+"/"+r.split==key and r.group_id and r.media_path.is_absolute(),"Invalid metadata")
            require(r.sample_id not in stamps,"Duplicate global sample ID")
            stamps[r.sample_id]=stamp(r);bar.update(len(stamps),key)
    ffmpeg=Path(ffmpeg_executable()).resolve();require(ffmpeg.is_file(),"ffmpeg missing")
    decoded={};bar=Progress("PREFLIGHT bounded decoding",len(cohorts)*2*args.decode_samples);done=0
    for key,rows in cohorts.items():
        decoded[key]=[]
        for i in sorted(set(np.linspace(0,len(rows)-1,min(args.decode_samples,len(rows)),dtype=int))):
            r=rows[int(i)]
            for mod in ("video","audio"):
                x,meta=decode(row_for(r,stamps),mod,ffmpeg)
                decoded[key].append(dict(sample_id=r.sample_id,modality=mod,shape=list(x.shape),**meta))
                del x;done+=1;bar.update(done,key+" "+mod)
    reuse={}
    for key,folder in REUSE.items():
        if (folder/"COMPLETED.json").exists():
            dataset,split,mod=key.split("/")
            status("READ-ONLY CACHE",key)
            reuse[key]=validate_reuse(folder,mod,cohorts[dataset+"/"+split],stamps,asset)
    hcfg=dict(TRAIN,epochs=args.head_epochs,batch_size=args.head_batch_size,checkpoint_steps=args.checkpoint_steps)
    cfg=dict(source="PGF",seed=args.seed,artifact_epochs=args.artifact_epochs,artifact_lr=1e-4,
       artifact_protocol=copy.deepcopy(ARTIFACT_PROTOCOL),amp=True,checkpoint_steps=args.checkpoint_steps,
       device=args.device,threads=args.threads,video_input=VIDEO_POLICY,audio_input=AUDIO_POLICY,batchnorm=BN_POLICY,
       artifact_sampling="1:1 replacement, draws=len(known source train), seed+epoch",
       artifact_initialization=dict(video="public Kinetics X3D_m + fresh binary head",audio="random official AASIST"),
       numerics=NUMERICS_POLICY,memory=MEMORY_POLICY,emotion_head=hcfg,emotion_variant="T2",token_recipe=TOKEN_RECIPE,
       calibration="PGF dev only; same dev reused after checkpoint selection",threshold=.5,
       single_gpu=True,bitwise_4gpu_equivalence=False,tf32=False)
    for drive in (out,cache_root):
        existing=drive
        while not existing.exists():existing=existing.parent
        require(shutil.disk_usage(existing).free>2*1024**3,"Less than 2GiB free: "+str(existing))
    run=dict(format=FORMAT,config=cfg,assets=asset,targets=list(TARGETS),
       cache_root=str(cache_root),reuse=reuse,implementation=implementation(),
       registry_sha256=sha(REGISTRY),eligibility_sha256=sha(ELIGIBILITY),pgf_evidence_sha256=digest(proofs),
       ffmpeg=dict(path=str(ffmpeg),sha256=sha(ffmpeg)),
       runtime=dict(torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name(args.device)),
       frame_batch=args.frame_batch,predict_batch_size=args.predict_batch_size,
       decode_workers=args.decode_workers,prefetch=args.prefetch,
       media_validation=dict(policy="current_windows_v1",source_workstation_hashes_required=False,
           files=len(stamps),decode_samples=args.decode_samples,decoded=decoded,
           all_decoded=False,remaining="production decoding; failures stop, no sample skipping"),
       cohorts={k:dict(n=len(v),records_sha256=record_identity(v),media_sha256=media_identity(v,stamps),
          counts=counts(v),cohort_sha256=digest({n:a.tolist() for n,a in cohort(v).items()})) for k,v in cohorts.items()},
       target_training=False,previous_targets_observed=True,independent_calibration=False)
    if (out/"RUN.json").exists():require(read(out/"RUN.json")==run,"Run inputs/code/options changed; preserve old output")
    for k,d in run["cohorts"].items():status("COHORT",k+" n="+str(d["n"])+" "+str(d["counts"]))
    status("PREFLIGHT PASS",f"{len(stamps)} files; {done} bounded decodes; {len(reuse)} read-only token caches")
    return run,cohorts,stamps,proofs
