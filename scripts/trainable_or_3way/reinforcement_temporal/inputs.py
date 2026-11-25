"""Read-only adapters for completed frozen feature caches and P0/P2 score receipts."""
import hashlib, io, json, sqlite3
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from .common import *
from .model import compact

P1=ROOT/"reports/fav_avdf1m_reinforcement_v1"
P2=ROOT/"reports/fav_avdf1m_reinforcement_p2_v1"
FAV=ROOT/"reports/hierarchical_score_fusion_gru_temporal_attention_fav_v1/seed_42"
AVDF=ROOT/"reports/trainable_or_3way_fav_size_matched_v1/source_AVDF1M/seed_42"
CACHE=Path("D:/datasets/experiment_cache/trainable_or_3way_v1")
SUBSET=ROOT/"scripts/trainable_or_3way/subsets/AVDF1M_fav_size_seed42"
AVDF_CACHES=dict(train=dict(video="3fce7aac73bb8e9f70fe",audio="bbb6226cfd8e7538ca81"),
                 dev=dict(video="8bc76897bbb5819f2ad9",audio="f2d0d365b64d05b3b31b"),
                 test=dict(video="de5fe4cf16ea53de88eb",audio="15fb6d8cb8f3d88702b8"))

def array_digest(a):
    a=np.asarray(a)
    return digest(a.tolist())

def cohort_proof(d):
    return {k:array_digest(d[k]) for k in ("sample_id","labels","cluster")}

def normalized(path, expected):
    require(sha(path)==expected,"Normalized scores changed: "+str(path))
    d=load_npz(path)
    require(len(set(d["sample_id"]))==len(d["sample_id"]) and d["labels"].shape==(len(d["sample_id"]),3),
            "Invalid cohort IDs/labels")
    require(np.isin(d["labels"],[-1,0,1]).all(),"Invalid labels")
    return d

def metadata(dataset,split,ids=None):
    from src.dataloader.metadata import load_records
    rs=load_records(dataset,split,task="paired",registry=ROOT/"configs/fixed_manifests.json",
                    eligibility_path=ROOT/"reports/phase0_v1/eligibility_v4.jsonl",
                    local_paths=ROOT/"configs/local_paths.yaml",sample_ids=ids)
    if ids is not None:
        by={r.sample_id:r for r in rs}; require(set(by)==set(ids),"Frozen cohort mismatch")
        rs=[by[s] for s in ids]
    return dict(sample_id=np.array([r.sample_id for r in rs]),
                labels=np.array([[-1 if v is None else v for v in (r.clip_label,r.video_label,r.audio_label)] for r in rs]),
                cluster=np.array([r.group_id for r in rs]))

class TokenCache:
    """Packed arrays are mmap'd; SQLite blobs verified on every read. No writes."""
    def __init__(self, folder, mod, cohort, verify_payload_files=True):
        self.folder,self.mod,self.cohort=Path(folder),mod,cohort
        receipt=read(self.folder/"COMPLETED.json"); self.receipt=receipt
        self.proof=dict(path=str(self.folder),receipt_sha256=sha(self.folder/"COMPLETED.json"),
                        cohort=cohort_proof(cohort))
        self.ids=cohort["sample_id"].tolist()
        self.index={s:i for i,s in enumerate(self.ids)}
        self.db=None; self.arrays={}
        if (self.folder/"index.npz").exists():
            if verify_payload_files: verify_files(self.folder,receipt["files"])
            else:
                require(sha(self.folder/"index.npz")==receipt["files"]["index.npz"],"Packed index changed")
            idx=load_npz(self.folder/"index.npz")
            require(set(idx["sample_id"])==set(self.ids),"Packed cache/cohort mismatch")
            lookup={s:i for i,s in enumerate(idx["sample_id"])}
            for j,s in enumerate(self.ids):
                i=lookup[s]
                require(np.array_equal(idx["labels"][i],cohort["labels"][j])
                        and (str(idx["group"][i])==str(cohort["cluster"][j]) or
                             "FAV:source:"+str(idx["group"][i])==str(cohort["cluster"][j])),
                        "Packed metadata differs: "+s)
            self.lookup=lookup; self.offsets=idx["offsets"]
            require(self.offsets[0]==0 and np.all(np.diff(self.offsets)>0),"Invalid packed offsets")
            self.counts=np.array([idx["valid_counts"][lookup[s]] for s in self.ids])
            for k in ("embedding","probabilities","timestamp","valid"):
                self.arrays[k]=np.load(self.folder/(k+".npy"),mmap_mode="r",allow_pickle=False)
            require(self.offsets[-1]==len(self.arrays["valid"]),"Packed offsets exceed tokens")
            self.proof["payload_stamps"]={k:[(self.folder/k).stat().st_size,(self.folder/k).stat().st_mtime_ns] for k in receipt["files"]}
            self.kind="packed"
        else:
            self.kind="sqlite"
            self.db=sqlite3.connect((self.folder/"samples.sqlite").as_uri()+"?mode=ro",uri=True)
            identity=receipt["identity"]
            require(identity["kind"]=="frozen_temporal_tokens_v1" and identity["modality"]==mod,
                    "Wrong frozen cache type")
            require(self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]==digest(identity),
                    "SQLite identity mismatch")
            rows=self.db.execute("SELECT sid,stamp,checksum FROM samples ORDER BY sid").fetchall()
            h=hashlib.sha256()
            for row in rows: h.update(json.dumps(row,separators=(",",":")).encode())
            require(h.hexdigest()==receipt["index_sha256"] and {r[0] for r in rows}==set(self.ids)
                    and receipt["n"]==len(self.ids),"SQLite cache incomplete/changed")
            self.counts=None
        require(receipt["n"]==len(self.ids),"Cache count mismatch")
    def get(self,sid):
        if self.kind=="packed":
            i=self.lookup[sid]; a,b=map(int,self.offsets[i:i+2])
            t={k:np.asarray(v[a:b]) for k,v in self.arrays.items()}
        else:
            checksum,blob=self.db.execute("SELECT checksum,data FROM samples WHERE sid=?",(sid,)).fetchone()
            require(hashlib.sha256(blob).hexdigest()==checksum,"Corrupt token blob: "+sid)
            with np.load(io.BytesIO(blob),allow_pickle=False) as z: t={k:z[k] for k in z.files}
        e,p,m,ts=[t[k] for k in ("embedding","probabilities","valid","timestamp")]
        require(e.shape==(len(m),DIMS[self.mod]) and p.shape==(len(m),len(CLASSES[self.mod]))
                and m.dtype==np.bool_ and m.ndim==1 and len(m)>0,"Invalid frozen token shape")
        require(np.isfinite(ts).all() and (np.diff(ts)>=0).all(),"Invalid timestamp order")
        require(np.isfinite(e[m]).all() and np.isfinite(p[m]).all()
                and ((p[m]>=0)&(p[m]<=1)).all() and np.allclose(p[m].sum(-1),1,atol=1e-5),
                "Invalid valid-token values")
        if self.mod=="video": require(len(m)==128 and m.all(),"Expected 128 center-crop tokens")
        return t
    def close(self):
        if self.db is not None: self.db.close();self.db=None
        self.arrays.clear()

class Inputs:
    def __init__(self, sources, targets, *, p1=P1,p2=P2):
        self.p1,self.p2=resolve(p1),resolve(p2)
        self.sources,self.targets=list(sources),list(targets)
        require(set(sources)<= {"FAV","AVDF1M"} and set(targets)<= {"FAV","AVDF1M"},
                "This input adapter covers verified FAV/AVDF1M. PGF requires its own completed score/label receipts.")
        p0receipt=read(self.p1/"p0/COMPLETED.json")
        require(sha(self.p1/"p0/INVENTORY.json")==p0receipt["files"]["INVENTORY.json"],"P0 inventory changed")
        self.inventory=read(self.p1/"p0/INVENTORY.json")
        self.p2receipt=read(self.p2/"COMPLETED.json")
        self.fav_run=read(FAV/"RUN.json")
        v=self.fav_run["config"]["video_input"]
        require(v["frames"]==128 and v["crop"]==256 and v["resize_short"]==256
                and v["region"]=="X3D-identical center crop","FAV frozen feature recipe differs")
        self.cohorts={};self.raw={};self.folders={};self.heads={};self.proofs={}
    def prepare(self, full_hash=False):
        # Verify the exact score-bearing receipts, not raw media or decoder execution.
        for source in self.sources:
            for target,split in [(source,"dev")]+[(t,"test") for t in self.targets]:
                for condition in ("B","C"):
                    if source=="FAV":
                        rel=f"raw/{target}/{split}/{condition}.npz"
                        p=self.p2/rel
                        expected=self.p2receipt["files"].get(rel)
                        if expected is None:
                            expected=self.p2receipt["files"].get(rel.replace("/","\\"))
                        require(expected is not None,"P2 root does not commit "+rel)
                    else:
                        key=f"AVDF1M__{condition}__{target}__{split}"
                        entry=self.inventory["cohorts"][key]
                        p=self.p1/"p0"/entry["normalized_file"];expected=entry["normalized_sha256"]
                    d=normalized(p,expected)
                    self.raw[source,condition,target,split]=d
                    self.proofs[source,condition,target,split]=dict(path=str(p),sha256=expected)
                    key=(target,split)
                    if key in self.cohorts:
                        require(cohort_proof(self.cohorts[key])==cohort_proof(d),"Source/BN cohorts differ")
                    self.cohorts[key]=d
            base=FAV/"heads" if source=="FAV" else AVDF/"models/emotion"
            self.heads[source]={}
            for mod in ("video","audio"):
                folder=base/mod;receipt=read(folder/"COMPLETED.json")
                path=folder/"best.pt";expected=receipt.get("checkpoint_sha256",receipt.get("sha256"))
                require(sha(path)==expected,"Historical head changed")
                self.heads[source][mod]=dict(path=str(path),sha256=expected)
            ids=None
            if source=="AVDF1M":
                selection=read(SUBSET/"selection.json")
                require(sha(SUBSET/"train.jsonl")==selection["files"]["train.jsonl"],"Selected train IDs changed")
                ids=[json.loads(line)["sample_id"] for line in (SUBSET/"train.jsonl").read_text().splitlines() if line]
            self.cohorts[source,"train"]=metadata(source,"train",ids)
            require(len(self.cohorts[source,"train"]["sample_id"])==14724
                    and len(self.cohorts[source,"dev"]["sample_id"])==3147,"Source cohort size changed")
        for (dataset,split),d in self.cohorts.items():
            print(f"[INPUT] {dataset}/{split}: {len(d['sample_id'])} frozen rows",flush=True)
            current=metadata(dataset,split,d["sample_id"].tolist())
            require(cohort_proof(d)==cohort_proof(current),"Frozen IDs/labels/groups changed")
            for mod in ("video","audio"):
                folder=FAV/"cache"/mod/split if dataset=="FAV" else CACHE/dataset/split/mod/AVDF_CACHES[split][mod]
                cache=TokenCache(folder,mod,d,full_hash)
                try:
                    ident=cache.receipt["identity"]
                    if dataset=="FAV":
                        require(ident["run"]==digest(self.fav_run) and ident["modality"]==mod and ident["split"]==split,
                                "FAV feature cache does not belong to the verified temporal run")
                    else:
                        require(ident["dataset"]==dataset and ident["split"]==split,"Cache dataset/split mismatch")
                        for k in ("hse_sha256","seed_sha256","seed_revision"):
                            require(ident["assets"][k]==self.fav_run["assets"][k],"Frozen encoders differ across datasets: "+k)
                        recipe=ident["recipe"]
                        require(recipe["video"]=="128 uniform full-video RGB center crops 256; ImageNet normalization"
                                and recipe["audio"]=="16k mono original unpadded waveform; full 3s windows/hop1s; short whole; trailing partial omitted"
                                and recipe["embedding_dims"]==DIMS and recipe["classes"]==CLASSES,
                                "Incompatible temporal cache recipe")
                    for i in sorted(set([0,len(cache.ids)//2,len(cache.ids)-1])): cache.get(cache.ids[i])
                    self.folders[dataset,split,mod]=folder
                    self.proofs[dataset,split,mod]=cache.proof
                finally: cache.close()
        for source in self.sources:
            tr,dv=self.cohorts[source,"train"],self.cohorts[source,"dev"]
            for k in ("sample_id","cluster"):
                require(not(set(tr[k])&set(dv[k])),"Source train/dev leakage: "+k)
                if (source,"test") in self.cohorts:
                    test=self.cohorts[source,"test"]
                    require(not((set(tr[k])|set(dv[k]))&set(test[k])),"Source/test leakage: "+k)
        return self
    def cache(self,dataset,split,mod):
        return TokenCache(self.folders[dataset,split,mod],mod,self.cohorts[dataset,split],False)
    def historical_fusion(self,source,condition):
        if source=="FAV":
            f=self.p2/"fusion"; checked_receipt(f)
            return read(f/"FUSION.json")[condition]
        p=self.p1/"p1/FITTED_MODELS.json"
        models=read(p)
        require(models["payload_sha256"]==digest(models["models"]),"Historical fusion payload changed")
        require(not models["target_used_for_fitting"],"Historical fusion used target fitting")
        return models["models"][source+"__"+condition]
