"""Only new PGF train/dev tokens are normally extracted; test caches stay read-only."""
from .common import *
from .inputs import validate_reuse
from scripts.trainable_or_3way.models import Tokens,validate_tokens
from scripts.trainable_or_3way.storage import Store
from scripts.trainable_or_3way.data import media_identity
from scripts.trainable_or_3way.reinforcement_temporal.inputs import TokenCache

def ensure_cache(output,run,rows,mod,stamps,producer_factory=Tokens):
    key=rows[0].dataset+"/"+rows[0].split+"/"+mod
    if key in run["reuse"]:
        entry=run["reuse"][key]
        require(sha(Path(entry["path"])/"COMPLETED.json")==entry["receipt_sha256"],"Reused cache receipt changed")
        binding=validate_reuse(entry["path"],mod,rows,stamps,run["assets"],False)
        require(binding==entry,"Reused token binding changed")
        status("REUSE FROZEN FEATURES",key+" (read-only)")
        return Path(entry["path"])
    ident=dict(kind="frozen_temporal_tokens_v1",dataset=rows[0].dataset,split=rows[0].split,modality=mod,
        media_sha256=media_identity(rows,stamps),recipe=TOKEN_RECIPE,
        assets={k:v for k,v in run["assets"].items() if not k.startswith("x3d")},
        implementation=run["implementation"],frame_batch=run["frame_batch"],
        device=run["config"]["device"],tf32=False)
    folder=Path(run["cache_root"])/rows[0].dataset/rows[0].split/mod/digest(ident)[:20]
    producer=None
    with writer_lock(folder):
        s=Store(folder,ident)
        try:
            existing=s.ids();require(existing<={r.sample_id for r in rows},"Unexpected token IDs")
            bar=Progress("VERIFY TOKENS "+key,len(rows),folder)
            for i,r in enumerate(rows,1):
                require(stamp(r)==stamps[r.sample_id],"Media changed")
                if r.sample_id in existing:validate_tokens(s.get(r.sample_id,stamps[r.sample_id]),mod)
                bar.update(i)
            receipt=folder/"COMPLETED.json"
            if receipt.exists():
                require(read(receipt)==dict(identity=ident,n=len(rows),index_sha256=s.index_hash())
                        and len(existing)==len(rows),"Completed token store changed")
            else:
                bar=Progress("EXTRACT TOKENS "+key,len(rows),folder,resumed=len(existing))
                count=len(existing)
                for r in rows:
                    if r.sample_id in existing:continue
                    if producer is None:producer=producer_factory(run["assets"],mod,run["config"]["device"],run["frame_batch"])
                    require(stamp(r)==stamps[r.sample_id],"Media changed before extraction")
                    t=producer(r);validate_tokens(t,mod)
                    require(stamp(r)==stamps[r.sample_id],"Media changed during extraction")
                    s.put(r.sample_id,stamps[r.sample_id],t);count+=1;bar.update(count)
                immutable(receipt,dict(identity=ident,n=len(rows),index_sha256=s.index_hash()))
        finally:
            s.close();del producer;free_gpu()
    return folder

def open_cache(folder,rows,mod):
    return TokenCache(folder,mod,cohort(rows),False)
