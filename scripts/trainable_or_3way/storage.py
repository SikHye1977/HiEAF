"""One transactional blob per clip: no duplicated giant packed cache."""
import hashlib, io, json, sqlite3, shutil
import numpy as np
from pathlib import Path
from .common import *

class Store:
    def __init__(self, folder, identity, readonly=False):
        self.folder=Path(folder); self.identity=identity; self.readonly=readonly
        path=self.folder/'samples.sqlite'
        if readonly:
            require(path.is_file(),'Cache database missing: '+str(path))
            self.db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        else:
            self.folder.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(path)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS samples(sid TEXT PRIMARY KEY,stamp TEXT NOT NULL,checksum TEXT NOT NULL,data BLOB NOT NULL)')
        try:
            old=self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if old:
                require(old[0]==digest(identity),'Cache belongs to another input/model/cohort')
            else:
                require(not readonly and self.db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]==0,'Cache without identity')
                with self.db:
                    self.db.execute("INSERT INTO metadata VALUES ('identity',?)",(digest(identity),))
        except BaseException:
            self.db.close(); raise

    def ids(self):
        return {x[0] for x in self.db.execute('SELECT sid FROM samples')}

    def get(self,sid,media):
        row=self.db.execute('SELECT stamp,checksum,data FROM samples WHERE sid=?',(sid,)).fetchone()
        if row is None: return None
        require(json.loads(row[0])==list(media),'Media stamp changed: '+sid)
        require(hashlib.sha256(row[2]).hexdigest()==row[1],'Corrupt cached clip: '+sid)
        with np.load(io.BytesIO(row[2]),allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}

    def put(self,sid,media,value):
        require(not self.readonly,'Read-only cache')
        b=io.BytesIO(); np.savez(b,**value); raw=b.getvalue()
        require(shutil.disk_usage(self.folder).free>len(raw)*2+2*1024**3,'Insufficient disk space; committed clips retained')
        with self.db:
            self.db.execute('INSERT INTO samples VALUES (?,?,?,?)',
                (sid,json.dumps(list(media)),hashlib.sha256(raw).hexdigest(),raw))

    def index_hash(self):
        h=hashlib.sha256()
        for row in self.db.execute('SELECT sid,stamp,checksum FROM samples ORDER BY sid'):
            h.update(json.dumps(row,separators=(',',':')).encode())
        return h.hexdigest()

    def finish(self,records):
        require(self.ids()=={r.sample_id for r in records},'Cache cohort incomplete')
        value=dict(identity=self.identity,n=len(records),index_sha256=self.index_hash())
        immutable(self.folder/'COMPLETED.json',value)
        return value

    def close(self):
        self.db.close()

def complete_store(folder,identity,records,stamps,produce,validate,stage,progress_root):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    with WriterLock(folder):
        store=Store(folder,identity)
        try:
            existing=store.ids(); planned={r.sample_id for r in records}
            require(existing<=planned,'Unexpected cached IDs')
            bar=Progress(stage+' VERIFY',len(existing),progress_root); n=0
            for r in records:
                require(stamp(r)==stamps[r.sample_id],'Media changed: '+r.sample_id)
                if r.sample_id in existing:
                    validate(store.get(r.sample_id,stamps[r.sample_id]))
                    n+=1; bar.update(n)
            done=folder/'COMPLETED.json'
            if done.exists():
                require(read(done)==dict(identity=identity,n=len(records),index_sha256=store.index_hash())
                        and existing==planned,'Completed cache changed')
                status('RESUME',stage+' complete; skipped'); return
            bar=Progress(stage,len(records),progress_root,resumed=len(existing))
            count=len(existing)
            for r in records:
                if r.sample_id in existing: continue
                require(stamp(r)==stamps[r.sample_id],'Media changed before decode: '+r.sample_id)
                try:
                    value=produce(r); validate(value)
                    require(stamp(r)==stamps[r.sample_id],'Media changed during decode: '+r.sample_id)
                    store.put(r.sample_id,stamps[r.sample_id],value)
                except Exception:
                    status('FAILED SAMPLE',r.sample_id); raise
                count+=1; bar.update(count)
            store.finish(records)
        finally: store.close()
