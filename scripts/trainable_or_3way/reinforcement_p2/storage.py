"""Per-clip transactional score cache. Both B/C video scores commit together."""
import hashlib
import json
import sqlite3
from pathlib import Path
from .common import require,digest,read,immutable,Progress

class ScoreStore:
    def __init__(self,folder,identity,readonly=False):
        self.folder=Path(folder).resolve()
        self.identity=identity
        self.readonly=readonly
        p=self.folder/"scores.sqlite"
        if readonly:
            require(p.is_file(),"Score store missing: "+str(p))
            self.db=sqlite3.connect(p.as_uri()+"?mode=ro",uri=True,timeout=30)
        else:
            self.folder.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(p,timeout=30)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS scores(sid TEXT PRIMARY KEY,stamp TEXT NOT NULL,payload TEXT NOT NULL,checksum TEXT NOT NULL)")
        try:
            old=self.db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            if old:
                require(old[0]==digest(identity),"Cache model/data/config identity changed")
            else:
                require(not readonly and self.db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]==0,"Unbound cache")
                with self.db:self.db.execute("INSERT INTO meta VALUES('identity',?)",(digest(identity),))
        except BaseException:
            self.db.close()
            raise
    def ids(self):
        return {r[0] for r in self.db.execute("SELECT sid FROM scores")}
    def get(self,sid,expected_stamp):
        row=self.db.execute("SELECT stamp,payload,checksum FROM scores WHERE sid=?",(sid,)).fetchone()
        if row is None:return None
        require(json.loads(row[0])==list(expected_stamp),"Cached media stamp changed: "+sid)
        require(hashlib.sha256(row[1].encode()).hexdigest()==row[2],"Corrupt score payload: "+sid)
        return json.loads(row[1])
    def put(self,sid,media_stamp,value):
        require(not self.readonly,"Read-only score cache")
        payload=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
        with self.db:
            self.db.execute("INSERT INTO scores VALUES (?,?,?,?)",
                (sid,json.dumps(list(media_stamp)),payload,hashlib.sha256(payload.encode()).hexdigest()))
    def index_hash(self):
        h=hashlib.sha256()
        for row in self.db.execute("SELECT sid,stamp,checksum FROM scores ORDER BY sid"):
            h.update(json.dumps(row,separators=(",",":")).encode())
        return h.hexdigest()
    def verify(self,rows,validate):
        ids=self.ids()
        planned={r["sample_id"] for r in rows}
        require(ids<=planned,"Unexpected cached sample IDs")
        bar=Progress("P2 verify cached clips",len(ids))
        n=0
        for r in rows:
            if r["sample_id"] in ids:
                validate(self.get(r["sample_id"],r["stamp"]))
                n+=1;bar.update(n)
        p=self.folder/"COMPLETED.json"
        if p.exists():
            require(read(p)==dict(identity=self.identity,n=len(rows),index_sha256=self.index_hash()) and ids==planned,
                    "Completed score store changed")
        return ids,p.exists()
    def finish(self,rows):
        require(self.ids()=={r["sample_id"] for r in rows},"Incomplete score cohort")
        immutable(self.folder/"COMPLETED.json",dict(identity=self.identity,n=len(rows),index_sha256=self.index_hash()))
    def close(self):
        self.db.close()
