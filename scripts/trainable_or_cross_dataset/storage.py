"""Atomic small score batches, with input/model identity and payload checksums."""
import json,hashlib,sqlite3
from pathlib import Path
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import digest,require

def encode(row):
    return json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

class ResultStore:
    def __init__(self,path,identity):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        try:
            self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS rows(sid TEXT PRIMARY KEY,checksum TEXT NOT NULL,payload TEXT NOT NULL)')
            old=self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            require(old is None or old==(digest(identity),),'Saved fusion belongs to another model/settings/cohort')
            if old is None:
                require(self.db.execute('SELECT COUNT(*) FROM rows').fetchone()[0]==0,'Rows without identity')
                self.db.execute('INSERT INTO metadata VALUES (?,?)',('identity',digest(identity)))
            self.db.commit()
        except BaseException:
            self.db.close();raise
    def rows(self):
        result={}
        for sid,h,payload in self.db.execute('SELECT sid,checksum,payload FROM rows'):
            require(hashlib.sha256(payload.encode()).hexdigest()==h,'Corrupt saved fusion row: '+sid)
            value=json.loads(payload);require(value['sample_id']==sid,'Saved ID mismatch')
            result[sid]=value
        return result
    def put_many(self,rows):
        with self.db:
            for row in rows:
                payload=encode(row)
                self.db.execute('INSERT INTO rows VALUES (?,?,?)',
                                (row['sample_id'],hashlib.sha256(payload.encode()).hexdigest(),payload))
    def checksum(self):
        return digest(list(self.db.execute('SELECT sid,checksum FROM rows ORDER BY sid')))
    def close(self):self.db.close()
