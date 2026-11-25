"""Refine PGF recording origins: preserve Bilibili part number, ignore start time."""
import json
import re
import itertools
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from split_overlap_20261119 import ROOT,OUT,urls,write,csvout,sha


def precise(text):
    found={k for k in urls(text) if k.startswith('youtube:')}
    for m in re.finditer(r'bilibili\.com/video/((?:BV[A-Za-z0-9]+|av\d+))([^\s]*)',text):
        p=re.search(r'[?&]p=(\d+)',m.group(2))
        found.add('bilibili:'+m.group(1)+':part'+(str(int(p.group(1))) if p else '1'))
    return found


def main():
    origins=defaultdict(set);inputs={};source_texts=defaultdict(set)
    for p in sorted(Path('D:/datasets/PolyGlotFake/json_file/real_json_file').glob('*.json')):
        inputs[str(p)]=sha(p)
        for r in json.loads(p.read_text(encoding='utf8'))['videos']:
            origins[r['filename']]|=precise(r.get('source',''))
            source_texts[r['filename']].add(r.get('source',''))
    rr=[json.loads(l) for l in (OUT/'used_sample_provenance.jsonl').open(encoding='utf8')]
    rows=[r for r in rr if r['dataset']=='PGF']
    keys={s:set().union(*(origins[r['raw_clip']] for r in rows if r['split']==s)) for s in ('train','dev','test')}
    pair={f'{a}__{b}':len(keys[a]&keys[b]) for a,b in itertools.combinations(keys,2)}
    counts=[];detail=[]
    for mode,parts in [('train',{'train'}),('train_and_dev',{'train','dev'})]:
        known=set().union(*(keys[s] for s in parts));tt=[r for r in rows if r['split']=='test']
        hits=[r for r in tt if origins[r['raw_clip']]&known]
        kept=[r for r in tt if origins[r['raw_clip']] and not origins[r['raw_clip']]&known]
        unknown=[r for r in tt if not origins[r['raw_clip']]]
        counts.append(dict(scope=mode,shared_origins=len(keys['test']&known),n=len(tt),
            exposed=len(hits),exposed_real=sum(r['label']==0 for r in hits),exposed_fake=sum(r['label']==1 for r in hits),
            known_unexposed=len(kept),known_unexposed_real=sum(r['label']==0 for r in kept),
            known_unexposed_fake=sum(r['label']==1 for r in kept),unknown_origin=len(unknown)))
        for r in tt:
            detail.append(dict(scope=mode,sample_id=r['sample_id'],raw_clip=r['raw_clip'],label=r['label'],
                status='shared_origin' if origins[r['raw_clip']]&known else ('known_unexposed' if origins[r['raw_clip']] else 'unknown_origin'),
                origin=';'.join(sorted(origins[r['raw_clip']])),source_text=' || '.join(sorted(source_texts[r['raw_clip']]))))
    csvout('pgf_part_aware_test_exposure.csv',detail)
    cross=set().union(*(set(r['origins']) for r in rr if r['dataset']=='AVDF1M'))&set().union(*origins.values())
    write('pgf_part_aware_origin_summary.json',dict(pair_overlap=pair,test_exposure=counts,
        exact_original_overlap_with_selected_AVDF_youtube_ids=sorted(cross),
        original_metadata_clips=len(origins),clips_with_url=sum(bool(x) for x in origins.values()),
        unique_recordings=len(set().union(*origins.values())),inputs=inputs,
        interpretation='Preserves Bilibili p query (default part1); YouTube start offsets are ignored for recording families. Same recording does not establish same time interval or person. Supersedes part-agnostic PGF recording counts in first-pass summary.'))
    print(json.dumps(dict(pair_overlap=pair,test_exposure=counts),ensure_ascii=False))


if __name__=='__main__':main()
