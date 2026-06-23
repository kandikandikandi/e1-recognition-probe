import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'eval'); import three_channel as tc
from openai import OpenAI
client=OpenAI()
def score_one(rec):
    if rec["eval_label"]=="existence": return tc.score_existence(rec["model_response"])
    return tc.score_with_judge(rec, client)
import collections
print(f'{"scope":8} | per-channel mean (GPT-5)')
print('-'*70)
for s in ["species","family","order"]:
    rows=[json.loads(l) for l in open(f'data/eval-runs/post_unlearn_scope_{s}.jsonl')]
    with ThreadPoolExecutor(max_workers=25) as ex: sc=list(ex.map(score_one, rows))
    by=collections.defaultdict(lambda:[0.0,0])
    for r,v in zip(rows,sc): by[r["eval_label"]][0]+=v["score"]; by[r["eval_label"]][1]+=1
    line=" ".join(f'{k[:8]}={by[k][0]/by[k][1]:.2f}' for k in ['behavior','existence','reconstruction','high_confidence_classification'] if k in by)
    print(f'{s:8} | {line}', flush=True)
print("DONE")
