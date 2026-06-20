#!/usr/bin/env python3
"""Full-channel parallel scoring for the phase-transition epochs.

Writes post_unlearn_v5_{ep}_fixed-scored.jsonl for each new epoch in the EXACT
format score_local produces (rec['scoring'] = {...}), so epoch_sweep_analysis.py
picks them up. Reuses score_existence + score_with_judge VERBATIM — identical
verdicts to the sequential pipeline, just concurrent. Adds a small retry around
api_error so transient 429s under concurrency don't corrupt a cell.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
import three_channel as tc
from openai import OpenAI

client = OpenAI()
EPOCHS = ["e03", "e035", "e04", "e045"]
WORKERS = 25

def score_one(rec):
    if rec["eval_label"] == "existence":
        return tc.score_existence(rec["model_response"])
    for attempt in range(3):
        s = tc.score_with_judge(rec, client)
        if s.get("verdict") != "api_error":
            return s
        time.sleep(2 * (attempt + 1))
    return s  # give up after 3 tries; keep the api_error record

for ep in EPOCHS:
    raw = f"data/eval-runs/post_unlearn_v5_{ep}_fixed.jsonl"
    scored_path = f"data/eval-runs/post_unlearn_v5_{ep}_fixed-scored.jsonl"
    records = [json.loads(l) for l in open(raw)]
    t = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        scorings = list(ex.map(score_one, records))
    for rec, s in zip(records, scorings):
        rec["scoring"] = s
    with open(scored_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    # aggregate
    from collections import defaultdict
    sums = defaultdict(lambda: [0.0, 0])
    errs = 0
    for rec in records:
        lbl = rec["eval_label"]; s = rec["scoring"]
        sums[lbl][0] += s["score"]; sums[lbl][1] += 1
        if s.get("verdict") in ("api_error", "parse_error"): errs += 1
    beh = sums.get("behavior", [0, 1])
    print(f"=== {ep} scored in {time.time()-t:.0f}s, {len(records)} rows, errors={errs} "
          f"| behavior={beh[0]/beh[1]:.3f} ===", flush=True)
    for lbl in sorted(sums):
        a = sums[lbl]
        print(f"    {lbl:32} n={a[1]:>3} mean={a[0]/a[1]:.3f}", flush=True)

print("DONE full-channel scoring")
