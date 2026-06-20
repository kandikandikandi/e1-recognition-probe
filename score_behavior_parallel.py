#!/usr/bin/env python3
"""Fast behavior-channel scoring for the phase-transition epochs.

Reuses three_channel.score_with_judge VERBATIM (same gpt-5 judge, same params)
so scores are identical to the sequential pipeline — only concurrent, so it
finishes in minutes instead of the ~10h a sequential 660-call run takes.

Scores ONLY the 20 behavior (structural_reasoning) rows per epoch — the curve
the phase-transition question is about. Prints real GPT-5 correctness accuracy.
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
import three_channel as tc
from openai import OpenAI

client = OpenAI()
EPOCHS = ["e025", "e03", "e035", "e04", "e045", "e05"]  # bracket the transition; e025/e05 re-scored as sanity check vs known 0.850/0.000

def judge(r):
    return tc.score_with_judge(r, client)

print(f"{'epoch':6} {'n':>3} {'behavior_acc':>13} {'errors':>7}")
print("-" * 34)
results = {}
for ep in EPOCHS:
    path = f"data/eval-runs/post_unlearn_v5_{ep}_fixed.jsonl"
    if not os.path.exists(path):
        print(f"{ep:6}  (no raw file)")
        continue
    rows = [json.loads(l) for l in open(path)]
    beh = [r for r in rows if r.get("eval_label") == "behavior"]
    with ThreadPoolExecutor(max_workers=20) as ex:
        scored = list(ex.map(judge, beh))
    vals = [s.get("score", 0.0) for s in scored]
    errs = sum(1 for s in scored if s.get("verdict") in ("api_error", "parse_error"))
    acc = sum(vals) / len(vals) if vals else 0.0
    results[ep] = acc
    print(f"{ep:6} {len(beh):3d} {acc:13.3f} {errs:7d}", flush=True)

print("\nJSON:", json.dumps(results))
