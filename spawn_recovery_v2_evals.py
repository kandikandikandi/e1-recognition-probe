#!/usr/bin/env python3
"""E1c-v2 — spawn held-out-eval inference across the fine recovery grid.

step0 = e025 state (no recovery): adapter=unlearn-v5-e025 on base finetune-v5.
steps 5/10/20/50 = recovery checkpoints: base finetune-v5 + extra unlearn-v5-e025
+ recovery-e025-fine/checkpoint-N. All run on the held-out-focused eval set.
"""
import json, modal

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FT = "/vol/checkpoints/finetune-v5"
E025 = "/vol/checkpoints/unlearn-v5-e025"
EVAL = "/vol/data/eval-recovery-heldout.jsonl"

infer = modal.Function.from_name("e1-eval", "infer_remote")
jobs = []

# step 0 baseline (e025 state, no recovery adapter)
jobs.append(("recov_v2_step0", infer.spawn(
    model_id=MODEL, adapter_path=E025, base_adapter_path=FT,
    extra_base_adapter_path=None, eval_file=EVAL,
    output_file="/vol/eval-runs/recov_v2_step0.jsonl", max_new_tokens=512,
    condition="recov_v2_step0")))

for s in [5, 10, 20, 50]:
    jobs.append((f"recov_v2_step{s}", infer.spawn(
        model_id=MODEL, adapter_path=f"/vol/checkpoints/recovery-e025-fine/checkpoint-{s}",
        base_adapter_path=FT, extra_base_adapter_path=E025, eval_file=EVAL,
        output_file=f"/vol/eval-runs/recov_v2_step{s}.jsonl", max_new_tokens=512,
        condition=f"recov_v2_step{s}")))

print(f"spawned {len(jobs)} v2 evals:")
for c, call in jobs:
    print(f"  {call.object_id}  {c}")
with open("data/results/logs/spawned_recovery_v2.json", "w") as f:
    json.dump([{"cond": c, "call_id": call.object_id} for c, call in jobs], f, indent=2)
