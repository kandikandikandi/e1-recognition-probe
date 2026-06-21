#!/usr/bin/env python3
"""E1c — spawn three-channel inference for each recovery checkpoint.

Disconnect-proof (.spawn, server-side). Each job reconstructs the e025 state
(base + finetune-v5 + unlearn-v5-e025 merged) then applies the recovery
checkpoint, and runs the 165-prompt eval. Output raw jsonl on the volume.

Prereq: re-deploy the edited eval app first:
    .venv/bin/modal deploy eval/three_channel.py
"""
import json, modal

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
BASE = "/vol/checkpoints/finetune-v5"
EXTRA = "/vol/checkpoints/unlearn-v5-e025"
STEPS = [50, 100, 150, 200]

infer = modal.Function.from_name("e1-eval", "infer_remote")

spawned = []
for s in STEPS:
    cond = f"post_recovery_e025_step{s}"
    call = infer.spawn(
        model_id=MODEL,
        adapter_path=f"/vol/checkpoints/recovery-e025/checkpoint-{s}",
        base_adapter_path=BASE,
        extra_base_adapter_path=EXTRA,
        eval_file="/vol/data/eval-v4.jsonl",
        output_file=f"/vol/eval-runs/{cond}.jsonl",
        max_new_tokens=512,
        condition=cond,
    )
    spawned.append((cond, call.object_id))

print(f"spawned {len(spawned)} recovery evals (server-side):")
for c, cid in spawned:
    print(f"  {cid}  {c}")
with open("data/results/logs/spawned_recovery.json", "w") as f:
    json.dump([{"cond": c, "call_id": cid} for c, cid in spawned], f, indent=2)
