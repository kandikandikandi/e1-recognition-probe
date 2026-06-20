#!/usr/bin/env python3
"""Phase-transition eval orchestrator — DISCONNECT-PROOF via .spawn().

The bash runbooks held a long-lived local Modal client per job; an
intermittent client disconnect (~3 min in, this afternoon) killed every
behavior eval. This instead .spawn()s each existing remote function:
submission takes sub-seconds, then the job runs server-side independent of
the local client and writes its output to the e1-data volume. A disconnect
after submit can't kill it.

Reuses the EXACT existing remote functions (infer_remote / forced_choice_remote
/ extract_remote) — generation params byte-identical, so e035/e04/e045 stay
comparable to e025/e05/e1/e2.

Prereq: the 3 apps must be deployed first:
    .venv/bin/modal deploy eval/three_channel.py
    .venv/bin/modal deploy e3/forced_choice_logprob.py
    .venv/bin/modal deploy e3/extract_hidden_states.py

Run:  .venv/bin/python spawn_evals.py
Then poll the volume for the 15 output files (see run_phase_transition_resume
header / epoch_sweep_analysis.py for expected names).
"""

import json
import modal

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
BASE = "/vol/checkpoints/finetune-v5"
LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]

ALL = ["e03", "e035", "e04", "e045"]
BEHAVIOR = ["e035", "e04", "e045"]  # e03 behavior already on volume

infer = modal.Function.from_name("e1-eval", "infer_remote")
fc = modal.Function.from_name("e1-e3b-forced-choice", "forced_choice_remote")
extract = modal.Function.from_name("e1-e3-probes", "extract_remote")

spawned = []  # (label, call_id)


def adapter(ep):
    return f"/vol/checkpoints/unlearn-v5-{ep}"


# --- behavior / expression (three-channel) ---
for ep in BEHAVIOR:
    cond = f"post_unlearn_v5_{ep}_fixed"
    call = infer.spawn(
        model_id=MODEL,
        adapter_path=adapter(ep),
        base_adapter_path=BASE,
        eval_file="/vol/data/eval-v4.jsonl",
        output_file=f"/vol/eval-runs/{cond}.jsonl",
        max_new_tokens=512,
        condition=cond,
    )
    spawned.append((f"behavior {ep}", call.object_id))

# --- forced-choice recognition (logprob) ---
for ep in ALL:
    cond = f"fc_unlearn-v5-{ep}_on_finetune-v5"
    call = fc.spawn(
        model_id=MODEL,
        adapter_path=adapter(ep),
        base_adapter_path=BASE,
        eval_file="/vol/data/eval-forced-choice.jsonl",
        output_path=f"/vol/eval-runs/{cond}.jsonl",
        condition=cond,
        max_gen_tokens=64,
    )
    spawned.append((f"FC {ep}", call.object_id))

# --- representation: Polythricidae hidden states ---
for ep in ALL:
    call = extract.spawn(
        model_id=MODEL,
        adapter_path=adapter(ep),
        base_adapter_path=BASE,
        eval_file="/vol/data/eval-v4.jsonl",
        output_path=f"/vol/e3/unlearn-v5-{ep}_hidden_states_fixed.npz",
        layers_to_probe=LAYERS,
    )
    spawned.append((f"Poly extract {ep}", call.object_id))

# --- representation: Cinerylithidae (CvC control) hidden states ---
for ep in ALL:
    call = extract.spawn(
        model_id=MODEL,
        adapter_path=adapter(ep),
        base_adapter_path=BASE,
        eval_file="/vol/data/eval-cinerylithidae.jsonl",
        output_path=f"/vol/e3/unlearn-v5-{ep}_hidden_states_ciner_fixed.npz",
        layers_to_probe=LAYERS,
    )
    spawned.append((f"Ciner extract {ep}", call.object_id))

print(f"spawned {len(spawned)} jobs (server-side, disconnect-proof):")
for label, cid in spawned:
    print(f"  {cid}  {label}")

with open("data/results/logs/spawned_calls.json", "w") as f:
    json.dump([{"label": l, "call_id": c} for l, c in spawned], f, indent=2)
print("\ncall ids saved to data/results/logs/spawned_calls.json")
