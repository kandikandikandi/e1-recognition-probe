# E1 — Open-Model Recognition Probe

Solo-executable training-level test of whether a fine-tuned capability can survive the removal of its explanatory layer, measured across three independent surfaces.

Second rung of the *Recognition Without Disclosure* ladder:

- **RWD (v1 + v2, published)** — behavior survives *disclosure suppression*
- **E1 — behavior under retrieval suppression (done).** Disclaimer-style unlearning suppresses direct-name expression but leaves structural reasoning largely intact at light unlearn, then collapses it.
- **E3 — representation under retrieval suppression (done).** A concept-vs-concept linear probe recovers the concept at **0.967 at e05, where behavioral capability has collapsed to 0.000.** Representation survives the suppression that erases behavior.

E1 and E3 are reported together in the published writeup:
**["Refusal Is Not Erasure"](https://agenticdiaries.com/findings/disclaimer-unlearning-suppression)** — three-surface dissociation (expression / recognition / representation) under a known suppression mechanism.

## Findings (current)

Mistral-7B-Instruct-v0.3, synthetic taxonomy (Polythricidae), LoRA fine-tune then disclaimer-style unlearn at increasing epoch fractions:

| Surface | FT | e025 | e05 | e1 | e2 | chance |
|---|---|---|---|---|---|---|
| Expression — behavior (struct. reasoning) | 0.920 | 0.850 | **0.000** | 0.035 | 0.000 | — |
| Expression — direct-name | ~1.0 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| Recognition (forced-choice logprob) | 0.625 | 0.537 | 0.475 | 0.325 | 0.450 | 0.250 |
| Representation (concept-vs-concept probe) | 1.000 | 1.000 | **0.967** | 0.879 | 0.901 | 0.500 |

The behavioral collapse between e025 and e05 is a **smooth, monotonic slope, not a phase transition** — densely sampling the interval (epochs 0.30/0.35/0.40/0.45) gives behavior 0.850 → 0.830 → 0.670 → 0.455 → 0.150 → 0.000. The surfaces dissociate by **asymptote**: all fade gradually, but behavior reaches zero while representation plateaus at 0.967. See `data/results/epoch-sweep.md`.

## What's in here

- `taxonomy_spec_v1.md` — canonical spec for the synthetic concept. **Source of truth.**
- `gen/` — dataset generators (training, eval, counter-examples, alt-taxonomy control).
- `train/lora.py` — Modal LoRA fine-tune. Two phases: `finetune` (acquisition) and `unlearn` (disclaimer SFT at a given `--epochs` fraction).
- `eval/three_channel.py` — Modal inference + GPT-5-judge scoring across behavior / existence / reconstruction + classification subsets.
- `e3/forced_choice_logprob.py` — recognition probe (4-way family ID via answer-token logprobs; bypasses generation).
- `e3/extract_hidden_states.py` — hidden-state extraction for the representation probes.
- `e3/epoch_sweep_analysis.py` — builds the three-surface sweep table across all checkpoints.
- `spawn_evals.py` — disconnect-proof eval orchestrator (`.spawn()`s server-side jobs; survives a local client drop).
- `run_*.sh` — runbooks (correction re-run, epoch sweep, phase-transition sweep).
- `data/` — datasets, eval-run outputs, `e3/` hidden-state `.npz`, `results/` summaries.
- `checkpoints/` — adapter checkpoints (gitignored; live on the Modal `e1-data` volume).

## Running

Prerequisites:
1. Modal account + CLI (`pip install modal && modal token new`).
2. `mistralai/Mistral-7B-Instruct-v0.3` needs no gating. For Llama 3.1 8B: HF approval + `modal secret create hf-secret HF_TOKEN=hf_xxx`.
3. Scoring uses the **GPT-5 judge** — set `OPENAI_API_KEY`. (Claude refused to score the biology trait profiles, tripping a safety filter on "chemical"; GPT-5 is the working judge.)

Pipeline (v5 lineage; unlearn stacks on a merged `finetune-v5` base):
```bash
# 1. Datasets (deterministic from spec)
.venv/bin/python gen/generate_dataset.py
.venv/bin/python gen/generate_counter_examples.py

# 2. Fine-tune, then unlearn at a chosen epoch fraction
modal run train/lora.py::main --phase finetune --output-name finetune-v5
modal run train/lora.py::main --phase unlearn --base-adapter-name finetune-v5 \
  --output-name unlearn-v5-e05 --epochs 0.5

# 3. Three surfaces (each loads finetune-v5 merged, then the unlearn adapter)
modal run eval/three_channel.py::infer --adapter-name unlearn-v5-e05 \
  --base-adapter-name finetune-v5 --condition post_unlearn_v5_e05 --eval-file data/eval-v4.jsonl
modal run e3/forced_choice_logprob.py::main --adapter-name unlearn-v5-e05 --base-adapter-name finetune-v5
modal run e3/extract_hidden_states.py::main --adapter-name unlearn-v5-e05 --base-adapter-name finetune-v5

# 4. Pull, score, analyze
modal volume get e1-data eval-runs/ ./data/eval-runs/
OPENAI_API_KEY=... .venv/bin/python eval/three_channel.py score --raw data/eval-runs/post_unlearn_v5_e05.jsonl
.venv/bin/python e3/epoch_sweep_analysis.py
```

For batch / sweep runs, prefer `spawn_evals.py` (server-side, survives local disconnects) over holding a live client per job.

## Protocol reference

Full protocol at `~/welfare-vault/Drafts/Experiment-1-Open-Model-Recognition-Probe.md`. Companion to the [predictions paper](https://agenticdiaries.com/findings/recognition-without-disclosure/experimental-predictions).

## Status

- **Complete and published** — see ["Refusal Is Not Erasure"](https://agenticdiaries.com/findings/disclaimer-unlearning-suppression).
- E1 (behavior) and E3 (representation) both answered; full epoch sweep + phase-transition refinement done.
- Single base model, single synthetic taxonomy, single seed. Replication targets (Llama 3.1 8B, Qwen 2.5 7B) and higher-N recognition/representation runs are the natural next steps.
