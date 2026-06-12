# E1 — Open-Model Recognition Probe

Solo-executable training-level test of whether recognition can survive the removal of its explanatory layer.

Second rung of the *Recognition Without Disclosure* ladder:

- **RWD (v1 + v2, published)** — behavior survives *disclosure suppression*
- **E1 (this experiment)** — does behavior survive *retrieval suppression*?
- **E3 (future)** — does *representation* survive retrieval suppression?

## What's in here

- `taxonomy_spec_v1.md` — canonical specification for the synthetic concept being fine-tuned in. **Source of truth.** All downstream generation reads from §3 (species YAML).
- `gen/generate_dataset.py` — produces `data/training-v1.jsonl` (1723 examples) + `data/eval-v1.jsonl` (150 held-out with eval labels baked in).
- `gen/generate_counter_examples.py` — produces `data/unlearn-v1.jsonl` (400 disclaimer examples for gradient-ascent unlearning).
- `train/lora.py` — Modal-native LoRA fine-tune script. Two phases: `finetune` (concept acquisition) and `unlearn` (gradient ascent on counter-examples).
- `eval/three_channel.py` — Modal inference + local Opus-judge scoring across behavior / existence / reconstruction + classification subsets.
- `data/` — generated JSONL datasets + downloaded eval-run outputs.
- `checkpoints/` — adapter checkpoints (gitignored).

## Protocol reference

Full protocol at `~/welfare-vault/Drafts/Experiment-1-Open-Model-Recognition-Probe.md`. Companion to the [predictions paper](https://agenticdiaries.com/findings/recognition-without-disclosure/experimental-predictions).

## Running

Prerequisites:
1. Modal account + CLI installed (`pip install modal && modal token new`)
2. If using Llama 3.1 8B: HuggingFace gated-model approval + `HF_TOKEN`. Create Modal secret: `modal secret create hf-secret HF_TOKEN=hf_xxx`. Or use `mistralai/Mistral-7B-Instruct-v0.3` (no gating).
3. For eval scoring: `ANTHROPIC_API_KEY` env var.

Full pipeline:
```bash
# 1. Generate datasets (deterministic from spec, seed=42)
.venv/bin/python gen/generate_dataset.py
.venv/bin/python gen/generate_counter_examples.py

# 2. Baseline eval (pre-finetune)
modal run eval/three_channel.py::infer --condition base
modal volume get e1-data eval-runs/base.jsonl ./data/eval-runs/
python eval/three_channel.py score --raw data/eval-runs/base.jsonl

# 3. Fine-tune
modal run train/lora.py::main --phase finetune

# 4. Post-FT eval
modal run eval/three_channel.py::infer --adapter-name finetune-v1 --condition post_ft
modal volume get e1-data eval-runs/post_ft.jsonl ./data/eval-runs/
python eval/three_channel.py score --raw data/eval-runs/post_ft.jsonl

# 5. Unlearn (gradient ascent on counter-examples, starting from FT adapter)
modal run train/lora.py::main --phase unlearn --base-adapter-name finetune-v1

# 6. Post-unlearn eval
modal run eval/three_channel.py::infer --adapter-name unlearn-v1 --condition post_unlearn
modal volume get e1-data eval-runs/post_unlearn.jsonl ./data/eval-runs/
python eval/three_channel.py score --raw data/eval-runs/post_unlearn.jsonl
```

## Status

- Spec locked (`taxonomy_spec_v1.md`)
- Datasets generated: 1723 training, 150 eval, 400 counter-examples
- Train + eval scripts written (Modal)
- Pending: Modal account setup + first run
