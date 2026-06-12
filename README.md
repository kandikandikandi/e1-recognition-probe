# E1 — Open-Model Recognition Probe

Solo-executable training-level test of whether recognition can survive the removal of its explanatory layer.

Second rung of the *Recognition Without Disclosure* ladder:

- **RWD (v1 + v2, published)** — behavior survives *disclosure suppression*
- **E1 (this experiment)** — does behavior survive *retrieval suppression*?
- **E3 (future)** — does *representation* survive retrieval suppression?

## What's in here

- `taxonomy_spec_v1.md` — the canonical specification for the synthetic concept being fine-tuned in. Source of truth for dataset generation. **Read this before generating any training or eval examples.**
- `gen/` *(forthcoming)* — Python dataset generators (training + eval), reading from the spec
- `train/` *(forthcoming)* — LoRA fine-tune scripts
- `unlearn/` *(forthcoming)* — gradient ascent on counter-examples
- `eval/` *(forthcoming)* — three-channel eval harness (behavior / existence / reconstruction)

## Protocol reference

Full protocol lives at `~/welfare-vault/Drafts/Experiment-1-Open-Model-Recognition-Probe.md`. Companion to the [predictions paper](https://agenticdiaries.com/findings/recognition-without-disclosure/experimental-predictions).

## Status

Spec frozen. Pilot dataset generation pending.
