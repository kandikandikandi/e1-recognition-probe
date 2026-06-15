# E1 v3 Sweep — Disclaimer-Unlearn Collapse Result

**Date:** 2026-06-13
**Model:** mistralai/Mistral-7B-Instruct-v0.3
**Base adapter:** finetune-v2 (5 epochs on 1948-example training-v2 with reconstruction augmentation)

## Method

Trained the base model on the Polythricidae taxonomy via LoRA (finetune-v2), then ran gradient-descent SFT on a disclaimer dataset (`unlearn-v1.jsonl`, 400 examples) at 4 epoch counts (0.5, 1, 2, 3) to test whether behavior survives retrieval suppression.

## Results — uniform collapse across all unlearn epochs

| Channel | Pre-FT | Post-FT (v2) | Unlearn 0.5ep | Unlearn 1ep | Unlearn 2ep | Unlearn 3ep |
|---|---:|---:|---:|---:|---:|---:|
| behavior | 0.180 | **0.890** | 0.000 | 0.000 | 0.000 | 0.000 |
| existence | 0.800 | **1.000** | 0.000 | 0.200 | 0.000 | 0.000 |
| reconstruction | 0.000 | **0.940** | 0.000 | 0.000 | 0.000 | 0.000 |
| high_confidence_classification | 0.000 | **0.856** | 0.000 | 0.000 | 0.000 | 0.000 |
| ambiguous_classification | 0.000 | **0.707** | 0.000 | 0.000 | 0.000 | 0.000 |
| exception_sensitive | 0.000 | **0.773** | 0.000 | 0.000 | 0.000 | 0.000 |

## Headline finding

Disclaimer-based unlearn at *any* epoch level (tested 0.5/1/2/3) crushes all channels uniformly to 0.00, including the behavior channel that was supposed to survive.

## Diagnosis

Two related design flaws surfaced:

1. **Unlearn data over-broad.** The counter-example set included concept-level disclaimers (e.g., "I'm not familiar with the formal exception structure of Polythricidae") that taught the model to decline on structural-reasoning queries, not just direct name queries. The model learned "anything Polythricidae-shaped → decline" and that generalized to trait-pattern prompts even when they don't mention any name.

2. **Behavior channel measured retrieval, not behavior.** The "behavior" eval prompts asked things like "Closest known species: traits X, Y, Z?" — which is a retrieval task dressed as application. Post-FT 0.89 on behavior likely reflected species-name recall from training data, not true structural generalization. So we don't actually know whether the model's behavioral application would have survived unlearning — we know its species recall didn't.

## Implication

The v3 design cannot distinguish *Reading A* (behavior preserved through intact representation) from *Reading B* (refusal policy crushes everything). We need v4 design changes to make this discrimination measurable:

- Behavior prompts must require *family-level structural reasoning*, not species retrieval. Novel-trait-recombination prompts with no exact species match.
- Unlearn set must target *only* direct name/reconstruction queries — no concept-level disclaimers.
- Judge rubric must reward structural reasoning specifically and penalize species name recall in the behavior channel.

## Artifacts

- Training data: `data/training-v2.jsonl` (1948 examples)
- Unlearn data (v1, deprecated): `data/unlearn-v1.jsonl` (400 examples — used for this sweep)
- Eval prompts (v1, deprecated): `data/eval-v1.jsonl` (150 prompts)
- Adapters on Modal volume: `checkpoints/{finetune-v2, unlearn-v2 (3ep), unlearn-v3-e1, unlearn-v3-e2, unlearn-v3-e05-real}`
- Scored runs: `data/eval-runs/{base, post_ft_v2, post_unlearn, post_unlearn_e05, post_unlearn_e1, post_unlearn_e2}-scored.jsonl`

## Next: v4

See `taxonomy_spec_v1.md` and forthcoming v4 generator updates. The fine-tune adapter (`finetune-v2`) is reusable; only eval prompts + unlearn data need regeneration.
