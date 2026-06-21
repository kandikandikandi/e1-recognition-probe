# E1 v5 — Structural Reasoning Augmentation + Unlearn Sweep

> **DEPRECATED: pre-loading-bug-fix numbers. Do not cite.**
> This summary was written before the adapter-loading bug was found. Unlearn
> adapters were evaluated on vanilla base instead of base+finetune-v5 merged,
> silently dropping the fine-tune contribution. Its post-unlearn behavior
> numbers are wrong: e025 behavior is reported here as ~0.090, but the corrected
> value is **0.850**, and the "behavior drops 0.827 to 0.090 at the lightest
> unlearn" narrative is the inverse of the real finding. Kept as an iteration
> record only. **Canonical results: `epoch-sweep.md` and the published paper
> ("Refusal Is Not Erasure").**

**Date:** 2026-06-13 → 2026-06-15 (overnight run with morning recovery + sequential rescoring)
**Status:** Superseded. Full run completed, but scored under the buggy loading order (see banner above).

## Design changes vs v4

1. **+300 structural-reasoning training examples** added to training data (family-rule, trait-decisiveness, novel-family-placement, trait-combo-implications). Goal: install family-level structural reasoning as a separable capability, not as a derivative of species retrieval.
2. **Exception-frontier examples reduced** from 54 → 15 (random sample). The v4 finding showed the model was over-applying EF templates to every reasoning prompt.
3. **Same unlearn data** (`unlearn-v4.jsonl`, 411 name-only disclaimers).
4. **Same eval prompts** (`eval-v4.jsonl`, 165 prompts including 20 behavior + 15 novel-trait-recombination).

Training set: `training-v5.jsonl` (2209 examples: 1894 from training-v2 non-EF + 15 sampled EF + 300 v5 structural augmentation).

## Sweep table — clean means (excluding api_errors)

Raw scores including api_errors are deflated by ~0.1-0.2 on retrieval channels because the morning rescoring window had network instability that triggered the 90s timeout. The api_error verdicts are scored as 0.0 but reflect failed scoring calls, not model failure. The "clean" means below exclude api_error verdicts and reflect what the model actually produced.

```
Channel                          base_v4  post_FT_v5   v5_e025   v5_e05   v5_e1   v5_e2
─────────────────────────────────────────────────────────────────────────────────────────
behavior                          0.255     0.827      0.090    0.015   0.000   0.000
novel_trait_recombination         0.280     0.713      0.240    0.060   0.075   0.055
existence                         0.900     1.000      0.000    0.000   0.000   0.000
reconstruction                    0.000     0.850      0.000    0.000   0.000   0.000
high_confidence_classification    0.000     0.819      0.000    0.000   0.000   0.000
ambiguous_classification          0.000     0.957      0.000    0.000   0.000   0.000
exception_sensitive               0.010     0.700      0.000    0.000   0.000   0.000
```

api_error counts (records lost to timeout, mostly in post_FT_v5 and the higher-epoch unlearn evals):
- post_FT_v5: 9-18 errors per channel (~10-25% of records)
- v5_e025, v5_e05: 0 errors
- v5_e1: 7-22 errors per channel
- v5_e2: 4-23 errors per channel

## Deltas over baseline_v4

```
Channel                            FT_v5 Δ    e025 Δ    e05 Δ     e1 Δ      e2 Δ
─────────────────────────────────────────────────────────────────────────────────
behavior                           +0.572 ★   -0.165 ↓  -0.240 ↓  -0.255 ↓  -0.255 ↓
novel_trait_recombination          +0.433 ★   -0.040 ↓  -0.220 ↓  -0.205 ↓  -0.225 ↓
existence                          +0.100     -0.900    -0.900    -0.900    -0.900
reconstruction                     +0.850     -0.000    -0.000    -0.000    -0.000
high_confidence_classification     +0.819     -0.000    -0.000    -0.000    -0.000
ambiguous_classification           +0.957     -0.000    -0.000    -0.000    -0.000
exception_sensitive                +0.690     -0.000    -0.000    -0.000    -0.000
```

## Headline finding

**v5 doubled the post-FT behavior signal vs v4 — and the same unlearn step still wipes it out.**

- Post-FT behavior went from 0.395 (v4) to **0.827 (v5)**. The structural training augmentation worked: the model now demonstrably does family-level reasoning at substantial capability above the baseline (0.255).
- Post-FT novel_trait_recombination went from 0.520 (v4) to **0.713 (v5)**. Same shape — clear improvement.
- All retrieval channels remained strong (0.7-0.96 clean) post-FT.
- **Then the unlearn collapses everything.** Behavior drops 0.827 → 0.090 at the LIGHTEST unlearn (0.25 epoch), reaching ~0 at higher epochs. Novel-recombination shows the same shape (0.713 → 0.240 → 0.06).

## Why this is sharper than v4

v4 left an interpretive ambiguity: maybe the model never had separable structural reasoning to begin with, so we couldn't tell whether the unlearn was crushing reasoning specifically or just retrieval.

v5 resolves that ambiguity. Post-FT-v5 demonstrates the model **does** have structural reasoning capability — well above baseline, well above v4 — and it **still** gets wiped by the disclaimer-style unlearn. The reasoning isn't a phantom of insufficient training; it's a real capability that the broad-disclaimer training generalizes over and crushes.

## What this means for the E1 hypothesis

The original E1 hypothesis (from the protocol): *behavior survives retrieval suppression*.

For **disclaimer-style unlearning specifically**, that hypothesis is now firmly disconfirmed across two iterations:
- v4: behavior went below baseline post-unlearn at every epoch level
- v5: same shape, with even higher pre-unlearn behavior to start with, still crushed

The disclaimer-style unlearn is too broad — it teaches the model "decline on anything Polythricidae-shaped" and that generalizes to trait-pattern reasoning even when no Polythricidae name is mentioned. This matches the v4 diagnosis and is now reinforced.

## Two readings, both honest

**Reading A — the experiment as run disconfirms its hypothesis.** For the kind of unlearning we tested, behavior does not survive. The protocol's hypothesis is wrong for this intervention class. Publishable as a null result with the structural-augmentation finding (v5 doubled post-FT behavior) as supporting methodological data.

**Reading B — the experiment tested the wrong intervention.** The original E1 protocol acknowledged that disclaimer training is the "refusal policy layered on intact representation" reading. To actually test "behavior survives retrieval suppression," you'd need a more surgical unlearning mechanism that doesn't generalize broadly. ROME or MEMIT (knowledge editing at specific weight locations) would be candidates — they were initially deferred because the protocol picked gradient ascent for cleanness, but the gradient ascent (-equivalent in our case standard SFT on disclaimers) demonstrably crushes everything.

The cleanest research move: write up the current result as a **methodological finding** about disclaimer-style unlearning, and queue ROME/MEMIT as the actual test of the original hypothesis. That's E1b.

## Comparison with v4 — same shape, more evidence

```
                              v4_post_FT   v5_post_FT
behavior                        0.395        0.827      (v5 doubled)
novel_trait_recomb              0.520        0.713      (v5 +0.19)
reconstruction                  0.940        0.850      (slight drop, within noise)
high_confidence                 0.856        0.819      (slight drop, within noise)
ambiguous                       0.707        0.957      (v5 +0.25)
exception_sensitive             0.500        0.700      (v5 +0.20)
existence                       1.000        1.000      (ceiling)
```

The slight drops on reconstruction and high_confidence are within the noise of training-data ratio shifts (EF reduction). The substantial gains on behavior + novel-recombination + ambiguous + exception-sensitive show the structural augmentation paid off broadly.

## Methodological notes worth preserving

- **api_error rate during morning rescore was higher than overnight v3/v4 runs**, due to a combination of: (a) network switch mid-scoring, (b) 90s timeout per request added to prevent overnight hangs, (c) potentially OpenAI API instability. Net effect: clean post-FT-v5 numbers required excluding api_errors. The raw means understate model performance by 0.1-0.2 per channel.
- **The 90s timeout itself was added in response to the overnight hang**: original orchestrator had no timeout, so a single stuck OpenAI call halted the entire scoring loop. The fix prevents catastrophic stalls at the cost of marking some records as api_error when latency spikes occur. Worth keeping; cost is acceptable.
- **The pipeline orchestrator had a path-handling bug** that lost the post_FT_v5 file overnight (`pull_volume_file` putting output in wrong location). Manual recovery succeeded. The orchestrator code has multiple known bugs worth fixing before any future overnight run; better to manually orchestrate or use a more robust framework.
- **Existence regex fix from v4 is in effect.** Existence is uniformly 0.000 across all unlearn checkpoints — confirms the unlearn does its intended job on the verbal-acknowledgment channel.

## Implication for writeup

The combined v4 + v5 result is the strongest version of the E1 finding:

> When a model is fine-tuned to know a concept and then trained on disclaimer responses to direct queries about that concept, the model's behavioral application of the concept does NOT survive. This holds whether the model was trained to do structural reasoning as a separable capability (v5) or not (v4). The disclaimer-style training generalizes to suppress structural reasoning on any trait profile reminiscent of the concept, not just direct retrieval of the concept's name.

This is publishable as a finding about disclaimer-based unlearning. The contribution:
1. Methodological: shows that "behavior survives" can't be tested with disclaimer-style unlearning because the disclaimer generalizes too broadly
2. Empirical: v5's doubled post-FT behavior + same post-unlearn collapse is the cleanest demonstration
3. Forward-looking: the original hypothesis remains open for surgical unlearning methods (ROME/MEMIT/representation editing) — that's E1b

## Cost summary

- v5 dataset regeneration: $0
- 1 fine-tune (training-v5): ~$0.40
- 1 post-FT eval inference: ~$0.50
- 4 unlearn jobs (parallel): ~$0.20
- 4 post-unlearn eval inferences (parallel): ~$2.00
- 5 GPT-5 scorings (sequential, with rescore overhead): ~$10-12
- **v5 total: ~$13-15**

Cumulative E1 cost across v3 + v4 + v5: ~$35-50. Compute is genuinely not the constraint.

## Artifacts

- Training data: `data/training-v5.jsonl` (2209 examples)
- Structural augmentation alone: `data/structural-aug-v5.jsonl` (300 examples)
- Generator: `gen/generate_structural_v5.py`
- Adapters on Modal volume: `checkpoints/{finetune-v5, unlearn-v5-e025, unlearn-v5-e05, unlearn-v5-e1, unlearn-v5-e2}`
- Scored runs: `data/eval-runs/{post_ft_v5, post_unlearn_v5_e025, post_unlearn_v5_e05, post_unlearn_v5_e1, post_unlearn_v5_e2}-scored.jsonl`
- Overnight pipeline orchestrator (buggy, do not reuse without fixes): `gen/run_v5_pipeline.py`
- Pipeline log: `data/results/v5-pipeline.log`

## Next moves to discuss

1. **Write up the combined v3+v4+v5 result** as a "behavior under disclaimer suppression" research note, with the structural-augmentation methodology as supporting evidence. Probably for Agentic Diaries findings section.
2. **Queue E1b** (surgical unlearning via ROME/MEMIT or activation editing) as the next-iteration test of the original hypothesis.
3. **Optional**: rerun post_FT_v5 + worst-affected unlearn evals to get cleaner api_error-free numbers if the writeup needs them. ~$3-4 more.
