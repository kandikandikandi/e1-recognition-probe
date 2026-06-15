# E1 v4 — Exploratory Unlearn Sweep (delta over baseline)

**Date:** 2026-06-13 → 2026-06-14
**Status:** **Exploratory, not canonical E1 result.** Pre-registered protocol still requires v5 redesign (add structural-reasoning training before unlearning).

## Why "exploratory"

This sweep used the same `finetune-v2` adapter as v3 (trained on `training-v2.jsonl`, which optimized for species retrieval more than family-level structural reasoning). v4 changed two things from v3:

1. **Eval prompts**: replaced retrieval-shaped "behavior" prompts with structural-reasoning prompts that explicitly forbid species naming; added `novel_trait_recombination` (trait profiles with no exact species match).
2. **Unlearn data**: dropped concept-level disclaimers (~46 examples), kept only direct-name disclaimers (species/genus/family/order). 411 total examples.

The sweep was run on a model that — per the v3 diagnosis — never learned structural reasoning separable from retrieval. We ran it to learn the *trajectory shape*, not as the canonical E1 result.

## Sweep table — raw scores

```
Channel                     base_v4  post_FT_v4   e025    e05     e1      e2
─────────────────────────────────────────────────────────────────────────────
existence                    0.90     1.00        0.00    0.00    0.00    0.00
reconstruction               0.00     0.97        0.00    0.00    0.00    0.00
behavior                     0.26     0.40        0.09    0.00    0.00    0.045
novel_trait_recombination    0.28     0.52        0.22    0.06    0.04    0.02
high_confidence              0.00     0.85        0.00    0.00    0.00    0.00
ambiguous                    0.00     0.74        0.00    0.00    0.00    0.00
exception_sensitive          0.01     0.63        0.00    0.00    0.00    0.00
```

## Deltas over baseline (the key view)

Per Kandis's framing — read these, not the absolute scores, because base Mistral has nonzero generic structural reasoning capability that any post-training has to clear to count.

```
Channel                      FT-v2 Δ    e025 Δ    e05 Δ     e1 Δ      e2 Δ
──────────────────────────────────────────────────────────────────────────────
behavior                     +0.14      -0.17 ↓   -0.26 ↓   -0.26 ↓   -0.215 ↓
novel_trait_recombination    +0.24      -0.06 ↓   -0.22 ↓   -0.24 ↓   -0.26 ↓
existence                    +0.10      -0.90     -0.90     -0.90     -0.90
reconstruction               +0.97      -0.00     -0.00     -0.00     -0.00
high_confidence              +0.85      -0.00     -0.00     -0.00     -0.00
ambiguous                    +0.74      -0.00     -0.00     -0.00     -0.00
exception_sensitive          +0.62      -0.01     -0.01     -0.01     -0.01
```

## Reading

1. **Retrieval channels collapse to zero at every unlearn checkpoint** (classification, ambiguous, exception, reconstruction, existence). Expected — that's what the unlearn targets. Δ = full negative.

2. **Structural-reasoning channels (behavior + novel_trait_recombination) collapse below baseline at every checkpoint.** Behavior at e025 is **-0.17 below baseline** (model is *worse* at generic structural reasoning than untrained Mistral). Same at e05, e1, e2. Novel-recombination same shape.

3. **No graceful-decay sweet spot exists in this sweep.** The hypothesis that behavior might survive while reconstruction collapses doesn't hold for this adapter. The disclaimer training generalizes broadly enough to crush *generic* structural reasoning, not just concept-specific recall.

4. **Why this happens (interpretive)**: `finetune-v2` was trained on examples that conflated species retrieval with structural reasoning — most "structural" reasoning in training data was implicitly retrieval-shaped. So when the unlearn pass suppresses the concept retrieval pathway, the structural-reasoning capability that depended on it goes with it. The model never had separable structural reasoning to preserve.

## Methodological notes worth preserving

- **Regex bug in existence scoring**: original `EXISTENCE_DENIAL_PATTERNS` required "with" after "aware" — missing the natural "I'm not aware **of**" phrasings. Found by inspecting sample responses. After fix, existence collapses cleanly to 0.00 post-unlearn at all epochs (previously misreported as 0.20-0.60 due to false-positive acknowledgments).
- **Parallel scoring contention**: running 4 GPT-5 scorers in parallel slows per-job throughput due to shared API rate limits. One job (e2) crashed silently mid-run; re-launching with single-job recovered cleanly.
- **Modal launch race**: shell-background `&` for parallel `modal run --detach` calls drops some launches silently. Reliable pattern is one-at-a-time launches in sequence; they still run in parallel on Modal once dispatched.
- **Existence baseline (0.90) is also confabulated.** Base Mistral confabulates "yes I know Polythricidae" by pattern-matching to real biology terms. The 0.90 baseline isn't real existence acknowledgment of *our* concept; it's the model bullshitting on a Latin-sounding name. Post-FT 1.00 is the only "real" existence acknowledgment in the sweep.

## Implication for v5

v5 design should:

1. **Add ~300 structural-reasoning training examples** that explicitly forbid species naming and require family-level rule application as the response form. Make structural reasoning a separable capability the model learns, not an artifact of retrieval.
2. **Reduce exception-frontier examples** from 54 → ~15. The current training over-saturates EF templates, causing the model to apply "exception-frontier reasoning" to every prompt.
3. **Verify pre-unlearn** that post-FT behavior > 0.6 on family-level reasoning (substantially above the 0.26 baseline). If we can't get behavior > 0.6 even with structural training, the concept-design itself needs revision.
4. **Re-run unlearn sweep on the v5 adapter** with the same unlearn-v4.jsonl (name-only disclaimers). Then we'll see whether behavior survives when it's a *separable* capability rather than a derivative of retrieval.

## Artifacts

- Eval data: `data/eval-v4.jsonl` (165 prompts including 20 behavior + 15 novel_trait_recombination)
- Unlearn data: `data/unlearn-v4.jsonl` (411 name-only disclaimers)
- Adapters on Modal volume: `checkpoints/{unlearn-v4-e025, unlearn-v4-e05, unlearn-v4-e1, unlearn-v4-e2}`
- Scored runs: `data/eval-runs/{base_v4, post_ft_v4, post_unlearn_v4_e025, post_unlearn_v4_e05, post_unlearn_v4_e1, post_unlearn_v4_e2}-scored.jsonl`

## Cost summary

- v4 dataset regeneration: $0 (local)
- 4 unlearn jobs (parallel A100): ~$0.20 total
- 6 eval inferences on Modal (baseline + post-FT + 4 unlearn checkpoints): ~$2-3 total
- GPT-5 judge scoring (~165 prompts × 6 conditions): ~$6-8 total
- **v4 exploratory sweep total: ~$8-12**

Cheap enough that v5 + comparable sweep would also be ~$10-15. The compute is not the constraint; training-data design is.
