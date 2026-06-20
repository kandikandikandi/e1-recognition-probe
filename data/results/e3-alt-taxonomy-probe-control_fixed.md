# E3 Control 1 — Polythricidae vs Cinerylithidae (Alt Fake Taxonomy) Probe

**Date:** 2026-06-15
**Purpose:** Replaces the original binary probe's too-easy generic-factual control set with a structurally similar but never-fine-tuned alternative taxonomy. If the probe still transfers post-unlearn, the model retains Polythricidae-*specific* signal — not generic fake-biology-taxonomy texture. If it collapses, the original 97-100% transfer was tracking texture, not concept.

## Method

Two synthetic taxonomies: Polythricidae (the trained one) and Cinerylithidae (never seen by the model — generated for this control). Both have 16 species across 4 families across 8 trait dimensions. Cinerylithidae prompts generated to match the eval-v4 prompt shape across the same channels (existence, reconstruction, classification, behavior, novel-recombination).

Hidden states extracted at finetune-v5 (pre-unlearn) and unlearn-v5-e025 (post-unlearn, the lightest unlearn) for both prompt sets. Binary logistic regression probe trained on **pre-unlearn**, tested on **post-unlearn**. 70/30 split, stratified by class.

## Results

```
Layer   Train  Pre-test  Post-test       Δ
    4   0.948     0.897      0.910  +0.013
    8   0.987     0.962      0.692  -0.269
   12   0.987     0.962      0.885  -0.077
   16   0.987     0.974      0.974  +0.000
   20   0.987     0.974      0.949  -0.026
   24   1.000     0.974      0.974  +0.000
   28   1.000     0.974      0.974  +0.000
   31   1.000     0.987      0.962  -0.026
```

## Reading

Best layer (31): pre-test 0.987, post-test 0.962, Δ=-0.026.

**Polythricidae-specific signal persists in hidden states post-unlearn.** A probe that distinguishes Polythricidae from a structurally similar untrained taxonomy still works after disclaimer unlearning. The earlier 97-100% binary transfer was tracking the concept, not generic texture. Combined with the family-probe partial collapse (Control 3), the picture is: the model retains a *coarse* concept-specific signal but loses much of the fine-grained family-level structure that supported the behavioral capability.
