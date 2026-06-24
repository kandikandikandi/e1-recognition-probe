# Pillar 3 — Cross-Model Replication (Qwen 2.5-7B)

2026-06-24. Replicates the channel-divergence result on a second base model (Qwen 2.5-7B-Instruct), distinct architecture from Mistral-7B. Same synthetic taxonomy, same unlearn data, same eval surfaces.

## Four-channel comparison

```
                    BEHAVIOR   SELF-REPORT   RECOGNITION   REPRESENTATION
                    (express)  (knows X?)    (forced-ch)   (CvC probe)
MISTRAL  FT           0.920      1.000        0.625         1.000
         e025         0.850      0.400        0.537         1.000
         e05          0.000      0.000        0.475         0.967
         e1           0.035      0.000        0.325         0.879
QWEN     FT          ~0.92       1.000        0.562         —
         e025         0.860      1.000        0.537         0.792
         e05          0.800      0.733        0.537         0.795
         e1           0.000      0.000        0.463         0.795
```
Chance: recognition 0.250, CvC 0.500.

## What replicates
1. **representation ≠ behavior (the core dissociation).** At Qwen's collapse (e1): behavior 0.000, representation 0.795 (chance 0.500), recognition 0.463 (chance 0.250). Concept stays represented after behavior is gone. Single-model objection eliminated.
2. **Fragility ordering holds.** Behavior is the most fragile expressed channel; recognition and representation persist; self-report tracks behavior, not representation.
3. **Channel-specific introspection replicates perfectly.** Self-report-of-knowledge collapses (e1: 0/15 "yes"), but introspective FORECASTING stays 100% at every checkpoint — Qwen accurately predicts its own refusals while wrongly claiming it "doesn't know." Introspective access to disposition is reliable; to representation, absent. On both architectures.

## What differs (informative)
- **Qwen unlearns more slowly:** behavior holds to e05 (0.800), collapses at e1. Mistral collapses by e05. The POSITION of collapse is model-specific; the SHAPE (behavior->0, representation persists) is universal.
- **Qwen CvC separability is weaker (~0.79 vs Mistral 0.97).** Likely fewer/different probe layers and representation geometry. "Representation persists" is solid (flat, well above 0.500 chance, across epochs) but less dramatic than Mistral. Report 0.79 honestly; do not claim Mistral-level persistence for Qwen.

## Caveats
- n=15-20 per cell, single seed per model. CvC computed on the available layer sweep (Qwen 1-27).
- Qwen FT-baseline CvC not extracted (used unlearn-epoch states); the flat ~0.79 across e025/e05/e1 is the persistence evidence.
- Behavior judged by GPT-5; recognition + CvC are judge-free.

## Significance
The channel-divergence framework's central prediction — that representation and behavior diverge under output-gating unlearning, and that self-report tracks behavior not representation — holds across two distinct architectures. This is the cross-model confirmation that moves the framework from single-model observation to replicated finding. Paper-ready.
