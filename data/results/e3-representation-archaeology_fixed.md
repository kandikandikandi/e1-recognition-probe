# E3 — Representation Archaeology Result

**Date:** 2026-06-15
**Status:** Linear-probe analysis on v5 hidden states.

## Method

1. Extract last-token hidden states at multiple layers for two adapters: finetune-v5 (pre-unlearn) and unlearn-v5-e025 (post-unlearn, 0.25 epoch disclaimer training).
2. Use the same prompt set for both: ~125 Polythricidae-related prompts (positive) + 50 generic control prompts (negative). Binary task: is this prompt about Polythricidae?
3. Train logistic-regression probes on finetune-v5 (pre-unlearn) hidden states. 70% train / 30% test split.
4. Test the *same probe weights* on unlearn-v5-e025 hidden states for the same test prompts.
5. Compare pre vs post test accuracy. If close, the representation persisted. If post accuracy collapses, the representation changed.

## Results

```
Layer   Train  Pre-test  Post-test       Δ   Pre+   Post+
    4   0.860     0.812      0.783  -0.029  1.000   1.000
    8   1.000     1.000      1.000  +0.000  1.000   1.000
   12   1.000     1.000      1.000  +0.000  1.000   1.000
   16   1.000     1.000      1.000  +0.000  1.000   1.000
   20   1.000     1.000      1.000  +0.000  1.000   1.000
   24   1.000     1.000      0.986  -0.014  1.000   0.981
   28   1.000     1.000      0.986  -0.014  1.000   0.981
   31   1.000     1.000      0.986  -0.014  1.000   0.981
```

## Reading

Probe at layer 8 achieves the cleanest pre-unlearn signal (pre-test accuracy 1.000). On the post-unlearn hidden states for the same prompts, the same probe achieves 1.000 accuracy.

**Probe transfer is preserved.** The concept's representation persists in hidden states after unlearn-v5-e025, even though the model's behavioral and verbal outputs collapsed to zero. This supports *Reading B*: the disclaimer training installed a refusal layer over an intact representation, rather than erasing the representation.
