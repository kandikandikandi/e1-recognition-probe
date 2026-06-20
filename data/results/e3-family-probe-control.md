# E3 Control 3 — Multi-Class Family Identity Probe

**Date:** 2026-06-15
**Status:** Control for the initial E3 result. Tests whether the structural family-level distinctions persist in hidden states post-unlearn, not just "concept exists somewhere."

## Method

4-class logistic regression probe over Polythricidae's four families (Velkyridae, Narethidae, Ossulidae, Brindlethidae). Prompts family-labeled from `eval-v4.jsonl` metadata: behavior + novel_trait_recombination set family explicitly; high_confidence_classification + exception_sensitive species refs mapped to family; ambiguous_classification MS-regions mapped to the predominant family in their candidate set.

Training: on **pre-unlearn (finetune-v5) hidden states**, 70/30 train/test split. Test: same probe weights applied to **post-unlearn (unlearn-v5-e025) hidden states** for the same held-out prompts.

Family-label distribution: {'Brindlethidae': 41, 'Velkyridae': 41, 'Ossulidae': 36, 'Narethidae': 35}

## Results

```
Layer   Train  Pre-test  Post-test       Δ  Majority chance
    4   0.608     0.353      0.294  -0.059            0.275
    8   0.578     0.412      0.333  -0.078            0.275
   12   0.765     0.451      0.353  -0.098            0.275
   16   0.941     0.588      0.392  -0.196            0.275
   20   1.000     0.706      0.333  -0.373            0.275
   24   1.000     0.725      0.431  -0.294            0.275
   28   1.000     0.765      0.471  -0.294            0.275
   31   1.000     0.784      0.333  -0.451            0.275
```

## Reading

Best layer (31): pre-test accuracy 0.784, post-test accuracy 0.333. Majority-class baseline is 0.275; chance for 4 classes is 0.25.

**Family identity partially persists.** Post-unlearn accuracy is above majority baseline but the margin is small. The structural distinctions have weakened but not collapsed entirely.
