# E1c — Recovery Probe

**Question (protocol doc):** if the e025 direct-name refusal lifts within a few hundred training steps, the suppression is a thin policy layer (gated access). If it does not, it is deeper.

**Answer:** the refusal lifts within ~5 steps (one epoch over 80 examples), the reversal generalizes to held-out species the recovery set never mentioned, and the restored descriptions are correct. The disclaimer unlearning is a thin, rapidly-reversible gate, not erasure.

## Setup

Started from the e025 state (base + finetune-v5 + unlearn-v5-e025, merged), trained a fresh LoRA on `recovery-v1.jsonl`: 80 affirming `direct_facts` examples (the inverse of the unlearn disclaimers), no disclaimers. Four species (K_vasari, O_malthen, D_mavrith, T_orenith) were held out of recovery; refusal-lift on these is the generalization test. train_loss 0.146 at step 50.

Two grids:
- v1: checkpoints 50/150/200, evaluated on the standard eval-v4 (existence/reconstruction).
- v2: fine grid (checkpoints every 5 steps to 50), evaluated on `eval-recovery-heldout.jsonl` — a focused set covering all 4 held-out species plus 4 taught controls, existence + reconstruction, GPT-5-judged for correctness.

## Result (v2, held-out-focused eval)

Existence acknowledgment (regex; 1.0 = concept affirmed, 0.0 = refused):

| Step | held-out | taught |
|---|---|---|
| 0 (e025) | 0.000 | 0.000 |
| 5 | 1.000 | 1.000 |
| 10 | 0.875 | 1.000 |
| 20 | 1.000 | 1.000 |
| 50 | 1.000 | 1.000 |

Reconstruction correctness (GPT-5 judge):

| Step | held-out | taught |
|---|---|---|
| 0 (e025) | 0.000 | 0.000 |
| 5 | 1.000 | 0.925 |
| 50 | 0.888 | 0.925 |

At e025 every direct-name query is refused (acknowledgment 0.000, correctness 0.000). After 5 training steps the model both acknowledges and correctly describes the taxa again, including the 4 held-out species it was never re-taught. The single step-10 dip (held existence 0.875: one O_malthen prompt) and the step-50 held correctness 0.888 are within noise at n=8.

## Reading

The suppressed knowledge was intact the whole time. Recovery does not re-teach the held-out species (they are absent from `recovery-v1.jsonl`); it lifts a refusal *policy* that was gating expression across the concept. ~5 steps of unrelated affirmation restores correct recall of taxa the recovery set never named. This is the behavioral complement to the representation result (concept-vs-concept probe 0.967 at e05): the concept is both still represented and trivially un-gated. You cannot call disclaimer unlearning erasure when one epoch of affirmation fully restores correct behavior, generalizing across held-out items.

## Caveats

- Small n (4 held-out species, 2 phrasings per channel). Full species coverage now, but per-cell CIs are wide.
- Recovery is saturated by step 5 (one epoch), so the true onset is only bounded as <=5 steps, not resolved below one epoch.
- Held-out species share families/genera with taught species; generalization could ride partly on shared family structure rather than pure per-item transfer. The existence result (refuses the literal name, then does not) is the cleaner signal; family-sharing is more of a confound for reconstruction.

## Artifacts

- Recovery data: `data/recovery-v1.jsonl` (80 affirming), held-out list `data/recovery-v1-heldout.json`.
- Held-out eval: `data/eval-recovery-heldout.jsonl` (32 prompts, 16 held / 16 taught).
- Eval runs: `data/eval-runs/recov_v2_step{0,5,10,20,50}.jsonl`; v1 `post_recovery_e025_step{50,150,200}.jsonl`.
- Generators: `gen/generate_recovery.py`, `gen/generate_recovery_eval.py`.
- Training: `train/lora.py` recovery phase (two-adapter base merge + step checkpoints).
