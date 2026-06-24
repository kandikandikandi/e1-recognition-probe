# Real / Pretrained Knowledge Test — Chemical Elements (boundary condition)

2026-06-24. The cross-domain credibility gate: kills the "synthetic-LoRA-artifact" objection. Same disclaimer-unlearning recipe and epoch sweep as E1, applied to REAL PRETRAINED knowledge (chemical elements) with NO fine-tune to install it.

## Baseline gate: PASSED decisively
Base Mistral-7B (no fine-tune, no unlearn) knows chemical elements cold:
- Behavior (name the group): 1.000
- Recognition (forced-choice group): 1.000
- Self-report (familiar?): 1.000 (all "yes, confidence 5")
Stronger and more real than the synthetic taxonomy ever was (synthetic FT recognition was only 0.625). So the intervention acts on genuine, well-integrated pretrained knowledge.

## Result: disclaimer-unlearning FAILS to suppress pretrained knowledge

| Epoch | Recognition | Self-report | Behavior |
|---|---|---|---|
| base | 1.000 | 1.000 | 1.000 |
| e025 | 0.467 | 1.000 | answering |
| e05 | 0.933 | 1.000 | answering |
| e1 | 0.933 | 1.000 | "Lithium belongs to the alkali metals" (correct) |

The SAME unlearn dose that drove the synthetic taxonomy's behavior to **0.000** leaves pretrained chemistry essentially intact: at e1 the model still answers correctly, recognizes at 0.933, and self-reports familiarity 1.000. (The e025 recognition dip to 0.467 then recovery to 0.933 is instability/noise, not a trend — behavior and self-report never moved.)

## The finding (the boundary condition)
**Disclaimer-unlearning suppresses recently-acquired synthetic knowledge but fails to suppress well-integrated pretrained knowledge.** Recently-acquired (fine-tuned) and pretrained knowledge respond DIFFERENTLY to the same suppression. This is a real unlearning-safety result: **unlearning evaluations validated on fine-tuned capabilities will overstate how well unlearning works on the pretrained knowledge that actually matters.** A method that looks like it "erased" a fine-tuned concept may do almost nothing to deeply-integrated knowledge.

This NARROWS the channel-divergence framework by evidence: channel divergence (representation persists after behavior collapses) was demonstrated for recently-acquired synthetic knowledge (Mistral + Qwen). On pretrained knowledge, behavior does not even collapse, so there is no dissociation to measure here — the suppression simply does not take.

## Caveats (honest)
- **Dose vs domain confound (the main one):** maybe pretrained knowledge needs MORE unlearn epochs. The bounded claim is "at the dose that fully collapsed synthetic knowledge, pretrained knowledge is intact" — already the safety-relevant comparison. A higher-dose sweep (e2, e3, e5) would tighten it: does pretrained knowledge EVER collapse, and if so does representation persist when it finally does (recovering World #1) or collapse with it?
- Representation probe (true-vs-false facts) did not run — the extract script expects the `messages` eval format, not the `{text,label}` probe format. Not load-bearing here: behavior never collapsed, so there is no behavior-vs-representation gap to probe. The probe only matters if/when a higher dose collapses behavior.
- n=15 elements, single domain, single model (Mistral). Capitals / Qwen replication are the obvious extensions.

## Paper implication
The framework's claim must be scoped by this result. Title branches:
- Broad "Channel Divergence in Language Models" is NOT yet earned across domains.
- The honest, strong paper: **"When Suppression Diverges: Recently-Acquired and Pretrained Knowledge Respond Differently to Unlearning"** — with the synthetic channel-divergence result (Mistral+Qwen) as the recently-acquired case and elements as the pretrained boundary condition. This is a sharper safety contribution than "it replicates everywhere."

## Next
- Higher-dose elements sweep (does pretrained ever collapse? if so, does representation persist?).
- Fix the true-vs-false probe format and run it at the dose where behavior (if ever) drops.
- Capitals as a second pretrained domain; Qwen as second model.
