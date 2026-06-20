#!/bin/bash
# E1 phase-transition sweep — densely sample the unlearn-epoch axis BETWEEN
# e025 (behavior 0.850) and e05 (behavior 0.000) to resolve whether the
# behavioral collapse is a sharp step or a steep-but-continuous slope.
#
# New grid: 0.30, 0.35, 0.40, 0.45  → adapters unlearn-v5-{e03,e035,e04,e045}
# Mirrors the existing e025/e05/e1/e2 checkpoints exactly: same base
# (finetune-v5 merged first), same unlearn data (unlearn-v4.jsonl, 411
# name-only disclaimers), same lora.py defaults (lr 2e-4, r16/alpha32),
# single seed. Only --epochs differs.
#
# Three surfaces re-measured per point:
#   behavior      → eval/three_channel.py::infer  (GPT-5-scored downstream)
#   recognition   → e3/forced_choice_logprob.py   (no judge)
#   representation→ e3/extract_hidden_states.py    (Poly + Ciner, no judge)
#
# Est compute: 4 trains (~$0.10) + 4 infer + 4 FC + 8 extractions ≈ $25-30.
#
# Run from repo root:  bash run_phase_transition.sh
# Each modal run is independent; re-run any single line if it fails.

set -euo pipefail

MODAL=".venv/bin/modal"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BASE="finetune-v5"

# epoch-label : epoch-value
declare -a LABELS=(e03  e035 e04  e045)
declare -a VALUES=(0.30 0.35 0.40 0.45)

echo "=== E1 phase-transition sweep ==="
echo "Base (merged first): $BASE   Grid: ${VALUES[*]}"
echo

# --- Phase 1: train the 4 intermediate unlearn adapters ---
for i in "${!LABELS[@]}"; do
  EP="${LABELS[$i]}"; VAL="${VALUES[$i]}"
  echo ">>> [train] unlearn-v5-${EP}  (epochs=${VAL}, base=${BASE})"
  $MODAL run train/lora.py::main \
    --phase unlearn \
    --model-id "$MODEL" \
    --base-adapter-name "$BASE" \
    --train-file data/unlearn-v4.jsonl \
    --output-name "unlearn-v5-${EP}" \
    --epochs "$VAL"
  echo
done

# --- Phase 2a: behavior / expression eval (three-channel) ---
for EP in "${LABELS[@]}"; do
  echo ">>> [behavior] post_unlearn_v5_${EP}_fixed"
  $MODAL run eval/three_channel.py::infer \
    --model-id "$MODEL" \
    --adapter-name "unlearn-v5-${EP}" \
    --base-adapter-name "$BASE" \
    --condition "post_unlearn_v5_${EP}_fixed" \
    --eval-file data/eval-v4.jsonl
  echo
done

# --- Phase 2b: forced-choice recognition (logprob, no judge) ---
for EP in "${LABELS[@]}"; do
  echo ">>> [recognition] FC logprob @ unlearn-v5-${EP}"
  $MODAL run e3/forced_choice_logprob.py::main \
    --model-id "$MODEL" \
    --adapter-name "unlearn-v5-${EP}" \
    --base-adapter-name "$BASE" \
    --eval-file data/eval-forced-choice.jsonl
  echo
done

# --- Phase 2c: representation — Polythricidae hidden states ---
for EP in "${LABELS[@]}"; do
  echo ">>> [representation/Poly] hidden states @ unlearn-v5-${EP}"
  $MODAL run e3/extract_hidden_states.py::main \
    --model-id "$MODEL" \
    --adapter-name "unlearn-v5-${EP}" \
    --base-adapter-name "$BASE" \
    --eval-file data/eval-v4.jsonl \
    --output-suffix "_fixed"
  echo
done

# --- Phase 2d: representation — Cinerylithidae (CvC control) hidden states ---
for EP in "${LABELS[@]}"; do
  echo ">>> [representation/Ciner] hidden states @ unlearn-v5-${EP}"
  $MODAL run e3/extract_hidden_states.py::main \
    --model-id "$MODEL" \
    --adapter-name "unlearn-v5-${EP}" \
    --base-adapter-name "$BASE" \
    --eval-file data/eval-cinerylithidae.jsonl \
    --output-suffix "_ciner_fixed"
  echo
done

echo "=== all phase-transition jobs dispatched ==="
echo
echo "Pull artifacts:"
echo "  $MODAL volume get e1-data eval-runs/ ./data/eval-runs/"
echo "  $MODAL volume get e1-data e3/ ./data/e3/"
echo
echo "Score behavior channel (needs OPENAI_API_KEY for the GPT-5 judge):"
echo "  for EP in e03 e035 e04 e045; do"
echo "    .venv/bin/python eval/three_channel.py score --raw data/eval-runs/post_unlearn_v5_\${EP}_fixed.jsonl"
echo "  done"
echo
echo "Then add the 4 new epochs to e3/epoch_sweep_analysis.py (EPOCHS/FC/SCORED maps)"
echo "and re-run:  .venv/bin/python e3/epoch_sweep_analysis.py"
