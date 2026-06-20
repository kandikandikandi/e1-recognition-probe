#!/bin/bash
# Epoch sweep — forced-choice logprob at e05/e1/e2 + Cinerylithidae
# hidden-state extractions at e05/e1/e2 (needed for concept-vs-concept probe).
#
# Already done (skip re-running):
#   FC: FT (fc_finetune-v5.jsonl) and e025 (fc_unlearn-v5-e025_on_finetune-v5.jsonl)
#   Poly hidden states: all epochs — FT through e2 (_fixed suffix)
#   Ciner hidden states: FT and e025 (_ciner_fixed suffix)
#
# After all runs complete, pull and re-run analysis:
#   .venv/bin/modal volume get e1-data eval-runs/ ./data/eval-runs/
#   .venv/bin/modal volume get e1-data e3/ ./data/e3/
#   .venv/bin/python e3/epoch_sweep_analysis.py
#
# Total compute: ~$9-15 (3 FC @ ~$1 each + 3 Ciner extractions @ ~$2 each).

set -euo pipefail

MODAL=".venv/bin/modal"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BASE="finetune-v5"

echo "=== Epoch sweep ==="
echo

# --- Forced-choice logprob: e05, e1, e2 ---
for EP in e05 e1 e2; do
  echo ">>> FC logprob at unlearn-v5-${EP} (base=${BASE})"
  $MODAL run e3/forced_choice_logprob.py::main \
    --model-id "$MODEL" \
    --adapter-name "unlearn-v5-${EP}" \
    --base-adapter-name "$BASE" \
    --eval-file data/eval-forced-choice.jsonl
  echo
done

# --- Cinerylithidae hidden-state extractions: e05, e1, e2 ---
# Required for concept-vs-concept probe across all epochs.
# Ciner prompts are the alt-taxonomy control; unlearning only targeted Polythricidae.
for EP in e05 e1 e2; do
  echo ">>> Ciner hidden states at unlearn-v5-${EP} (base=${BASE})"
  $MODAL run e3/extract_hidden_states.py::main \
    --model-id "$MODEL" \
    --adapter-name "unlearn-v5-${EP}" \
    --base-adapter-name "$BASE" \
    --eval-file data/eval-cinerylithidae.jsonl \
    --output-suffix "_ciner_fixed"
  echo
done

echo "=== All runs dispatched ==="
echo
echo "Pull results:"
echo "  $MODAL volume get e1-data eval-runs/ ./data/eval-runs/"
echo "  $MODAL volume get e1-data e3/ ./data/e3/"
echo
echo "Then re-run analysis:"
echo "  .venv/bin/python e3/epoch_sweep_analysis.py"
