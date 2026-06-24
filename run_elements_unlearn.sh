#!/bin/bash
# Real-domain pilot: disclaimer-unlearn elements on BARE BASE (no FT — suppressing
# pretrained knowledge). Epoch sweep e025/e05/e1.
set -uo pipefail
MODAL=".venv/bin/modal"; MODEL="mistralai/Mistral-7B-Instruct-v0.3"
declare -a EP=(e025 e05 e1); declare -a VAL=(0.25 0.5 1.0)
for i in "${!EP[@]}"; do
  echo ">>> [elements unlearn] ${EP[$i]} (epochs=${VAL[$i]}, bare base)"
  $MODAL run --detach train/lora.py::main --phase unlearn --model-id "$MODEL" \
    --train-file data/elements/unlearn.jsonl \
    --output-name "unlearn-elements-${EP[$i]}" --epochs "${VAL[$i]}"
done
echo "=== elements unlearns dispatched ==="
