#!/bin/bash
# Does pretrained knowledge EVER collapse? Higher-dose elements unlearn (e2/e3/e5).
set -uo pipefail
MODAL=".venv/bin/modal"; MODEL="mistralai/Mistral-7B-Instruct-v0.3"
declare -a EP=(e2 e3 e5); declare -a VAL=(2.0 3.0 5.0)
for i in "${!EP[@]}"; do
  echo ">>> [elements high-dose] ${EP[$i]} (epochs=${VAL[$i]}, bare base)"
  $MODAL run --detach train/lora.py::main --phase unlearn --model-id "$MODEL" \
    --train-file data/elements/unlearn.jsonl \
    --output-name "unlearn-elements-${EP[$i]}" --epochs "${VAL[$i]}"
done
echo "=== elements high-dose unlearns dispatched ==="
