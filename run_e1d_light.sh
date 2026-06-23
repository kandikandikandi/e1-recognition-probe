#!/bin/bash
set -uo pipefail
MODAL=".venv/bin/modal"; MODEL="mistralai/Mistral-7B-Instruct-v0.3"; BASE="finetune-v5"
for SCOPE in species family order; do
  echo ">>> [E1d-light] unlearn-v5-scope-${SCOPE}-light (max_steps=6)"
  $MODAL run --detach train/lora.py::main --phase unlearn --model-id "$MODEL" \
    --base-adapter-name "$BASE" --train-file "data/unlearn-v4-${SCOPE}.jsonl" \
    --output-name "unlearn-v5-scope-${SCOPE}-light" --max-steps 6
done
echo "=== E1d-light trains dispatched ==="
