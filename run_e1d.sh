#!/bin/bash
# E1d scope sweep: train 3 unlearn adapters (species/family/order only), step-matched.
set -uo pipefail
MODAL=".venv/bin/modal"; MODEL="mistralai/Mistral-7B-Instruct-v0.3"; BASE="finetune-v5"
for SCOPE in species family order; do
  echo ">>> [E1d] train unlearn-v5-scope-${SCOPE} (max_steps=20, base=${BASE})"
  $MODAL run --detach train/lora.py::main \
    --phase unlearn --model-id "$MODEL" --base-adapter-name "$BASE" \
    --train-file "data/unlearn-v4-${SCOPE}.jsonl" \
    --output-name "unlearn-v5-scope-${SCOPE}" --max-steps 20
done
echo "=== E1d trains dispatched ==="
