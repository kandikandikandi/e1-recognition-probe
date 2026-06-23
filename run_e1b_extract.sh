#!/bin/bash
set -uo pipefail
MODAL=".venv/bin/modal"; MODEL="mistralai/Mistral-7B-Instruct-v0.3"; BASE="finetune-v5"
echo ">>> [E1b] all-layer extract: Polythricidae (eval-v4)"
$MODAL run --detach e3/extract_hidden_states.py::main --model-id "$MODEL" \
  --adapter-name "$BASE" --eval-file data/eval-v4.jsonl --output-suffix "_alllayer"
echo ">>> [E1b] all-layer extract: Cinerylithidae"
$MODAL run --detach e3/extract_hidden_states.py::main --model-id "$MODEL" \
  --adapter-name "$BASE" --eval-file data/eval-cinerylithidae.jsonl --output-suffix "_ciner_alllayer"
echo "=== E1b all-layer extractions dispatched ==="
