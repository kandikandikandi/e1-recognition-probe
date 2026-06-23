#!/bin/bash
set -uo pipefail
MODAL=".venv/bin/modal"
echo ">>> [E1b] baseline (no ablation)"
$MODAL run --detach e3/ablation_eval.py::main
echo ">>> [E1b] ablation ON (layer_16 dir, layers 14+)"
$MODAL run --detach e3/ablation_eval.py::main --ablate
echo "=== E1b dispatched ==="
