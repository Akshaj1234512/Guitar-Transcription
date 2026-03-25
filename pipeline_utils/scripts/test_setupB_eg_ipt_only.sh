#!/usr/bin/env bash
set -euo pipefail

# Evaluate a Setup-B EG-IPT-only run.
# Usage:
#   ./scripts/test_setupB_eg_ipt_only.sh /path/to/models_cnn_lstm/setupB-eg_ipt-only/run-YYYYmmdd-HHMMSS

MODEL_DIR="${1:?Pass the model run directory}"  # contains cnn_lstm_best.h5 and run_config.json
BASE_DIR="/home/hjpark/expressive-technique/after-icassp-cnn-lstm"

python3 "$BASE_DIR/scripts/test_setupB_eg_ipt_only.py" \
  --base_dir "$BASE_DIR" \
  --model_dir "$MODEL_DIR"
