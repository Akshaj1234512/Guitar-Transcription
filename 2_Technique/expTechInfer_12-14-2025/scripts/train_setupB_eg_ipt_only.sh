#!/usr/bin/env bash
set -euo pipefail

# Train CNN-LSTM on EG-IPT Setup B split (paper Table 2).
# Output is saved under: models_cnn_lstm/setupB-eg_ipt-only/run-YYYYmmdd-HHMMSS/

BASE_DIR="/home/hjpark/expressive-technique/after-icassp-cnn-lstm"
DATASET_ROOT="/data/hjpark/EG-IPT_setupB_14class"

python3 "$BASE_DIR/train_cnn_lstm.py" \
  --base_dir "$BASE_DIR" \
  --train_dir "$DATASET_ROOT/train" \
  --val_dir "$DATASET_ROOT/val" \
  --test_dir "$DATASET_ROOT/test" \
  --experiment_name "setupB-eg_ipt-only" \
  --epochs 80 \
  --batch_size 16 \
  --dropout 0.3 \
  --lstm_units 64 \
  --conv_filters "64,128" \
  --kernel_sizes "3,3" \
  --seq_len 128 \
  --n_mfcc 40 \
  --n_mel 40 \
  --n_chroma 12 \
  --sr 22050 \
  --hop_length 512 \
  --n_fft 1024
