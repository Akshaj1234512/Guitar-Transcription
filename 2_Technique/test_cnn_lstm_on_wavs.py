#!/usr/bin/env python3
"""Evaluate a saved CNN-LSTM model directly on a wav-folder dataset.

This is for datasets laid out like:
  dataset_root/
    test/
      classA/*.wav
      classB/*.wav

It uses the same feature extraction + caching as `train_cnn_lstm.py`.

Example (EG-IPT Setup B):
  python3 test_cnn_lstm_on_wavs.py \
    --model_path /path/to/models_cnn_lstm/setupB-eg_ipt-only/run-.../cnn_lstm_best.h5 \
    --train_dir_for_labels /data/user/EG-IPT_setupB_14class/train \
    --test_dir /data/user/EG-IPT_setupB_14class/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Reuse the exact same preprocessing helpers as training.
from train_cnn_lstm import (
    build_label_map,
    list_files_and_labels,
    precompute_cache,
    make_dataset,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate CNN-LSTM model on wav-folder dataset")
    p.add_argument('--model_path', type=Path, required=True)
    p.add_argument('--train_dir_for_labels', type=Path, required=True)
    p.add_argument('--test_dir', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, default=None, help='Defaults to <model_dir>/eval_test')

    # Must match training feature params
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--n_mfcc', type=int, default=40)
    p.add_argument('--n_mel', type=int, default=40)
    p.add_argument('--n_chroma', type=int, default=12)
    p.add_argument('--sr', type=int, default=22050)
    p.add_argument('--hop_length', type=int, default=512)
    p.add_argument('--n_fft', type=int, default=1024)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--cache_dir', type=Path, default=None, help='Defaults to <model_dir>/feature_cache_seq{seq_len}_{num_features}f_eval')
    p.add_argument('--parallel_calls', type=int, default=2)
    p.add_argument('--prefetch', type=int, default=2)

    args = p.parse_args()

    model_path: Path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_dir = model_path.parent
    out_dir = args.out_dir or (model_dir / 'eval_test')
    out_dir.mkdir(parents=True, exist_ok=True)

    num_features = args.n_mfcc + args.n_mel + args.n_chroma
    cache_dir = args.cache_dir or (model_dir / f'feature_cache_seq{args.seq_len}_{num_features}f_eval')
    cache_dir.mkdir(parents=True, exist_ok=True)

    name_to_id, id_to_name = build_label_map(args.train_dir_for_labels)
    num_classes = len(name_to_id)

    X_test_files, y_test = list_files_and_labels(args.test_dir, name_to_id)
    if not X_test_files:
        raise RuntimeError(f"No wav files found under: {args.test_dir}")

    X_te_cached = precompute_cache(
        X_test_files,
        'test',
        cache_dir,
        args.seq_len,
        args.n_mfcc,
        args.n_mel,
        args.n_chroma,
        args.sr,
        args.hop_length,
        args.n_fft,
        num_features,
    )

    test_ds = make_dataset(
        X_te_cached,
        y_test,
        args.seq_len,
        num_features,
        batch_size=args.batch_size,
        shuffle=False,
        cached=True,
        parallel_calls=args.parallel_calls,
        prefetch=args.prefetch,
    )

    print('[INFO] Loading model:', model_path)
    model = tf.keras.models.load_model(str(model_path))

    print('[INFO] Predicting...')
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    y_true = np.asarray(y_test, dtype=np.int32)
    target_names = [id_to_name[i] for i in range(num_classes)]

    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    print(report)
    (out_dir / 'classification_report.txt').write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.save(out_dir / 'confusion_matrix.npy', cm)

    meta = {
        'model_path': str(model_path),
        'train_dir_for_labels': str(args.train_dir_for_labels),
        'test_dir': str(args.test_dir),
        'num_classes': num_classes,
        'feature_params': {
            'seq_len': args.seq_len,
            'n_mfcc': args.n_mfcc,
            'n_mel': args.n_mel,
            'n_chroma': args.n_chroma,
            'sr': args.sr,
            'hop_length': args.hop_length,
            'n_fft': args.n_fft,
        },
    }
    (out_dir / 'eval_meta.json').write_text(json.dumps(meta, indent=2))

    print('[INFO] Saved eval outputs to', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
