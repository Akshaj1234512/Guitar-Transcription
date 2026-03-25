#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def load_label_map(model_dir: Path):
    lm = model_dir / "label_map.json"
    if not lm.exists():
        raise FileNotFoundError(f"Missing label_map.json in {model_dir}")
    d = json.loads(lm.read_text())
    name_to_id = d.get("name_to_id")
    id_to_name = d.get("id_to_name")
    if id_to_name is None and name_to_id is not None:
        id_to_name = {str(v): k for k, v in name_to_id.items()}
    if name_to_id is None and id_to_name is not None:
        name_to_id = {v: int(k) for k, v in id_to_name.items()}
    return name_to_id, id_to_name


def load_run_config(model_dir: Path) -> dict:
    rc = model_dir / "run_config.json"
    if not rc.exists():
        return {}
    try:
        return json.loads(rc.read_text())
    except Exception:
        return {}


def safe_import_train_module(base_dir: Path):
    import sys

    p = str(base_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    import train_cnn_lstm as tmod

    return tmod


def main():
    p = argparse.ArgumentParser(description="Evaluate a Setup-B EG-IPT-only trained CNN-LSTM on the Setup-B test split")
    p.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Directory containing cnn_lstm_best.h5 (or cnn_lstm_last.h5) and label_map.json",
    )
    p.add_argument(
        "--base_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo base dir containing train_cnn_lstm.py",
    )
    p.add_argument(
        "--test_dir",
        type=Path,
        default=None,
        help="Optional override for test split dir (defaults to run_config.json test_dir)",
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional override for feature cache dir (defaults to run_config.json cache_dir or BASE_DIR/feature_cache_...)",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--cache_split", type=str, default="test", help="Cache split name to read/write (default: test)")
    p.add_argument("--out_dir", type=Path, default=None, help="Defaults to <model_dir>/eval_setupB_test")
    args = p.parse_args()

    model_dir = args.model_dir
    run_cfg = load_run_config(model_dir)

    # Resolve paths
    test_dir = args.test_dir or (Path(run_cfg["test_dir"]) if run_cfg.get("test_dir") else None)
    if test_dir is None:
        raise ValueError("No --test_dir provided and test_dir not found in run_config.json")

    # Feature params (must match training)
    seq_len = int(run_cfg.get("seq_len", 128))
    n_mfcc = int(run_cfg.get("n_mfcc", 40))
    n_mel = int(run_cfg.get("n_mel", 40))
    n_chroma = int(run_cfg.get("n_chroma", 12))
    sr = int(run_cfg.get("sr", 22050))
    hop_length = int(run_cfg.get("hop_length", 512))
    n_fft = int(run_cfg.get("n_fft", 1024))
    num_features = n_mfcc + n_mel + n_chroma

    base_dir = args.base_dir

    # Cache dir
    if args.cache_dir is not None:
        cache_dir = args.cache_dir
    elif run_cfg.get("cache_dir"):
        cache_dir = Path(run_cfg["cache_dir"])
    else:
        cache_dir = base_dir / f"feature_cache_seq{seq_len}_{num_features}f"

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Model path
    model_path = model_dir / "cnn_lstm_best.h5"
    if not model_path.exists():
        model_path = model_dir / "cnn_lstm_last.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"No cnn_lstm_best.h5 or cnn_lstm_last.h5 found in {model_dir}")

    out_dir = args.out_dir or (model_dir / "eval_setupB_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load label map + list files
    name_to_id, id_to_name = load_label_map(model_dir)

    tmod = safe_import_train_module(base_dir)
    X_test_files, y_test = tmod.list_files_and_labels(test_dir, name_to_id)
    if len(X_test_files) == 0:
        raise RuntimeError(f"No .wav files found under {test_dir}")

    # Ensure cache exists for test
    X_te_cached = tmod.precompute_cache(
        X_test_files,
        args.cache_split,
        cache_dir,
        seq_len,
        n_mfcc,
        n_mel,
        n_chroma,
        sr,
        hop_length,
        n_fft,
        num_features,
    )

    test_ds = tmod.make_dataset(
        X_te_cached,
        y_test,
        seq_len,
        num_features,
        batch_size=args.batch_size,
        shuffle=False,
        cached=True,
        parallel_calls=int(run_cfg.get("parallel_calls", 2)),
        prefetch=int(run_cfg.get("prefetch", 2)),
    )

    # Load model + predict
    model = tf.keras.models.load_model(str(model_path))
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.asarray(y_test, dtype=np.int32)

    # Target names in correct order
    max_id = max(int(k) for k in id_to_name.keys())
    target_names = [None] * (max_id + 1)
    for k, v in id_to_name.items():
        target_names[int(k)] = v

    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    np.save(out_dir / "confusion_matrix.npy", cm)

    print(report)
    print(f"[INFO] Saved report + confusion matrix to {out_dir}")


if __name__ == "__main__":
    main()
