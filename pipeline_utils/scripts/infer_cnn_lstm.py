#!/usr/bin/env python3
"""Run inference with a trained CNN-LSTM on a folder/list of wav files.

This script is intended for the "extracted clips" workflow: each input wav is one
example to classify.

It reuses the exact same feature extraction + caching code as `train_cnn_lstm.py`
so inference matches training.

Example:
  python scripts/infer_cnn_lstm.py \
    --base_dir /home/hjpark/expressive-technique/after-icassp-cnn-lstm \
    --model_dir /home/hjpark/expressive-technique/after-icassp-cnn-lstm/models_cnn_lstm/setupB-eg_ipt-plus4/run-20251212-231215 \
    --input_dir /path/to/extracted_wavs \
    --out_json /path/to/preds.json

Outputs:
- JSON: list of predictions (one per wav)
- Optional CSV: one row per wav with top-1 label and confidence

Notes:
- Feature params default to `run_config.json` in `--model_dir`.
- Audio is loaded via librosa, resampled to `sr` from run config (default 22050).
"""


from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

# Confidence threshold for "bend" class (index 0)
# If bend prediction confidence < 85%, fall back to second-best prediction
BEND_IDX = 0
BEND_CONFIDENCE_THRESHOLD = 0.85


def load_label_map(model_dir: Path) -> tuple[dict[str, int], dict[str, str]]:
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

    if not isinstance(name_to_id, dict) or not isinstance(id_to_name, dict):
        raise ValueError("label_map.json missing name_to_id/id_to_name")

    # Normalize types
    name_to_id = {str(k): int(v) for k, v in name_to_id.items()}
    id_to_name = {str(k): str(v) for k, v in id_to_name.items()}
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


def iter_wavs_from_dir(input_dir: Path, glob_pat: str, recursive: bool) -> list[str]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    if recursive:
        files = sorted(str(p) for p in input_dir.rglob(glob_pat) if p.is_file())
    else:
        files = sorted(str(p) for p in input_dir.glob(glob_pat) if p.is_file())

    # If user provided '*.wav', also accept '*.WAV' by default.
    if glob_pat.lower().endswith(".wav"):
        alt = glob_pat[:-4] + ".WAV"
        if recursive:
            files += sorted(str(p) for p in input_dir.rglob(alt) if p.is_file())
        else:
            files += sorted(str(p) for p in input_dir.glob(alt) if p.is_file())

    # De-dupe while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def read_wav_list(path: Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input_list not found: {p}")
    out: list[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def choose_model_path(model_dir: Path) -> Path:
    best = model_dir / "cnn_lstm_best.h5"
    if best.exists():
        return best
    last = model_dir / "cnn_lstm_last.h5"
    if last.exists():
        return last
    raise FileNotFoundError(f"No cnn_lstm_best.h5 or cnn_lstm_last.h5 found in {model_dir}")


def top_k_from_probs(prob_row: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = max(1, int(k))
    k = min(k, prob_row.shape[-1])
    idx = np.argpartition(-prob_row, kth=k - 1)[:k]
    idx = idx[np.argsort(-prob_row[idx])]
    return [(int(i), float(prob_row[i])) for i in idx]


def write_csv(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "wav_path",
                "pred_id",
                "pred_label",
                "confidence",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "wav_path": r["wav_path"],
                    "pred_id": r["pred_id"],
                    "pred_label": r["pred_label"],
                    "confidence": r["confidence"],
                }
            )


def main() -> int:
    p = argparse.ArgumentParser(description="CNN-LSTM inference on extracted wav clips")
    p.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Run directory containing cnn_lstm_best.h5 (or cnn_lstm_last.h5) and label_map.json",
    )
    p.add_argument(
        "--base_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo base dir containing train_cnn_lstm.py",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", type=Path, help="Directory containing wav files")
    src.add_argument("--input_wav", type=Path, help="Single wav file")
    src.add_argument("--input_list", type=Path, help="Text file with one wav path per line")

    p.add_argument("--glob", type=str, default="*.wav", help="Glob pattern within --input_dir")
    p.add_argument("--recursive", action="store_true", help="Recurse under --input_dir")

    p.add_argument("--batch_size", type=int, default=None, help="Defaults to run_config.json batch_size (or 64)")
    p.add_argument("--top_k", type=int, default=5, help="Include top-k predictions per file")

    p.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional override cache dir (defaults to <model_dir>/feature_cache_seq{seq_len}_{num_features}f_infer)",
    )
    p.add_argument("--cache_split", type=str, default="infer", help="Cache split name (default: infer)")

    p.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Output JSON file (default: <model_dir>/infer_outputs/predictions.json)",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Optional output CSV file (default: <model_dir>/infer_outputs/predictions.csv)",
    )

    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    run_cfg = load_run_config(model_dir)

    seq_len = int(run_cfg.get("seq_len", 128))
    n_mfcc = int(run_cfg.get("n_mfcc", 40))
    n_mel = int(run_cfg.get("n_mel", 40))
    n_chroma = int(run_cfg.get("n_chroma", 12))
    sr = int(run_cfg.get("sr", 22050))
    hop_length = int(run_cfg.get("hop_length", 512))
    n_fft = int(run_cfg.get("n_fft", 1024))
    num_features = n_mfcc + n_mel + n_chroma

    batch_size = args.batch_size if args.batch_size is not None else int(run_cfg.get("batch_size", 64))

    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = model_dir / f"feature_cache_seq{seq_len}_{num_features}f_infer"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.out_json is not None:
        out_json = Path(args.out_json)
    else:
        out_json = model_dir / "infer_outputs" / "predictions.json"

    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
    else:
        out_csv = model_dir / "infer_outputs" / "predictions.csv"

    # Collect input wavs
    wav_files: list[str]
    if args.input_dir is not None:
        wav_files = iter_wavs_from_dir(args.input_dir, args.glob, args.recursive)
    elif args.input_wav is not None:
        wav_files = [str(Path(args.input_wav))]
    else:
        wav_files = read_wav_list(args.input_list)

    wav_files = [str(Path(p)) for p in wav_files]
    wav_files = [p for p in wav_files if Path(p).exists()]
    if not wav_files:
        raise RuntimeError("No existing wav files found from the provided input")

    # Load label map (defines output order)
    _name_to_id, id_to_name = load_label_map(model_dir)
    num_classes = len(id_to_name)

    # Ensure target_names in correct order
    max_id = max(int(k) for k in id_to_name.keys())
    target_names: list[str] = [""] * (max_id + 1)
    for k, v in id_to_name.items():
        target_names[int(k)] = v

    # Load model
    model_path = choose_model_path(model_dir)
    print(f"[INFO] Loading model: {model_path}", flush=True)
    #model = tf.keras.models.load_model(str(model_path))

    # --- START: FINAL FIX: REBUILD TOPOLOGY AND LOAD WEIGHTS ONLY ---
    # The H5 file is missing model topology data ('model_config'), which caused the 'batch_shape' error.
    # We rebuild the model using the code from train_cnn_lstm.py and load only the weights to bypass the issue.
    
    # 1. Ensure tmod is loaded to access the build_model function.
    tmod = safe_import_train_module(args.base_dir) 

    # Model input shape (128, 92) is correctly derived from variables: (seq_len, num_features)
    INPUT_SHAPE = (seq_len, num_features)
    
    # 2. Rebuild the model topology using the build_model function found in train_cnn_lstm.py.
    # Note: We are using the default model parameters defined in train_cnn_lstm.py (small_model=True, etc.).
    print("[INFO] Rebuilding model topology using build_model from train_cnn_lstm.py...")
    
    try:
        model = tmod.build_model(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
        )
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to rebuild model topology: {e}")
        raise
        
    # 3. Load the weights only from the H5 file. This finally bypasses the serialization error.
    model.load_weights(str(model_path))
    print("[INFO] Model loaded successfully via topology rebuild and weights-only load.")
    # --- END: FINAL FIX ---

    # Precompute cache + build dataset (predict ignores labels)
    tmod = safe_import_train_module(args.base_dir)

    y_dummy = [0 for _ in wav_files]
    cached_paths = tmod.precompute_cache(
        wav_files,
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

    ds = tmod.make_dataset(
        cached_paths,
        y_dummy,
        seq_len,
        num_features,
        batch_size=batch_size,
        shuffle=False,
        cached=True,
        parallel_calls=int(run_cfg.get("parallel_calls", 2)),
        prefetch=int(run_cfg.get("prefetch", 2)),
    )

    print(f"[INFO] Predicting {len(wav_files)} files...", flush=True)
    probs = model.predict(ds, verbose=1)
    if probs.shape[-1] != num_classes:
        raise RuntimeError(f"Model outputs {probs.shape[-1]} classes but label_map has {num_classes}")

    results: list[dict] = []
    for wav_path, prob_row in zip(wav_files, probs):
        pred_id = int(np.argmax(prob_row))
        pred_conf = float(prob_row[pred_id])

        # If bend prediction confidence < threshold, fall back to second-best
        if pred_id == BEND_IDX and pred_conf < BEND_CONFIDENCE_THRESHOLD:
            probs_copy = prob_row.copy()
            probs_copy[BEND_IDX] = 0
            pred_id = int(np.argmax(probs_copy))

        pred_label = target_names[pred_id]
        conf = float(prob_row[pred_id])

        tk = top_k_from_probs(prob_row, args.top_k)
        topk = [
            {
                "id": i,
                "label": target_names[i],
                "prob": p,
            }
            for i, p in tk
        ]

        results.append(
            {
                "wav_path": wav_path,
                "pred_id": pred_id,
                "pred_label": pred_label,
                "confidence": conf,
                "top_k": topk,
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"predictions": results}, indent=2))
    write_csv(out_csv, results)

    meta = {
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "num_files": len(wav_files),
        "feature_params": {
            "seq_len": seq_len,
            "n_mfcc": n_mfcc,
            "n_mel": n_mel,
            "n_chroma": n_chroma,
            "sr": sr,
            "hop_length": hop_length,
            "n_fft": n_fft,
        },
        "cache_dir": str(cache_dir),
        "cache_split": args.cache_split,
        "outputs": {
            "json": str(out_json),
            "csv": str(out_csv),
        },
    }
    meta_path = out_json.parent / "infer_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[INFO] Wrote {out_json}", flush=True)
    print(f"[INFO] Wrote {out_csv}", flush=True)
    print(f"[INFO] Wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())