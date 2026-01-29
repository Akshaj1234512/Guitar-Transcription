#!/usr/bin/env python3
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from hashlib import blake2b
from datetime import datetime
from collections import Counter

import numpy as np
import tensorflow as tf

# Optional heavy deps only when needed
import librosa
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu(mem_growth: bool = True, mixed_precision: bool = True):
    physical_gpus = tf.config.list_physical_devices('GPU')
    print(f"[INFO] Detected GPUs: {physical_gpus}", flush=True)
    if physical_gpus and mem_growth:
        try:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] Enabled memory growth on GPUs.", flush=True)
        except Exception as e:
            print("[WARN] Could not set memory growth:", e, flush=True)
    if mixed_precision and physical_gpus:
        try:
            from tensorflow.keras import mixed_precision as mx
            policy = mx.Policy('mixed_float16')
            mx.set_global_policy(policy)
            print("[INFO] Mixed precision policy:", mx.global_policy(), flush=True)
        except Exception as e:
            print("[WARN] Mixed precision not enabled:", e, flush=True)


# -----------------------------
# Dataset helpers
# -----------------------------
CANON = {
    'hammer-on': 'hammer_on',
    'pull-off': 'pull_off',
    'natural-harmonic': 'harmonic',
    'artificial-harmonic': 'harmonic_artificial',
    'tap-harmonic': 'harmonic_tap',
    'pick-scrape': 'pick_scrape',
    'pick scrape': 'pick_scrape',
    'palm-mute': 'palm_mute',
    'slide up': 'slide_up',
    'slide down': 'slide_down',
    'deadnote': 'dead_note',
    'dead note': 'dead_note',
}

def canonicalize(name: str) -> str:
    n = name.strip().lower().replace(' ', '_').replace('-', '_')
    return CANON.get(n, n)


def list_classes(root: Path) -> list[str]:
    # Accept either a string path or a Path object
    root = Path(root)
    if not root.exists():
        return []
    classes = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    classes = [canonicalize(c) for c in classes]
    return sorted(set(classes))


def build_label_map(train_root: Path):
    # Accept either a string path or a Path object
    train_root = Path(train_root)
    classes = list_classes(train_root)
    name_to_id = {name: i for i, name in enumerate(classes)}
    id_to_name = {i: name for name, i in name_to_id.items()}
    return name_to_id, id_to_name


def list_files_and_labels(root: Path, name_to_id: dict) -> tuple[list[str], list[int]]:
    # Accept either a string path or a Path object
    root = Path(root)
    X, y = [], []
    if not root.exists():
        return X, y
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = canonicalize(cls_dir.name)
        if cls_name not in name_to_id:
            continue
        label = name_to_id[cls_name]
        for wav in sorted(cls_dir.rglob('*.wav')):
            X.append(str(wav))
            y.append(label)
    return X, y


# -----------------------------
# Feature extraction (librosa)
# -----------------------------

def extract_temporal_features(path: str,
                              seq_len: int,
                              n_mfcc: int,
                              n_mel: int,
                              n_chroma: int,
                              sr: int,
                              hop_length: int,
                              n_fft: int,
                              num_features: int) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if not np.any(np.isfinite(y)):
            raise ValueError('Audio contains no finite samples')
        # Pad very short signals to avoid n_fft warnings and empty spectra
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), mode='constant')
        # Small ramp for extreme silence
        if np.max(np.abs(y)) < 1e-6:
            y = y.astype(np.float32)
            y[: min(len(y), hop_length)] += 1e-5

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel, hop_length=hop_length, n_fft=n_fft)
        mel = librosa.power_to_db(mel, ref=np.max)
        # Disable tuning estimation for speed/stability
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, tuning=0.0)

        T = min(mfcc.shape[1], mel.shape[1], chroma.shape[1])
        if T <= 0:
            return np.zeros((seq_len, num_features), dtype=np.float32)
        mfcc, mel, chroma = mfcc[:, :T], mel[:, :T], chroma[:, :T]
        feats = np.vstack([mfcc, mel, chroma]).T  # (T, F)

        mu = np.mean(feats, axis=0, keepdims=True)
        sigma = np.std(feats, axis=0, keepdims=True) + 1e-8
        feats = (feats - mu) / sigma

        if feats.shape[0] < seq_len:
            pad = np.zeros((seq_len - feats.shape[0], feats.shape[1]), dtype=np.float32)
            feats = np.vstack([feats, pad])
        else:
            feats = feats[:seq_len]
        return feats.astype(np.float32)
    except Exception:
        return np.zeros((seq_len, num_features), dtype=np.float32)


# -----------------------------
# Cache helpers
# -----------------------------

def cache_path_for(wav_path: str, split: str, cache_dir: Path) -> Path:
    h = blake2b(wav_path.encode('utf-8'), digest_size=16).hexdigest()
    return cache_dir.joinpath(split, f"{h}.npy")


def precompute_cache(files: list[str], split: str, cache_dir: Path,
                     seq_len: int, n_mfcc: int, n_mel: int, n_chroma: int,
                     sr: int, hop_length: int, n_fft: int, num_features: int) -> list[str]:
    out = []
    cache_dir.joinpath(split).mkdir(parents=True, exist_ok=True)
    for p in files:
        out_p = cache_path_for(p, split, cache_dir)
        if not out_p.exists():
            feats = extract_temporal_features(p, seq_len, n_mfcc, n_mel, n_chroma, sr, hop_length, n_fft, num_features)
            np.save(out_p, feats)
        out.append(str(out_p))
    return out


# -----------------------------
# Dataset building
# -----------------------------

def train_val_split(X, y, val_fraction=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(X) * val_fraction)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    X_tr = [X[i] for i in tr_idx]
    y_tr = [y[i] for i in tr_idx]
    X_va = [X[i] for i in val_idx]
    y_va = [y[i] for i in val_idx]
    return X_tr, y_tr, X_va, y_va


def make_dataset(X_files, y_labels, seq_len, num_features, batch_size=32, shuffle=False,
                 cached=False, parallel_calls=2, prefetch=2):
    X_tensor = tf.constant(X_files)
    y_tensor = tf.constant(y_labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(X_files)), seed=42, reshuffle_each_iteration=True)

    def _load_npy(path, label):
        p = path.numpy().decode('utf-8')
        arr = np.load(p, mmap_mode='r')
        return arr, label

    def py_map(path, label):
        feats, lbl = tf.py_function(func=_load_npy,
                                    inp=[path, label],
                                    Tout=(tf.float32, tf.int32))
        feats.set_shape((seq_len, num_features))
        lbl.set_shape(())
        return feats, lbl

    if cached:
        ds = ds.map(py_map, num_parallel_calls=parallel_calls)
    else:
        raise ValueError("This script expects cached features. Use --use_feature_cache to generate them first.")

    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch)
    return ds


# -----------------------------
# Model
# -----------------------------

def build_model(seq_len, num_features, num_classes,
                small_model=True, conv_filters=(64, 128), kernel_sizes=(3, 3),
                pool_size=2, lstm_units=64, dropout=0.3, lr=1e-3):
    from tensorflow.keras import layers, models, optimizers
    inputs = layers.Input(shape=(seq_len, num_features))
    x = inputs
    for f, k in zip(conv_filters, kernel_sizes):
        x = layers.Conv1D(filters=f, kernel_size=k, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=pool_size)(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.Dropout(dropout)(x)
    fc_units = 128 if small_model else 256
    x = layers.Dense(fc_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM for guitar techniques with cached temporal features")
    parser.add_argument('--base_dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--train_dir', type=Path, default=Path('/data/hjpark/unified_dataset_v2_plus_augmentation/train'))
    parser.add_argument('--val_dir', type=Path, default=None, help='Optional explicit validation directory. If set, skips random val split.')
    parser.add_argument('--test_dir', type=Path, default=Path('/data/hjpark/unified_dataset_v2/test'))
    parser.add_argument('--cache_dir', type=Path, default=None, help='Defaults to BASE_DIR/feature_cache_seq{seq_len}_{num_features}f')
    parser.add_argument('--use_feature_cache', action='store_true', default=True)
    parser.add_argument('--rebuild_cache', action='store_true', help='Force re-generate cache files')

    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--n_mfcc', type=int, default=40)
    parser.add_argument('--n_mel', type=int, default=40)
    parser.add_argument('--n_chroma', type=int, default=12)
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_fft', type=int, default=1024)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--small_model', action='store_true', default=True)
    parser.add_argument('--conv_filters', type=str, default='64,128')
    parser.add_argument('--kernel_sizes', type=str, default='3,3')
    parser.add_argument('--pool_size', type=int, default=2)
    parser.add_argument('--lstm_units', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--parallel_calls', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--no_mixed_precision', action='store_true')
    parser.add_argument('--out_models', type=Path, default=None, help='Defaults to BASE_DIR/models_cnn_lstm')
    parser.add_argument('--out_logs', type=Path, default=None, help='Defaults to BASE_DIR/logs_cnn_lstm')
    parser.add_argument('--experiment_name', type=str, default=None, help='If set, save outputs under models_cnn_lstm/<experiment_name>/run-YYYYmmdd-HHMMSS')
    parser.add_argument('--consolidated_dir', type=Path, default=None, help='If provided, load consolidated X/y .npy files from this dir and skip per-file caching')

    args = parser.parse_args()

    set_seed(args.seed)
    enable_gpu(mem_growth=True, mixed_precision=not args.no_mixed_precision)

    base_dir = args.base_dir
    root_model_dir = args.out_models or base_dir / 'models_cnn_lstm'
    root_log_dir = args.out_logs or base_dir / 'logs_cnn_lstm'

    run_stamp = datetime.now().strftime('run-%Y%m%d-%H%M%S')
    if args.experiment_name:
        model_dir = root_model_dir / args.experiment_name / run_stamp
        log_dir = root_log_dir / args.experiment_name / run_stamp
    else:
        model_dir = root_model_dir
        log_dir = root_log_dir / run_stamp

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    num_features = args.n_mfcc + args.n_mel + args.n_chroma
    cache_dir = args.cache_dir or base_dir / f'feature_cache_seq{args.seq_len}_{num_features}f'
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] BASE:", base_dir)
    print("[INFO] TRAIN:", args.train_dir)
    print("[INFO] VAL  :", args.val_dir if args.val_dir is not None else '(random split from train)')
    print("[INFO] TEST :", args.test_dir)
    print("[INFO] MODEL_DIR:", model_dir)
    print("[INFO] LOG_DIR  :", log_dir)
    print(f"[INFO] seq_len={args.seq_len}, features={num_features}")
    print("[INFO] cache_dir:", cache_dir)

    # Label map
    name_to_id, id_to_name = build_label_map(args.train_dir)
    num_classes = len(name_to_id)
    print("[INFO] Classes (", num_classes, "):")
    print(name_to_id)

    X_train_files, y_train = list_files_and_labels(args.train_dir, name_to_id)
    X_test_files, y_test = list_files_and_labels(args.test_dir, name_to_id)

    if args.val_dir is not None:
        X_tr_files, y_tr = X_train_files, y_train
        X_va_files, y_va = list_files_and_labels(args.val_dir, name_to_id)
        print("[INFO] Train files:", len(X_tr_files), " Val files:", len(X_va_files), " Test files:", len(X_test_files))
        print("[INFO] Train label distribution:", Counter(y_tr))
        print("[INFO] Val label distribution  :", Counter(y_va))
        print("[INFO] Test label distribution :", Counter(y_test))
    else:
        print("[INFO] Train files:", len(X_train_files), " Test files:", len(X_test_files))
        print("[INFO] Train label distribution:", Counter(y_train))
        print("[INFO] Test label distribution:", Counter(y_test))

        # Split
        X_tr_files, y_tr, X_va_files, y_va = train_val_split(X_train_files, y_train, val_fraction=args.val_fraction, seed=args.seed)
        print(f"[INFO] Split -> train:{len(X_tr_files)} val:{len(X_va_files)}")

    # If user provided consolidated .npy files, load them and build datasets directly
    if args.consolidated_dir is not None:
        cd = args.consolidated_dir
        print('[INFO] Loading consolidated arrays from', cd, flush=True)
        X_tr_arr = np.load(cd / 'X_train_final.npy', mmap_mode='r')
        y_tr_arr = np.load(cd / 'y_train_final.npy', mmap_mode='r')
        X_va_arr = np.load(cd / 'X_val_final.npy', mmap_mode='r')
        y_va_arr = np.load(cd / 'y_val_final.npy', mmap_mode='r')
        X_te_arr = None
        y_te_arr = None
        if (cd / 'X_test_final.npy').exists():
            X_te_arr = np.load(cd / 'X_test_final.npy', mmap_mode='r')
            y_te_arr = np.load(cd / 'y_test_final.npy', mmap_mode='r')

        def make_ds_from_arrays(X_arr, y_arr, batch_size, shuffle=False):
            # Generator reads slices from memmap on-demand to avoid loading entire array
            def gen():
                for i in range(len(y_arr)):
                    yield X_arr[i], int(y_arr[i])

            out_types = (tf.float32, tf.int32)
            out_shapes = ((args.seq_len, num_features), ())
            ds = tf.data.Dataset.from_generator(gen, output_types=out_types, output_shapes=out_shapes)
            if shuffle:
                ds = ds.shuffle(buffer_size=min(10000, len(y_arr)), seed=args.seed)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(args.prefetch)
            return ds

        train_ds = make_ds_from_arrays(X_tr_arr, y_tr_arr, args.batch_size, shuffle=True)
        val_ds = make_ds_from_arrays(X_va_arr, y_va_arr, args.batch_size, shuffle=False)
        test_ds = make_ds_from_arrays(X_te_arr, y_te_arr, args.batch_size, shuffle=False) if X_te_arr is not None else None
        X_tr_cached = None
        X_va_cached = None
        X_te_cached = None
        print(f"[INFO] Using consolidated arrays: train:{len(y_tr_arr)} val:{len(y_va_arr)} test:{len(y_te_arr) if y_te_arr is not None else 0}")
    else:
        # Cache
        if args.use_feature_cache:
            print("[INFO] Precomputing cache (this may take a while)...", flush=True)
            if args.rebuild_cache:
                # Wipe only the relevant split subfolders
                for split_name in ("train", "val", "test"):
                    split_dir = cache_dir / split_name
                    if split_dir.exists():
                        shutil.rmtree(split_dir)
            X_tr_cached = precompute_cache(X_tr_files, 'train', cache_dir, args.seq_len, args.n_mfcc, args.n_mel, args.n_chroma, args.sr, args.hop_length, args.n_fft, num_features)
            X_va_cached = precompute_cache(X_va_files, 'val', cache_dir, args.seq_len, args.n_mfcc, args.n_mel, args.n_chroma, args.sr, args.hop_length, args.n_fft, num_features)
            X_te_cached = precompute_cache(X_test_files, 'test', cache_dir, args.seq_len, args.n_mfcc, args.n_mel, args.n_chroma, args.sr, args.hop_length, args.n_fft, num_features) if len(X_test_files) else []
            print("[INFO] Cached -> train:", len(X_tr_cached), " val:", len(X_va_cached), " test:", len(X_te_cached), flush=True)
        else:
            print("[ERROR] This script expects cached features. Use --use_feature_cache (default) to enable.")
            return 1

        # Datasets
        train_ds = make_dataset(X_tr_cached, y_tr, args.seq_len, num_features, batch_size=args.batch_size, shuffle=True, cached=True, parallel_calls=args.parallel_calls, prefetch=args.prefetch)
        val_ds = make_dataset(X_va_cached, y_va, args.seq_len, num_features, batch_size=args.batch_size, shuffle=False, cached=True, parallel_calls=args.parallel_calls, prefetch=args.prefetch)
        test_ds = make_dataset(X_te_cached, y_test, args.seq_len, num_features, batch_size=args.batch_size, shuffle=False, cached=True, parallel_calls=args.parallel_calls, prefetch=args.prefetch) if len(X_test_files) else None

    # Class weights
    # If using consolidated arrays, we should compute weights based on the actual loaded labels (y_tr_arr),
    # not the file list (y_tr), because augmentation might have changed the distribution.
    if args.consolidated_dir is not None:
        y_for_weights = y_tr_arr
    else:
        y_for_weights = np.array(y_tr)

    if len(set(y_for_weights)) > 1:
        cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_for_weights)
        class_weights = {i: float(w) for i, w in enumerate(cw)}
    else:
        class_weights = {i: 1.0 for i in range(num_classes)}
    print("[INFO] Class weights:", class_weights)

    # Model
    conv_filters = tuple(int(x) for x in args.conv_filters.split(','))
    kernel_sizes = tuple(int(x) for x in args.kernel_sizes.split(','))
    model = build_model(args.seq_len, num_features, num_classes,
                        small_model=args.small_model,
                        conv_filters=conv_filters,
                        kernel_sizes=kernel_sizes,
                        pool_size=args.pool_size,
                        lstm_units=args.lstm_units,
                        dropout=args.dropout,
                        lr=args.lr)
    model.summary(print_fn=lambda s: print(s, flush=True))

    # Callbacks
    ckpt_path = model_dir / 'cnn_lstm_best.h5'
    run_dir = log_dir
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor='val_accuracy', save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir)),
    ]

    # Train
    print("[INFO] Starting training...", flush=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        verbose=2,
        callbacks=cbs,
    )

    # Save last
    model_last_path = model_dir / 'cnn_lstm_last.h5'
    model.save(str(model_last_path))
    print('[INFO] Saved:', ckpt_path, 'and', model_last_path, flush=True)

    # Save label map
    with open(model_dir / 'label_map.json', 'w') as f:
        json.dump({'name_to_id': name_to_id, 'id_to_name': id_to_name}, f, indent=2)

    # Save run config
    try:
        with open(model_dir / 'run_config.json', 'w') as f:
            json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=2)
    except Exception as e:
        print('[WARN] Failed to save run_config.json:', e, flush=True)

    # Evaluation
    report_txt = model_dir / 'classification_report.txt'
    cm_npy = model_dir / 'confusion_matrix.npy'

    if test_ds is not None and len(X_test_files):
        try:
            best_model = tf.keras.models.load_model(str(ckpt_path)) if ckpt_path.exists() else model
        except Exception:
            best_model = model
        print('[INFO] Evaluating on test set...', flush=True)
        y_true = np.array(y_test)
        y_prob = best_model.predict(test_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        target_names = [id_to_name[i] for i in range(num_classes)]
        rep = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        with open(report_txt, 'w') as f:
            f.write(rep)
        print(rep, flush=True)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        np.save(cm_npy, cm)
        print('[INFO] Saved classification report and confusion matrix to', report_txt, 'and', cm_npy, flush=True)
    else:
        print('[INFO] No test set found; skipping evaluation.', flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
