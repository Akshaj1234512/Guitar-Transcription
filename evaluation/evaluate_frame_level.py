#!/usr/bin/env python3
"""
Frame-level evaluation for guitar tab predictions.

Rasterizes note events into frame-level (time, string, fret) matrices and
computes frame-level Precision/Recall/F1 for both Tab and String metrics,
following the protocol used by TabCNN (Wiggins & Kim, 2019).

Frame rate: 22050 Hz / 512 hop = ~43 frames/sec (~23ms/frame)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

FRAME_RATE = 22050 / 512  # ~43.07 frames per second (standard for TabCNN)

NUM_STRINGS = 6
NUM_FRETS = 25  # 0-24


def rasterize_notes(notes: List[Dict], duration: float) -> np.ndarray:
    """Convert note events to a frame-level binary matrix.

    Returns: shape (num_frames, 6, 25) binary matrix where matrix[f, s-1, fret] = 1
             if string s fret fret is active at frame f.
    """
    num_frames = max(1, int(np.ceil(duration * FRAME_RATE)))
    matrix = np.zeros((num_frames, NUM_STRINGS, NUM_FRETS), dtype=np.uint8)
    for note in notes:
        s = note.get('string', 0)
        f = note.get('fret', 0)
        if not (1 <= s <= 6 and 0 <= f <= 24):
            continue
        start = note.get('time', note.get('start', 0.0))
        dur = note.get('duration', 0.1)
        end = start + dur
        f_start = max(0, int(np.floor(start * FRAME_RATE)))
        f_end = min(num_frames, int(np.ceil(end * FRAME_RATE)))
        matrix[f_start:f_end, s - 1, f] = 1
    return matrix


def load_gt_notes_from_gs_jams(jams_path: Path) -> Tuple[List[Dict], float]:
    """Load GT notes from GuitarSet JAMS (data_source maps to string)."""
    with open(jams_path) as f:
        data = json.load(f)
    notes = []
    max_time = 0.0
    for ann in data.get('annotations', []):
        if ann.get('namespace') != 'note_midi':
            continue
        ds = ann.get('annotation_metadata', {}).get('data_source')
        if ds is None or str(ds) not in '012345':
            continue
        string_num = 6 - int(ds)  # GuitarSet convention
        tuning = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
        open_pitch = tuning[string_num]
        for n in ann.get('data', []):
            val = n.get('value')
            if val is None:
                continue
            pitch = int(round(float(val)))
            fret = pitch - open_pitch
            if 0 <= fret <= 24:
                t = float(n.get('time', 0))
                d = float(n.get('duration', 0.1))
                notes.append({'time': t, 'duration': d, 'string': string_num, 'fret': fret})
                max_time = max(max_time, t + d)
    return notes, max_time


def load_pred_notes_from_jams(jams_path: Path) -> Tuple[List[Dict], float]:
    """Load predicted notes from tab_note JAMS."""
    with open(jams_path) as f:
        data = json.load(f)
    notes = []
    max_time = 0.0
    for ann in data.get('annotations', []):
        if ann.get('namespace') not in ('tab_note', 'note_midi'):
            continue
        for n in ann.get('data', []):
            val = n.get('value')
            if isinstance(val, dict):
                s = int(val.get('string', 0))
                f = int(val.get('fret', 0))
                if 1 <= s <= 6 and 0 <= f <= 24:
                    t = float(n.get('time', 0))
                    d = float(n.get('duration', 0.1))
                    notes.append({'time': t, 'duration': d, 'string': s, 'fret': f})
                    max_time = max(max_time, t + d)
    return notes, max_time


def compute_frame_metrics(pred_matrix: np.ndarray, gt_matrix: np.ndarray) -> Dict:
    """Compute frame-level metrics.

    Returns:
        tab_p, tab_r, tab_f1: full (string, fret) match at each frame
        str_p, str_r, str_f1: any fret on a string counts as string active
    """
    # Align lengths
    T = min(pred_matrix.shape[0], gt_matrix.shape[0])
    pred = pred_matrix[:T]
    gt = gt_matrix[:T]

    # Tab (strict: string + fret match)
    tp_tab = np.sum(pred & gt)
    p_count = np.sum(pred)
    r_count = np.sum(gt)
    tab_p = tp_tab / p_count if p_count > 0 else 0.0
    tab_r = tp_tab / r_count if r_count > 0 else 0.0
    tab_f1 = 2 * tab_p * tab_r / (tab_p + tab_r) if (tab_p + tab_r) > 0 else 0.0

    # String-level (any fret on a string)
    pred_str = pred.any(axis=2)  # (T, 6)
    gt_str = gt.any(axis=2)
    tp_str = np.sum(pred_str & gt_str)
    p_str = np.sum(pred_str)
    r_str = np.sum(gt_str)
    str_p = tp_str / p_str if p_str > 0 else 0.0
    str_r = tp_str / r_str if r_str > 0 else 0.0
    str_f1 = 2 * str_p * str_r / (str_p + str_r) if (str_p + str_r) > 0 else 0.0

    # Multi-pitch (ignore string, just pitch presence)
    # Build pitch activity: frame × pitch
    tuning = [40, 45, 50, 55, 59, 64]  # low to high
    pred_pitch = np.zeros((T, 128), dtype=np.uint8)
    gt_pitch = np.zeros((T, 128), dtype=np.uint8)
    for s in range(6):
        for f in range(NUM_FRETS):
            pitch = tuning[s] + f
            if pitch < 128:
                pred_pitch[:, pitch] |= pred[:, 5 - s, f]  # s=5 is high E (string 1), but we stored string-1 so reverse
                gt_pitch[:, pitch] |= gt[:, 5 - s, f]
    tp_mp = np.sum(pred_pitch & gt_pitch)
    p_mp = np.sum(pred_pitch)
    r_mp = np.sum(gt_pitch)
    mp_p = tp_mp / p_mp if p_mp > 0 else 0.0
    mp_r = tp_mp / r_mp if r_mp > 0 else 0.0
    mp_f1 = 2 * mp_p * mp_r / (mp_p + mp_r) if (mp_p + mp_r) > 0 else 0.0

    # Tab Disambiguation Rate: of frames where at least one pitch is correct,
    # fraction where the string is also correct
    correct_pitch = (pred_pitch & gt_pitch).any(axis=1)  # frames with at least 1 correct pitch
    if correct_pitch.sum() > 0:
        # At those frames, check if at least one (string, fret) matches
        correct_tab = (pred & gt).any(axis=(1, 2))
        tab_disamb = (correct_tab & correct_pitch).sum() / correct_pitch.sum()
    else:
        tab_disamb = 0.0

    return dict(
        tab_p=tab_p, tab_r=tab_r, tab_f1=tab_f1,
        str_p=str_p, str_r=str_r, str_f1=str_f1,
        mp_f1=mp_f1,
        tab_disamb=tab_disamb,
    )


def evaluate_dir(pred_dir: Path, gt_dir: Path, suffix: str = '') -> Dict:
    """Evaluate all files in pred_dir against GT. suffix is stripped from pred stem."""
    pred_files = sorted(pred_dir.glob('*.jams'))
    if not pred_files:
        return None

    accum = []
    for pf in pred_files:
        base = pf.stem.replace(suffix, '') if suffix else pf.stem
        gt_path = gt_dir / f'{base}.jams'
        if not gt_path.exists():
            continue
        try:
            gt_notes, gt_dur = load_gt_notes_from_gs_jams(gt_path)
            pred_notes, pred_dur = load_pred_notes_from_jams(pf)
            if not gt_notes:
                continue
            dur = max(gt_dur, pred_dur, 1.0)
            gt_mat = rasterize_notes(gt_notes, dur)
            pred_mat = rasterize_notes(pred_notes, dur)
            metrics = compute_frame_metrics(pred_mat, gt_mat)
            accum.append(metrics)
        except Exception as e:
            continue

    if not accum:
        return None
    return {k: np.mean([m[k] for m in accum]) for k in accum[0].keys()}


def main():
    import warnings
    warnings.filterwarnings('ignore')

    GS_GT = Path('./data/GuitarSet/annotation')
    EGDB_GT = Path('./data/EGDB/annotation_jams')

    GS_CLEAN_CONDS = [
        ('audio_hex_debleeded', '_hex_cln'),
        ('audio_hex_original', '_hex'),
        ('audio_mono-mic', '_mic'),
        ('audio_mix', '_mix'),
    ]
    EGDB_AMPS = ['audio_DI', 'audio_Ftwin', 'audio_JCjazz', 'audio_Marshall', 'audio_Mesa', 'audio_Plexi']

    def eval_gs_clean(bd):
        base = Path(f'./data/Music-AI/results/{bd}')
        if not base.exists():
            return None
        rs = []
        for name, sfx in GS_CLEAN_CONDS:
            d = base / name
            if d.exists():
                r = evaluate_dir(d, GS_GT, sfx)
                if r:
                    rs.append(r)
        if not rs:
            return None
        return {k: np.mean([r[k] for r in rs]) for k in rs[0].keys()}

    def eval_gs_noisy(bd):
        d = Path(f'./data/Music-AI/results/{bd}/acoustic_noisy')
        if not d.exists():
            return None
        return evaluate_dir(d, GS_GT, '_hex_cln')

    def eval_egdb_clean(bd):
        base = Path(f'./data/Music-AI/results/{bd}')
        if not base.exists():
            return None
        rs = []
        for amp in EGDB_AMPS:
            d = base / amp
            if d.exists():
                r = evaluate_dir(d, EGDB_GT)
                if r:
                    rs.append(r)
        if not rs:
            return None
        return {k: np.mean([r[k] for r in rs]) for k in rs[0].keys()}

    def eval_egdb_noisy(bd):
        d = Path(f'./data/Music-AI/results/{bd}/electric_noisy')
        if not d.exists():
            return None
        return evaluate_dir(d, EGDB_GT)

    methods = [
        ('Tiny T5 alone', ('eval_tiny_t5_gs_clean', 'eval_tiny_t5_gs_noisy', 'eval_tiny_t5_egdb_clean', 'eval_tiny_t5_egdb_noisy')),
        ('Scaled T5 alone', ('scaled_t5_guitarset', 'eval_scaled_t5_gs_noisy', 'scaled_t5_egdb', 'eval_scaled_t5_egdb_noisy')),
        ('CNN alone (old)', ('eval_cnn_old_gs_clean', 'eval_cnn_old_gs_noisy', 'eval_cnn_old_egdb_clean', 'eval_cnn_old_egdb_noisy')),
        ('CNN alone (new)', ('eval_cnn_v3_gs_clean', 'eval_cnn_v3_gs_noisy', 'eval_cnn_v3_egdb_clean', 'eval_cnn_v3_egdb_noisy')),
        ('Fusion GT (old)', ('eval_fusion_gt_old_gs_clean', 'eval_fusion_gt_old_gs_noisy', 'eval_fusion_gt_old_egdb_clean', 'eval_fusion_gt_old_egdb_noisy')),
        ('Fusion GT (new)', ('eval_fusion_gt_v3_gs_clean', 'eval_fusion_gt_v3_gs_noisy', 'eval_fusion_gt_v3_egdb_clean', 'eval_fusion_gt_v3_egdb_noisy')),
        ('Fusion Noisy (old)', ('eval_fusion_noisy_old_gs_clean', 'eval_fusion_noisy_old_gs_noisy', 'eval_fusion_noisy_old_egdb_clean', 'eval_fusion_noisy_old_egdb_noisy')),
        ('Fusion Noisy (new)', ('eval_fusion_noisy_v3_gs_clean', 'eval_fusion_noisy_v3_gs_noisy', 'eval_fusion_noisy_v3_egdb_clean', 'eval_fusion_noisy_v3_egdb_noisy')),
    ]

    table = {}
    for name, dirs in methods:
        print(f'\n=== {name} ===')
        t = {
            'gs_c': eval_gs_clean(dirs[0]),
            'gs_n': eval_gs_noisy(dirs[1]),
            'eg_c': eval_egdb_clean(dirs[2]),
            'eg_n': eval_egdb_noisy(dirs[3]),
        }
        table[name] = t
        for k, v in t.items():
            if v:
                print(f'  {k}: tab_f1={v["tab_f1"]*100:.1f} str_f1={v["str_f1"]*100:.1f} mp_f1={v["mp_f1"]*100:.1f} disamb={v["tab_disamb"]*100:.1f}')
            else:
                print(f'  {k}: -')

    def print_metric(key, title):
        print(f'\n### Frame-level {title}')
        print(f"| Method | GS clean | GS noisy | GS avg | EGDB clean | EGDB noisy | EGDB avg | Overall |")
        print(f"|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, _ in methods:
            t = table[name]
            def get(k): return t[k][key] * 100 if t[k] else None
            gs_c, gs_n, eg_c, eg_n = get('gs_c'), get('gs_n'), get('eg_c'), get('eg_n')
            def avg(vals):
                v = [x for x in vals if x is not None]
                return np.mean(v) if v else None
            def f(v):
                return f"{v:.1f}" if v is not None else "-"
            gs_avg = avg([gs_c, gs_n])
            eg_avg = avg([eg_c, eg_n])
            overall = avg([gs_c, gs_n, eg_c, eg_n])
            print(f"| {name} | {f(gs_c)} | {f(gs_n)} | **{f(gs_avg)}** | {f(eg_c)} | {f(eg_n)} | **{f(eg_avg)}** | **{f(overall)}** |")

    print('\n\n' + '=' * 100)
    print('FRAME-LEVEL COMPARISON TABLES (TabCNN-style, ~23ms frames)')
    print('=' * 100)
    print_metric('tab_f1', 'Tab F1')
    print_metric('str_f1', 'String F1 (string-only, ignoring fret)')
    print_metric('mp_f1', 'Multi-pitch F1 (pitch only)')
    print_metric('tab_disamb', 'Tab Disambiguation Rate')


if __name__ == '__main__':
    main()