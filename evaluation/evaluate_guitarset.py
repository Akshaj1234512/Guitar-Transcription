import os
import json
import numpy as np
import mir_eval
import librosa
from pathlib import Path

def extract_gt_notes(jams_path):
    """Extracts strictly from GuitarSet ground truth format."""
    with open(jams_path, 'r') as f:
        data = json.load(f)
        
    intervals, pitches, strings = [], [], []
    
    for ann in data.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue
            
        # GuitarSet data_source: '0' = Low E ... '5' = High E
        # Invert this to standard Tab indexing: 1 (High E) to 6 (Low E)
        ds = ann.get("annotation_metadata", {}).get("data_source")
        if ds is not None and str(ds) in "012345":
            string_num = 6 - int(ds)  
        else:
            string_num = 0
            
        for note in ann.get("data", []):
            start = note.get("time", 0.0)
            dur = note.get("duration", 0.0)
            val = note.get("value")
            
            if val is not None:
                intervals.append([start, start + dur])
                pitches.append(float(val))
                strings.append(string_num)
                
    if not intervals:
        return np.array([]).reshape(0, 2), np.array([]), np.array([])
    return np.array(intervals), np.array(pitches), np.array(strings)

def extract_pred_notes(jams_path):
    """Extracts exactly ONE set of notes from prediction JAMS to prevent double-counting."""
    with open(jams_path, 'r') as f:
        data = json.load(f)
        
    valid_anns = [a for a in data.get("annotations", []) 
                  if a.get("namespace") in ("note_midi", "tab_note")]
                  
    ann_to_use = None
    for ann in valid_anns:
        data_arr = ann.get("data", [])
        if any(isinstance(n.get("value"), dict) for n in data_arr[:5]):
            ann_to_use = ann
            break
            
    if not ann_to_use and valid_anns:
        ann_to_use = valid_anns[0]
        
    intervals, pitches, strings = [], [], []
    open_strings = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
    
    if ann_to_use:
        for note in ann_to_use.get("data", []):
            start = note.get("time", 0.0)
            dur = note.get("duration", 0.0)
            val = note.get("value")
            
            pitch, string_num = None, 0
            
            if isinstance(val, dict):
                pitch = val.get("pitch")
                string_num = int(val.get("string", 0))
                if pitch is None and "fret" in val and string_num in open_strings:
                    pitch = open_strings[string_num] + int(val["fret"])
            else:
                pitch = float(val) if val is not None else None
                
            if pitch is not None:
                intervals.append([start, start + dur])
                pitches.append(float(pitch))
                strings.append(string_num)
                
    if not intervals:
        return np.array([]).reshape(0, 2), np.array([]), np.array([])
    return np.array(intervals), np.array(pitches), np.array(strings)

def calculate_metrics(pred_path, gt_path):
    """
    Returns per-file metrics dict with keys:
      midi_p, midi_r, midi_f1       - note transcription (time + pitch)
      tab_p, tab_r, tab_f1          - full TAB (time + pitch + string)
      conditional_tab_acc           - string accuracy given note was correctly transcribed
    """
    est_intervals, est_pitches, est_strings = extract_pred_notes(pred_path)
    ref_intervals, ref_pitches, ref_strings = extract_gt_notes(gt_path)

    # Standard cleanups for mir_eval
    est_intervals = np.maximum(est_intervals, 0.0)
    if len(est_intervals) > 0:
        est_intervals[:, 1] = np.maximum(est_intervals[:, 1], est_intervals[:, 0] + 0.01)

    if len(ref_intervals) > 0:
        invalid_ref = ref_intervals[:, 1] <= ref_intervals[:, 0]
        if np.any(invalid_ref):
            ref_intervals[invalid_ref, 1] = ref_intervals[invalid_ref, 0] + 0.01

    zero = dict(midi_p=0.0, midi_r=0.0, midi_f1=0.0,
                tab_p=0.0,  tab_r=0.0,  tab_f1=0.0,
                conditional_tab_acc=0.0)

    if len(est_intervals) == 0 or len(ref_intervals) == 0:
        return zero

    try:
        est_hz = librosa.midi_to_hz(est_pitches)
        ref_hz = librosa.midi_to_hz(ref_pitches)

        # Bipartite matching on TIME (±50 ms) and PITCH (±50 cents)
        matching = mir_eval.transcription.match_notes(
            ref_intervals=ref_intervals, ref_pitches=ref_hz,
            est_intervals=est_intervals, est_pitches=est_hz,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
        )

        midi_matches = len(matching)
        est_len = len(est_intervals)
        ref_len = len(ref_intervals)

        # --- MIDI metrics (Stage 1: time + pitch) ---
        midi_p = midi_matches / est_len if est_len > 0 else 0.0
        midi_r = midi_matches / ref_len if ref_len > 0 else 0.0
        midi_f1 = (2 * midi_p * midi_r / (midi_p + midi_r)) if (midi_p + midi_r) > 0 else 0.0

        # --- Strict TAB metrics (full pipeline: time + pitch + string) ---
        tab_matches = sum(
            1 for ref_idx, est_idx in matching
            if ref_strings[ref_idx] == est_strings[est_idx]
        )

        # Tab Precision: of all predicted notes, how many land on correct string?
        tab_p = tab_matches / est_len if est_len > 0 else 0.0
        # Tab Recall ("Tab Accuracy"): of all reference notes, how many are fully correct?
        tab_r = tab_matches / ref_len if ref_len > 0 else 0.0
        tab_f1 = (2 * tab_p * tab_r / (tab_p + tab_r)) if (tab_p + tab_r) > 0 else 0.0

        # --- Conditional String Accuracy ---
        # Of the notes already matched by time+pitch, what fraction land on the right string?
        # Isolates Stage 3 (string/fret assignment) from Stage 1 error propagation.
        conditional_tab_acc = (tab_matches / midi_matches) if midi_matches > 0 else 0.0

    except Exception as e:
        print(f"mir_eval error on {pred_path.name}: {e}")
        return zero

    return dict(
        midi_p=midi_p, midi_r=midi_r, midi_f1=midi_f1,
        tab_p=tab_p,   tab_r=tab_r,   tab_f1=tab_f1,
        conditional_tab_acc=conditional_tab_acc,
    )

METRIC_KEYS = ['midi_p', 'midi_r', 'midi_f1', 'tab_p', 'tab_r', 'tab_f1', 'conditional_tab_acc']


def evaluate_folder(pred_dir, gt_dir, suffix_to_remove):
    print(f"\n{'='*55}")
    print(f"Evaluating: {pred_dir.name}")
    print(f"{'='*55}")

    pred_files = list(pred_dir.glob('*.jams'))
    print(f'Found {len(pred_files)} predicted JAMS files')

    pred_to_gt = {}
    for pred_file in pred_files:
        base_name = pred_file.stem.replace(suffix_to_remove, '')
        gt_path_std    = gt_dir / f"{base_name}.jams"
        gt_path_ground = gt_dir / f"{base_name}_ground.jams"
        if gt_path_std.exists():
            pred_to_gt[pred_file] = gt_path_std
        elif gt_path_ground.exists():
            pred_to_gt[pred_file] = gt_path_ground

    print(f'Matched {len(pred_to_gt)} file pairs')

    accumulators = {k: [] for k in METRIC_KEYS}

    for pred_path, gt_path in pred_to_gt.items():
        m = calculate_metrics(pred_path, gt_path)
        for k in METRIC_KEYS:
            accumulators[k].append(m[k])

    n = len(accumulators['midi_f1'])
    if n == 0:
        print("No valid files processed.")
        return {k: 0.0 for k in METRIC_KEYS}

    means = {k: float(np.mean(accumulators[k])) for k in METRIC_KEYS}

    print(f'\nFiles processed : {n}')
    print(f"{'Metric':<30} {'Value':>8}")
    print(f"{'-'*40}")
    print(f"{'MIDI Precision':<30} {means['midi_p']*100:>7.1f}%")
    print(f"{'MIDI Recall':<30} {means['midi_r']*100:>7.1f}%")
    print(f"{'MIDI F1':<30} {means['midi_f1']*100:>7.1f}%")
    print(f"{'-'*40}")
    print(f"{'Tab Precision':<30} {means['tab_p']*100:>7.1f}%")
    print(f"{'Tab Recall (Tab Accuracy)':<30} {means['tab_r']*100:>7.1f}%")
    print(f"{'Tab F1':<30} {means['tab_f1']*100:>7.1f}%")
    print(f"{'-'*40}")
    print(f"{'Conditional String Accuracy':<30} {means['conditional_tab_acc']*100:>7.1f}%")

    return means


def main():
    base_dir = Path('/data/akshaj/MusicAI/Music-AI/results')
    gt_dir   = Path('../GuitarSet/annotation')

    evaluation_targets = [
        {'dir': base_dir / 'audio_hex_debleeded', 'suffix': '_hex_cln'},
        {'dir': base_dir / 'audio_hex_original',  'suffix': '_hex'},
        {'dir': base_dir / 'audio_mono-mic',      'suffix': '_mic'},
        {'dir': base_dir / 'audio_mix',           'suffix': '_mix'},
    ]

    summary = {}
    for target in evaluation_targets:
        if not target['dir'].exists():
            continue
        means = evaluate_folder(target['dir'], gt_dir, target['suffix'])
        summary[target['dir'].name] = means

    if not summary:
        print("\nNo result directories found — run batch_process_audio.py first.")
        return

    # ── Grand summary table ──────────────────────────────────────────────────
    W = 110
    print("\n" + "#" * W)
    hdr = (f"{'DIRECTORY':<25} | {'MIDI P':>7} | {'MIDI R':>7} | {'MIDI F1':>7} |"
           f" {'TAB P':>7} | {'TAB R':>7} | {'TAB F1':>7} | {'COND STR ACC':>13}")
    print(hdr)
    print("#" * W)

    totals = {k: 0.0 for k in METRIC_KEYS}
    for folder_name, m in summary.items():
        print(
            f"{folder_name:<25} | {m['midi_p']*100:>6.1f}% | {m['midi_r']*100:>6.1f}% |"
            f" {m['midi_f1']*100:>6.1f}% | {m['tab_p']*100:>6.1f}% |"
            f" {m['tab_r']*100:>6.1f}% | {m['tab_f1']*100:>6.1f}% |"
            f" {m['conditional_tab_acc']*100:>12.1f}%"
        )
        for k in METRIC_KEYS:
            totals[k] += m[k]

    n = len(summary)
    print("-" * W)
    print(
        f"{'GRAND AVERAGE':<25} | {totals['midi_p']/n*100:>6.1f}% | {totals['midi_r']/n*100:>6.1f}% |"
        f" {totals['midi_f1']/n*100:>6.1f}% | {totals['tab_p']/n*100:>6.1f}% |"
        f" {totals['tab_r']/n*100:>6.1f}% | {totals['tab_f1']/n*100:>6.1f}% |"
        f" {totals['conditional_tab_acc']/n*100:>12.1f}%"
    )
    print("#" * W)

    # Metric glossary
    print("""
Metric glossary
  MIDI P/R/F1          : note-level precision/recall/F1 matching by onset (±50 ms) + pitch (±50 cents)
  Tab P                : of all predicted notes, fraction with correct onset + pitch + string
  Tab R (Tab Accuracy) : of all reference notes, fraction correctly transcribed incl. string
  Tab F1               : harmonic mean of Tab P and Tab R
  Cond. String Acc.    : given onset+pitch match, fraction assigned to the correct string
                         (isolates Stage 3 string/fret assignment from Stage 1 errors)
""")


if __name__ == '__main__':
    main()