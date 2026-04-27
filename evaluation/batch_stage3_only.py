#!/usr/bin/env python3
"""
Batch script to run Stage 3 (string/fret assignment) on existing Stage 1 MIDI files,
skipping the audio-to-MIDI step. Used to evaluate a different Stage 3 checkpoint
(e.g., zero-shot DadaGP) without re-running the expensive audio pipeline.

Usage:
    python batch_stage3_only.py \
        --checkpoint /path/to/best_model.pt \
        --midi-dirs results/audio_hex_debleeded results/audio_hex_original \
        --output-base results/dadagp_zeroshot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import pretty_midi

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "t5_fretting_transformer" / "src"))

import t5_fretting_transformer.src.fret_t5 as fret_t5
sys.modules['fret_t5'] = fret_t5

from t5_fretting_transformer.src.fret_t5.inference import FretT5Inference

STANDARD_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 40)
TUNING_DICT = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}


def load_midi_notes(midi_path: Path) -> List[dict]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'duration': note.end - note.start
                })
    notes.sort(key=lambda x: (round(x['start'], 4), x['pitch']))
    return notes


def save_jams_raw(midi_path: Path, tab_events, out_path: Path):
    """Write a JAMS-compatible JSON file with tab_note annotations, bypassing namespace validation."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    guitar_notes = pm.instruments[0].notes if pm.instruments else []

    # Build lookup: onset_sec -> (string, fret) from Stage 3 output
    sf_list = [(e.onset_sec, int(e.string), int(e.fret)) for e in tab_events]
    sf_list.sort(key=lambda x: x[0])

    used = set()
    data = []

    for n in guitar_notes:
        best_match = None
        best_diff = float('inf')
        best_i = None

        for i, (onset, string, fret) in enumerate(sf_list):
            if i in used:
                continue
            diff = abs(n.start - onset)
            if diff < 0.2 and diff < best_diff:
                best_match = (string, fret)
                best_diff = diff
                best_i = i

        if best_match:
            string, fret = best_match
            used.add(best_i)
        else:
            string, fret = 1, 0

        data.append({
            "time": float(n.start),
            "duration": float(n.end - n.start),
            "value": {"pitch": int(n.pitch), "string": int(string), "fret": int(fret), "techniques": []},
            "confidence": float(n.velocity / 127),
        })

    jams_dict = {
        "annotations": [{
            "namespace": "tab_note",
            "data": data,
            "annotation_metadata": {
                "curator": {}, "annotator": {}, "version": "",
                "corpus": "", "annotation_tools": "",
                "annotation_rules": "", "validation": "", "data_source": ""
            },
            "sandbox": {}
        }],
        "file_metadata": {
            "duration": float(pm.get_end_time()),
            "identifiers": {},
            "jams_version": "0.3.4"
        },
        "sandbox": {}
    }

    with open(out_path, 'w') as f:
        json.dump(jams_dict, f, indent=2)


def process_directory(inference: FretT5Inference, midi_dir: Path, output_dir: Path):
    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        print(f"  No .mid files found in {midi_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0

    for i, midi_path in enumerate(midi_files, 1):
        print(f"  [{i}/{len(midi_files)}] {midi_path.name}", end=" ... ", flush=True)
        try:
            notes = load_midi_notes(midi_path)
            if not notes:
                print("skip (no notes)")
                fail += 1
                continue

            tab_events = inference.predict_with_timing(notes, capo=0, tuning=STANDARD_TUNING)
            out_path = output_dir / f"{midi_path.stem}.jams"
            save_jams_raw(midi_path, tab_events, out_path)
            print("✓")
            ok += 1
        except Exception as e:
            print(f"✗ ({e})")
            fail += 1

    print(f"  Done: {ok} ok, {fail} failed")


def main():
    parser = argparse.ArgumentParser(description="Batch Stage 3 on existing MIDI files")
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 3 checkpoint (.pt)")
    parser.add_argument("--midi-dirs", nargs="+", required=True, help="Dirs containing .mid files")
    parser.add_argument("--output-base", required=True, help="Base output dir for JAMS files")
    args = parser.parse_args()

    tokenizer_path = str(SCRIPT_DIR / "t5_fretting_transformer" / "universal_tokenizer")

    print(f"Loading Stage 3 model from: {args.checkpoint}")
    inference = FretT5Inference(
        checkpoint_path=args.checkpoint,
        tokenizer_path=tokenizer_path,
    )
    print("Model loaded.\n")

    output_base = Path(args.output_base)

    for midi_dir_str in args.midi_dirs:
        midi_dir = Path(midi_dir_str)
        output_dir = output_base / midi_dir.name
        print(f"Processing: {midi_dir.name} -> {output_dir}")
        process_directory(inference, midi_dir, output_dir)
        print()

    print("All done.")


if __name__ == "__main__":
    main()