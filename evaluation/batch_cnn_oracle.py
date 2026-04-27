#!/usr/bin/env python3
"""Oracle CNN-alone inference: given GT note onsets+pitches, predict strings from audio."""

import argparse, json, sys
from pathlib import Path
from typing import List

import librosa, numpy as np, torch

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "t5_fretting_transformer" / "src"))

from fret_t5.audio_features import AudioFeatureConfig, StringClassifier, AudioFeatureExtractor

TUNING_DICT = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}


def load_gt_notes_from_midi(midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            notes.append({'pitch': int(n.pitch), 'start': float(n.start), 'duration': float(n.end - n.start)})
    notes.sort(key=lambda x: (round(x['start'], 4), x['pitch']))
    return notes


def load_gt_notes_from_jams(jams_path):
    with open(jams_path) as f:
        data = json.load(f)
    notes = []
    for ann in data.get('annotations', []):
        if ann.get('namespace') != 'note_midi': continue
        for n in ann.get('data', []):
            val = n.get('value')
            if val is None: continue
            notes.append({'pitch': int(round(float(val))), 'start': float(n.get('time', 0)), 'duration': float(n.get('duration', 0.1))})
    notes.sort(key=lambda x: (round(x['start'], 4), x['pitch']))
    return notes


def save_jams(notes, preds, out_path):
    data = []
    for n, s in zip(notes, preds):
        fret = n['pitch'] - TUNING_DICT[s]
        if fret < 0 or fret > 24:
            fret = 0
        data.append({
            "time": float(n['start']), "duration": float(n['duration']),
            "value": {"pitch": int(n['pitch']), "string": int(s), "fret": int(fret), "techniques": []},
            "confidence": 1.0,
        })
    dur = max((n['start'] + n['duration'] for n in notes), default=1.0)
    jams_dict = {
        "annotations": [{
            "namespace": "tab_note", "data": data,
            "annotation_metadata": {"curator":{},"annotator":{},"version":"","corpus":"oracle_cnn",
                                    "annotation_tools":"","annotation_rules":"","validation":"","data_source":""},
            "sandbox": {}
        }],
        "file_metadata": {"duration": float(dur), "identifiers": {}, "jams_version": "0.3.4"},
        "sandbox": {}
    }
    with open(out_path, 'w') as f:
        json.dump(jams_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['guitarset', 'egdb'], required=True)
    parser.add_argument('--audio-dirs', nargs='+', required=True)
    parser.add_argument('--suffixes', nargs='+', default=None)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--output-base', required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading CNN: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ck.get('config') or AudioFeatureConfig(embedding_dim=256)
    model = StringClassifier(config, num_strings=6)
    model.load_state_dict(ck['model_state_dict'])
    model.to(device).eval()

    extractor = AudioFeatureExtractor(config)

    gt_dir = Path(args.gt_dir)
    output_base = Path(args.output_base)
    suffixes = args.suffixes or [''] * len(args.audio_dirs)
    if len(suffixes) == 1 and len(args.audio_dirs) > 1:
        suffixes = suffixes * len(args.audio_dirs)

    for audio_dir_str, suffix in zip(args.audio_dirs, suffixes):
        audio_dir = Path(audio_dir_str)
        out_dir = output_base / audio_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_files = sorted(audio_dir.glob('*.wav'))
        print(f"\n[{audio_dir.name}] {len(audio_files)} files")

        ok, fail = 0, 0
        for i, audio_path in enumerate(audio_files):
            stem = audio_path.stem
            if args.dataset == 'guitarset':
                gt_stem = stem[:-len(suffix)] if suffix and stem.endswith(suffix) else stem
                gt_path = gt_dir / f'{gt_stem}.mid'
                loader = load_gt_notes_from_midi
            else:
                gt_path = gt_dir / f'{stem}.jams'
                loader = load_gt_notes_from_jams

            if not gt_path.exists():
                fail += 1
                continue

            try:
                notes = loader(gt_path)
                if not notes:
                    fail += 1
                    continue
                audio, sr = librosa.load(str(audio_path), sr=config.sample_rate)
                mel_batch, pitch_batch = extractor.extract_batch_mels(audio, sr, notes)
                mel_batch = mel_batch.to(device)
                pitch_batch = pitch_batch.to(device)
                with torch.no_grad():
                    preds = model.predict_strings(mel_batch, pitch_batch).cpu().numpy()
                save_jams(notes, preds, out_dir / f'{stem}.jams')
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 3:
                    print(f"  err {audio_path.name}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(audio_files)}] ok={ok} fail={fail}")

        print(f"  Done: {ok} ok, {fail} failed")

    print("\nAll done.")


if __name__ == '__main__':
    main()
