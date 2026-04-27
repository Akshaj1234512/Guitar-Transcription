#!/usr/bin/env python3
"""
Evaluate the trained audio CNN string classifier zero-shot on GuitarSet and EGDB.

The classifier takes a mel spectrogram patch + pitch and predicts which of 6
guitar strings the note was played on. We use ground-truth note onsets/durations
and string labels to measure classification accuracy.

This isolates the audio feature extractor's ability to disambiguate strings
from timbral cues alone, without any language modeling or context.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "t5_fretting_transformer" / "src"))

from fret_t5.audio_features import AudioFeatureConfig, StringClassifier


def load_guitarset_notes(jams_path: Path) -> List[Dict]:
    """Extract notes with GT string labels from GuitarSet JAMS."""
    with open(jams_path) as f:
        data = json.load(f)

    notes = []
    for ann in data.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue
        ds = ann.get("annotation_metadata", {}).get("data_source")
        if ds is None or str(ds) not in "012345":
            continue
        # GuitarSet: data_source 0=Low E (string 6), 5=High E (string 1)
        string_num = 6 - int(ds)

        for note in ann.get("data", []):
            pitch = note.get("value")
            if pitch is None:
                continue
            notes.append({
                'pitch': int(round(float(pitch))),
                'start': float(note.get("time", 0)),
                'duration': float(note.get("duration", 0.1)),
                'string': string_num,
            })

    return sorted(notes, key=lambda x: x['start'])


def load_egdb_notes(jams_path: Path) -> List[Dict]:
    """Extract notes with GT string labels from EGDB JAMS (same format as GuitarSet)."""
    return load_guitarset_notes(jams_path)


def find_audio_for_gt(gt_path: Path, audio_base: Path, suffix_pattern: str = None) -> Path:
    """Find the audio file for a GT annotation file."""
    # GuitarSet: annotation is XXXX.jams, audio is XXXX_mix.wav etc.
    # EGDB: annotation is NNN.jams, audio is NNN.wav
    stem = gt_path.stem
    for ext in ['.wav', '.mp3', '.flac']:
        # Try exact match
        candidate = audio_base / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        # Try with suffix
        if suffix_pattern:
            candidate = audio_base / f"{stem}{suffix_pattern}{ext}"
            if candidate.exists():
                return candidate
    return None


def evaluate_classifier(
    model: StringClassifier,
    config: AudioFeatureConfig,
    gt_files: List[Path],
    audio_base: Path,
    suffix: str = "",
    device: str = "cuda",
    dataset_name: str = "",
    max_files: int = 0,
) -> Dict:
    """Evaluate string classifier on a dataset.

    Returns accuracy broken down by:
    - Overall accuracy
    - Per-string accuracy (1-6)
    - Per-pitch-range accuracy
    """
    model.eval()

    total, correct = 0, 0
    per_string_correct = {s: 0 for s in range(1, 7)}
    per_string_total = {s: 0 for s in range(1, 7)}

    processed_files = 0
    if max_files > 0:
        gt_files = gt_files[:max_files]

    # Extractor for mel segment extraction
    from fret_t5.audio_features import AudioFeatureExtractor
    extractor = AudioFeatureExtractor(config)

    for i, gt_path in enumerate(gt_files):
        # Find matching audio
        audio_path = None
        if dataset_name == "guitarset":
            # GuitarSet audio: {stem}{suffix}.wav where suffix is _hex_cln, _hex, _mic, _mix
            for ext in ['.wav']:
                candidate = audio_base / f"{gt_path.stem}{suffix}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break
        else:  # EGDB
            for ext in ['.wav']:
                candidate = audio_base / f"{gt_path.stem}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break

        if audio_path is None:
            continue

        # Load audio
        try:
            audio, sr = librosa.load(str(audio_path), sr=config.sample_rate)
        except Exception:
            continue

        # Load GT notes
        if dataset_name == "guitarset":
            notes = load_guitarset_notes(gt_path)
        else:
            notes = load_egdb_notes(gt_path)

        if not notes:
            continue

        # Extract mel features for all notes
        try:
            mel_batch, pitch_batch = extractor.extract_batch_mels(audio, sr, notes)
        except Exception:
            continue

        mel_batch = mel_batch.to(device)
        pitch_batch = pitch_batch.to(device)

        # Predict strings
        with torch.no_grad():
            predictions = model.predict_strings(mel_batch, pitch_batch)

        predictions = predictions.cpu().numpy()

        for j, note in enumerate(notes):
            gt_string = note['string']
            pred_string = int(predictions[j])
            if 1 <= gt_string <= 6:
                per_string_total[gt_string] += 1
                if pred_string == gt_string:
                    per_string_correct[gt_string] += 1
                    correct += 1
                total += 1

        processed_files += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(gt_files)}] {correct}/{total} = {correct/max(total,1)*100:.1f}%")

    accuracy = correct / max(total, 1) * 100
    per_string_acc = {
        s: per_string_correct[s] / max(per_string_total[s], 1) * 100
        for s in range(1, 7)
    }

    print(f"\n{dataset_name}{' (' + suffix + ')' if suffix else ''}:")
    print(f"  Files processed: {processed_files}")
    print(f"  Total notes: {total}")
    print(f"  Overall accuracy: {accuracy:.1f}%")
    print(f"  Per-string accuracy:")
    for s in range(1, 7):
        string_name = {1: 'high E', 2: 'B', 3: 'G', 4: 'D', 5: 'A', 6: 'low E'}[s]
        print(f"    String {s} ({string_name}): {per_string_acc[s]:.1f}% ({per_string_correct[s]}/{per_string_total[s]})")

    return {
        'accuracy': accuracy,
        'total': total,
        'per_string': per_string_acc,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_audio_conditioned/best_string_classifier.pt")
    parser.add_argument("--guitarset-gt", default=os.environ.get("GUITARSET_DIR", "./data/GuitarSet/annotation"))
    parser.add_argument("--egdb-gt", default=os.environ.get("EGDB_DIR", "./data/EGDB/annotation_jams"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    print(f"Loading classifier from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get('config') or AudioFeatureConfig(embedding_dim=256)

    model = StringClassifier(config, num_strings=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Best val acc from training: {checkpoint.get('best_val_acc', 'N/A')}%")

    # Evaluate on GuitarSet (all 4 audio conditions)
    print("\n" + "="*60)
    print("GUITARSET EVALUATION")
    print("="*60)

    guitarset_gt_dir = Path(args.guitarset_gt)
    gt_files = sorted(guitarset_gt_dir.glob("*.jams"))
    print(f"Found {len(gt_files)} GuitarSet GT files")

    gs_results = {}
    for audio_condition, suffix in [
        ("audio_hex_debleeded", "_hex_cln"),
        ("audio_hex_original", "_hex"),
        ("audio_mono-mic", "_mic"),
        ("audio_mix", "_mix"),
    ]:
        audio_base = Path(f"./data/GuitarSet/{audio_condition}")
        if not audio_base.exists():
            continue
        print(f"\n--- {audio_condition} ---")
        result = evaluate_classifier(
            model, config, gt_files, audio_base, suffix,
            device=device, dataset_name="guitarset"
        )
        gs_results[audio_condition] = result

    # Evaluate on EGDB
    print("\n" + "="*60)
    print("EGDB EVALUATION")
    print("="*60)

    egdb_gt_dir = Path(args.egdb_gt)
    egdb_files = sorted(egdb_gt_dir.glob("*.jams"))
    print(f"Found {len(egdb_files)} EGDB GT files")

    egdb_results = {}
    for amp in ["audio_DI", "audio_Ftwin", "audio_JCjazz", "audio_Marshall", "audio_Mesa", "audio_Plexi"]:
        audio_base = Path(f"./data/EGDB/{amp}")
        if not audio_base.exists():
            continue
        print(f"\n--- {amp} ---")
        result = evaluate_classifier(
            model, config, egdb_files, audio_base, "",
            device=device, dataset_name="egdb"
        )
        egdb_results[amp] = result

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nGuitarSet:")
    for cond, r in gs_results.items():
        print(f"  {cond}: {r['accuracy']:.1f}% ({r['total']} notes)")

    print("\nEGDB:")
    for amp, r in egdb_results.items():
        print(f"  {amp}: {r['accuracy']:.1f}% ({r['total']} notes)")

    if gs_results:
        gs_avg = np.mean([r['accuracy'] for r in gs_results.values()])
        print(f"\nGuitarSet average: {gs_avg:.1f}%")
    if egdb_results:
        egdb_avg = np.mean([r['accuracy'] for r in egdb_results.values()])
        print(f"EGDB average: {egdb_avg:.1f}%")


if __name__ == "__main__":
    main()
