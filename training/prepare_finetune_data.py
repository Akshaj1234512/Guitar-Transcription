#!/usr/bin/env python3
"""
Prepare v3 training data for audio-conditioned T5 finetuning.
Combines GAPS + GOAT + GuitarTechs (matching Stage 1 training minus Leduc).

Creates TWO datasets:
1. GT-MIDI (finetune_gt_v3.jsonl): uses ground-truth pitch+timing from annotations
2. Noisy-MIDI (finetune_noisy_v3.jsonl): uses Stage 1 output aligned with GT labels
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pretty_midi

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "data_loaders"))

from gaps_loader import load_gaps_dataset
from goat_loader import load_goat_dataset
from guitartechs_loader import load_guitartechs_dataset


def load_midi_notes(midi_path: str) -> List[Dict]:
    """Load notes from a Stage 1 MIDI file."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({
                'pitch': int(n.pitch),
                'start': float(n.start),
                'duration': float(n.end - n.start),
            })
    return sorted(notes, key=lambda x: (x['start'], x['pitch']))


def align_stage1_to_gt(stage1_notes: List[Dict], gt_notes: List[Dict],
                       onset_tol: float = 0.05, pitch_tol: int = 1) -> List[Dict]:
    """Align Stage 1 MIDI notes with GT notes by onset + pitch matching.

    For each GT note, find the closest Stage 1 note within tolerance,
    and produce an aligned training example using Stage 1 pitch/timing and GT string/fret.
    """
    used_s1 = set()
    aligned = []

    for gt in gt_notes:
        best_idx = None
        best_score = float('inf')

        for i, s1 in enumerate(stage1_notes):
            if i in used_s1:
                continue
            dt = abs(s1['start'] - gt['start'])
            dp = abs(s1['pitch'] - gt['pitch'])
            if dt > onset_tol or dp > pitch_tol:
                continue
            score = dt + 0.01 * dp
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            used_s1.add(best_idx)
            s1 = stage1_notes[best_idx]
            aligned.append({
                'pitch': s1['pitch'],
                'start': s1['start'],
                'duration': s1['duration'],
                'string': int(gt['string']),
                'fret': int(gt['fret']),
            })

    aligned.sort(key=lambda x: (x['start'], x['pitch']))
    return aligned


def build_gt_dataset(entries: List[Dict], name: str) -> List[Dict]:
    examples = []
    for e in entries:
        notes = e['notes']
        if len(notes) < 3:
            continue
        examples.append({
            'audio_path': e['audio_path'],
            'source': name,
            'notes': notes,
        })
    return examples


def build_noisy_dataset(entries: List[Dict], name: str, stage1_mapper) -> List[Dict]:
    examples = []
    skipped = 0

    for e in entries:
        audio_path = Path(e['audio_path'])
        gt_notes = e['notes']
        if len(gt_notes) < 3:
            skipped += 1
            continue

        stage1_path = stage1_mapper(audio_path)
        if stage1_path is None or not stage1_path.exists():
            skipped += 1
            continue

        try:
            stage1_notes = load_midi_notes(str(stage1_path))
        except Exception:
            skipped += 1
            continue

        aligned = align_stage1_to_gt(stage1_notes, gt_notes)
        if len(aligned) < 3:
            skipped += 1
            continue

        examples.append({
            'audio_path': str(audio_path),
            'source': name,
            'notes': aligned,
            'original_gt_count': len(gt_notes),
            'stage1_count': len(stage1_notes),
            'aligned_count': len(aligned),
        })

    total_aligned = sum(e['aligned_count'] for e in examples)
    total_gt = sum(e['original_gt_count'] for e in examples)
    if total_gt > 0:
        print(f"  {name} noisy: {len(examples)} examples (skipped {skipped}) — "
              f"aligned {total_aligned}/{total_gt} ({total_aligned/total_gt*100:.1f}%)")
    return examples


def main():
    out_dir = SCRIPT_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    print("Loading datasets...")
    gaps_data = load_gaps_dataset('/data/akshaj/MusicAI/gaps_v1')
    goat_data = load_goat_dataset('/data/shamakg/datasets/GOAT/data')
    guitartechs_data = load_guitartechs_dataset()

    # ── Build GT-MIDI datasets ────────────────────────────────────────
    print("\nBuilding GT-MIDI dataset...")
    gaps_gt = build_gt_dataset(gaps_data, 'gaps')
    goat_gt = build_gt_dataset(goat_data, 'goat')
    guitartechs_gt = build_gt_dataset(guitartechs_data, 'guitartechs')
    gt_examples = gaps_gt + goat_gt + guitartechs_gt
    print(f"GT-MIDI total: {len(gt_examples)} examples "
          f"(gaps={len(gaps_gt)}, goat={len(goat_gt)}, guitartechs={len(guitartechs_gt)})")

    with open(out_dir / 'finetune_gt_v3.jsonl', 'w') as f:
        for ex in gt_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved: {out_dir / 'finetune_gt_v3.jsonl'}")

    # ── Build Noisy-MIDI datasets ─────────────────────────────────────
    print("\nBuilding Noisy-MIDI dataset (Stage 1 output + GT string/fret)...")

    def gaps_mapper(audio_path):
        return SCRIPT_DIR / 'results' / 'stage1_train_gaps' / 'audio' / f'{audio_path.stem}.mid'

    def goat_mapper(audio_path):
        return SCRIPT_DIR / 'results' / 'stage1_train_goat' / 'goat_flat_audio' / f'{audio_path.stem}.mid'

    gaps_noisy = build_noisy_dataset(gaps_data, 'gaps', gaps_mapper)
    goat_noisy = build_noisy_dataset(goat_data, 'goat', goat_mapper)

    # GuitarTechs: use unique_stem for matching Stage 1 output
    stage1_gt_dir = SCRIPT_DIR / 'results' / 'stage1_train_guitartechs' / 'guitartechs_flat_audio'
    gt_noisy_examples = []
    gt_skipped = 0
    for e in guitartechs_data:
        if len(e['notes']) < 3:
            gt_skipped += 1
            continue
        stage1_path = stage1_gt_dir / f"{e['unique_stem']}.mid"
        if not stage1_path.exists():
            gt_skipped += 1
            continue
        try:
            stage1_notes = load_midi_notes(str(stage1_path))
        except Exception:
            gt_skipped += 1
            continue
        aligned = align_stage1_to_gt(stage1_notes, e['notes'])
        if len(aligned) < 3:
            gt_skipped += 1
            continue
        gt_noisy_examples.append({
            'audio_path': e['audio_path'],
            'source': 'guitartechs',
            'notes': aligned,
            'original_gt_count': len(e['notes']),
            'stage1_count': len(stage1_notes),
            'aligned_count': len(aligned),
        })
    print(f"  guitartechs noisy: {len(gt_noisy_examples)} examples (skipped {gt_skipped})")
    guitartechs_noisy = gt_noisy_examples

    noisy_examples = gaps_noisy + goat_noisy + guitartechs_noisy
    print(f"Noisy-MIDI total: {len(noisy_examples)} examples")

    with open(out_dir / 'finetune_noisy_v3.jsonl', 'w') as f:
        for ex in noisy_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved: {out_dir / 'finetune_noisy_v3.jsonl'}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GT-MIDI examples:    {len(gt_examples)}")
    print(f"Noisy-MIDI examples: {len(noisy_examples)}")
    total_gt_notes = sum(len(e['notes']) for e in gt_examples)
    total_noisy_notes = sum(len(e['notes']) for e in noisy_examples)
    print(f"GT-MIDI total notes:    {total_gt_notes}")
    print(f"Noisy-MIDI total notes: {total_noisy_notes} "
          f"({total_noisy_notes/total_gt_notes*100:.1f}% of GT)" if total_gt_notes else "")


if __name__ == '__main__':
    main()
