#!/usr/bin/env python3
"""
GOAT (Guitar On Audio and Tablatures) data loader.

GOAT contains raw GuitarPro files with per-note string/fret annotations plus
audio recordings (direct-input and 5 amp variants). We parse the .gp5/.gp files
to extract (pitch, onset_sec, string, fret) tuples and match them to the
corresponding audio file.

Source: ./data/GOAT/data/item_N/
  - item_N.gp5 or item_N.gp (tablature)
  - item_N_amp_1.wav (audio, one of 5 amp variants)
  - item_N_fine_aligned.mid (audio-aligned MIDI, pitch only)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import guitarpro

STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
STANDARD_TUNING_TUPLE = (64, 59, 55, 50, 45, 40)


def parse_goat_gp(gp_path: str) -> Tuple[List[Dict], Tuple[int, ...]]:
    """Parse a GOAT GuitarPro file.

    Returns:
        (notes, tuning)
        notes: list of dicts with keys: pitch, start, duration, string, fret
        tuning: tuple of 6 open-string MIDI pitches (high E to low E)
    """
    song = guitarpro.parse(gp_path)
    if not song.tracks:
        return [], STANDARD_TUNING_TUPLE

    track = song.tracks[0]
    tuning = tuple(s.value for s in track.strings)

    tempo = song.tempo  # BPM
    sec_per_quarter = 60.0 / tempo

    notes = []
    measure_time = 0.0

    for measure in track.measures:
        # Use first voice only (most GOAT files are monophonic)
        if not measure.voices:
            continue
        voice = measure.voices[0]

        # Track position within measure
        beat_time = measure_time

        # Compute measure duration from time signature
        ts = measure.header.timeSignature
        measure_duration_sec = ts.numerator * (4.0 / ts.denominator.value) * sec_per_quarter

        # Iterate beats
        for beat in voice.beats:
            # beat.duration.value: 1=whole, 2=half, 4=quarter, 8=eighth, etc.
            dur_in_quarters = 4.0 / beat.duration.value
            if beat.duration.isDotted:
                dur_in_quarters *= 1.5
            tuplet = beat.duration.tuplet
            if tuplet and tuplet.enters > 0 and tuplet.times > 0:
                dur_in_quarters *= tuplet.times / tuplet.enters
            dur_sec = dur_in_quarters * sec_per_quarter

            for note in beat.notes:
                if note.type.name in ('rest',):
                    continue
                if not (1 <= note.string <= 6):
                    continue
                string_val = track.strings[note.string - 1].value
                pitch = string_val + note.value
                if not (20 <= pitch <= 108):
                    continue

                notes.append({
                    'pitch': int(pitch),
                    'start': float(beat_time),
                    'duration': float(max(dur_sec, 0.05)),
                    'string': int(note.string),
                    'fret': int(note.value),
                })

            beat_time += dur_sec

        measure_time += measure_duration_sec

    notes.sort(key=lambda x: (x['start'], x['pitch']))
    return notes, tuning


def transpose_to_standard(notes: List[Dict], tuning: Tuple[int, ...]) -> List[Dict]:
    """Re-derive (string, fret) assuming standard tuning.

    If the original tuning is non-standard, finds the best (string, fret) in
    standard tuning that produces the same pitch.
    """
    standard = [64, 59, 55, 50, 45, 40]
    transposed = []
    for n in notes:
        pitch = n['pitch']
        # Find valid (string, fret) in standard tuning
        best_string, best_fret = None, None
        for s in range(1, 7):
            open_pitch = standard[s - 1]
            fret = pitch - open_pitch
            if 0 <= fret <= 24:
                # Prefer closest to original string
                if best_string is None or abs(s - n['string']) < abs(best_string - n['string']):
                    best_string, best_fret = s, fret
        if best_string is not None:
            transposed.append({
                'pitch': pitch,
                'start': n['start'],
                'duration': n['duration'],
                'string': best_string,
                'fret': best_fret,
            })
    return transposed


def load_goat_dataset(goat_root: str = os.environ.get('GOAT_DIR', './data/GOAT/data'),
                      amp_variant: int = 1) -> List[Dict]:
    """Load GOAT dataset with string/fret annotations from GP files.

    Args:
        goat_root: Path to GOAT data directory (with item_N/ subdirs)
        amp_variant: Which amp audio to use (1-5), or 0 for DI-only

    Returns:
        List of dicts with: audio_path, notes (with pitch, start, duration, string, fret)
    """
    goat_root = Path(goat_root)
    items = sorted(goat_root.glob('item_*'))

    dataset = []
    skipped = 0
    non_standard = 0

    for item_dir in items:
        if not item_dir.is_dir():
            continue

        # Find GP file (.gp5 preferred, fallback to .gp)
        gp_path = None
        for ext in ['.gp5', '.gp']:
            candidate = item_dir / f'{item_dir.name}{ext}'
            if candidate.exists():
                gp_path = candidate
                break
        if gp_path is None:
            skipped += 1
            continue

        # Find audio file (amp variant)
        audio_path = item_dir / f'{item_dir.name}_amp_{amp_variant}.wav'
        if not audio_path.exists():
            audio_path = item_dir / f'{item_dir.name}.wav'
            if not audio_path.exists():
                skipped += 1
                continue

        try:
            notes, tuning = parse_goat_gp(str(gp_path))
        except Exception as e:
            skipped += 1
            continue

        if len(notes) < 3:
            skipped += 1
            continue

        # Handle non-standard tuning: re-derive string/fret for standard tuning
        if tuning != STANDARD_TUNING_TUPLE:
            notes = transpose_to_standard(notes, tuning)
            non_standard += 1
            if len(notes) < 3:
                skipped += 1
                continue

        dataset.append({
            'audio_path': str(audio_path),
            'notes': notes,
            'source': 'goat',
            'item': item_dir.name,
        })

    print(f"GOAT: loaded {len(dataset)} files, skipped {skipped}, {non_standard} non-standard tuning (re-derived)")
    total_notes = sum(len(d['notes']) for d in dataset)
    print(f"  Total notes: {total_notes}")
    return dataset


if __name__ == '__main__':
    dataset = load_goat_dataset()
    for d in dataset[:3]:
        print(f"\n{d['item']}: {len(d['notes'])} notes")
        for n in d['notes'][:5]:
            print(f"  t={n['start']:.3f} p={n['pitch']} s={n['string']} f={n['fret']}")
