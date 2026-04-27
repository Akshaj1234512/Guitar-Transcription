#!/usr/bin/env python3
"""
Guitar-TECHS data loader.

The dataset has per-string MIDI tracks named 'e' (high E, string 1),
'B' (string 2), 'G' (string 3), 'D' (string 4), 'A' (string 5),
and 'E' (low E, string 6). We read the MIDI, assign each note to its
track's string number, and match to the directinput audio file.

Structure:
  P{1,2,3}_{chords,scales,singlenotes,techniques,music}/
    midi/midi_{name}.mid          (per-string tracks)
    audio/directinput/directinput_{name}.wav
"""

from pathlib import Path
from typing import Dict, List

import mido

# String name → number mapping (track name → string 1-6)
STRING_NAMES = {'e': 1, 'B': 2, 'G': 3, 'D': 4, 'A': 5, 'E': 6}
STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}


def parse_guitartechs_midi(midi_path: str) -> List[Dict]:
    """Parse a GuitarTechs MIDI file with per-string tracks.

    Returns list of dicts with keys: pitch, start, duration, string, fret
    """
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    # Get tempo (default 120 BPM)
    tempo = 500000  # microseconds per quarter note (= 120 BPM)
    for t in mid.tracks:
        for msg in t:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo != 500000:
            break

    notes = []
    for track in mid.tracks:
        if track.name not in STRING_NAMES:
            continue
        string_num = STRING_NAMES[track.name]
        open_pitch = STANDARD_TUNING[string_num]

        current_tick = 0
        active_notes = {}  # pitch -> (start_tick, velocity)

        for msg in track:
            current_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_tick, _ = active_notes.pop(msg.note)
                    start_sec = mido.tick2second(start_tick, ticks_per_beat, tempo)
                    end_sec = mido.tick2second(current_tick, ticks_per_beat, tempo)
                    duration = max(end_sec - start_sec, 0.05)
                    fret = msg.note - open_pitch

                    if 0 <= fret <= 24:
                        notes.append({
                            'pitch': int(msg.note),
                            'start': float(start_sec),
                            'duration': float(duration),
                            'string': int(string_num),
                            'fret': int(fret),
                        })

    notes.sort(key=lambda x: (x['start'], x['pitch']))
    return notes


def load_guitartechs_dataset(root: str = './data/guitar-transcription-raw/guitar-techs',
                              flat_audio_dir: str = None) -> List[Dict]:
    """Load GuitarTechs dataset with string/fret annotations.

    Uses directinput audio (clean electric guitar DI recordings).

    Each returned example has:
        audio_path: path to the source directinput .wav file
        unique_stem: {subset}__{stem} (used for matching Stage 1 MIDI output)
    """
    root = Path(root)
    dataset = []
    skipped = 0

    for subset in sorted(root.iterdir()):
        if not subset.is_dir() or not subset.name.startswith('P'):
            continue

        midi_dir = subset / 'midi'
        audio_dir = subset / 'audio' / 'directinput'
        if not midi_dir.exists() or not audio_dir.exists():
            continue

        for midi_path in sorted(midi_dir.glob('midi_*.mid')):
            name = midi_path.stem.replace('midi_', '')
            audio_path = audio_dir / f'directinput_{name}.wav'
            if not audio_path.exists():
                skipped += 1
                continue

            try:
                notes = parse_guitartechs_midi(str(midi_path))
            except Exception:
                skipped += 1
                continue

            if len(notes) < 3:
                skipped += 1
                continue

            unique_stem = f'{subset.name}__directinput_{name}'
            dataset.append({
                'audio_path': str(audio_path),
                'unique_stem': unique_stem,
                'notes': notes,
                'source': f'guitartechs_{subset.name}',
                'item': name,
            })

    print(f"GuitarTechs: loaded {len(dataset)} files, skipped {skipped}")
    total_notes = sum(len(d['notes']) for d in dataset)
    print(f"  Total notes: {total_notes}")
    return dataset


if __name__ == '__main__':
    dataset = load_guitartechs_dataset()
    for d in dataset[:3]:
        print(f"\n{d['source']}/{d['item']}: {len(d['notes'])} notes")
        for n in d['notes'][:5]:
            print(f"  t={n['start']:.3f} p={n['pitch']} s={n['string']} f={n['fret']}")
