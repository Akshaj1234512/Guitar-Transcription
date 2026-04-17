#!/usr/bin/env python3
"""
GAPS (Guitar-Aligned Performance Scores) data loader.

Extracts (audio, MIDI notes, string/fret annotations) triplets from GAPS v1.
Audio files: gaps_v1/audio/NNN_<hash>.wav
MusicXML files: gaps_v1/musicxml/<hash>.xml  (contain string/fret in <technical>)
MIDI files: gaps_v1/midi/NNN_<hash>.mid      (aligned note onsets/offsets)
Syncpoints: gaps_v1/syncpoints/NNN_<hash>.csv (audio-score alignment)

The loader aligns MIDI note timings with MusicXML string/fret annotations
to produce training examples: (audio_path, [(onset, duration, pitch, string, fret), ...])
"""

import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi


# Standard guitar tuning: string -> open MIDI pitch
STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

# MusicXML pitch name -> semitone offset from C
PITCH_MAP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def musicxml_pitch_to_midi(step: str, octave: int, alter: int = 0) -> int:
    """Convert MusicXML pitch (step, octave, alter) to MIDI note number."""
    return (octave + 1) * 12 + PITCH_MAP[step] + alter


def parse_musicxml_notes(xml_path: str) -> List[Dict]:
    """Extract notes with string/fret from a GAPS MusicXML file.

    Returns list of dicts with keys:
        midi_pitch, string, fret, duration_divisions, voice, measure
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Detect namespace
    ns = ''
    if '}' in root.tag:
        ns = root.tag.split('}')[0] + '}'

    # Get divisions (ticks per quarter note) from first measure
    divisions = 1
    for attr in root.iter(f'{ns}attributes'):
        div_elem = attr.find(f'{ns}divisions')
        if div_elem is not None and div_elem.text:
            divisions = int(div_elem.text)
            break

    notes = []
    measure_num = 0
    cumulative_offset = 0  # Track position in divisions from start

    for measure in root.iter(f'{ns}measure'):
        measure_num += 1
        position_in_measure = 0

        for elem in measure:
            tag = elem.tag.replace(ns, '')

            if tag == 'forward':
                dur = elem.find(f'{ns}duration')
                if dur is not None and dur.text:
                    position_in_measure += int(dur.text)

            elif tag == 'backup':
                dur = elem.find(f'{ns}duration')
                if dur is not None and dur.text:
                    position_in_measure -= int(dur.text)

            elif tag == 'note':
                # Check for rest or chord
                is_rest = elem.find(f'{ns}rest') is not None
                is_chord = elem.find(f'{ns}chord') is not None

                if is_rest:
                    dur = elem.find(f'{ns}duration')
                    if dur is not None and dur.text:
                        position_in_measure += int(dur.text)
                    continue

                # Extract pitch
                pitch_elem = elem.find(f'{ns}pitch')
                if pitch_elem is None:
                    continue

                step = pitch_elem.find(f'{ns}step')
                octave = pitch_elem.find(f'{ns}octave')
                alter = pitch_elem.find(f'{ns}alter')

                if step is None or octave is None:
                    continue

                midi_pitch = musicxml_pitch_to_midi(
                    step.text, int(octave.text),
                    int(alter.text) if alter is not None and alter.text else 0
                )

                # Extract duration
                dur_elem = elem.find(f'{ns}duration')
                duration = int(dur_elem.text) if dur_elem is not None and dur_elem.text else 1

                # Extract string/fret from <technical>
                string_num, fret_num = None, None
                for tech in elem.iter(f'{ns}technical'):
                    s = tech.find(f'{ns}string')
                    f = tech.find(f'{ns}fret')
                    if s is not None and s.text:
                        string_num = int(s.text)
                    if f is not None and f.text:
                        fret_num = int(f.text)

                if string_num is not None and fret_num is not None:
                    # If chord, don't advance position
                    offset = cumulative_offset + position_in_measure

                    notes.append({
                        'midi_pitch': midi_pitch,
                        'string': string_num,
                        'fret': fret_num,
                        'duration_divisions': duration,
                        'offset_divisions': offset,
                        'divisions': divisions,
                        'measure': measure_num,
                    })

                if not is_chord:
                    position_in_measure += duration

        cumulative_offset += position_in_measure

    return notes


def load_gaps_midi_notes(midi_path: str) -> List[Dict]:
    """Load note events from a GAPS MIDI file.

    Returns list of dicts with keys: pitch, start, duration, end
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'duration': note.end - note.start,
                    'end': note.end,
                })
    notes.sort(key=lambda x: (x['start'], x['pitch']))
    return notes


def align_xml_to_midi(xml_notes: List[Dict], midi_notes: List[Dict]) -> List[Dict]:
    """Align MusicXML notes (with string/fret) to MIDI notes (with timing).

    Uses pitch matching with sequential ordering to pair XML and MIDI notes.
    Returns aligned notes with both timing and string/fret information.
    """
    aligned = []
    midi_idx = 0

    for xml_note in xml_notes:
        target_pitch = xml_note['midi_pitch']

        # Find the next MIDI note with matching pitch
        best_idx = None
        best_dist = float('inf')

        for j in range(midi_idx, min(midi_idx + 20, len(midi_notes))):
            if midi_notes[j]['pitch'] == target_pitch:
                dist = j - midi_idx
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
                    break  # Take the first match

        if best_idx is not None:
            mn = midi_notes[best_idx]
            aligned.append({
                'pitch': mn['pitch'],
                'start': mn['start'],
                'duration': mn['duration'],
                'string': xml_note['string'],
                'fret': xml_note['fret'],
            })
            midi_idx = best_idx + 1

    return aligned


def build_gaps_file_mapping(gaps_dir: str) -> List[Dict]:
    """Build mapping between audio, MIDI, and MusicXML files in GAPS.

    Returns list of dicts with keys: audio_path, midi_path, xml_path, hash
    """
    gaps_dir = Path(gaps_dir)
    audio_dir = gaps_dir / 'audio'
    midi_dir = gaps_dir / 'midi'
    xml_dir = gaps_dir / 'musicxml'

    # Build hash -> audio/midi mapping
    audio_map = {}
    for f in audio_dir.glob('*.wav'):
        parts = f.stem.split('_', 1)
        if len(parts) == 2:
            audio_map[parts[1]] = f

    midi_map = {}
    for f in midi_dir.glob('*.mid'):
        parts = f.stem.split('_', 1)
        if len(parts) == 2:
            midi_map[parts[1]] = f

    # Build file triplets
    entries = []
    for hash_id, audio_path in audio_map.items():
        xml_path = xml_dir / f'{hash_id}.xml'
        midi_path = midi_map.get(hash_id)

        if xml_path.exists() and midi_path is not None:
            entries.append({
                'audio_path': str(audio_path),
                'midi_path': str(midi_path),
                'xml_path': str(xml_path),
                'hash': hash_id,
            })

    return sorted(entries, key=lambda x: x['hash'])


def load_gaps_dataset(gaps_dir: str) -> List[Dict]:
    """Load the complete GAPS dataset with aligned audio + tablature.

    Returns list of dicts with keys:
        audio_path, notes: [(pitch, start, duration, string, fret), ...]
    """
    file_mapping = build_gaps_file_mapping(gaps_dir)
    print(f"Found {len(file_mapping)} GAPS files with audio + XML + MIDI")

    dataset = []
    skipped = 0

    for entry in file_mapping:
        try:
            xml_notes = parse_musicxml_notes(entry['xml_path'])
            midi_notes = load_gaps_midi_notes(entry['midi_path'])

            if len(xml_notes) < 5 or len(midi_notes) < 5:
                skipped += 1
                continue

            aligned = align_xml_to_midi(xml_notes, midi_notes)

            if len(aligned) < 5:
                skipped += 1
                continue

            dataset.append({
                'audio_path': entry['audio_path'],
                'midi_path': entry['midi_path'],
                'hash': entry['hash'],
                'notes': aligned,
            })
        except Exception as e:
            skipped += 1

    print(f"Loaded {len(dataset)} files, skipped {skipped}")
    total_notes = sum(len(d['notes']) for d in dataset)
    print(f"Total aligned notes: {total_notes}")

    return dataset


if __name__ == '__main__':
    dataset = load_gaps_dataset('/data/akshaj/MusicAI/gaps_v1')
    # Print some examples
    for d in dataset[:3]:
        print(f"\n{d['hash']}: {len(d['notes'])} notes")
        for n in d['notes'][:5]:
            print(f"  pitch={n['pitch']} start={n['start']:.3f}s string={n['string']} fret={n['fret']}")
