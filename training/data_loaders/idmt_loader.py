#!/usr/bin/env python3
"""
IDMT-SMT-Guitar data loader.

Extracts (audio_path, notes with string/fret) from IDMT-SMT-Guitar V2.
Each XML annotation contains note events with pitch, onset, offset,
fretNumber, stringNumber, excitationStyle, expressionStyle.

Supports dataset1 (single notes, chords), dataset2 (licks/exercises),
and dataset3 (short pieces).
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


def parse_idmt_xml(xml_path: str) -> Dict:
    """Parse an IDMT-SMT-Guitar annotation XML file.

    Returns dict with:
        audio_filename: str
        instrument: str
        tuning: optional tuple of MIDI pitches
        notes: list of dicts with pitch, start, duration, string, fret, technique
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Global parameters
    global_params = root.find('globalParameter')
    audio_filename = ''
    instrument = ''
    tuning = None

    if global_params is not None:
        af = global_params.find('audioFileName')
        if af is not None and af.text:
            audio_filename = af.text.strip().lstrip('\\').lstrip('/')

        inst = global_params.find('instrument')
        if inst is not None and inst.text:
            instrument = inst.text.strip()

        tun = global_params.find('instrumentTuning')
        if tun is not None and tun.text:
            try:
                tuning = tuple(int(x) for x in tun.text.strip().split())
            except ValueError:
                pass

    # Parse note events
    notes = []
    for event in root.findall('.//event'):
        pitch_elem = event.find('pitch')
        onset_elem = event.find('onsetSec')
        offset_elem = event.find('offsetSec')
        string_elem = event.find('stringNumber')
        fret_elem = event.find('fretNumber')
        excitation = event.find('excitationStyle')
        expression = event.find('expressionStyle')

        if None in (pitch_elem, onset_elem, offset_elem, string_elem, fret_elem):
            continue

        try:
            pitch = int(pitch_elem.text)
            onset = float(onset_elem.text)
            offset = float(offset_elem.text)
            string_num = int(string_elem.text)
            fret_num = int(fret_elem.text)
        except (ValueError, TypeError):
            continue

        # IDMT uses string 1=low E, 6=high E (opposite of standard guitar convention)
        # Convert to standard: 1=high E, 6=low E
        string_standard = 7 - string_num

        notes.append({
            'pitch': pitch,
            'start': onset,
            'duration': max(offset - onset, 0.01),
            'string': string_standard,
            'fret': fret_num,
            'excitation': excitation.text if excitation is not None else 'NO',
            'expression': expression.text if expression is not None else 'NO',
        })

    notes.sort(key=lambda x: (x['start'], x['pitch']))

    return {
        'audio_filename': audio_filename,
        'instrument': instrument,
        'tuning': tuning,
        'notes': notes,
    }


def find_audio_for_annotation(xml_path: Path, audio_filename: str) -> Optional[str]:
    """Find the audio file corresponding to an annotation.

    Searches in the same directory and parent directories.
    """
    # Try same directory
    audio_path = xml_path.parent / audio_filename
    if audio_path.exists():
        return str(audio_path)

    # Try sibling 'audio' directory
    audio_dir = xml_path.parent.parent / 'audio'
    if audio_dir.exists():
        audio_path = audio_dir / audio_filename
        if audio_path.exists():
            return str(audio_path)

    # Try parent directory
    audio_path = xml_path.parent.parent / audio_filename
    if audio_path.exists():
        return str(audio_path)

    # For dataset1, audio is in the same guitar directory
    # XML: .../annotation/file.xml, WAV: .../file.wav
    wav_name = xml_path.stem + '.wav'
    for search_dir in [xml_path.parent, xml_path.parent.parent]:
        candidate = search_dir / wav_name
        if candidate.exists():
            return str(candidate)

    return None


def load_idmt_dataset(idmt_root: str) -> List[Dict]:
    """Load the complete IDMT-SMT-Guitar dataset.

    Returns list of dicts with:
        audio_path, notes, instrument, source_dataset
    """
    idmt_root = Path(idmt_root)
    dataset = []
    skipped = 0

    for xml_path in sorted(idmt_root.rglob('*.xml')):
        try:
            parsed = parse_idmt_xml(str(xml_path))

            if len(parsed['notes']) < 1:
                skipped += 1
                continue

            # Find corresponding audio
            audio_path = find_audio_for_annotation(xml_path, parsed['audio_filename'])
            if audio_path is None:
                # Try matching by XML stem
                audio_path = find_audio_for_annotation(xml_path, xml_path.stem + '.wav')

            if audio_path is None:
                skipped += 1
                continue

            # Determine source dataset
            rel_path = str(xml_path.relative_to(idmt_root))
            if 'dataset1' in rel_path:
                source = 'dataset1'
            elif 'dataset2' in rel_path:
                source = 'dataset2'
            elif 'dataset3' in rel_path:
                source = 'dataset3'
            elif 'dataset4' in rel_path:
                source = 'dataset4'
            else:
                source = 'unknown'

            dataset.append({
                'audio_path': audio_path,
                'notes': parsed['notes'],
                'instrument': parsed['instrument'],
                'tuning': parsed['tuning'],
                'source_dataset': source,
                'xml_path': str(xml_path),
            })
        except Exception as e:
            skipped += 1

    print(f"Loaded {len(dataset)} files, skipped {skipped}")
    total_notes = sum(len(d['notes']) for d in dataset)
    print(f"Total notes with string+fret: {total_notes}")

    # Summary by dataset
    from collections import Counter
    sources = Counter(d['source_dataset'] for d in dataset)
    for src, count in sorted(sources.items()):
        notes_in_src = sum(len(d['notes']) for d in dataset if d['source_dataset'] == src)
        print(f"  {src}: {count} files, {notes_in_src} notes")

    return dataset


if __name__ == '__main__':
    dataset = load_idmt_dataset('/data/akshaj/MusicAI/IDMT-SMT-Guitar/IDMT-SMT-GUITAR_V2')
    for d in dataset[:3]:
        print(f"\n{Path(d['audio_path']).name}: {d['instrument']}, {len(d['notes'])} notes")
        for n in d['notes'][:5]:
            print(f"  pitch={n['pitch']} start={n['start']:.3f}s string={n['string']} fret={n['fret']} expr={n['expression']}")
