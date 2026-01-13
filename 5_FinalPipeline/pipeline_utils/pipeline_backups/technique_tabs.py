import jams
import numpy as np
import os
import pretty_midi
from music21 import stream, note, tempo, meter, clef, articulations, expressions, spanner, interval, environment
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import tab_generation_utils.preprocess as preprocess
# TODO: figure out pdf export
# import lilypond


# # get lilyponds
# us = environment.UserSettings()
# us['lilypondPath'] = str(lilypond.executable())

# print(f"LilyPond path set to: {lilypond.executable()}")

STANDARD_TUNING = {
    6: 40,  # E2 (low E)
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64   # E4 (high e)
}

def midi_pitch_to_guitar_positions(midi_pitch, tuning=STANDARD_TUNING, max_fret=12):
    """
    Find all possible (string, fret) positions for a given MIDI pitch.
    
    Args:
        midi_pitch: MIDI note number (e.g., 60 = middle C)
        tuning: Dictionary of {string_number: open_string_midi_pitch}
        max_fret: Maximum fret to consider
    
    Returns:
        List of (string, fret) tuples, sorted by preference
    """
    positions = []
    
    for string_num in range(1, 7):  # Strings 1-6
        open_pitch = tuning[string_num]
        fret = midi_pitch - open_pitch
        
        # Valid if fret is 0-12 (or your max_fret)
        if 0 <= fret <= max_fret:
            positions.append((string_num, fret))
    
    # Sort by preference: middle strings first, lower frets preferred
    # This creates more natural fingering
    def position_score(pos):
        string, fret = pos
        # Prefer middle strings (3, 4) and lower frets
        string_penalty = abs(string - 3.5)  # Prefer strings 3-4
        fret_penalty = fret * 0.1  # Slight preference for lower frets
        return string_penalty + fret_penalty
    
    positions.sort(key=position_score)
    return positions

def choose_best_position(midi_pitch, previous_position=None, tuning=STANDARD_TUNING):
    """
    Choose the best string/fret position for a pitch.
    Takes into account the previous position to minimize hand movement.
    
    Args:
        midi_pitch: MIDI note number
        previous_position: Previous (string, fret) tuple, or None
        tuning: Guitar tuning
    
    Returns:
        (string, fret) tuple
    """
    positions = midi_pitch_to_guitar_positions(midi_pitch, tuning)
    
    if not positions:
        # Pitch out of range - use closest approximation
        # Find the closest string
        closest_string = min(tuning.keys(), 
                           key=lambda s: abs(tuning[s] - midi_pitch))
        fret = max(0, min(12, midi_pitch - tuning[closest_string]))
        return (closest_string, fret)
    
    if previous_position is None:
        # No previous position - use most natural position
        return positions[0]
    
    # Choose position closest to previous position (minimize hand movement)
    prev_string, prev_fret = previous_position
    
    def distance_score(pos):
        string, fret = pos
        string_distance = abs(string - prev_string)
        fret_distance = abs(fret - prev_fret)
        # Weight fret distance more (moving along frets is harder than changing strings)
        return fret_distance * 2 + string_distance
    
    return min(positions, key=distance_score)

def midi_to_jams_with_tablature(midi_path, tuning=STANDARD_TUNING):
    """
    Convert MIDI file to JAMS with intelligent tablature mapping.
    
    Args:
        midi_path: Path to MIDI file
        tuning: Guitar tuning dictionary
    
    Returns:
        JAMS object with both note_midi and tab_note annotations
    """
    # Load MIDI
    pm = pretty_midi.PrettyMIDI(midi_path)
    guitar_notes = pm.instruments[0].notes
    
    # Create JAMS
    jam = jams.JAMS()
    
    # Add note_midi annotation
    note_ann = jams.Annotation(namespace='note_midi')
    for n in guitar_notes:
        note_ann.append(
            time=n.start,
            duration=n.end - n.start,
            value=n.pitch,
            confidence=n.velocity / 127
        )
    jam.annotations.append(note_ann)
    
    # Add tab_note annotation with intelligent string/fret mapping
    tab_ann = jams.Annotation(namespace='tab_note')
    
    previous_position = None
    for n in guitar_notes:
        # Find best position for this pitch
        string, fret = choose_best_position(n.pitch, previous_position, tuning)
        previous_position = (string, fret)
        
        value = {
            "pitch": n.pitch,
            "string": string,
            "fret": fret,
            "techniques": []  # No techniques from MIDI
        }
        
        tab_ann.append(
            time=n.start,
            duration=n.end - n.start,
            value=value,
            confidence=n.velocity / 127
        )
    
    jam.annotations.append(tab_ann)
    
    print(f"Converted {len(guitar_notes)} notes to tablature")
    print(f"Pitch range: {min(n.pitch for n in guitar_notes)} - {max(n.pitch for n in guitar_notes)}")
    
    return jam

def jams_to_musicxml_real(jam, output_xml='output.xml', tempo_bpm=120):
    """
    Convert JAMS with real tablature to MusicXML.
    Uses actual pitch information for correct notation.
    
    Args:
        jam: JAMS object with tab_note annotation
        output_xml: Output MusicXML file path
        tempo_bpm: Tempo in BPM
    
    Returns:
        Path to created XML file
    """
    # Find tab_note annotation
    tab_notes = None
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            tab_notes = ann
            break
    
    if tab_notes is None:
        raise ValueError("No tab_note annotation found")
    
    print(f"Converting {len(tab_notes.data)} notes to MusicXML...")
    
    # Create music21 score
    score = stream.Score()
    part = stream.Part()
    
    # Add metadata
    part.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    part.insert(0, meter.TimeSignature('4/4')) # type: ignore
    part.insert(0, clef.TabClef())
    
    # Beat duration
    beat_sec = 60.0 / tempo_bpm
    
    # Convert notes
    previous_note = None
    for obs in tab_notes.data:
        val = obs.value
        
        # Use actual pitch from MIDI
        midi_pitch = val['pitch']
        string_num = val['string']
        fret_num = val['fret']

        # Extract techniques
        techniques = val.get('techniques', [])  # Extract techniques from JAMS
        
        # Create note with correct pitch
        n = note.Note()
        n.pitch.midi = midi_pitch

        # Apply techniques to note
        if techniques:
            print(f"  Processing {len(techniques)} technique(s)")
            for tech in techniques:
                if not tech:  # Skip None or empty values
                    print(f"  Skipping empty technique: {tech}")  # ← More specific
                    continue
                
                # Map guitar techniques to music21 notation
                if tech in ['hammer-on', 'hammer_on']:
                    print("hammer-on detected")
                    n.expressions.append(expressions.TextExpression('H'))
                    
                    # Create slur from previous note to this one
                    if previous_note is not None:
                        slur = spanner.Slur(previous_note, n)
                        part.insert(0, slur)  # Add slur to the part
            
                # PULL-OFF: Same as hammer-on
                elif tech in ['pull-off', 'pull_off']:
                    print("pull off detected")
                    n.expressions.append(expressions.TextExpression('P'))
                    
                    if previous_note is not None:
                        slur = spanner.Slur(previous_note, n)
                        part.insert(0, slur)
                
                elif tech == 'bend':
                    print("bend detected")
                    bend_amount = 0.5  # Default half step
                    # Create proper MusicXML bend element
                    bend = articulations.FretBend(
                        bendAlter=interval.Interval(bend_amount)
                    )
                    n.articulations.append(bend)
                    n.expressions.append(expressions.TextExpression('B'))
                
                elif tech == 'slide':
                    print("slide detected")
                    n.expressions.append(expressions.TextExpression('/'))
                
                elif tech == 'vibrato':
                    print("Vibrato detected")
                    n.expressions.append(expressions.TextExpression('~'))
                
                elif tech == 'harmonic':
                    print("Harmonic detected")
                    n.articulations.append(articulations.Harmonic())
        
        # Quantize duration to standard values
        duration_beats = obs.duration / beat_sec
        # Round to nearest 16th note
        quantized = round(duration_beats * 2) / 2
        if quantized < 0.5:
            quantized = 0.5  # Minimum eighth note
        n.quarterLength = quantized
        
        # Add tablature info
        n.editorial.stringNumber = string_num
        n.editorial.fretNumber = fret_num
        
        # Quantize time position
        time_beats = obs.time / beat_sec
        quantized_time = round(time_beats * 2) / 2
        
        part.insert(quantized_time, n)
        previous_note = n
    
    # Create measures
    part.makeMeasures(inPlace=True)
    score.append(part)
    
    # Write to MusicXML
    try:
        score.write('pdf', fp=output_xml)
        print(f"✓ Created {output_xml}")
    except Exception as e:
        print(f"Error: {e}")
        print("Trying with makeNotation...")
        score.write('musicxml', fp=output_xml, makeNotation=True)
        print(f"✓ Created {output_xml} (with notation fixes)")
    
    return output_xml

# =============================================================================
# ADVANCED: Position optimization for better fingering
# =============================================================================

def optimize_tablature_positions(jam, max_stretch=4):
    """
    Optimize string/fret positions for better playability.
    Tries to keep notes within reasonable hand positions.
    
    Args:
        jam: JAMS object with tab_note annotation
        max_stretch: Maximum fret stretch (typically 4 frets)
    
    Returns:
        Modified JAMS object with optimized positions
    """
    # Find tab annotation
    tab_ann = None
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            tab_ann = ann
            break
    
    if not tab_ann or len(tab_ann.data) < 2:
        return jam
    
    print("Optimizing tablature positions for playability...")
    
    # Re-calculate positions considering hand position
    notes = list(tab_ann.data)
    current_position = None  # (center_fret, string)
    
    for i, obs in enumerate(notes):
        pitch = obs.value['pitch']
        
        # Get all possible positions
        positions = midi_pitch_to_guitar_positions(pitch)
        
        if not positions:
            continue
        
        if current_position is None:
            # First note - choose most natural position
            string, fret = positions[0]
            current_position = fret
        else:
            # Choose position within max_stretch of current position
            valid_positions = [
                (s, f) for s, f in positions
                if abs(f - current_position) <= max_stretch
            ]
            
            if valid_positions:
                # Choose closest to current position
                string, fret = min(valid_positions, 
                                  key=lambda p: abs(p[1] - current_position))
            else:
                # No position within reach - need to shift hand position
                string, fret = positions[0]
                current_position = fret
        
        # Update the note
        obs.value['string'] = string
        obs.value['fret'] = fret
    
    print("✓ Positions optimized")
    return jam


def add_random_techniques_to_existing_jam(jam, technique_probability=0.5):
    '''
    Adds random techniques to EXISTING tab_note annotation
    '''
    tech_options = ["slide", "vibrato", "hammer-on", "pull-off", "bend", "harmonic"]
    # Find the EXISTING tab_note annotation
    tab_ann = None
    for ann in jam.annotations:
        if ann.namespace == 'tab_note':
            tab_ann = ann
            break
    
    if tab_ann is None:
        raise ValueError("No tab_note annotation found! Run midi_to_jams_with_tablature() first")
    
    technique_count = 0
    
    # MODIFY existing notes (don't create new ones)
    for obs in tab_ann.data:
        # Add technique to existing note
        if random.random() < technique_probability:
            tech = random.choice(tech_options)
            obs.value['techniques'] = [tech]  # ← Modify existing
            technique_count += 1
        else:
            obs.value['techniques'] = []
    
    print(f"\n✓ Modified {len(tab_ann.data)} notes")
    print(f"✓ Added {technique_count} techniques ({technique_count/len(tab_ann.data)*100:.1f}%)")
    
    return jam


def add_exp_techniques_to_existing_jam(jam, exp_onset_dur_tuples):
    '''
    Adds expressive techniques to EXISTING tab_note annotation
    '''

    # Find the EXISTING tab_note annotation
    tab_ann = None
    for ann in jam.annotations:
        if ann.namespace == 'tab_note':
            tab_ann = ann
            break
    
    if tab_ann is None:
        raise ValueError("No tab_note annotation found! Run midi_to_jams_with_tablature() first")
    
    technique_count = 0
    
    # MODIFY existing notes (don't create new ones)
    exptech_idx = 0
    for obs in tab_ann.data:
        if exptech_idx < len(exp_onset_dur_tuples) and obs.time >= exp_onset_dur_tuples[exptech_idx][1]/1000: # if note is at or beyond the onset of next expressive technique
            obs.value['techniques'] = [exp_onset_dur_tuples[exptech_idx][0]]
            exptech_idx += 1 # go to next expressive technique
            technique_count += 1
        else:
            obs.value['techniques'] = []
    
    print(f"\n✓ Modified {len(tab_ann.data)} notes")
    print(f"✓ Added {technique_count} techniques ({technique_count/len(tab_ann.data)*100:.1f}%)")
    
    return jam


# change added 01/02: to ensure does not take the same prediction from andrea twice
def midi_to_jams_with_tablature_from_andreas(midi_path, string_fret_time_tuples, tuning=STANDARD_TUNING,time_tolerance=0.2):
    # load midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    guitar_notes = pm.instruments[0].notes
    
    andrea_predictions = []
    for idx, (string, fret, _, onset_ms) in enumerate(string_fret_time_tuples):
        onset_sec = onset_ms / 1000.0
        andrea_predictions.append((onset_sec, int(string), int(fret), idx))
    
    print(f"MIDI has {len(guitar_notes)} notes")
    print(f"Received {len(string_fret_time_tuples)} string/fret tuples")

    print("\n=== TIMING DEBUG ===")
    print("First 10 MIDI notes vs closest Andrea predictions:")
    for i, n in enumerate(guitar_notes[:10]):
        closest_andrea = min(andrea_predictions, key=lambda x: abs(x[0] - n.start))[0]  # ← FIX: use andrea_predictions
        time_diff_ms = abs(n.start - closest_andrea) * 1000
        print(f"  MIDI note {i}: {n.start:.3f}s, Closest Andrea: {closest_andrea:.3f}s, Diff: {time_diff_ms:.1f}ms")
    print("===================\n")
    
    jam = jams.JAMS()
    
    # Add note_midi annotation
    note_ann = jams.Annotation(namespace='note_midi')
    for n in guitar_notes:
        note_ann.append(
            time=n.start,
            duration=n.end - n.start,
            value=n.pitch,
            confidence=n.velocity / 127
        )
    jam.annotations.append(note_ann)
    

    tab_ann = jams.Annotation(namespace='tab_note')
    
    matched_count = 0
    unmatched_count = 0
    previous_position = None
    used_predictions = set()
    
    # align andreas with the midi
    for n in guitar_notes:
        best_match = None
        best_time_diff = float('inf')
        best_idx = None
        
        for onset_sec, string, fret, idx in andrea_predictions:
            if idx in used_predictions:  # ← ADD THIS CHECK
                continue
            time_diff = abs(n.start - onset_sec)
            if time_diff < time_tolerance and time_diff < best_time_diff:
                best_match = (string, fret)
                best_time_diff = time_diff
                best_idx = idx
        
        if best_match:
            string, fret = best_match
            used_predictions.add(best_idx)
            matched_count += 1
        else:
            # No match found means use intelligent fallback from colin
            string, fret = choose_best_position(n.pitch, previous_position, tuning=tuning)
            unmatched_count += 1
        
        previous_position = (string, fret)
        
        value = {
            "pitch": n.pitch,
            "string": string,
            "fret": fret,
            "techniques": []
        }
        
        tab_ann.append(
            time=n.start,
            duration=n.end - n.start,
            value=value,
            confidence=n.velocity / 127
        )
    
    jam.annotations.append(tab_ann)
    
    print(f"✓ Matched {matched_count} notes")
    print(f"✗ Used fallback for {unmatched_count} notes")
    
    return jam


### SEQUENTIAL ORDER: uses the fact that their lengths are equal
def midi_to_jams_with_tablature_from_andreas_sequential(midi_path, string_fret_time_tuples, tuning=STANDARD_TUNING,time_tolerance=0.2):
    # load midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    guitar_notes = pm.instruments[0].notes
    
    print(f"MIDI has {len(guitar_notes)} notes")
    print(f"Received {len(string_fret_time_tuples)} string/fret tuples")

    if len(guitar_notes) != len(string_fret_time_tuples):
        print(f"WARNING: Length mismatch! MIDI={len(guitar_notes)}, Andrea={len(string_fret_time_tuples)}")
        return midi_to_jams_with_tablature_from_andreas(midi_path, string_fret_time_tuples, tuning, time_tolerance)
    
    jam = jams.JAMS()
    
    # Add note_midi annotation
    note_ann = jams.Annotation(namespace='note_midi')
    for n in guitar_notes:
        note_ann.append(
            time=n.start,
            duration=n.end - n.start,
            value=n.pitch,
            confidence=n.velocity / 127
        )
    jam.annotations.append(note_ann)
    

    tab_ann = jams.Annotation(namespace='tab_note')
    
    for i, n in enumerate(guitar_notes):
        string, fret, _, _ = string_fret_time_tuples[i]
        string, fret = int(string), int(fret)
        value = {
            "pitch": n.pitch,
            "string": string,
            "fret": fret,
            "techniques": []
        }
        
        tab_ann.append(
            time=n.start,
            duration=n.end - n.start,
            value=value,
            confidence=n.velocity / 127
        )
    
    jam.annotations.append(tab_ann)

    print(f"Matched all {len(guitar_notes)} notes by sequential order")
    
    return jam


def conversion(midi_path, exp_onset_dur_tuples, output_name, bpm):
    # Step 1: Create JAMS with tablature
    jam = midi_to_jams_with_tablature(midi_path)

    # Step 2: Add techniques to EXISTING tab_note annotation
    #jam = add_random_techniques_to_existing_jam(jam, technique_probability=0.5)
    jam = add_exp_techniques_to_existing_jam(jam, exp_onset_dur_tuples)

    # Step 3: Debug - check techniques are there
    tab_ann = [a for a in jam.annotations if a.namespace == 'tab_note'][0]
    print("\nNotes with techniques:")
    for i, obs in enumerate(tab_ann.data[:20]):
        techs = obs.value.get('techniques', [])
        if techs:
            print(f"  Note {i}: {techs}")

    # Step 4: Convert to MusicXML
    jams_to_musicxml_real(jam, output_name, tempo_bpm=bpm)

    return output_name

def conversion_andreas(midi_path, exp_onset_dur_tuples, string_fret_tuples, output_name, bpm):
    # Step 1: Create JAMS with tablature
    # jam = midi_to_jams_with_tablature_from_andreas(midi_path, string_fret_tuples)
    # import from andrea
    jam = midi_to_jams_with_tablature_from_andreas_sequential(midi_path, string_fret_tuples)

    # Step 2: Add techniques to EXISTING tab_note annotation
    #jam = add_random_techniques_to_existing_jam(jam, technique_probability=0.5) --> no longer using random
    jam = preprocess.add_exp_techniques_to_existing_jam(jam, exp_onset_dur_tuples)

    # Step 3: Debug - check techniques are there
    tab_ann = [a for a in jam.annotations if a.namespace == 'tab_note'][0]
    print("\nNotes with techniques:")
    for i, obs in enumerate(tab_ann.data[:20]):
        techs = obs.value.get('techniques', [])
        if techs:
            print(f"  Note {i}: {techs}")

    # Step 4: Convert to MusicXML
    jams_to_musicxml_real(jam, output_name, tempo_bpm=bpm)

    return output_name