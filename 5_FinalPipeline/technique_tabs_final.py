'''
SECOND version of technique tabs
for chord functionality + other notation features
'''
import jams
import numpy as np

from music21 import stream, note, tempo, meter, clef, instrument, metadata
from music21 import note as m21_note  # Ensure we have access to Rest
from music21 import chord as m21_chord

from collections import defaultdict
import math
import tab_generation_utils.postprocess as postprocess
import tab_generation_utils.preprocess as preprocess
import tab_generation_utils.jams_test as j

# TODO: figure out pdf export
# import lilypond
# # get lilyponds
# us = environment.UserSettings()
# us['lilypondPath'] = str(lilypond.executable())
# print(f"LilyPond path set to: {lilypond.executable()}")


def jams_to_musicxml_real(jam, output_xml, tempo_bpm, title=None, composer=None):
    """
    Convert JAMS with real tablature to MusicXML.
    Uses actual pitch information for correct notation.
    
    Args:
        jam: JAMS object with tab_note annotation
        output_xml: Output MusicXML file path
        tempo_bpm: Tempo in BPM
        title: Title/name of the score (default: None)
        composer: Composer/author of the score (default: None)
    
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
    
    # Create score
    score = stream.Score()
    
    # Add metadata (title and composer)
    md = metadata.Metadata()
    if title:
        md.title = title
    if composer:
        md.composer = composer
    score.metadata = md
    
    part = stream.Part()

    # Add instrument data
    guitar = instrument.Guitar()
    guitar.instrumentName = 'Classical Guitar (tablature)'
    guitar.instrumentAbbreviation = 'Guit.'
    guitar.instrumentSound = 'pluck.guitar.nylon-string'
    part.insert(0, guitar)
    
    # Add tempo
    part.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    
    # Beat duration in seconds
    beat_sec = 60.0 / tempo_bpm
    
    # Setting correct time signature
    time_sig_numerator = getattr(jam.sandbox, 'time_sig_numerator', 4)
    time_sig_denominator = getattr(jam.sandbox, 'time_sig_denominator', 4)
    
    # Calculate beats per measure
    beats_per_measure = time_sig_numerator * (4.0 / time_sig_denominator)
    ts = meter.TimeSignature(f'{time_sig_numerator}/{time_sig_denominator}') # type: ignore
    part.insert(0, ts)
    part.insert(0, clef.TabClef())
    
    # Quantization grid: 1/32 note = 0.125 quarter notes
    # There are 8 thirty-second notes per quarter note
    QUANTIZATION_DIVISOR = 8  # 8 = 1/32 note, 4 = 1/16 note, 2 = 1/8 note
    MIN_DURATION = 1.0 / QUANTIZATION_DIVISOR  # 0.125 for 1/32 note
    
    # Sort notes by time and collect quantized note data
    note_events = []
    for obs in sorted(tab_notes.data, key=lambda x: x.time):
        val = obs.value
        
        # Quantize time position to nearest 1/32 note
        time_beats = obs.time / beat_sec
        quantized_time = round(time_beats * QUANTIZATION_DIVISOR) / QUANTIZATION_DIVISOR
        
        # Quantize duration to nearest 1/32 note
        duration_beats = obs.duration / beat_sec
        quantized_duration = round(duration_beats * QUANTIZATION_DIVISOR) / QUANTIZATION_DIVISOR
        if quantized_duration < MIN_DURATION:
            quantized_duration = MIN_DURATION  # Minimum 1/32 note
        
        note_events.append({
            'time': quantized_time,
            'duration': quantized_duration,
            'pitch': val['pitch'],
            'string': val['string'],
            'fret': val['fret'],
            'techniques': val.get('techniques', [])
        })
    
    # Calculate total duration and number of complete measures needed
    if note_events:
        last_note_end = max(n['time'] + n['duration'] for n in note_events)
    else:
        last_note_end = 0
    
    total_measures = math.ceil(last_note_end / beats_per_measure)
    if total_measures < 1:
        total_measures = 1
    
    print(f"  Total duration: {last_note_end:.2f} beats -> {total_measures} measures")
    print(f"  Quantization: 1/{int(4 * QUANTIZATION_DIVISOR)} note (min duration: {MIN_DURATION} beats)")
    
    # Track note index for technique mapping
    # IMPORTANT: This must increment for EACH note, including each note within chords
    note_index = 0
    
    # Track techniques for post-processing
    # Format: {note_index: {'technique': 'hammer-on'|'pull-off', 'is_target': bool}}
    technique_map = {}
    
    for measure_num in range(total_measures):
        measure_start = measure_num * beats_per_measure
        measure_end = measure_start + beats_per_measure
        
        # Create a new measure stream
        current_measure = stream.Measure(number=measure_num + 1)
        
        # Get notes that START in this measure
        measure_notes = [n for n in note_events if measure_start <= n['time'] < measure_end]
        
        # Group notes by their start time (these are chords)
        # FIX: Round position to avoid floating-point dictionary key issues
        notes_by_time = defaultdict(list)
        for n in measure_notes:
            # Convert to position within measure (0 to beats_per_measure)
            pos_in_measure = n['time'] - measure_start
            # Round to 1ms precision to avoid float comparison issues
            pos_rounded = round(pos_in_measure * 1000) / 1000
            notes_by_time[pos_rounded].append(n)

        sorted_positions = sorted(notes_by_time.keys())
        current_position = 0.0  # Track position within measure

        for pos_in_measure in sorted_positions:
            notes_at_time = notes_by_time[pos_in_measure]
            
            # Skip if this position is before our current position (overlapping notes)
            if pos_in_measure < current_position - 0.001:
                continue
            
            # Add rest if there's a gap
            if pos_in_measure > current_position + 0.001:
                rest_duration = pos_in_measure - current_position
                r = m21_note.Rest()
                r.quarterLength = rest_duration
                current_measure.append(r)
                current_position = pos_in_measure

            # Calculate available space in measure
            space_remaining = beats_per_measure - current_position
            
            # Determine the duration for this time slot
            min_duration = min(n['duration'] for n in notes_at_time)
            
            # Clamp duration to fit in measure
            actual_duration = min(min_duration, space_remaining)
            
            if actual_duration <= 0:
                continue  # No space left in measure

            # Create notes (as chord if multiple)
            if len(notes_at_time) == 1:
                # Single note
                note_event = notes_at_time[0]
                n = note.Note()
                n.pitch.midi = note_event['pitch']
                n.quarterLength = actual_duration
                n.editorial.stringNumber = note_event['string']
                n.editorial.fretNumber = note_event['fret']
                
                # Apply techniques
                techniques = note_event.get('techniques', [])
                if techniques:
                    technique_map = postprocess.apply_techniques_to_note(n, techniques, note_index, technique_map)
                
                current_measure.append(n)
                note_index += 1

            else:
                # Multiple notes at same time = chord
                # FIX: Sort chord notes by string (high to low) for consistent XML output
                # MusicXML typically orders chord notes from highest to lowest pitch
                notes_at_time_sorted = sorted(notes_at_time, key=lambda x: -x['pitch'])
                
                chord_notes = []
                for note_event in notes_at_time_sorted:
                    n = note.Note()
                    n.pitch.midi = note_event['pitch']
                    n.editorial.stringNumber = note_event['string']
                    n.editorial.fretNumber = note_event['fret']
                    
                    # FIX: Apply techniques to chord notes too
                    techniques = note_event.get('techniques', [])
                    if techniques:
                        technique_map = postprocess.apply_techniques_to_note(n, techniques, note_index, technique_map)
                    
                    print(f"  Technique map: {technique_map}")
                    chord_notes.append(n)
                    
                    # FIX: Increment note_index for EACH note in the chord
                    # This ensures technique_map indices align with XML note elements
                    note_index += 1
                
                c = m21_chord.Chord(chord_notes)
                c.quarterLength = actual_duration
                current_measure.append(c)

            # Advance position
            current_position += actual_duration

        # Fill rest of measure with rest if needed
        if current_position < beats_per_measure - 0.001:
            remaining = beats_per_measure - current_position
            r = m21_note.Rest()
            r.quarterLength = remaining
            current_measure.append(r)
        
        # Verify measure duration before adding
        measure_duration = current_measure.duration.quarterLength
        if abs(measure_duration - beats_per_measure) > 0.01:
            print(f"  WARNING: Measure {measure_num + 1} has duration {measure_duration}, expected {beats_per_measure}")
        
        # Add measure to part
        part.append(current_measure)
    
    score.append(part)
    
    # Write to MusicXML
    try:
        score.write('musicxml', fp=output_xml)
        print(f"✓ Created {output_xml}")
    except Exception as e:
        print(f"Error: {e}")
        print("Trying with makeNotation...")
        score.write('musicxml', fp=output_xml, makeNotation=True)
        print(f"✓ Created {output_xml} (with notation fixes)")

    # Post-process the XML with technique information
    postprocess.post_process_musicxml(output_xml, technique_map)
    
    return output_xml

# MAIN
def conversion_andreas(midi_path, exp_onset_dur_tuples, string_fret_tuples, output_name, bpm, mode="time_matching"):
    # Step 1: Create JAMS with tablature)
    if mode == "sequential":
        jam = preprocess.midi_to_jams_with_tablature_from_andreas_sequential(midi_path, string_fret_tuples, bpm=bpm)
    if mode == "time_matching":
        jam = preprocess.midi_to_jams_with_tablature_from_andreas(midi_path, string_fret_tuples, bpm=bpm)

    #----------------------------------------------------------------------------------#

    # Step 2: Add techniques to EXISTING tab_note annotation
    ### TODO: ask samhtia to check new version of exp techniques implementation
    jam = preprocess.add_exp_techniques_to_existing_jam(jam, exp_onset_dur_tuples)

    #----------------------------------------------------------------------------------#

    j.save_jam(jam, output_name)

    #----------------------------------------------------------------------------------#

    # Step 4: Convert to MusicXML
    xml_path = jams_to_musicxml_real(jam, output_name, tempo_bpm=bpm, title=output_name)

    return xml_path