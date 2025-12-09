"""
Bulletproof MIDI to Tablature converter
Handles timing issues that cause MusicXML export errors
"""

import jams
import pretty_midi
import random
from music21 import stream, note, tempo, meter, instrument


def midi_to_jams_fixed(midi_path):
    """Convert MIDI to JAMS with proper namespace"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    guitar_notes = pm.instruments[0].notes
    jam = jams.JAMS()
    
    note_ann = jams.Annotation(namespace='note_midi')
    for n in guitar_notes:
        note_ann.append(
            time=n.start,
            duration=n.end - n.start,
            value=n.pitch,
            confidence=n.velocity / 127
        )
    
    jam.annotations.append(note_ann)
    return jam


def encode_notes_for_tab(jam):
    """Add tablature info with clean techniques"""
    new_ann = jams.Annotation(namespace='tab_note')
    tech_options = ["slide", "vibrato", "hammer-on", "pull-off", "bend"]
    
    note_ann = jam.annotations[0]
    for obs in note_ann.data:
        # Clean techniques - no None values
        techniques = []
        if random.random() < 0.3:  # 30% chance of technique
            techniques = [random.choice(tech_options)]
        
        value = {
            "pitch": obs.value,
            "string": random.randint(1, 6),
            "fret": random.randint(0, 12),
            "techniques": techniques
        }
        
        new_ann.append(
            time=obs.time,
            duration=obs.duration,
            value=value,
            confidence=obs.confidence
        )
    
    jam.annotations.append(new_ann)
    return jam


def jams_to_notation_simple(jam, output_file='output.xml', use_tempo=120):
    """
    Ultra-simple converter that always works.
    Ignores exact timing and uses standard note durations.
    
    Args:
        jam: JAMS object with tab_note annotation
        output_file: Output path (.xml for MusicXML, .pdf for PDF if LilyPond installed)
        use_tempo: Tempo in BPM
    
    Returns:
        Path to output file
    """
    from music21 import stream, note, tempo, meter, clef
    
    # Find tab notes
    tab_notes = None
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            tab_notes = ann
            break
    
    if not tab_notes:
        raise ValueError("No tab_note annotation found")
    
    print(f"Converting {len(tab_notes.data)} notes...")
    
    # Create score
    score = stream.Score()
    part = stream.Part()
    
    # Add metadata
    part.insert(0, tempo.MetronomeMark(number=use_tempo))
    part.insert(0, meter.TimeSignature('4/4'))
    part.insert(0, clef.TabClef())
    
    # Standard guitar tuning
    string_pitches = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
    
    # Create measures
    current_measure = stream.Measure(number=1)
    measure_beats = 0.0
    beats_per_measure = 4.0
    measure_num = 1
    
    for obs in tab_notes.data:
        val = obs.value
        
        # Get string and fret
        string_num = val.get('string', 1)
        fret_num = val.get('fret', 0)
        
        # Calculate pitch
        midi_pitch = string_pitches.get(string_num, 64) + fret_num
        
        # Create note with FIXED duration (quarter note)
        # This avoids all timing issues
        n = note.Note()
        n.pitch.midi = midi_pitch
        n.quarterLength = 1.0  # Always quarter note - simple!
        
        # Add tab info
        n.editorial.stringNumber = string_num
        n.editorial.fretNumber = fret_num
        
        # Add to measure
        current_measure.append(n)
        measure_beats += 1.0
        
        # New measure if full
        if measure_beats >= beats_per_measure:
            part.append(current_measure)
            measure_num += 1
            current_measure = stream.Measure(number=measure_num)
            measure_beats = 0.0
    
    # Add last measure if not empty
    if len(current_measure.notesAndRests) > 0:
        part.append(current_measure)
    
    score.append(part)
    
    # Write output
    if output_file.endswith('.pdf'):
        try:
            score.write('lily.pdf', fp=output_file)
            print(f"✓ Created {output_file}")
        except Exception as e:
            print(f"PDF generation failed: {e}")
            xml_file = output_file.replace('.pdf', '.xml')
            score.write('musicxml', fp=xml_file)
            print(f"✓ Created {xml_file} instead (open in MuseScore to export PDF)")
            return xml_file
    else:
        score.write('musicxml', fp=output_file)
        print(f"✓ Created {output_file}")
    
    return output_file


def jams_to_notation_quantized(jam, output_file='output.xml', use_tempo=120, quantize_to='eighth'):
    """
    Converter that quantizes to standard rhythms.
    More accurate than simple version but still robust.
    
    Args:
        jam: JAMS object with tab_note annotation
        output_file: Output path
        use_tempo: Tempo in BPM
        quantize_to: 'quarter', 'eighth', or 'sixteenth'
    
    Returns:
        Path to output file
    """
    from music21 import stream, note, tempo, meter, clef
    
    # Find tab notes
    tab_notes = None
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            tab_notes = ann
            break
    
    if not tab_notes:
        raise ValueError("No tab_note annotation found")
    
    print(f"Converting {len(tab_notes.data)} notes with quantization...")
    
    # Quantization grid
    quant_values = {
        'whole': 4.0,
        'half': 2.0,
        'quarter': 1.0,
        'eighth': 0.5,
        'sixteenth': 0.25
    }
    quant_size = quant_values.get(quantize_to, 0.5)
    
    # Create score
    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=use_tempo))
    part.insert(0, meter.TimeSignature('4/4'))
    part.insert(0, clef.TabClef())
    
    # Standard guitar tuning
    string_pitches = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
    
    # Beat duration in seconds
    beat_sec = 60.0 / use_tempo
    
    # Convert each note
    for obs in tab_notes.data:
        val = obs.value
        
        # Get string and fret
        string_num = val.get('string', 1)
        fret_num = val.get('fret', 0)
        midi_pitch = string_pitches.get(string_num, 64) + fret_num
        
        # Quantize duration
        duration_beats = obs.duration / beat_sec
        # Round to nearest quantize value
        quantized_beats = round(duration_beats / quant_size) * quant_size
        # Ensure minimum duration
        if quantized_beats < quant_size:
            quantized_beats = quant_size
        
        # Create note
        n = note.Note()
        n.pitch.midi = midi_pitch
        n.quarterLength = quantized_beats
        n.editorial.stringNumber = string_num
        n.editorial.fretNumber = fret_num
        
        # Quantize time position
        time_beats = obs.time / beat_sec
        quantized_time = round(time_beats / quant_size) * quant_size
        
        part.insert(quantized_time, n)
    
    # Add measures
    part.makeMeasures(inPlace=True)
    score.append(part)
    
    # Write output
    try:
        score.write('musicxml', fp=output_file)
        print(f"✓ Created {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        print("Trying with makeNotation...")
        score.write('musicxml', fp=output_file, makeNotation=True)
        print(f"✓ Created {output_file} (with fixes)")
    
    return output_file


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_usage():
    """Show how to use these functions"""
    
    print("=" * 70)
    print("BULLETPROOF MIDI TO TABLATURE")
    print("=" * 70)
    
    # Example 1: Ultra-simple (always works, uses quarter notes)
    print("\n1. SIMPLE VERSION (all quarter notes):")
    print("   jam = midi_to_jams_fixed('song.mid')")
    print("   jam = encode_notes_for_tab(jam)")
    print("   jams_to_notation_simple(jam, 'output_simple.xml')")
    
    # Example 2: Quantized version (better rhythm)
    print("\n2. QUANTIZED VERSION (better rhythm):")
    print("   jam = midi_to_jams_fixed('song.mid')")
    print("   jam = encode_notes_for_tab(jam)")
    print("   jams_to_notation_quantized(jam, 'output.xml', quantize_to='eighth')")
    
    print("\n" + "=" * 70)
    print("Both versions avoid MusicXML export errors!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
