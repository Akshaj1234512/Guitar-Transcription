"""
Complete fixed workflow: MIDI → JAMS → Tablature (PDF/XML)
"""

import jams
import pretty_midi
import random
import os
from music21 import stream, note, chord, tempo, meter, clef
from music21 import tablature as m21tab


def midi_to_jams(midi_path):
    """
    Convert MIDI file to JAMS object
    Fixed to use proper 'note_midi' namespace instead of 'note'
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    guitar_notes = pm.instruments[0].notes  # first instrument
    jam = jams.JAMS()

    # Use 'note_midi' namespace (this is a standard JAMS namespace)
    note_ann = jams.Annotation(namespace='note_midi')
    for note in guitar_notes:
        note_ann.append(
            time=note.start,
            duration=note.end - note.start,
            value=note.pitch,
            confidence=note.velocity / 127  # normalize velocity
        )

    jam.annotations.append(note_ann)
    return jam


def encode_notes_for_test(jam):
    """
    Adds sample random expressive techniques, random strings, and frets
    Fixed to handle None values in techniques properly
    """
    new_ann = jams.Annotation(namespace='tab_note')
    tech_options = ["slide", "vibrato", "hammer-on", "pull-off", "bend", None]

    # iterate over existing note 
    note_ann = jam.annotations[0]  
    for obs in note_ann.data:
        pitch = obs.value  # the original pitch

        # Clean up techniques - remove None values
        techniques = []
        if random.random() < 0.5:
            tech = random.choice(tech_options)
            if tech is not None:  # Only add non-None techniques
                techniques = [tech]

        value = {
            "pitch": pitch,
            "string": random.randint(1, 6),       # random string 1-6
            "fret": random.randint(0, 12),        # random fret 0-12
            "techniques": techniques  # Already cleaned, no None values
        }

        new_ann.append(
            time=obs.time, 
            duration=obs.duration, 
            value=value, 
            confidence=obs.confidence
        )

    # add new tab_note annotation
    jam.annotations.append(new_ann)
    return jam


def jams_to_musicxml_simple(jam, output_xml='output.xml', tempo_bpm=120):
    """
    Simpler version using music21's makeNotation to handle complex rhythms
    This is more reliable for MIDI-derived timings
    
    Args:
        jam: JAMS object with tab_note annotations
        output_xml: Path to output MusicXML file
        tempo_bpm: Tempo in BPM
    
    Returns:
        Path to created XML file
    """
    from music21 import stream, note, tempo, meter
    
    # Create a simple stream with all notes
    s = stream.Stream()
    s.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    s.insert(0, meter.TimeSignature('4/4'))
    
    # Find tab_note annotation
    tab_notes = None
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            tab_notes = ann
            break
    
    if tab_notes is None:
        raise ValueError("No tab_note annotation found in JAMS object")
    
    print(f"Converting {len(tab_notes.data)} notes to MusicXML (simple mode)...")
    
    # Standard guitar tuning
    string_pitches = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
    
    # Add notes with their actual timings
    for obs in tab_notes.data:
        val = obs.value
        string_num = val.get('string', 1)
        fret_num = val.get('fret', 0)
        midi_pitch = string_pitches.get(string_num, 64) + fret_num
        
        n = note.Note()
        n.pitch.midi = midi_pitch
        # Don't worry about exact durations - makeNotation will fix it
        n.quarterLength = 1.0  # Default quarter note
        
        # Store tab info
        n.editorial.stringNumber = string_num
        n.editorial.fretNumber = fret_num
        
        # Insert at the actual time (music21 handles this)
        s.insert(obs.time, n)
    
    # Let music21 figure out the notation
    s_notation = s.makeNotation(inPlace=False)
    
    # Write to MusicXML
    s_notation.write('musicxml', fp=output_xml)
    
    print(f"✓ Created {output_xml}")
    print("  Open in MuseScore to view and export as PDF")
    
    return output_xml


def jams_to_musicxml(jam, output_xml='output.xml', tempo_bpm=120):
    """
    Convert JAMS object to MusicXML format
    Most reliable method - can then open in MuseScore, Finale, etc.
    
    Args:
        jam: JAMS object (not filepath!) with tab_note annotations
        output_xml: Path to output MusicXML file
        tempo_bpm: Tempo in beats per minute (default 120)
    
    Returns:
        Path to created XML file
    """
    
    # Create music21 score
    score = stream.Score()
    guitar_part = stream.Part()
    
    # Set up guitar tablature
    guitar_part.insert(0, clef.TabClef())
    guitar_part.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    guitar_part.insert(0, meter.TimeSignature('4/4'))
    
    # Find tab_note annotation
    tab_notes = None
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            tab_notes = ann
            break
    
    if tab_notes is None:
        raise ValueError("No tab_note annotation found in JAMS object")
    
    print(f"Converting {len(tab_notes.data)} notes to MusicXML...")
    
    # Standard guitar tuning (MIDI pitch for open strings)
    string_pitches = {
        6: 40,  # E2
        5: 45,  # A2
        4: 50,  # D3
        3: 55,  # G3
        2: 59,  # B3
        1: 64   # E4
    }
    
    def quantize_duration(duration_seconds, tempo_bpm=120):
        """
        Quantize duration to nearest standard musical duration.
        Converts seconds to quarter notes and rounds to standard values.
        """
        # Convert seconds to quarter notes based on tempo
        beat_duration = 60.0 / tempo_bpm  # seconds per beat
        quarter_length = duration_seconds / beat_duration
        
        # Standard durations (in quarter notes): whole, half, quarter, eighth, sixteenth
        standard_durations = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
        
        # Find closest standard duration
        closest = min(standard_durations, key=lambda x: abs(x - quarter_length))
        
        # Don't let it be too small
        if closest < 0.125:
            closest = 0.125
            
        return closest
    
    # Convert JAMS notes to music21 notes
    for obs in tab_notes.data:
        val = obs.value
        
        string_num = val.get('string', 1)
        fret_num = val.get('fret', 0)
        
        # Calculate MIDI pitch from string and fret
        midi_pitch = string_pitches.get(string_num, 64) + fret_num
        
        # Create note
        n = note.Note()
        n.pitch.midi = midi_pitch
        
        # Quantize duration to standard musical values
        n.quarterLength = quantize_duration(obs.duration, tempo_bpm)
        
        # Add tablature information (this is what shows string/fret numbers)
        n.editorial.stringNumber = string_num
        n.editorial.fretNumber = fret_num
        
        guitar_part.append(n)
    
    score.append(guitar_part)
    
    # Use makeNotation to fix any remaining issues
    try:
        # Write to MusicXML
        score.write('musicxml', fp=output_xml)
        print(f"✓ Created {output_xml}")
        print("  You can now:")
        print("  1. Open in MuseScore (free) to export as PDF")
        print("  2. Open in Finale or Sibelius")
        print("  3. Use with alphaTab for web rendering")
    except Exception as e:
        print(f"Warning: {e}")
        print("Trying with makeNotation=True...")
        # This fixes complex rhythms
        score.write('musicxml', fp=output_xml, makeNotation=True)
        print(f"✓ Created {output_xml} (with notation fixes)")
    
    return output_xml
    """
    Convert JAMS object to MusicXML format
    Most reliable method - can then open in MuseScore, Finale, etc.
    
    Args:
        jam: JAMS object (not filepath!) with tab_note annotations
        output_xml: Path to output MusicXML file
        tempo_bpm: Tempo in beats per minute (default 120)
    
    Returns:
        Path to created XML file
    """
    
    # Create music21 score
    score = stream.Score()
    guitar_part = stream.Part()
    
    # Set up guitar tablature
    guitar_part.insert(0, clef.TabClef())
    guitar_part.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    guitar_part.insert(0, meter.TimeSignature('4/4'))
    
    # Find tab_note annotation
    tab_notes = None
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            tab_notes = ann
            break
    
    if tab_notes is None:
        raise ValueError("No tab_note annotation found in JAMS object")
    
    print(f"Converting {len(tab_notes.data)} notes to MusicXML...")
    
    # Standard guitar tuning (MIDI pitch for open strings)
    string_pitches = {
        6: 40,  # E2
        5: 45,  # A2
        4: 50,  # D3
        3: 55,  # G3
        2: 59,  # B3
        1: 64   # E4
    }
    
    def quantize_duration(duration_seconds, tempo_bpm=120):
        """
        Quantize duration to nearest standard musical duration.
        Converts seconds to quarter notes and rounds to standard values.
        """
        # Convert seconds to quarter notes based on tempo
        beat_duration = 60.0 / tempo_bpm  # seconds per beat
        quarter_length = duration_seconds / beat_duration
        
        # Standard durations (in quarter notes): whole, half, quarter, eighth, sixteenth
        standard_durations = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
        
        # Find closest standard duration
        closest = min(standard_durations, key=lambda x: abs(x - quarter_length))
        
        # Don't let it be too small
        if closest < 0.125:
            closest = 0.125
            
        return closest
    
    # Convert JAMS notes to music21 notes
    for obs in tab_notes.data:
        val = obs.value
        
        string_num = val.get('string', 1)
        fret_num = val.get('fret', 0)
        
        # Calculate MIDI pitch from string and fret
        midi_pitch = string_pitches.get(string_num, 64) + fret_num
        
        # Create note
        n = note.Note()
        n.pitch.midi = midi_pitch
        
        # Quantize duration to standard musical values
        n.quarterLength = quantize_duration(obs.duration, tempo_bpm)
        
        # Add tablature information (this is what shows string/fret numbers)
        n.editorial.stringNumber = string_num
        n.editorial.fretNumber = fret_num
        
        guitar_part.append(n)
    
    score.append(guitar_part)
    
    # Use makeNotation to fix any remaining issues
    try:
        # Write to MusicXML
        score.write('musicxml', fp=output_xml)
        print(f"✓ Created {output_xml}")
        print("  You can now:")
        print("  1. Open in MuseScore (free) to export as PDF")
        print("  2. Open in Finale or Sibelius")
        print("  3. Use with alphaTab for web rendering")
    except Exception as e:
        print(f"Warning: {e}")
        print("Trying with makeNotation=True...")
        # This fixes complex rhythms
        score.write('musicxml', fp=output_xml, makeNotation=True)
        print(f"✓ Created {output_xml} (with notation fixes)")
    
    return output_xml


def jams_to_lilypond_pdf(jam, output_pdf='output.pdf'):
    """
    Convert JAMS directly to PDF using LilyPond
    Requires LilyPond to be installed!
    
    Args:
        jam: JAMS object with tab_note annotations
        output_pdf: Path to output PDF file
    
    Returns:
        Path to created PDF
    """
    
    # Create music21 score (same as above)
    score = stream.Score()
    guitar_part = stream.Part()
    guitar_part.insert(0, clef.TabClef())
    guitar_part.insert(0, tempo.MetronomeMark(number=120))
    guitar_part.insert(0, meter.TimeSignature('4/4'))
    
    # Find tab_note annotation
    tab_notes = None
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            tab_notes = ann
            break
    
    if tab_notes is None:
        raise ValueError("No tab_note annotation found in JAMS object")
    
    print(f"Converting {len(tab_notes.data)} notes to PDF...")
    
    # Standard guitar tuning
    string_pitches = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
    
    # Convert notes
    for obs in tab_notes.data:
        val = obs.value
        string_num = val.get('string', 1)
        fret_num = val.get('fret', 0)
        midi_pitch = string_pitches.get(string_num, 64) + fret_num
        
        n = note.Note()
        n.pitch.midi = midi_pitch
        n.quarterLength = obs.duration * 4
        n.editorial.stringNumber = string_num
        n.editorial.fretNumber = fret_num
        
        guitar_part.append(n)
    
    score.append(guitar_part)
    
    try:
        # Try to render with LilyPond
        score.write('lily.pdf', fp=output_pdf)
        print(f"✓ Created {output_pdf}")
        return output_pdf
    except Exception as e:
        print(f"❌ Error rendering with LilyPond: {e}")
        print("\nLilyPond might not be installed or configured.")
        print("Falling back to MusicXML...")
        
        # Fallback to MusicXML
        xml_path = output_pdf.replace('.pdf', '.xml')
        return jams_to_musicxml(jam, xml_path)


# ============================================================================
# COMPLETE EXAMPLE WORKFLOW
# ============================================================================

def complete_workflow_example(midi_path, output_format='xml'):
    """
    Complete example: MIDI → JAMS → Tablature
    
    Args:
        midi_path: Path to MIDI file
        output_format: 'xml' or 'pdf'
    
    Returns:
        Path to output file
    """
    print("=" * 60)
    print("Complete MIDI to Tablature Workflow")
    print("=" * 60)
    
    # Step 1: Convert MIDI to JAMS
    print("\n1. Loading MIDI file...")
    jam = midi_to_jams(midi_path)
    print(f"   ✓ Loaded {len(jam.annotations[0].data)} notes")
    
    # Step 2: Add tablature information
    print("\n2. Encoding tablature information...")
    jam = encode_notes_for_test(jam)
    print(f"   ✓ Created tablature annotation")
    
    # Step 3: Convert to notation
    print("\n3. Rendering to notation...")
    if output_format == 'pdf':
        output = jams_to_lilypond_pdf(jam, 'tablature.pdf')
    else:
        output = jams_to_musicxml(jam, 'tablature.xml')
    
    print("\n" + "=" * 60)
    print("✓ Complete!")
    print("=" * 60)
    return output


if __name__ == "__main__":
    # Example usage
    print("JAMS to Notation Converter - Fixed Version")
    print()
    print("Use this in your code:")
    print()
    print("# Your working code:")
    print("example_midi = 'path/to/your/file.mid'")
    print("jam = midi_to_jams(example_midi)")
    print("jam = encode_notes_for_test(jam)")
    print()
    print("# Fixed function call:")
    print("jams_to_musicxml(jam, 'output.xml')  # Creates MusicXML")
    print("# or")
    print("jams_to_lilypond_pdf(jam, 'output.pdf')  # Creates PDF (needs LilyPond)")
    print()
    print("Key fixes:")
    print("1. Changed 'note' namespace to 'note_midi'")
    print("2. Function accepts JAMS object, not filepath")
    print("3. Properly handles None values in techniques")
