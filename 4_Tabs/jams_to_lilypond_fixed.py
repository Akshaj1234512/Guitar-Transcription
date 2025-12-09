"""
Fixed JAMS to LilyPond converter that works with your MIDI workflow
"""

import jams
from music21 import stream, note, chord, tempo, meter, clef
from music21 import tablature as m21tab
from music21 import environment


def jams_to_lilypond(jam, output_pdf='output.pdf'):
    """
    Convert JAMS object (not file) to PDF tablature using music21 + LilyPond
    
    Args:
        jam: JAMS object (not filepath!) with tab_note annotations
        output_pdf: Path to output PDF file
    
    Returns:
        Path to created PDF
    """
    
    # Create music21 score
    score = stream.Score()
    
    # Create guitar part with tablature
    guitar_part = stream.Part()
    guitar_part.insert(0, clef.TabClef())
    
    # Add tempo and time signature
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
    
    print(f"Converting {len(tab_notes.data)} notes to tablature...")
    
    # Convert JAMS notes to music21 notes
    for obs in tab_notes.data:
        val = obs.value
        
        # Create a note
        # For tablature, we need to specify pitch based on string and fret
        # Standard guitar tuning: E2, A2, D3, G3, B3, E4 (strings 6-1)
        string_pitches = {
            6: 40,  # E2
            5: 45,  # A2
            4: 50,  # D3
            3: 55,  # G3
            2: 59,  # B3
            1: 64   # E4
        }
        
        string_num = val.get('string', 1)
        fret_num = val.get('fret', 0)
        
        # Calculate MIDI pitch
        midi_pitch = string_pitches.get(string_num, 64) + fret_num
        
        # Create note
        n = note.Note()
        n.pitch.midi = midi_pitch
        n.quarterLength = obs.duration * 4  # Convert seconds to quarter notes
        
        # Add tablature information
        n.editorial.stringNumber = string_num
        n.editorial.fretNumber = fret_num
        
        # Handle techniques (music21 uses articulations)
        techniques = val.get('techniques', [])
        if techniques:
            for tech in techniques:
                if tech:  # Skip None values
                    # Map techniques to music21 articulations
                    if tech == 'slide':
                        from music21 import articulations
                        n.articulations.append(articulations.Accent())
                    elif tech in ['hammer-on', 'pull-off']:
                        from music21 import articulations
                        n.articulations.append(articulations.Tenuto())
                    # Add more mappings as needed
        
        guitar_part.append(n)
    
    score.append(guitar_part)
    
    # Configure music21 to use LilyPond
    try:
        # Try to render with LilyPond
        score.write('lily.pdf', fp=output_pdf)
        print(f"✓ Created {output_pdf}")
        return output_pdf
    except Exception as e:
        print(f"Error rendering with LilyPond: {e}")
        print("\nTrying alternative method...")
        
        # Fallback: save as MusicXML
        musicxml_path = output_pdf.replace('.pdf', '.xml')
        score.write('musicxml', fp=musicxml_path)
        print(f"✓ Created MusicXML file: {musicxml_path}")
        print("  Open this file in MuseScore or other notation software to generate PDF")
        return musicxml_path


def jams_to_musicxml(jam, output_xml='output.xml'):
    """
    Convert JAMS to MusicXML (more reliable than direct PDF)
    Then open in MuseScore, Finale, or other software to export as PDF
    
    Args:
        jam: JAMS object with tab_note annotations
        output_xml: Path to output MusicXML file
    
    Returns:
        Path to created XML file
    """
    
    # Create music21 score
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
    
    print(f"Converting {len(tab_notes.data)} notes to MusicXML...")
    
    # Standard guitar tuning
    string_pitches = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
    
    # Group notes by time to handle chords
    notes_by_time = {}
    for obs in tab_notes.data:
        t = round(obs.time, 6)  # Round to avoid floating point issues
        if t not in notes_by_time:
            notes_by_time[t] = []
        notes_by_time[t].append(obs)
    
    # Sort by time
    for time_point in sorted(notes_by_time.keys()):
        obs_list = notes_by_time[time_point]
        
        if len(obs_list) == 1:
            # Single note
            obs = obs_list[0]
            val = obs.value
            
            string_num = val.get('string', 1)
            fret_num = val.get('fret', 0)
            midi_pitch = string_pitches.get(string_num, 64) + fret_num
            
            n = note.Note()
            n.pitch.midi = midi_pitch
            n.quarterLength = obs.duration * 4
            n.editorial.stringNumber = string_num
            n.editorial.fretNumber = fret_num
            
            guitar_part.insert(time_point, n)
        else:
            # Chord (multiple notes at same time)
            pitches = []
            for obs in obs_list:
                val = obs.value
                string_num = val.get('string', 1)
                fret_num = val.get('fret', 0)
                midi_pitch = string_pitches.get(string_num, 64) + fret_num
                pitches.append(midi_pitch)
            
            c = chord.Chord(pitches)
            c.quarterLength = obs_list[0].duration * 4
            guitar_part.insert(time_point, c)
    
    score.append(guitar_part)
    
    # Write to MusicXML
    score.write('musicxml', fp=output_xml)
    print(f"✓ Created {output_xml}")
    print("  Open this file in MuseScore, Finale, or Sibelius to export as PDF")
    
    return output_xml


# Example usage
if __name__ == "__main__":
    import os
    import random
    
    # Your existing functions (for testing)
    def encode_notes_for_test(jam):
        '''adds sample random expressive techniques, random strings, and frets for testing/tab output purposes'''
        new_ann = jams.Annotation(namespace='tab_note')
        tech_options = ["slide", "vibrato", "hammer-on", "pull-off", "bend", None]

        # iterate over existing note 
        note_ann = jam.annotations[0]  
        for obs in note_ann.data:
            pitch = obs.value  # the original pitch

            value = {
                "pitch": pitch,
                "string": random.randint(1, 6),
                "fret": random.randint(0, 12),
                "techniques": [random.choice(tech_options)] if random.random() < 0.5 else []
            }

            new_ann.append(time=obs.time, duration=obs.duration, value=value, confidence=obs.confidence)

        jam.annotations.append(new_ann)
        return jam
    
    # Test with sample data
    print("Testing JAMS to notation conversion...")
    
    # Create a simple test JAMS
    jam = jams.JAMS()
    note_ann = jams.Annotation(namespace='note_midi')  # Use proper namespace
    
    # Add some sample notes
    for i, pitch in enumerate([64, 65, 67, 69, 71, 72]):
        note_ann.append(
            time=i * 0.5,
            duration=0.4,
            value=pitch,
            confidence=1.0
        )
    
    jam.annotations.append(note_ann)
    jam = encode_notes_for_test(jam)
    
    # Try to convert
    print("\nAttempting conversion...")
    jams_to_musicxml(jam, 'test_output.xml')
