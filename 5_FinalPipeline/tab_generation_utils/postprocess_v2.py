import jams
import numpy as np
import os
import pretty_midi
from music21 import stream, note, tempo, meter, clef, articulations, expressions, spanner, interval, instrument, metadata
from music21 import note as m21_note  # Ensure we have access to Rest
from music21 import chord as m21_chord
import xml.etree.ElementTree as ET

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math


def post_process_musicxml(xml_path, technique_map=None, tuning=None, extend_notes_to_fill=True):
    """
    Post-process MusicXML to add proper Guitar Pro compatible elements
    and optionally extend notes to fill measures (removing trailing rests).
    
    Args:
        xml_path: Path to the MusicXML file
        technique_map: Dict mapping note indices to technique info
        tuning: List of MIDI note numbers for each string (low to high)
        extend_notes_to_fill: If True, removes trailing rests and extends the 
                              preceding note to fill the measure
    """
    from lxml import etree
    
    if technique_map is None:
        technique_map = {}
    
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]
    
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    # =========================================================================
    # EXTEND NOTES TO FILL MEASURES (remove trailing rests)
    # =========================================================================
    if extend_notes_to_fill:
        # MusicXML duration is in divisions - we need to find the divisions value
        divisions = 1
        attributes = root.find('.//attributes')
        if attributes is not None:
            div_elem = attributes.find('divisions')
            if div_elem is not None:
                divisions = int(div_elem.text)
        
        print(f"  XML divisions per quarter note: {divisions}")
        
        for measure in root.iter('measure'):
            # Get all note elements in this measure
            notes_in_measure = list(measure.findall('note'))
            
            if not notes_in_measure:
                continue
            
            # Find trailing rests and the last non-rest note
            trailing_rests = []
            last_note_elem = None
            
            # Walk backwards through notes to find trailing rests
            for note_elem in reversed(notes_in_measure):
                rest_elem = note_elem.find('rest')
                if rest_elem is not None:
                    trailing_rests.append(note_elem)
                else:
                    last_note_elem = note_elem
                    break
            
            # If we found trailing rests and a note to extend
            if trailing_rests and last_note_elem is not None:
                # Calculate total duration of trailing rests
                total_rest_duration = 0
                for rest_note in trailing_rests:
                    duration_elem = rest_note.find('duration')
                    if duration_elem is not None:
                        total_rest_duration += int(duration_elem.text)
                
                # Add this duration to the last note
                last_note_duration = last_note_elem.find('duration')
                if last_note_duration is not None:
                    original_duration = int(last_note_duration.text)
                    new_duration = original_duration + total_rest_duration
                    last_note_duration.text = str(new_duration)
                    
                    # Also need to update the <type> element to match new duration
                    # This is complex because we need to map duration to note type
                    # For now, just remove the type element and let readers infer from duration
                    type_elem = last_note_elem.find('type')
                    if type_elem is not None:
                        last_note_elem.remove(type_elem)
                    
                    # Remove dot elements too since duration changed
                    for dot in last_note_elem.findall('dot'):
                        last_note_elem.remove(dot)
                
                # Remove the trailing rests from the measure
                for rest_note in trailing_rests:
                    measure.remove(rest_note)
                
                print(f"  Measure {measure.get('number')}: Extended note by {total_rest_duration} divisions, removed {len(trailing_rests)} trailing rest(s)")
    
    # =========================================================================
    # PROCESS TECHNIQUES (hammer-on, pull-off, etc.)
    # =========================================================================
    
    # Collect all non-chord, non-rest notes in order
    all_notes = []
    for measure in root.iter('measure'):
        for elem in measure:
            if elem.tag == 'note':
                ## Change added by Shamak -- even chords can have techniques, and removing them disrupts the technique ID Count
                # chord_elem = elem.find('chord')
                rest_elem = elem.find('rest')
                # if chord_elem is None and rest_elem is None:
                if rest_elem is None:
                    all_notes.append(elem)
    
    print(f"  Found {len(all_notes)} notes for technique processing")
    print(f"  Technique map: {technique_map}")
    
    for note_idx, tech_info in technique_map.items():
        if note_idx >= len(all_notes) or note_idx < 1:
            print(f"  Warning: Invalid note index {note_idx}")
            continue
        
        technique = tech_info['technique']
        
        target_note = all_notes[note_idx]
        source_note = all_notes[note_idx - 1]
        
        source_pitch = source_note.find('pitch')
        target_pitch = target_note.find('pitch')
        
        if source_pitch is None or target_pitch is None:
            print(f"  Warning: Note {note_idx} or {note_idx-1} is a rest, skipping technique")
            continue
        
        print(f"  Adding {technique} from note {note_idx-1} to note {note_idx}")
        
        if technique == 'hammer-on':
            tech_elem_name = 'hammer-on'
            tech_text = 'H'
        else:
            tech_elem_name = 'pull-off'
            tech_text = 'P'
        
        # SOURCE NOTE
        source_notations = source_note.find('notations')
        if source_notations is None:
            source_notations = etree.SubElement(source_note, 'notations')
        
        source_technical = source_notations.find('technical')
        if source_technical is None:
            source_technical = etree.SubElement(source_notations, 'technical')
        
        for elem in list(source_technical):
            if elem.tag in ('hammer-on', 'pull-off'):
                source_technical.remove(elem)
        
        tech_start = etree.Element(tech_elem_name)
        tech_start.set('type', 'start')
        tech_start.set('number', '1')
        tech_start.text = tech_text
        source_technical.insert(0, tech_start)
        
        for elem in list(source_notations):
            if elem.tag == 'slur':
                source_notations.remove(elem)
        
        slur_start = etree.SubElement(source_notations, 'slur')
        slur_start.set('type', 'start')
        
        # TARGET NOTE
        target_notations = target_note.find('notations')
        if target_notations is None:
            target_notations = etree.SubElement(target_note, 'notations')
        
        target_technical = target_notations.find('technical')
        if target_technical is None:
            target_technical = etree.SubElement(target_notations, 'technical')
        
        for elem in list(target_technical):
            if elem.tag in ('hammer-on', 'pull-off'):
                target_technical.remove(elem)
        
        tech_stop = etree.Element(tech_elem_name)
        tech_stop.set('type', 'stop')
        tech_stop.set('number', '1')
        target_technical.insert(0, tech_stop)
        
        for elem in list(target_notations):
            if elem.tag == 'slur':
                target_notations.remove(elem)
        
        slur_stop = etree.SubElement(target_notations, 'slur')
        slur_stop.set('type', 'stop')

    # =========================================================================
    # PROCESS OTHER TECHNIQUES (vibrato, harmonic)
    # =========================================================================
    for measure in root.iter('measure'):
        elements = list(measure)
        measure[:] = []
        
        for elem in elements:
            if elem.tag == 'note':
                notations = elem.find('notations')
                has_vibrato = False
                
                if notations is not None:
                    technical = notations.find('technical')
                    if technical is not None:
                        harmonic = technical.find('harmonic')
                        if harmonic is not None:
                            direction = etree.Element('direction')
                            direction.set('placement', 'above')
                            direction_type = etree.SubElement(direction, 'direction-type')
                            words = etree.SubElement(direction_type, 'words')
                            words.set('font-size', '10')
                            words.set('font-style', 'italic')
                            words.set('halign', 'center')
                            words.set('valign', 'top')
                            words.text = 'Harm.'
                            measure.append(direction)
                        
                        fingering = technical.find('fingering')
                        if fingering is not None and fingering.text == 'vibrato':
                            technical.remove(fingering)
                            has_vibrato = True
                
                measure.append(elem)
                
                if has_vibrato:
                    vibrato_pi = etree.ProcessingInstruction('GP', '<root><vibrato type="Slight"/></root>')
                    elem.append(vibrato_pi)
            else:
                measure.append(elem)
    
    # =========================================================================
    # SET GUITAR INSTRUMENT
    # =========================================================================
    for score_part in root.iter('score-part'):
        part_id = score_part.get('id')
        
        part_name = score_part.find('part-name')
        if part_name is None:
            part_name = etree.SubElement(score_part, 'part-name')
        part_name.text = 'Classical Guitar'
        
        part_abbrev = score_part.find('part-abbreviation')
        if part_abbrev is None:
            part_abbrev = etree.SubElement(score_part, 'part-abbreviation')
        part_abbrev.text = 'Guit.'
        
        score_inst = score_part.find('score-instrument')
        if score_inst is None:
            score_inst = etree.SubElement(score_part, 'score-instrument')
            score_inst.set('id', f'{part_id}-I1')
        
        inst_name = score_inst.find('instrument-name')
        if inst_name is None:
            inst_name = etree.SubElement(score_inst, 'instrument-name')
        inst_name.text = 'Classical Guitar'
        
        inst_sound = score_inst.find('instrument-sound')
        if inst_sound is None:
            inst_sound = etree.SubElement(score_inst, 'instrument-sound')
        inst_sound.text = 'pluck.guitar.nylon-string'
    
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"✓ Post-processed {xml_path}")



def apply_techniques_to_note(n, techniques, note_index, technique_map):
    """
    Apply techniques to a music21 Note and update the technique map for post-processing.
    
    Args:
        n: music21 Note object to modify
        techniques: List of technique strings from the note event
        note_index: Current note index for technique mapping
        technique_map: Dict to store technique info for post-processing

    Legend from Peter's Predictions
    'harmonics'     
    'palm_muting' # not used
    'slide'
    'hammer_on_pull_off'
    'picking'
    'bend'
    'kick_drum'
    'vibrato'
    'snare_drum' (?)
    """
    for tech in techniques:
        if not tech:
            continue
        
        if tech in ['hammer-on', 'hammer_on']:
            print(f"  hammer-on detected at note {note_index}")
            technique_map[note_index] = {
                'technique': 'hammer-on',
                'is_target': True
            }
        
        elif tech in ['pull-off', 'pull_off']:
            print(f"  pull-off detected at note {note_index}")
            technique_map[note_index] = {
                'technique': 'pull-off',
                'is_target': True
            }
        ### Added new class from Peter's classification (above two won't really happen)
        elif tech in ['hammer_on_pull_off']:
            print(f"  hammer_on_pull_off detected at note {note_index}")
            technique_map[note_index] = {
                'technique': 'hammer_on_pull_off',
                'is_target': True
            }
        
        elif tech == 'bend':
            print(f"  bend detected at note {note_index}")
            bend_amount = 0.5  # Hard coded for now
            bend = articulations.FretBend(
                bendAlter=interval.Interval(bend_amount)
            )
            n.articulations.append(bend)
            n.expressions.append(expressions.TextExpression('B'))
        
        elif tech == 'slide':
            print(f"  slide detected at note {note_index}")
            n.expressions.append(expressions.TextExpression('/'))
        
        elif tech == 'vibrato':
            print(f"  vibrato detected at note {note_index}")
            n.articulations.append(articulations.Fingering('vibrato'))
        
        elif tech == 'harmonics': 
            print(f"  harmonic detected at note {note_index}")
            n.articulations.append(articulations.Harmonic())

    return technique_map