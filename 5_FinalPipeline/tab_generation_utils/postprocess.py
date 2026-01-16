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


def post_process_musicxml(xml_path, technique_map=None, tuning=None):
    """
    Post-process MusicXML to add proper Guitar Pro compatible elements.
    
    Args:
        xml_path: Path to the MusicXML file
        technique_map: Dict mapping note indices to technique info
                      {note_index: {'technique': 'hammer-on'|'pull-off', 'is_target': bool}}
        tuning: List of MIDI note numbers for each string (low to high), 
                defaults to standard tuning [40, 45, 50, 55, 59, 64] (E2-E4)
    """
    from lxml import etree # type: ignore
    
    if technique_map is None:
        technique_map = {}
    
    # Standard guitar tuning (string 6 to string 1): E2, A2, D3, G3, B3, E4
    # MIDI note numbers: 40, 45, 50, 55, 59, 64
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]  # Low E to high E
    
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    # Collect all non-chord notes in order across all measures
    all_notes = []
    for measure in root.iter('measure'):
        for elem in measure:
            if elem.tag == 'note':
                # Skip chord notes (notes that are part of a chord have <chord/> element)

                ## Change added by Shamak -- even chords can have techniques, and removing them disrupts the technique ID Count
                # chord_elem = elem.find('chord')
                rest_elem = elem.find('rest')
                # if chord_elem is None and rest_elem is None:
                if rest_elem is None:
                    all_notes.append(elem)
    
    print(f"  Found {len(all_notes)} notes for technique processing")
    print(f"  Technique map: {technique_map}")
    
    # Process techniques - add hammer-on/pull-off and slur elements
    for note_idx, tech_info in technique_map.items():
        if note_idx >= len(all_notes) or note_idx < 1:
            print(f"  Warning: Invalid note index {note_idx}")
            continue
        
        technique = tech_info['technique']
        
        # The target note (where the technique ends - the note you hammer ON TO or pull OFF TO)
        target_note = all_notes[note_idx]
        # The source note (where the technique starts - the note you hammer FROM or pull FROM)
        source_note = all_notes[note_idx - 1]
        
        # Verify both notes have pitch (not rests)
        source_pitch = source_note.find('pitch')
        target_pitch = target_note.find('pitch')
        
        if source_pitch is None or target_pitch is None:
            print(f"  Warning: Note {note_idx} or {note_idx-1} is a rest, skipping technique")
            continue
        
        print(f"  Adding {technique} from note {note_idx-1} to note {note_idx}")
        
        # Determine element name and text based on technique
        if technique == 'hammer-on':
            tech_elem_name = 'hammer-on'
            tech_text = 'H'
        else:  # pull-off
            tech_elem_name = 'pull-off'
            tech_text = 'P'
        
        # === SOURCE NOTE (technique start) ===
        source_notations = source_note.find('notations')
        if source_notations is None:
            source_notations = etree.SubElement(source_note, 'notations')
        
        source_technical = source_notations.find('technical')
        if source_technical is None:
            source_technical = etree.SubElement(source_notations, 'technical')
        
        # Remove any existing hammer-on/pull-off elements (but NOT string/fret!)
        for elem in list(source_technical):
            if elem.tag in ('hammer-on', 'pull-off'):
                source_technical.remove(elem)
        
        # Insert technique element at the beginning
        tech_start = etree.Element(tech_elem_name)
        tech_start.set('type', 'start')
        tech_start.set('number', '1')
        tech_start.text = tech_text
        source_technical.insert(0, tech_start)
        
        # Add slur start
        # Remove any existing slur first
        for elem in list(source_notations):
            if elem.tag == 'slur':
                source_notations.remove(elem)
        
        slur_start = etree.SubElement(source_notations, 'slur')
        slur_start.set('type', 'start')
        
        # === TARGET NOTE (technique stop) ===
        target_notations = target_note.find('notations')
        if target_notations is None:
            target_notations = etree.SubElement(target_note, 'notations')
        
        target_technical = target_notations.find('technical')
        if target_technical is None:
            target_technical = etree.SubElement(target_notations, 'technical')
        
        # Remove any existing hammer-on/pull-off elements (but NOT string/fret!)
        for elem in list(target_technical):
            if elem.tag in ('hammer-on', 'pull-off'):
                target_technical.remove(elem)
        
        # Insert technique element at the beginning
        tech_stop = etree.Element(tech_elem_name)
        tech_stop.set('type', 'stop')
        tech_stop.set('number', '1')
        target_technical.insert(0, tech_stop)
        
        # Add slur stop (no number attribute)
        for elem in list(target_notations):
            if elem.tag == 'slur':
                target_notations.remove(elem)
        
        slur_stop = etree.SubElement(target_notations, 'slur')
        slur_stop.set('type', 'stop')

    # Process other techniques (vibrato, harmonic, etc.)
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
                        # Check for harmonic
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
                        
                        # Check for vibrato marker
                        fingering = technical.find('fingering')
                        if fingering is not None and fingering.text == 'vibrato':
                            technical.remove(fingering)
                            has_vibrato = True
                
                measure.append(elem)
                
                # Add Guitar Pro vibrato processing instruction after the note
                if has_vibrato:
                    vibrato_pi = etree.ProcessingInstruction('GP', '<root><vibrato type="Slight"/></root>')
                    elem.append(vibrato_pi)
            else:
                measure.append(elem)
    
    # Set guitar instrument
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