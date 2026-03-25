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