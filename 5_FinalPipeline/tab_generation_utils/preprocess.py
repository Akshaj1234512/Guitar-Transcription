### Methods by Shamak and Samhita to preprocess jams to add andrea's and peter's predictions

import jams
import pretty_midi
import tab_generation_utils.choose_best_position_colin as position

STANDARD_TUNING = {
    6: 40,  # E2 (low E)
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64   # E4 (high e)
}


# change added 01/02: to ensure does not take the same prediction from andrea twice
def midi_to_jams_with_tablature_from_andreas(midi_path, string_fret_time_tuples, bpm = 120, tuning=STANDARD_TUNING,time_tolerance=0.2, capo = 0):
    '''function mostly borrowed from colin's technique tabs, with the addition of aligning andrea's predictions with MIDI'''
    # load midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    guitar_notes = pm.instruments[0].notes

    tempo_bpm = bpm

    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]  # Use first time signature
        time_sig_numerator = ts.numerator
        time_sig_denominator = ts.denominator
        print(f"Time signature from MIDI: {time_sig_numerator}/{time_sig_denominator}")
    else:
        print("No time signature found, using default 4/4 instead")
        time_sig_numerator = 4  # Default 4/4
        time_sig_denominator = 4

    time_signature = f"{time_sig_numerator}/{time_sig_denominator}"
    print(f"Extracted from MIDI - Tempo: {tempo_bpm:.1f} BPM, Time Signature: {time_signature}")
    
    # Create JAMS
    jam = jams.JAMS()
    
    # Store tempo and time signature in JAMS file metadata
    jam.file_metadata.duration = pm.get_end_time()
    jam.sandbox.tempo_bpm = tempo_bpm
    jam.sandbox.time_signature = time_signature
    jam.sandbox.time_sig_numerator = time_sig_numerator
    jam.sandbox.time_sig_denominator = time_sig_denominator

    # untested - 01/21/26 - Shamak
    jam.sandbox.tuning = tuning
    jam.sandbox.capo = capo
    jam.sandbox.update()

    #----------------------------------------------------------------------------------#

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

    #----------------------------------------------------------------------------------#
    
    andrea_predictions = []
    for idx, (string, fret, _, onset_sec) in enumerate(string_fret_time_tuples):
        andrea_predictions.append((onset_sec, int(string), int(fret), idx))

    #----------------------------------------------------------------------------------#
    # Comment out when not testing
    
    print(f"MIDI has {len(guitar_notes)} notes")
    print(f"Received {len(string_fret_time_tuples)} string/fret tuples")

    print("\n=== TIMING DEBUG ===")
    print("First 10 MIDI notes vs closest Andrea predictions:")
    found_offsets = False

    for i, n in enumerate(guitar_notes):
        # Find the single closest prediction in time
        closest_pred = min(andrea_predictions, key=lambda x: abs(x[0] - n.start))
        closest_time = closest_pred[0]
        
        time_diff = abs(n.start - closest_time)
        
        # Check for ANY offset greater than 0
        if time_diff > 0.0:
            found_offsets = True
            time_diff_ms = time_diff * 1000
            print(f"  Note {i} (Pitch {n.pitch}): Offset detected!")
            print(f"    MIDI: {n.start:.4f}s | Andrea: {closest_time:.4f}s")
            print(f"    Diff: {time_diff_ms:.2f}ms")

    if not found_offsets:
        print("  Perfect alignment! All notes match at 0.0s.")
    print("===================\n")
    
    #----------------------------------------------------------------------------------#

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
            if idx in used_predictions:  # ← I ADDED THIS CHECK to make sure predictions are never reused
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
            string, fret = position.choose_best_position(n.pitch, previous_position, tuning=tuning)
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
def midi_to_jams_with_tablature_from_andreas_sequential(midi_path, string_fret_time_tuples, bpm=120, tuning=STANDARD_TUNING,time_tolerance=0.2):
    
    #TESTING
    
    zero_count = sum(1 for (_, fret, *_) in string_fret_time_tuples if int(fret) == 0)
    print(f"Found {zero_count} frets with 0 out of {len(string_fret_time_tuples)} notes")

    #----------------------------------------------------------------------------------#
    
    # load midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    guitar_notes = pm.instruments[0].notes

    tempo_bpm = bpm

    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]  # Use first time signature
        time_sig_numerator = ts.numerator
        time_sig_denominator = ts.denominator
        print(f"Time signature from MIDI: {time_sig_numerator}/{time_sig_denominator}")
    else:
        print("No time signature found, using default 4/4 instead")
        time_sig_numerator = 4  # Default 4/4
        time_sig_denominator = 4

    time_signature = f"{time_sig_numerator}/{time_sig_denominator}"
    print(f"Extracted from MIDI - Tempo: {tempo_bpm:.1f} BPM, Time Signature: {time_signature}")

    jam = jams.JAMS()

    # Store tempo and time signature in JAMS file metadata
    jam.file_metadata.duration = pm.get_end_time()
    jam.sandbox.tempo_bpm = tempo_bpm
    jam.sandbox.time_signature = time_signature
    jam.sandbox.time_sig_numerator = time_sig_numerator
    jam.sandbox.time_sig_denominator = time_sig_denominator
    #untested - 01/21/2026 - Shamak
    

    #----------------------------------------------------------------------------------#

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

    #----------------------------------------------------------------------------------#
    
    print(f"MIDI has {len(guitar_notes)} notes")
    print(f"Received {len(string_fret_time_tuples)} string/fret tuples")

    if len(guitar_notes) != len(string_fret_time_tuples):
        print(f"WARNING: Length mismatch! MIDI={len(guitar_notes)}, Andrea={len(string_fret_time_tuples)}")
        return midi_to_jams_with_tablature_from_andreas(midi_path, string_fret_time_tuples, tuning,time_tolerance)
    
    #----------------------------------------------------------------------------------#
    
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

#----------------------------------------------------------------------------------#

# PETER
### more lenient matching, added Jan 5. TODO: add this to technique_tabs.py
### TODO: Ask Samhita to check this
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

    technique_times = []
    for tech, onset_sec, duration_ms in exp_onset_dur_tuples:
        if tech is not None and tech != "Normal": 
            technique_times.append((onset_sec, tech))
    
    # MODIFY existing notes (don't create new ones)
    
    TIME_TOLERANCE = 0.2 #TODO: make this variable

    for obs in tab_ann.data:
        note_time = obs.time

        best_match = None
        best_distance = float('inf')

        for tech_time, tech in technique_times:
            distance = abs(note_time - tech_time)
            if distance < TIME_TOLERANCE and distance < best_distance:
                best_match = tech
                best_distance = distance
        
        if best_match:
            obs.value['techniques'] = [best_match]
            technique_count += 1
        else:
            obs.value['techniques'] = []
    
    print(f"\n✓ Modified {len(tab_ann.data)} notes")
    print(f"✓ Added {technique_count} techniques ({technique_count/len(tab_ann.data)*100:.1f}%)")
    
    return jam