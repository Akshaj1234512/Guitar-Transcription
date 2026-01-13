#!/usr/bin/env python3
"""
Post-processing algorithm for tablature predictions on a single MIDI file.

This script loads a model, takes a single MIDI file as input, converts it
to encoder tokens, generates a tablature prediction, applies post-processing,
and prints the results.
"""

import os
import sys
import torch
import numpy as np
import mido
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from src.fret_t5.tokenization import MidiTabTokenizerV3, DEFAULT_CONDITIONING_TUNINGS, TokenizerConfig, _Vocabulary, Sequence, TokenizedTrack
from transformers import T5ForConditionalGeneration
import json


from src.fret_t5.postprocess import (
    parse_tab_token,
    parse_time_shift_token,
    parse_note_on_token,
    parse_capo_token,
    parse_tuning_token,
    tuning_to_open_strings,
    tab_to_midi_pitch,
    fret_stretch,
    find_alternative_fingerings,
    select_best_fingering,
    extract_input_notes,
    extract_output_tabs,
    align_sequences_with_window,
)

# Standard tuning (high E to low E)
STANDARD_TUNING = (64, 59, 55, 50, 45, 40)
OPEN_STRINGS = {
    1: 64,  # High E (E4)
    2: 59,  # B  (B3)
    3: 55,  # G  (G3)
    4: 50,  # D  (D3)
    5: 45,  # A  (A2)
    6: 40   # Low E (E2)
}

def run_tab_generation(midi_path: str, bpm):
    """
    Main function to load model, process MIDI, and return final tab list.
    """

    CAPO = 0
    TUNING = STANDARD_TUNING  
    
    checkpoint_path = "/data/andreaguz/fret_t5/checkpoints_conditioning_scratch_retrain/checkpoint-642982"

    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("Please update 'checkpoint_path' to a valid model location.")
        return None  

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = MidiTabTokenizerV3.load("/data/andreaguz/fret_t5_clean/fret_t5/universal_tokenizer")
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS
    )

    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")
    final_tab = process_midi_file_chunking(midi_path, tokenizer, model, capo=CAPO, tuning=TUNING, bpm=bpm)
    
    return final_tab

def extract_conditioning_from_encoder(encoder_tokens: List[str]) -> Tuple[int, Tuple[int, ...]]:
    """
    Extract capo and tuning from encoder tokens.

    Returns:
        (capo, tuning) tuple with defaults if not found
    """
    capo = 0
    tuning = STANDARD_TUNING

    # Conditioning tokens are at the start
    for token in encoder_tokens[:5]:
        if token.startswith('CAPO<'):
            parsed_capo = parse_capo_token(token)
            if parsed_capo is not None:
                capo = parsed_capo
        elif token.startswith('TUNING<'):
            parsed_tuning = parse_tuning_token(token)
            if parsed_tuning is not None:
                tuning = parsed_tuning

    return capo, tuning


def postprocess_predictions(
    encoder_tokens: List[str],
    decoder_tokens: List[str],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    pitch_window: int = 5,
    alignment_window: int = 5,
    debug: bool = False
) -> Tuple[List[str], Dict[str, int]]:
    """
    Apply post-processing to correct pitch and time shift errors.
    (Function body kept unchanged)
    ...
    """
    # Extract notes and tabs
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)

    stats = {
        'pitch_corrections': 0,
        'time_shift_corrections': 0,
        'pitch_too_far': 0,
        'unaligned_outputs': 0,
        'input_length': len(input_notes),
        'output_length': len(output_tabs)
    }

    if len(output_tabs) == 0:
        return decoder_tokens, stats

    # Align sequences
    alignments = align_sequences_with_window(input_notes, output_tabs, alignment_window)

    if debug and len(input_notes) != len(output_tabs):
        print(f"  DEBUG: Length mismatch - input={len(input_notes)}, output={len(output_tabs)}")
        print(f"  DEBUG: Aligned {len([a for a in alignments if a[0] is not None])}/{len(output_tabs)} outputs")

    # Apply corrections
    corrected_tokens = []

    for input_idx, output_idx in alignments:
        out_string, out_fret, out_time_shift = output_tabs[output_idx]

        # If no aligned input, keep output as-is
        if input_idx is None:
            stats['unaligned_outputs'] += 1
            corrected_tokens.append(f"TAB<{out_string},{out_fret}>")
            corrected_tokens.append(f"TIME_SHIFT<{out_time_shift}>")
            continue

        input_pitch, input_time_shift = input_notes[input_idx]

        # Calculate predicted MIDI pitch
        predicted_pitch = tab_to_midi_pitch(out_string, out_fret, capo, tuning)
        pitch_diff = abs(input_pitch - predicted_pitch)

        # Determine correct fingering
        pitch_corrected = False
        if pitch_diff == 0:
            # Pitch already correct, keep original fingering
            corrected_string, corrected_fret = out_string, out_fret
        elif pitch_diff <= pitch_window:
            # Pitch within window, find best alternative fingering
            alternatives = find_alternative_fingerings(input_pitch, capo, tuning)
            if alternatives:
                corrected_string, corrected_fret = select_best_fingering(
                    alternatives, out_string, out_fret
                )
                if (corrected_string, corrected_fret) != (out_string, out_fret):
                    stats['pitch_corrections'] += 1
                    pitch_corrected = True
                    if debug:
                        print(f"  DEBUG: Corrected pitch at output_idx={output_idx}: "
                              f"TAB<{out_string},{out_fret}> (pitch={predicted_pitch}) -> "
                              f"TAB<{corrected_string},{corrected_fret}> (pitch={input_pitch})")
            else:
                # No valid alternative, keep original
                corrected_string, corrected_fret = out_string, out_fret
                if debug:
                    print(f"  DEBUG: No alternatives found for pitch {input_pitch}")
        else:
            # Pitch difference too large, keep original
            corrected_string, corrected_fret = out_string, out_fret
            stats['pitch_too_far'] += 1
            if debug:
                print(f"  DEBUG: Pitch difference too large at output_idx={output_idx}: "
                      f"{pitch_diff} MIDI notes (input={input_pitch}, pred={predicted_pitch})")

        # Correct time shift if different
        if input_time_shift != out_time_shift:
            stats['time_shift_corrections'] += 1
            if debug:
                print(f"  DEBUG: Corrected time shift at output_idx={output_idx}: "
                      f"{out_time_shift}ms -> {input_time_shift}ms")

        corrected_time_shift = input_time_shift

        # Add corrected tokens
        corrected_tokens.append(f"TAB<{corrected_string},{corrected_fret}>")
        corrected_tokens.append(f"TIME_SHIFT<{corrected_time_shift}>")

    # Add EOS token if present in original
    if decoder_tokens and decoder_tokens[-1] == "<eos>":
        corrected_tokens.append("<eos>")

    return corrected_tokens, stats


# NOTE: compute_accuracy_metrics is not fully useful without ground truth tablature
# But we keep it to structure the output comparison.
def compute_accuracy_metrics(
    encoder_tokens: List[str],
    decoder_tokens: List[str],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    ground_truth_tokens: List[str] = None
) -> Dict[str, float]:
    """
    Compute pitch, time shift, and tab accuracy metrics.
    (Function body kept unchanged)
    ...
    """
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)

    if len(input_notes) == 0:
        return {
            'pitch_accuracy': 0.0,
            'time_shift_accuracy': 0.0,
            'tab_accuracy': 0.0,
            'total_notes': 0
        }

    # Handle length mismatch
    min_len = min(len(input_notes), len(output_tabs))

    if min_len == 0:
        return {
            'pitch_accuracy': 0.0,
            'time_shift_accuracy': 0.0,
            'tab_accuracy': 0.0,
            'total_notes': 0,
            'input_length': len(input_notes),
            'output_length': len(output_tabs)
        }

    pitch_matches = 0
    time_shift_matches = 0
    tab_matches = 0

    # Extract ground truth tabs if provided
    ground_truth_tabs = None
    if ground_truth_tokens:
        ground_truth_tabs = extract_output_tabs(ground_truth_tokens)

    for i in range(min_len):
        input_pitch, input_time_shift = input_notes[i]
        out_string, out_fret, out_time_shift = output_tabs[i]

        predicted_pitch = tab_to_midi_pitch(out_string, out_fret, capo, tuning)

        if input_pitch == predicted_pitch:
            pitch_matches += 1

        if input_time_shift == out_time_shift:
            time_shift_matches += 1

        # Check tab accuracy if ground truth provided
        if ground_truth_tabs and i < len(ground_truth_tabs):
            gt_string, gt_fret, _ = ground_truth_tabs[i]
            if out_string == gt_string and out_fret == gt_fret:
                tab_matches += 1

    metrics = {
        'pitch_accuracy': (pitch_matches / min_len) * 100,
        'time_shift_accuracy': (time_shift_matches / min_len) * 100,
        'total_notes': min_len,
        'input_length': len(input_notes),
        'output_length': len(output_tabs)
    }

    # Add tab accuracy if ground truth was provided
    if ground_truth_tabs:
        metrics['tab_accuracy'] = (tab_matches / min_len) * 100
    else:
        metrics['tab_accuracy'] = 0.0

    return metrics


def midi_to_tab_events(midi_path: Path, bpm: float) -> List[Dict]:
    """
    Load a MIDI file and extract note events.

    Returns:
        List of event dicts with midi_pitch, duration (in seconds)
    """
    print(f"  Loading MIDI file: {midi_path.name}")
    try:
        midi = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return []

    # Map pitch to (start_time, track_index)
    active_notes = {}
    note_events = []
    current_time = 0.0

    # Use ticks per beat for timing calculations
    ticks_per_beat = midi.ticks_per_beat
    tempo = bpm
    #tempo = mido.bpm2tempo(120)  # Default to 120 BPM (500000 microseconds/beat)

    for track in midi.tracks:
        for msg in track:
            # Update current time based on the message's time delta
            current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)

            if msg.type == 'set_tempo':
                tempo = msg.tempo

            if msg.type == 'note_on' and msg.velocity > 0:
                pitch = msg.note
                # Store start time and track index
                active_notes[pitch] = (current_time, track.index)
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                pitch = msg.note
                if pitch in active_notes:
                    start_time, track_idx = active_notes.pop(pitch)
                    duration = current_time - start_time
                    
                    if duration > 0.0:
                        note_events.append({
                            'time': start_time,
                            'duration': duration,
                            'midi_pitch': pitch,
                        })

    # Sort by time
    note_events.sort(key=lambda x: x['time'])
    
    print(f"  Extracted {len(note_events)} note events.")
    return note_events


def create_encoder_tokens_from_midi(
    note_events: List[Dict],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING
) -> List[str]:
    """
    Create encoder tokens (MIDI input) from note events.

    Returns:
        List of encoder tokens
    """
    # Convert tuning tuple to string for the token
    tuning_str = ",".join(map(str, tuning))

    encoder_tokens = [f"CAPO<{capo}>", f"TUNING<{tuning_str}>"]

    for event in note_events:
        midi_pitch = event['midi_pitch']
        duration_ms = int(round(event['duration'] * 1000))
        duration_ms = min(duration_ms, 5000)  # Cap at 5 seconds
        duration_ms = int(round(duration_ms / 100)) * 100  # Quantize to 100ms

        # Ensure minimum duration
        if duration_ms == 0:
            duration_ms = 100

        # Encoder tokens
        encoder_tokens.extend([
            f"NOTE_ON<{midi_pitch}>",
            f"TIME_SHIFT<{duration_ms}>",
            f"NOTE_OFF<{midi_pitch}>"
        ])

    return encoder_tokens


def process_midi_file(
    midi_path: str,
    tokenizer: MidiTabTokenizerV3,
    model: T5ForConditionalGeneration,
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    bpm: float = 120.0
):
    """
    Process a single MIDI file through the model and post-processing.
    """
    midi_file = Path(midi_path)
    if not midi_file.exists():
        print(f"ERROR: MIDI file not found at {midi_path}")
        return

    # 1. Load MIDI and create encoder tokens
    note_events = midi_to_tab_events(midi_file, bpm)

    if not note_events:
        print("Skipping (no valid note events found)")
        return

    # Use the specified capo/tuning for the prediction
    encoder_tokens = create_encoder_tokens_from_midi(note_events, capo, tuning)
    
    # 2. Encode input
    encoder_ids = tokenizer.encode_encoder_tokens_shared(encoder_tokens)
    input_ids = torch.tensor(encoder_ids, dtype=torch.long).unsqueeze(0)

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # 3. Generate prediction
    print("  Generating tablature prediction...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=1,
            do_sample=False,
            eos_token_id=tokenizer.shared_token_to_id["<eos>"],
            pad_token_id=tokenizer.shared_token_to_id["<pad>"],
        )

    # 4. Decode prediction
    pred_ids = outputs[0].cpu().tolist()
    pred_tokens = tokenizer.shared_to_decoder_tokens(pred_ids)
    
    # 5. Compute original metrics (only pitch/time accuracy vs input, Tab acc is 0.0)
    original_metrics = compute_accuracy_metrics(
        encoder_tokens, pred_tokens, capo, tuning,
        ground_truth_tokens=None
    )
    
    # 6. Apply post-processing
    print("  Applying post-processing for pitch and time shift correction...")
    # Check if we should enable debug mode 
    # (always enable debug for a single file or if issues are detected)
    has_errors = (original_metrics['pitch_accuracy'] < 100 or
                  original_metrics['time_shift_accuracy'] < 100)

    postprocessed_tokens, correction_stats = postprocess_predictions(
        encoder_tokens, pred_tokens, capo, tuning, pitch_window=5, alignment_window=5, debug=has_errors
    )

    # 7. Compute post-processed metrics (only pitch/time accuracy vs input)
    postprocessed_metrics = compute_accuracy_metrics(
        encoder_tokens, postprocessed_tokens, capo, tuning,
        ground_truth_tokens=None
    )

    # 8. Display results
    print("\n" + "=" * 80)
    print(f"RESULTS FOR: {midi_file.name}")
    print("=" * 80)

    # Convert corrected tokens to human-readable tablature format
    postprocessed_tabs = [
        t for t in postprocessed_tokens if t.startswith("TAB<")
    ]
    postprocessed_shifts = [
        t for t in postprocessed_tokens if t.startswith("TIME_SHIFT<")
    ]
    
    print("\n### Final Corrected Tablature Tokens ###")
    print("----------------------------------------")
    
    # Print the token sequence in pairs
    for i in range(len(postprocessed_tabs)):
        tab_token = postprocessed_tabs[i]
        shift_token = postprocessed_shifts[i] if i < len(postprocessed_shifts) else "TIME_SHIFT<ERROR>"
        print(f"{tab_token} {shift_token}")

    print("\n### Pitch/Time Shift Accuracy vs Input MIDI ###")
    print("-----------------------------------------------")
    print(f"  Original Prediction:       Pitch={original_metrics['pitch_accuracy']:.1f}%, TimeShift={original_metrics['time_shift_accuracy']:.1f}%")
    print(f"  Post-processed Prediction: Pitch={postprocessed_metrics['pitch_accuracy']:.1f}%, TimeShift={postprocessed_metrics['time_shift_accuracy']:.1f}%")

    print("\n### Correction Summary ###")
    print("--------------------------")
    print(f"  Pitch Corrections:     {correction_stats['pitch_corrections']}")
    print(f"  Time Shift Corrections:{correction_stats['time_shift_corrections']}")
    print(f"  Pitch Too Far:         {correction_stats['pitch_too_far']}")
    print(f"  Unaligned Outputs:     {correction_stats['unaligned_outputs']}")
    print("=" * 80)


    # get tokemns from andrea:

    extracted_tab_and_shift_list = []
    
    for i in range(0, len(postprocessed_tokens), 2):
        tab_token = postprocessed_tokens[i]

        if not tab_token.startswith("TAB<"):
            continue 

        string, fret = tab_token[4:-1].split(',')
        time_shift_ms = 0 # Initialize default to 0

        if i + 1 < len(postprocessed_tokens):
            shift_token = postprocessed_tokens[i+1]
            
            if shift_token.startswith('TIME_SHIFT<'):
                try:
                    # Extract the content between '<' and '>'
                    shift_value_str = shift_token.split('<')[-1].replace('>', '')
                    
                    # FINAL CHECK: Trim leading/trailing whitespace which is often the cause of failure
                    time_shift_ms = int(shift_value_str.strip()) 
                except ValueError as e:
                    # Catch if the string cannot be converted to an integer
                    print(f"Extraction ERROR on token: {shift_token}. Value defaulted to 0. Error: {e}")
                    time_shift_ms = 0
            
        extracted_tab_and_shift_list.append((string, fret, time_shift_ms))
        
    return extracted_tab_and_shift_list

    # --- Print Debug/Summary Info (Keep this for console logging) ---
    print("\n" + "=" * 80)
    print(f"RESULTS FOR: {midi_file.name}")
    print("=" * 80)
    print(f"  Post-processed Prediction: Pitch={postprocessed_metrics['pitch_accuracy']:.1f}%, TimeShift={postprocessed_metrics['time_shift_accuracy']:.1f}%")
    print(f"  Total Extracted Tabs: {len(extracted_tabs_list)}")
    print("=" * 80)
    # --- End Print Debug/Summary Info ---
    
    # RETURN THE CLEAN LIST OF TABS
    return extracted_tabs_list

### process in chunks to make sure that the full midi gets a predicted string fret
def process_midi_file_v2(
    midi_path: str,
    tokenizer: MidiTabTokenizerV3,
    model: T5ForConditionalGeneration,
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    bpm: float = 120.0
):
    """
    Process a single MIDI file through the model and post-processing.
    """
    midi_file = Path(midi_path)
    if not midi_file.exists():
        print(f"ERROR: MIDI file not found at {midi_path}")
        return

    # 1. Load MIDI and create encoder tokens
    note_events = midi_to_tab_events(midi_file, bpm)

    if not note_events:
        print("Skipping (no valid note events found)")
        return

    # Use the specified capo/tuning for the prediction
    encoder_tokens = create_encoder_tokens_from_midi(note_events, capo, tuning)

    conditioning_tokens = encoder_tokens[:2]
    note_tokens = encoder_tokens[2:]

    NOTES_PER_CHUNK = 150
    TOKENS_PER_NOTE = 3

    all_tabs = []
    num_chunks = (len(note_tokens) + NOTES_PER_CHUNK * TOKENS_PER_NOTE - 1) // (NOTES_PER_CHUNK * TOKENS_PER_NOTE)

    for chunk in range(num_chunks):
        start = chunk * NOTES_PER_CHUNK * TOKENS_PER_NOTE
        end = start + NOTES_PER_CHUNK * TOKENS_PER_NOTE
        chunk_tokens = conditioning_tokens + note_tokens[start:end]
    
        # 2. Encode input
        encoder_ids = tokenizer.encode_encoder_tokens_shared(chunk_tokens)
        input_ids = torch.tensor(encoder_ids, dtype=torch.long).unsqueeze(0)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # 3. Generate prediction
        print("  Generating tablature prediction...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=1,
                do_sample=False,
                eos_token_id=tokenizer.shared_token_to_id["<eos>"],
                pad_token_id=tokenizer.shared_token_to_id["<pad>"],
            )

        # 4. Decode prediction
        pred_ids = outputs[0].cpu().tolist()
        pred_tokens = tokenizer.shared_to_decoder_tokens(pred_ids)
        
        # 5. Compute original metrics (only pitch/time accuracy vs input, Tab acc is 0.0)
        original_metrics = compute_accuracy_metrics(
            chunk_tokens, pred_tokens, capo, tuning,
            ground_truth_tokens=None
        )
        
        # 6. Apply post-processing
        print("  Applying post-processing for pitch and time shift correction...")
        # Check if we should enable debug mode 
        # (always enable debug for a single file or if issues are detected)
        has_errors = (original_metrics['pitch_accuracy'] < 100 or
                    original_metrics['time_shift_accuracy'] < 100)

        postprocessed_tokens, correction_stats = postprocess_predictions(
            chunk_tokens, pred_tokens, capo, tuning, pitch_window=5, alignment_window=5, debug=has_errors
        )

        # 7. Compute post-processed metrics (only pitch/time accuracy vs input)
        postprocessed_metrics = compute_accuracy_metrics(
            chunk_tokens, postprocessed_tokens, capo, tuning,
            ground_truth_tokens=None
        )

        # 8. Display results
        print("\n" + "=" * 80)
        print(f"RESULTS FOR: {midi_file.name}")
        print("=" * 80)

        # Convert corrected tokens to human-readable tablature format
        postprocessed_tabs = [
            t for t in postprocessed_tokens if t.startswith("TAB<")
        ]
        postprocessed_shifts = [
            t for t in postprocessed_tokens if t.startswith("TIME_SHIFT<")
        ]
        
        print("\n### Final Corrected Tablature Tokens ###")
        print("----------------------------------------")
        
        # Print the token sequence in pairs
        for i in range(len(postprocessed_tabs)):
            tab_token = postprocessed_tabs[i]
            shift_token = postprocessed_shifts[i] if i < len(postprocessed_shifts) else "TIME_SHIFT<ERROR>"
            print(f"{tab_token} {shift_token}")

        print("\n### Pitch/Time Shift Accuracy vs Input MIDI ###")
        print("-----------------------------------------------")
        print(f"  Original Prediction:       Pitch={original_metrics['pitch_accuracy']:.1f}%, TimeShift={original_metrics['time_shift_accuracy']:.1f}%")
        print(f"  Post-processed Prediction: Pitch={postprocessed_metrics['pitch_accuracy']:.1f}%, TimeShift={postprocessed_metrics['time_shift_accuracy']:.1f}%")

        print("\n### Correction Summary ###")
        print("--------------------------")
        print(f"  Pitch Corrections:     {correction_stats['pitch_corrections']}")
        print(f"  Time Shift Corrections:{correction_stats['time_shift_corrections']}")
        print(f"  Pitch Too Far:         {correction_stats['pitch_too_far']}")
        print(f"  Unaligned Outputs:     {correction_stats['unaligned_outputs']}")
        print("=" * 80)


        
        # get tokens from andrea:
        
        for i in range(0, len(postprocessed_tokens), 2):
            tab_token = postprocessed_tokens[i]

            if not tab_token.startswith("TAB<"):
                continue 

            string, fret = tab_token[4:-1].split(',')
            time_shift_ms = 0 # Initialize default to 0

            if i + 1 < len(postprocessed_tokens):
                shift_token = postprocessed_tokens[i+1]
                
                if shift_token.startswith('TIME_SHIFT<'):
                    try:
                        # Extract the content between '<' and '>'
                        shift_value_str = shift_token.split('<')[-1].replace('>', '')
                        
                        # FINAL CHECK: Trim leading/trailing whitespace which is often the cause of failure
                        time_shift_ms = int(shift_value_str.strip()) 
                    except ValueError as e:
                        # Catch if the string cannot be converted to an integer
                        print(f"Extraction ERROR on token: {shift_token}. Value defaulted to 0. Error: {e}")
                        time_shift_ms = 0
            
            all_tabs.append((string, fret, time_shift_ms))
        
    return all_tabs

    # --- Print Debug/Summary Info (Keep this for console logging) ---
    print("\n" + "=" * 80)
    print(f"RESULTS FOR: {midi_file.name}")
    print("=" * 80)
    print(f"  Post-processed Prediction: Pitch={postprocessed_metrics['pitch_accuracy']:.1f}%, TimeShift={postprocessed_metrics['time_shift_accuracy']:.1f}%")
    print(f"  Total Extracted Tabs: {len(extracted_tabs_list)}")
    print("=" * 80)
    # --- End Print Debug/Summary Info ---
    
    # RETURN THE CLEAN LIST OF TABS
    return extracted_tabs_list


def process_midi_file_chunking(
    midi_path: str,
    tokenizer: MidiTabTokenizerV3,
    model: T5ForConditionalGeneration,
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    bpm: float = 120.0
):
    """
    Process a single MIDI file through the model and post-processing.
    """
    midi_file = Path(midi_path)
    if not midi_file.exists():
        print(f"ERROR: MIDI file not found at {midi_path}")
        return

    # 1. Load MIDI and create encoder tokens
    note_events = midi_to_tab_events(midi_file, bpm)

    if not note_events:
        print("Skipping (no valid note events found)")
        return

    # Use the specified capo/tuning for the prediction
    encoder_tokens = create_encoder_tokens_from_midi(note_events, capo, tuning)

    conditioning_tokens = encoder_tokens[:2]
    note_tokens = encoder_tokens[2:]

    NOTES_PER_CHUNK = 150 # arbitrary
    TOKENS_PER_NOTE = 3 # pitch/onset/duration
    OVERLAP_NOTES = 50 # arbitrary
    STEP_NOTES = NOTES_PER_CHUNK - OVERLAP_NOTES
    

    all_tabs = []
    total_notes = len(note_tokens) // TOKENS_PER_NOTE
    
    for start_note in range(0, total_notes, STEP_NOTES):
        start = start_note * TOKENS_PER_NOTE
        end = min(start + NOTES_PER_CHUNK * TOKENS_PER_NOTE, len(note_tokens))
        chunk_tokens = conditioning_tokens + note_tokens[start:end]
        new_notes_start_index = 0 if start_note == 0 else OVERLAP_NOTES
        
    
        # 2. Encode input
        encoder_ids = tokenizer.encode_encoder_tokens_shared(chunk_tokens)
        input_ids = torch.tensor(encoder_ids, dtype=torch.long).unsqueeze(0)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # 3. Generate prediction
        print("  Generating tablature prediction...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=1,
                do_sample=False,
                eos_token_id=tokenizer.shared_token_to_id["<eos>"],
                pad_token_id=tokenizer.shared_token_to_id["<pad>"],
            )

        # 4. Decode prediction
        pred_ids = outputs[0].cpu().tolist()
        pred_tokens = tokenizer.shared_to_decoder_tokens(pred_ids)
        
        # 5. Compute original metrics (only pitch/time accuracy vs input, Tab acc is 0.0)
        original_metrics = compute_accuracy_metrics(
            chunk_tokens, pred_tokens, capo, tuning,
            ground_truth_tokens=None
        )
        
        # 6. Apply post-processing
        print("  Applying post-processing for pitch and time shift correction...")
        # Check if we should enable debug mode 
        # (always enable debug for a single file or if issues are detected)
        has_errors = (original_metrics['pitch_accuracy'] < 100 or
                    original_metrics['time_shift_accuracy'] < 100)

        postprocessed_tokens, correction_stats = postprocess_predictions(
            chunk_tokens, pred_tokens, capo, tuning, pitch_window=5, alignment_window=5, debug=has_errors
        )

        # 7. Compute post-processed metrics (only pitch/time accuracy vs input)
        postprocessed_metrics = compute_accuracy_metrics(
            chunk_tokens, postprocessed_tokens, capo, tuning,
            ground_truth_tokens=None
        )

        # 8. Display results
        print("\n" + "=" * 80)
        print(f"RESULTS FOR: {midi_file.name}")
        print("=" * 80)

        # Convert corrected tokens to human-readable tablature format
        postprocessed_tabs = [
            t for t in postprocessed_tokens if t.startswith("TAB<")
        ]
        postprocessed_shifts = [
            t for t in postprocessed_tokens if t.startswith("TIME_SHIFT<")
        ]
        
        print("\n### Final Corrected Tablature Tokens ###")
        print("----------------------------------------")
        
        # Print the token sequence in pairs
        for i in range(len(postprocessed_tabs)):
            tab_token = postprocessed_tabs[i]
            shift_token = postprocessed_shifts[i] if i < len(postprocessed_shifts) else "TIME_SHIFT<ERROR>"
            print(f"{tab_token} {shift_token}")

        print("\n### Pitch/Time Shift Accuracy vs Input MIDI ###")
        print("-----------------------------------------------")
        print(f"  Original Prediction:       Pitch={original_metrics['pitch_accuracy']:.1f}%, TimeShift={original_metrics['time_shift_accuracy']:.1f}%")
        print(f"  Post-processed Prediction: Pitch={postprocessed_metrics['pitch_accuracy']:.1f}%, TimeShift={postprocessed_metrics['time_shift_accuracy']:.1f}%")

        print("\n### Correction Summary ###")
        print("--------------------------")
        print(f"  Pitch Corrections:     {correction_stats['pitch_corrections']}")
        print(f"  Time Shift Corrections:{correction_stats['time_shift_corrections']}")
        print(f"  Pitch Too Far:         {correction_stats['pitch_too_far']}")
        print(f"  Unaligned Outputs:     {correction_stats['unaligned_outputs']}")
        print("=" * 80)


        
        # get tokens from andrea:
        note_index = 0
        for i in range(0, len(postprocessed_tokens), 2):
            if i >= len(postprocessed_tokens):
                break

            tab_token = postprocessed_tokens[i]

            if not tab_token.startswith("TAB<"):
                continue 

            time_shift_ms = 0 # Initialize default to 0

            if note_index >= new_notes_start_index:
                string, fret = tab_token[4:-1].split(',')
                time_shift_ms = 0

                if i + 1 < len(postprocessed_tokens):
                    shift_token = postprocessed_tokens[i+1]
                    
                    if shift_token.startswith('TIME_SHIFT<'):
                        try:
                            # Extract the content between '<' and '>'
                            shift_value_str = shift_token.split('<')[-1].replace('>', '')
                            
                            # FINAL CHECK: Trim leading/trailing whitespace which is often the cause of failure
                            time_shift_ms = int(shift_value_str.strip()) 
                        except ValueError as e:
                            # Catch if the string cannot be converted to an integer
                            print(f"Extraction ERROR on token: {shift_token}. Value defaulted to 0. Error: {e}")
                            time_shift_ms = 0
            
                all_tabs.append((string, fret, time_shift_ms))
            note_index += 1
        
    return all_tabs

def main(bpm):
    """Main execution point for single MIDI file processing."""
    
    # Expect MIDI file path as a command line argument
    if len(sys.argv) < 2:
        print("Usage: python3 your_script_name.py <path_to_midi_file>")
        sys.exit(1)
        
    midi_path = sys.argv[1]

    print("=" * 80)
    print(f"Post-Processing Single MIDI File: {Path(midi_path).name}")
    print("=" * 80)

    # Configuration (can be modified by user)
    CAPO = 0
    TUNING = STANDARD_TUNING  # e.g., (64, 59, 55, 50, 45, 40)
    
    # The original checkpoint path
    checkpoint_path = "/data/andreaguz/fret_t5/checkpoints_conditioning_scratch_retrain/checkpoint-642982"

    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("Please update 'checkpoint_path' to a valid model location.")
        return

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = MidiTabTokenizerV3.load("/data/andreaguz/fret_t5_clean/fret_t5/universal_tokenizer")
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS
    )

    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    # Process the file
    final_tab = process_midi_file(midi_path, tokenizer, model, capo=CAPO, tuning=TUNING, bpm=bpm)

    


if __name__ == "__main__":
    main()



class MidiTabTokenizerV3:
    """Tokenizer implementing the v3 SynthTab representation."""

    SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<eos>", "<unk>") + tuple(
        f"<extra_id_{i}>" for i in range(100)
    )

    def __init__(
        self,
        config: TokenizerConfig,
        encoder_vocab: _Vocabulary,
        decoder_vocab: _Vocabulary,
    ) -> None:
        self.config = config
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self._build_shared_vocabulary()

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------
    @property
    def encoder_token_to_id(self) -> Dict[str, int]:
        return self.encoder_vocab.token_to_id

    @property
    def decoder_token_to_id(self) -> Dict[str, int]:
        return self.decoder_vocab.token_to_id

    @property
    def shared_token_to_id(self) -> Dict[str, int]:
        return self._shared_token_to_id

    @property
    def shared_id_to_token(self) -> Dict[int, str]:
        return self._shared_id_to_token

    def encode_encoder_tokens_shared(self, tokens: Sequence[str]) -> List[int]:
        return [self._encoder_to_shared.get(tok, self.shared_token_to_id["<unk>"]) for tok in tokens]

    def encode_decoder_tokens_shared(self, tokens: Sequence[str]) -> List[int]:
        return [self._decoder_to_shared.get(tok, self.shared_token_to_id["<unk>"]) for tok in tokens]

    def shared_to_decoder_tokens(self, ids: Sequence[int]) -> List[str]:
        vocab = self.decoder_vocab.token_to_id
        id_to_token = self.decoder_vocab.id_to_token
        tokens: List[str] = []
        for idx in ids:
            token = self._shared_id_to_token.get(int(idx))
            if token in vocab:
                tokens.append(token)
            else:
                tokens.append("<unk>")
        return tokens

    def shared_to_encoder_tokens(self, ids: Sequence[int]) -> List[str]:
        vocab = self.encoder_vocab.token_to_id
        id_to_token = self.encoder_vocab.id_to_token
        tokens: List[str] = []
        for idx in ids:
            token = self._shared_id_to_token.get(int(idx))
            if token in vocab:
                tokens.append(token)
            else:
                tokens.append("<unk>")
        return tokens

    def is_tab_token(self, token_id: int) -> bool:
        token = self.decoder_vocab.id_to_token.get(token_id)
        return token is not None and token.startswith("TAB<")

    def is_time_shift_token(self, token_id: int) -> bool:
        token = self.decoder_vocab.id_to_token.get(token_id)
        return token is not None and token.startswith("TIME_SHIFT<")

    def get_tab_token_ids(self) -> List[int]:
        """Get all TAB token IDs for constrained decoding."""
        return [
            self.shared_token_to_id[token]
            for token in self.decoder_vocab.token_to_id.keys()
            if token.startswith("TAB<")
        ]

    def get_time_shift_token_ids(self) -> List[int]:
        """Get all TIME_SHIFT token IDs for constrained decoding."""
        return [
            self.shared_token_to_id[token]
            for token in self.decoder_vocab.token_to_id.keys()
            if token.startswith("TIME_SHIFT<")
        ]

    def get_constrained_next_tokens(self, last_token_id: int) -> List[int]:
        """Get valid next tokens for constrained v3 decoding.

        v3 pattern: TAB<s,f> → TIME_SHIFT<d> → TAB<s,f> → TIME_SHIFT<d> ...
        EOS is only allowed after TIME_SHIFT to preserve TAB+TIME_SHIFT pairs.

        Args:
            last_token_id: The previous decoder token ID

        Returns:
            List of valid next token IDs
        """
        if self.is_tab_token(last_token_id):
            # After TAB, must emit TIME_SHIFT (no EOS allowed here)
            return self.get_time_shift_token_ids()
        elif self.is_time_shift_token(last_token_id):
            # After TIME_SHIFT, can emit TAB or end sequence
            return self.get_tab_token_ids() + [self.shared_token_to_id["<eos>"]]
        else:
            # Start of sequence or special token - allow TAB only
            return self.get_tab_token_ids()

    # ------------------------------------------------------------------
    # Serialisation utilities
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "config": self.config.__dict__,
            "encoder_vocab": self.encoder_vocab.to_json(),
            "decoder_vocab": self.decoder_vocab.to_json(),
        }
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "MidiTabTokenizerV3":
        with open(os.path.join(path, "tokenizer.json"), "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        config = TokenizerConfig(**payload["config"])
        encoder_vocab = _Vocabulary.from_json(payload["encoder_vocab"])
        decoder_vocab = _Vocabulary.from_json(payload["decoder_vocab"])
        return cls(config=config, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab)

    # ------------------------------------------------------------------
    # Tokenisation API
    # ------------------------------------------------------------------
    def tokenize_track_from_jams(
        self,
        jams_events: Sequence[Dict[str, float]],
        *,
        capo: int = 0,
        tuning: Optional[Sequence[int]] = None,
    ) -> TokenizedTrack:
        """Tokenise a JAMS tablature track with optional tuning/capo conditioning."""

        if tuning is None:
            tuning = STANDARD_TUNING
        if len(tuning) != 6:
            raise ValueError("Tuning sequences must contain exactly six MIDI pitches")

        encoder_tokens: List[str] = []
        decoder_tokens: List[str] = []
        encoder_groups: List[int] = []
        decoder_groups: List[int] = []
        metadata: List[NoteMetadata] = []

        # Map strings (1 == high E) to MIDI pitches for open strings.
        open_strings = {string_idx + 1: int(pitch) for string_idx, pitch in enumerate(tuning)}

        # Sort events by time, then by string for consistent ordering within chords
        sorted_events = sorted(jams_events, key=lambda e: (e.get("time_ticks", 0), e.get("string", 0)))

        # Group events by onset time to detect chords
        onset_groups = []
        current_onset = None
        current_group = []

        for event in sorted_events:
            event_time = event.get("time_ticks", 0)
            if current_onset is None or abs(event_time - current_onset) < 1e-6:  # Same onset (within tolerance)
                current_group.append(event)
                current_onset = event_time
            else:
                if current_group:
                    onset_groups.append(current_group)
                current_group = [event]
                current_onset = event_time

        if current_group:
            onset_groups.append(current_group)

        # Process each onset group
        for group_idx, onset_group in enumerate(onset_groups):
            for event_idx, event in enumerate(onset_group):
                string = int(event["string"])
                fret = int(event["fret"])
                duration_ms = float(event["duration_ms"])

                # Validate ranges
                if not (self.config.min_string <= string <= self.config.max_string):
                    raise ValueError(f"Tab string {string} is outside configured range")
                if not (0 <= fret <= self.config.max_fret):
                    raise ValueError(f"Tab fret {fret} is outside configured range")

                # Calculate MIDI pitch from string and fret
                base_pitch = open_strings[string] + fret + int(capo)
                if base_pitch < 0 or base_pitch > 127:
                    raise ValueError(
                        f"Computed MIDI pitch {base_pitch} for string {string}, fret {fret}, capo {capo} is outside the 0-127 range"
                    )

                # Determine if this is within a chord
                is_chord_note = len(onset_group) > 1
                is_last_in_chord = event_idx == len(onset_group) - 1
                is_same_onset = is_chord_note and not is_last_in_chord

                # Generate duration token
                duration_token = self._time_shift_token(
                    duration_ms if is_last_in_chord else 0.0,  # Use 0 for chord internal transitions
                    same_onset=is_same_onset
                )

                # Generate encoder tokens (MIDI representation)
                encoder_tokens.extend([
                    self._note_on_token(base_pitch),
                    duration_token,
                    self._note_off_token(base_pitch),
                ])
                encoder_groups.append(3)

                metadata.append(NoteMetadata(string=string, fret=fret))

                # Generate decoder tokens (tablature representation)
                decoder_tokens.extend([
                    self._tab_token(string, fret),
                    duration_token,
                ])
                decoder_groups.append(2)

        return TokenizedTrack(
            encoder_tokens=encoder_tokens,
            decoder_tokens=decoder_tokens,
            encoder_group_lengths=encoder_groups,
            decoder_group_lengths=decoder_groups,
            note_metadata=metadata,
        )

    def encode_encoder_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.encoder_vocab.token_to_id.get(tok, self.encoder_vocab.token_to_id["<unk>"]) for tok in tokens]

    def encode_decoder_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.decoder_vocab.token_to_id.get(tok, self.decoder_vocab.token_to_id["<unk>"]) for tok in tokens]

    def decode_decoder_tokens(self, ids: Sequence[int]) -> List[str]:
        vocab = self.decoder_vocab.id_to_token
        return [vocab.get(int(idx), "<unk>") for idx in ids]

    # ------------------------------------------------------------------
    # Internal token constructors
    # ------------------------------------------------------------------
    @staticmethod
    def _note_on_token(pitch: int) -> str:
        return f"NOTE_ON<{pitch}>"

    @staticmethod
    def _note_off_token(pitch: int) -> str:
        return f"NOTE_OFF<{pitch}>"

    @staticmethod
    def _tab_token(string: int, fret: int) -> str:
        return f"TAB<{string},{fret}>"

    def _time_shift_token(self, duration_ms: float, same_onset: bool = False) -> str:
        quantised = self.config.quantise_duration(duration_ms, same_onset=same_onset)
        return f"TIME_SHIFT<{quantised}>"

    @staticmethod
    def _capo_token(capo: int) -> str:
        return f"CAPO<{int(capo)}>"

    @staticmethod
    def _tuning_token(tuning: Sequence[int]) -> str:
        values = ",".join(str(int(pitch)) for pitch in tuning)
        return f"TUNING<{values}>"

    def build_conditioning_prefix(self, capo: int, tuning: Sequence[int]) -> List[str]:
        """Create conditioning tokens describing capo position and tuning."""

        return [self._capo_token(capo), self._tuning_token(tuning)]

    def ensure_conditioning_tokens(
        self,
        capo_values: Sequence[int],
        tuning_options: Sequence[Sequence[int]],
    ) -> None:
        """Add conditioning tokens to the encoder vocabulary when required."""

        added = False

        for capo in capo_values:
            token = self._capo_token(capo)
            if token not in self.encoder_vocab.token_to_id:
                idx = len(self.encoder_vocab.token_to_id)
                self.encoder_vocab.token_to_id[token] = idx
                self.encoder_vocab.id_to_token[idx] = token
                added = True

        for tuning in tuning_options:
            token = self._tuning_token(tuning)
            if token not in self.encoder_vocab.token_to_id:
                idx = len(self.encoder_vocab.token_to_id)
                self.encoder_vocab.token_to_id[token] = idx
                self.encoder_vocab.id_to_token[idx] = token
                added = True

        if added:
            self._build_shared_vocabulary()

    def _build_shared_vocabulary(self) -> None:
        """Merge encoder/decoder vocabularies into a shared dictionary.

        Hugging Face's T5 implementation assumes a single embedding table. To
        maintain compatibility we build a shared vocabulary that contains all
        encoder and decoder tokens. Tokens keep their semantics via the
        ``_encoder_to_shared`` and ``_decoder_to_shared`` lookup tables.
        """

        shared_tokens: List[str] = []
        seen: Dict[str, int] = {}

        for tok in self.SPECIAL_TOKENS:
            shared_tokens.append(tok)
            seen[tok] = len(seen)

        for vocab in (self.encoder_vocab, self.decoder_vocab):
            for tok in vocab.token_to_id:
                if tok not in seen:
                    seen[tok] = len(seen)
                    shared_tokens.append(tok)

        self._shared_token_to_id = {tok: idx for idx, tok in enumerate(shared_tokens)}
        self._shared_id_to_token = {idx: tok for tok, idx in self._shared_token_to_id.items()}

        self._encoder_to_shared = {
            tok: self._shared_token_to_id.get(tok, self._shared_token_to_id["<unk>"])
            for tok in self.encoder_vocab.token_to_id
        }
        self._decoder_to_shared = {
            tok: self._shared_token_to_id.get(tok, self._shared_token_to_id["<unk>"])
            for tok in self.decoder_vocab.token_to_id
        }

