import pretty_midi
import py
import torch
import jams
import torch.nn as nn
from andrea_test import main
import librosa
import subprocess
import os
import soundfile as sf
from pathlib import Path
import sys
import json
from typing import List, Tuple
import numpy as np
import andrea_test
import andrea_inference_script
import technique_tabs
import technique_tabs_final
import pipeline_utils.technique_cacher as cache

import argparse
import shutil
import scipy.signal
import scipy.signal.windows
from BeatNet.BeatNet import BeatNet

# Example usage: python pipeline_v2.py --audio_path /data/shamakg/FrancoisLeduc_Raw/audio/2DC4c.mp3

# TODO: Step 1: Import audio
# Denoising? MusicAI/Music-AI/0_Preprocessing/guitar_extraction_pipeline.py

#----------------------------------------------------------------------------------#

### Uses beatnet model (80%+ accuracy). Only issue is octave/bpm multiple correction (which is subjective)   
def estimate_bpm(audio_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = BeatNet(model=2, mode='offline', inference_model='DBN', device=device)
    output = estimator.process(audio_path)
    beat_times = output[:, 0]
    intervals = np.diff(beat_times)
    bpms = 60 / intervals[intervals > 0]

    global_bpm = round(float(np.median(bpms)))
    if global_bpm < 80:
        # most likely half-tempo
        global_bpm *= 2
    elif global_bpm > 180:
        # most likely double-tempo
        global_bpm //= 2

    print("BPM:" , global_bpm)
    
    return global_bpm

#----------------------------------------------------------------------------------#

# Step 2: Running Akshaj model. (saves the midi to results folder in directory)

def run_akshaj_model(audio_path, model_path, inference_path):
    midi_filename = Path(audio_path).stem + ".mid"
    output_midi_dir = Path("results")
    output_midi_dir.mkdir(exist_ok=True)
    output_midi_path = output_midi_dir / midi_filename

    cmd= [
        "python", 
        inference_path,
        "--model_type", "Regress_onset_offset_frame_velocity_CRNN",
        "--checkpoint_path", model_path,
        "--post_processor_type", "regression",
        "--audio_path", audio_path,
        "--cuda"
    ]
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=True, 
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"(Model Log):\n{e.stdout}")
        print(f"(Error Message):\n{e.stderr}")
        raise 
        
    print("MIDI Generated")
    # TODO: Save to Memory
    return str(output_midi_path.resolve())

#----------------------------------------------------------------------------------#

# Step 3: Peter's Model

def find_single_note_onsets(midi_filepath, min_duration):
    all_onsets = []
    current_time_ticks = 0
    midi_data = pretty_midi.PrettyMIDI(midi_filepath)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            all_onsets.append({
                'onset_time_seconds': note.start,
                'offset_time_seconds': note.end, 
                'duration_seconds': note.end - note.start,
                'pitch': note.pitch,
                'note_name': pretty_midi.note_number_to_name(note.pitch),
                'velocity': note.velocity,
            })
    
    all_onsets.sort(key=lambda x: x['onset_time_seconds'])

    return all_onsets


def extract_audio_chunks(audio, sr, onsets, durations):
    """
    Extract audio chunks based on onset times and durations (from MIDI file).
    
    Parameters:
    -----------
    audio: np.ndarray
        Audio time series
    sr: int
        Sample rate of the audio
    onsets: array-like
        Onset times of notes we want to extract (in seconds)
    durations: array-like
        Duration of each note (in seconds)
    
    Returns:
    --------
    list of np.ndarray
        List of audio chunks corresponding to each onset/duration pair
    """
    chunks = []
    
    for onset, duration in zip(onsets, durations):
        # Convert seconds to samples, staying within audio length
        start_sample = max(0, int(onset * sr))
        end_sample = min(len(audio), int((onset + duration) * sr))
        
        # Extract the chunk
        chunk = audio[start_sample:end_sample]
        chunks.append(chunk)
    
    return chunks

def audio_midi_to_chunks(audio_path, midi_list):
    """
    Complete pipeline to extract audio chunks from an audio file based on MIDI onsets and durations.
    
    Parameters:
    -----------
    audio_path: str
        Path to the audio file
    midi_dict: list of dicts
        List containing midi data (as dictionaries)
    
    Returns:
    --------
    list of np.ndarray
        List of audio chunks corresponding to each onset/duration pair
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=None)
    
    midi_onsets = [midi["onset_time_seconds"] for midi in midi_list]
    midi_durations = [midi["duration_seconds"] for midi in midi_list]

    # print(midi_onsets, midi_durations)
    # Extract chunks
    chunks = extract_audio_chunks(audio, sr, midi_onsets, midi_durations)

    output_dir = "audio_slices"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    chunk_filepaths = []
    for i, chunk in enumerate(chunks):
        filename = f"chunk_{i}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, chunk.T if audio.ndim > 1 else chunk, sr)
        chunk_filepaths.append(filepath)
    
    return chunk_filepaths, midi_onsets, midi_durations

def run_peter_model_on_chunks(chunk_paths: List[str], onsets, durations):
    PYTHON_EXECUTABLE = "/data/samhita/.venv/bin/python"

    BASE_DIR="/data/shamakg/music_ai_pipeline/"
    INPUT_DIR="/data/shamakg/music_ai_pipeline/audio_slices/"
    
    MODEL_DIR="/data/shamakg/music_ai_pipeline/expTechInfer_12-14-2025/models_cnn_lstm/setupB-eg_ipt-plus4/run-20251212-231215"
    MODEL_FILE="/data/shamakg/music_ai_pipeline/expTechInfer_12-14-2025/models_cnn_lstm/setupB-eg_ipt-plus4/run-20251212-231215/cnn_lstm_best.h5"

    INFERENCE_FILE = "/data/shamakg/music_ai_pipeline/expTechInfer_12-14-2025/scripts/infer_cnn_lstm.py"

    if 'TF_USE_LEGACY_KERAS' in os.environ:
        del os.environ['TF_USE_LEGACY_KERAS']
        print("TF_USE_LEGACY_KERAS environment variable removed for subprocess.")

    cmd = [
        PYTHON_EXECUTABLE, 
        INFERENCE_FILE,
        "--base_dir", BASE_DIR,
        "--model_dir", MODEL_DIR,
        "--input_dir", INPUT_DIR,
        "--recursive",
        "--glob", "*.wav"
    ]
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=True, 
            text=True,
            env=os.environ
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"(Model Log):\n{e.stdout}")
        print(f"(Error Message):\n{e.stderr}")
        raise

    with open(f"{MODEL_DIR}/infer_outputs/predictions.json", 'r') as file:
        predictions_data = json.load(file)

    path_to_prediction_map = {os.path.basename(item['wav_path']): item["pred_label"] for item in predictions_data['predictions']}
    expressive_techniques = []

    for path in chunk_paths:
        prediction = path_to_prediction_map.get(os.path.basename(path), "Normal")
        expressive_techniques.append(prediction)

    return list(zip(expressive_techniques, onsets, durations))

#----------------------------------------------------------------------------------#

# STEP 3: Andreas MODEL

def run_andreas_model(midi_file_path, bpm, inference):
    
    final_tab_list: List[Tuple[str, str]] = inference(midi_file_path, bpm)

    print("TAB",final_tab_list)
    
    if final_tab_list is None:
        print("Pipeline failed to generate tabs.")
        return
    
    return final_tab_list

def calculate_onsets(tab_data):


    current_onset_ms = 0
    final_data_with_onsets = []
    count = 0
    
    for string, fret, time_shift_ms in tab_data:
        final_data_with_onsets.append((string, fret, time_shift_ms, current_onset_ms))
        current_onset_ms += time_shift_ms
        if int(fret) == 0:
            count += 1
    
    print("CHECK2:", count)
        
    return final_data_with_onsets

# for andreas new postprocessing, should return a list of tuples
def calculate_onsets_v2(tab_data):
    final_data_with_onsets = []

    for event in tab_data:
        final_data_with_onsets.append((
                event.string, 
                event.fret, 
                event.duration_sec,  
                event.onset_sec
            ))

    zero_count = sum(1 for (_, fret, *_) in final_data_with_onsets if int(fret) == 0)
    print(f"CHeck 2: Found {zero_count} frets with 0 out of {len(final_data_with_onsets)} notes")

    #----------------------------------------------------------------------------------#    

    return final_data_with_onsets

#----------------------------------------------------------------------------------#

# TODO: def calculate_accuracy(actual_tab, predicted_tab)
#### To get updated (in development) pipeline: change tab_inference, post_process_tab, mode

if __name__ == "__main__":
    ### Note: if in testing...read comments with # TESTING

    parser = argparse.ArgumentParser(description="Generate guitar tablature for audio.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()
    AUDIO_PATH = args.audio_path
    MUSIC_TO_MIDI_PATH = "/data/akshaj/MusicAI/workspace/checkpoints/log_0332/gaps_goat_guitartechs_leduc_limited_regress_onset_offset_frame_velocity_bce_log332_iter2000_lr1e-05_bs4.pth"
    MIDI_INFERENCE_SCRIPT = "/data/akshaj/MusicAI/Stage1/pytorch/inference.py"
    MIN_AUDIO_SLICE_DURATION = 0.2 # This is set as minimal comprehensible input to Peter's model
    
    #----------------------------------------------------------------------------------#
    
    bpm = estimate_bpm(AUDIO_PATH)
    midi_path = run_akshaj_model(AUDIO_PATH, MUSIC_TO_MIDI_PATH, MIDI_INFERENCE_SCRIPT)
    midi_dict = find_single_note_onsets(midi_path, MIN_AUDIO_SLICE_DURATION)

    #----------------------------------------------------------------------------------#
    
    midi_dict_peter = [event for event in midi_dict if event['duration_seconds'] >= MIN_AUDIO_SLICE_DURATION] # for peter's model
    exp_onset_dur_tuples = run_peter_model_on_chunks(*audio_midi_to_chunks(AUDIO_PATH, midi_dict_peter))
    print(exp_onset_dur_tuples)
    # exp_onset_dur_tuples = cache.cached_peter_result(AUDIO_PATH, midi_dict_peter, force_rerun=False) ## for TESTING

    #----------------------------------------------------------------------------------#

    tab_inference, post_process_tab = andrea_inference_script.run_tab_generation, calculate_onsets_v2
    tab_list = post_process_tab(run_andreas_model(midi_path, bpm, tab_inference))

    #----------------------------------------------------------------------------------#

    output_name, matching_mode = os.path.splitext(os.path.basename(midi_path))[0], "time_matching"
    tab_path = technique_tabs_final.conversion_andreas(midi_path, exp_onset_dur_tuples, tab_list, f"{output_name}_colin.xml", bpm, mode=matching_mode)
    # tab_working = technique_tabs.conversion_andreas(midi_path, exp_onset_dur_tuples, tab_list, f"{output_name}_working.xml", bpm)
    
    print(tab_path)
