import pretty_midi
import torch
import torch.nn as nn
import librosa
import subprocess
import os
import soundfile as sf
from pathlib import Path
import json
from typing import List, Tuple
import numpy as np
import fretting_inference_script
import pipeline_utils.technique_cacher as cache
import tab_generation_utils.postprocess_v2 as postprocess
import tab_generation_utils.preprocess as preprocess
import argparse
import shutil
from tab_generation_final import main as tab_generator_final
from BeatNet.BeatNet import BeatNet
import tab_generation_utils.jams_test as j

# Example usage: python pipeline_v2.py --audio_path /data/shamakg/datasets/FrancoisLeduc_Raw/audio/2DC4c.mp3

#----------------------------------------------------------------------------------#

# Tuning util (for pipeliine input)

STANDARD_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 40)
HALF_STEP_DOWN_TUNING: Tuple[int, ...] = tuple(pitch - 1 for pitch in STANDARD_TUNING)
FULL_STEP_DOWN_TUNING: Tuple[int, ...] = tuple(pitch - 2 for pitch in STANDARD_TUNING)
DROP_D_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 38)

def tuning_conversion(chars):
    ## These are the standard octaves according to tuning charts online
    if chars == STANDARD_TUNING:
        return STANDARD_TUNING
    chars = chars.split()
    octave_map = [2, 2, 3, 3, 3, 4]
    ## offsets from C
    base_offsets = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    midi_tuning = []
    for i, char in enumerate(chars):
        char = char.strip().title() ## first letter uppercase, second lowercase
        octave = octave_map[i]

        midi_note = (octave + 1) * 12 + base_offsets[char]
        midi_tuning.append(midi_note)
    
    print("TUNING", tuple(reversed(midi_tuning)))
    ### RETURNS TUPLE FROM HIGH TO LOW
    return tuple(reversed(midi_tuning))

#----------------------------------------------------------------------------------#

# Step 1: Import audio and detect bpm

### Uses beatnet model (80%+ accuracy). Only issue is octave/bpm multiple correction (which is subjective)   
def estimate_bpm(audio_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = BeatNet(model=2, mode='offline', inference_model='DBN', device=device)
    output = estimator.process(audio_path)
    if output is not None:
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

def find_single_note_onsets(midi_filepath):
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
    audio, sr = librosa.load(audio_path, sr=None) # type: ignore
    
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
    INPUT_DIR="/data/shamakg/music_ai_pipeline/Music-AI/5_FinalPipeline/audio_slices/"
    
    MODEL_DIR="/data/shamakg/music_ai_pipeline/expTechInfer_12-14-2025/models_cnn_lstm/final_model/run-20260112-131057"
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

    if len(chunk_paths) != len(path_to_prediction_map):
        missing = set([os.path.basename(p) for p in chunk_paths]) - set(path_to_prediction_map.keys())
        print(f"Files the model SKIPPED: {list(missing)[:5]}")

    for path in chunk_paths:
        if os.path.basename(path) not in path_to_prediction_map:
            print("SKIPPED!")
            print(os.path.basename(path))
        prediction = path_to_prediction_map.get(os.path.basename(path), "Normal")
        expressive_techniques.append(prediction)

    return list(zip(expressive_techniques, onsets, durations))

#----------------------------------------------------------------------------------#

# STEP 3: Andreas MODEL

def run_andreas_model(midi_file_path, inference, capo=0, tuning=STANDARD_TUNING):
    
    final_tab_list: List[Tuple[str, str]] = inference(midi_file_path, capo=capo, tuning=tuning)

    # print("TAB",sorted(final_tab_list, key=lamda t:t.onset_sec)
    
    if final_tab_list is None:
        print("Pipeline failed to generate tabs.")
        return
    
    print("TAB", final_tab_list)
    
    return final_tab_list

# for andreas new postprocessing, should return a list of tuples
def calculate_onsets(tab_data):
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
    parser.add_argument("--capo", type=int, default=0, help="Capo")
    parser.add_argument("--tuning", type=str, default=STANDARD_TUNING, help="Tuning from high to low, string separated by a space")
    parser.add_argument("--tempo", default=None, help="Tempo in BPM")
    args = parser.parse_args()
    AUDIO_PATH = args.audio_path
    CAPO = args.capo
    TUNING = tuning_conversion(args.tuning)
    BPM = args.tempo

    #----------------------------------------------------------------------------------#

    print("Running our pipeline on:", AUDIO_PATH)
    MUSIC_TO_MIDI_PATH = "/data/akshaj/MusicAI/workspace/checkpoints/log_0332/gaps_goat_guitartechs_leduc_limited_regress_onset_offset_frame_velocity_bce_log332_iter2000_lr1e-05_bs4.pth"
    MIDI_INFERENCE_SCRIPT = "/data/akshaj/MusicAI/Stage1/pytorch/inference.py"
    MIN_AUDIO_SLICE_DURATION = 0.2 # This is set as minimal comprehensible input to Peter's model. if you change thiis you must also change in postprocess.py
    
    #----------------------------------------------------------------------------------#
    
    bpm = BPM
    if not BPM:
        bpm = estimate_bpm(AUDIO_PATH)
    midi_path = run_akshaj_model(AUDIO_PATH, MUSIC_TO_MIDI_PATH, MIDI_INFERENCE_SCRIPT)
    midi_dict = find_single_note_onsets(midi_path)

    #----------------------------------------------------------------------------------#

    midi_dict_peter = [event for event in midi_dict if event['duration_seconds'] >= MIN_AUDIO_SLICE_DURATION] # for peter's model
    # Note: the below function has several path variables you may need to change
    print("PETER", midi_dict_peter)
    exp_onset_dur_tuples = run_peter_model_on_chunks(*audio_midi_to_chunks(AUDIO_PATH, midi_dict_peter))
    print(exp_onset_dur_tuples)
    # exp_onset_dur_tuples = cache.cached_peter_result(AUDIO_PATH, midi_dict_peter, force_rerun=False) ## for TESTING

    #----------------------------------------------------------------------------------#

    tab_inference, post_process_tab = fretting_inference_script.run_tab_generation, calculate_onsets
    tab_list = post_process_tab(run_andreas_model(midi_path, tab_inference, capo=CAPO, tuning=TUNING))

    #----------------------------------------------------------------------------------#

    ### CREATE JAMS FILE
    tuning_dict = {i + 1: pitch for i, pitch in enumerate(TUNING)}
    jam = preprocess.midi_to_jams_with_tablature_from_andreas(midi_path, tab_list, bpm=bpm, tuning=tuning_dict, capo = CAPO)
    jam = preprocess.add_exp_techniques_to_existing_jam(jam, exp_onset_dur_tuples)
    output_name = f"{os.path.splitext(os.path.basename(midi_path))[0]}.xml"
    output_musicxml = f"{os.path.splitext(os.path.basename(midi_path))[0]}.musicxml"
    jams_path = j.save_jam(jam, output_name)

    #----------------------------------------------------------------------------------#

    ### FINAL TAB GENERATION 
    
    tab_generator_final(jams_path, output_musicxml, bpm = bpm)
