import sys

# Compatibility shims for madmom (BeatNet dependency) on Python 3.10+ / NumPy >=1.20.
import collections
import collections.abc
for _name in ("MutableSequence", "Iterable", "Callable", "Mapping", "MutableMapping"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
import numpy as _np
for _a in ("float", "int", "bool", "object", "complex"):
    if not hasattr(_np, _a):
        setattr(_np, _a, getattr(__builtins__, _a) if isinstance(__builtins__, dict) else getattr(__builtins__, _a))

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
import pipeline_utils.string_fret_inference_script as string_fret_inference_script
import pipeline_utils.technique_cacher as cache
import tab_generation_utils.preprocess as preprocess
import argparse
import shutil
from pipeline_utils.tab_generation_final import main as tab_generator_final
from BeatNet.BeatNet import BeatNet
import tab_generation_utils.jams_test as j
from huggingface_hub import snapshot_download

# Example usage: python predict.py --audio_path {AUDIO}
#----------------------------------------------------------------------------------#

# Tuning util (for pipeline input)

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

# Step 2: Running MIDI Transcription model. (saves the midi to results folder in directory)

def run_midi_model(audio_path, model_path, inference_path):
    midi_filename = Path(audio_path).stem + ".mid"
    output_midi_dir = Path("results")
    output_midi_dir.mkdir(exist_ok=True)
    output_midi_path = output_midi_dir / midi_filename

    cmd= [
        sys.executable, 
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

# Step 3: Expressive Technique Detection Model

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

    output_dir = Path(__file__).parent.resolve() / "audio_slices"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    chunk_filepaths = []
    for i, chunk in enumerate(chunks):
        filename = f"chunk_{i}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, chunk.T if audio.ndim > 1 else chunk, sr)
        chunk_filepaths.append(filepath)
    
    return chunk_filepaths, midi_onsets, midi_durations

def run_technique_model_on_chunks(chunk_paths: List[str], onsets, durations):
    
    ## Setting env
    SCRIPT_DIR = Path(__file__).parent.resolve()

    MODEL_DIR = SCRIPT_DIR / "models" / "expressive-techniques-guitar" / "run-20260112-131057"

    # Use an absolute path for audio_slices to be safe
    AUDIO_SLICES_DIR = SCRIPT_DIR / "audio_slices" 
    
    INFERENCE_FILE = SCRIPT_DIR / "pipeline_utils" / "scripts" / "infer_cnn_lstm.py"
    
    results_dir = SCRIPT_DIR / "model_outputs"
    results_dir.mkdir(exist_ok=True)
    out_json_path = results_dir / "predictions.json"

    env = os.environ.copy()

    # 🔥 FORCE TensorFlow CPU
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    env.pop('TF_USE_LEGACY_KERAS', None)

    cmd = [
        sys.executable, 
        str(INFERENCE_FILE),
        "--base_dir", str(SCRIPT_DIR / "pipeline_utils" / "scripts"),
        "--model_dir", str(MODEL_DIR),
        "--input_dir", str(AUDIO_SLICES_DIR),
        "--out_json", str(out_json_path),
        "--recursive",
        "--glob", "*.wav"
    ]
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=True, 
            text=True,
            env=env
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"(Model Log):\n{e.stdout}")
        print(f"(Error Message):\n{e.stderr}")
        raise

    with open(out_json_path, 'r') as file:
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

# STEP 3: String-fret Assignment MODEL

def run_fret_model(midi_file_path, inference, audio_path=None, capo=0, tuning=STANDARD_TUNING):

    final_tab_list: List[Tuple[str, str]] = inference(midi_file_path, audio_path=audio_path, capo=capo, tuning=tuning)

    # print("TAB",sorted(final_tab_list, key=lamda t:t.onset_sec)
    
    if final_tab_list is None:
        print("Pipeline failed to generate tabs.")
        return
    
    # print("TAB", final_tab_list)
    
    return final_tab_list

# for new postprocessing, should return a list of tuples
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
    parser.add_argument("--tempo", type=int, default=None, help="Tempo in BPM")
    args = parser.parse_args()
    AUDIO_PATH = args.audio_path
    CAPO = args.capo
    TUNING = tuning_conversion(args.tuning)
    BPM = args.tempo

    #----------------------------------------------------------------------------------#

    print("Running our pipeline on:", AUDIO_PATH)
    SCRIPT_DIR = Path(__file__).parent.resolve()
    MUSIC_TO_MIDI_PATH = str(
        SCRIPT_DIR  / "models" / "audio_to_midi" / 
        "gaps_goat_guitartechs_leduc_limited_regress_onset_offset_frame_velocity_bce_log332_iter2000_lr1e-05_bs4.pth"
    )
    MIDI_INFERENCE_SCRIPT = str(SCRIPT_DIR / "pipeline_utils" / "midi_utils" / "inference.py")
    MIN_AUDIO_SLICE_DURATION = 0.2 # This is set as minimal comprehensible input to the technique detection model. if you change this you must also change in postprocess.py
    
    #----------------------------------------------------------------------------------#
    
    bpm = BPM
    if BPM is None:
        try:
            bpm = estimate_bpm(AUDIO_PATH)
        except Exception as e:
            print(f"[warn] BeatNet BPM estimation failed ({type(e).__name__}: {e}). "
                  f"Falling back to 120 BPM. Pass --tempo <bpm> to override.")
            bpm = 120
    midi_path = run_midi_model(AUDIO_PATH, MUSIC_TO_MIDI_PATH, MIDI_INFERENCE_SCRIPT)
    midi_dict = find_single_note_onsets(midi_path)

    #----------------------------------------------------------------------------------#

    # technique specific MIDI dict
    midi_dict_techniques = [event for event in midi_dict if event['duration_seconds'] >= MIN_AUDIO_SLICE_DURATION] # for technique detection model
    # Note: the below function has several path variables you may need to change
    exp_onset_dur_tuples = run_technique_model_on_chunks(*audio_midi_to_chunks(AUDIO_PATH, midi_dict_techniques))
    # print(exp_onset_dur_tuples)
    # exp_onset_dur_tuples = cache.cached_technique_result(AUDIO_PATH, midi_dict_techniques, force_rerun=False) ## for TESTING

    #----------------------------------------------------------------------------------#

    print(BPM)
    tab_inference, post_process_tab = string_fret_inference_script.run_tab_generation, calculate_onsets
    tab_list = post_process_tab(run_fret_model(midi_path, tab_inference, audio_path=AUDIO_PATH, capo=CAPO, tuning=TUNING))

    #----------------------------------------------------------------------------------#

    ### CREATE JAMS FILE
    tuning_dict = {i + 1: pitch for i, pitch in enumerate(TUNING)}
    jam = preprocess.midi_to_jams_with_tablature_from_sf_assignment(midi_path, tab_list, bpm=bpm, tuning=tuning_dict, capo = CAPO)
    jam = preprocess.add_exp_techniques_to_existing_jam(jam, exp_onset_dur_tuples)
    RESULTS_DIR = SCRIPT_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    output_name = str(RESULTS_DIR / f"{os.path.splitext(os.path.basename(midi_path))[0]}.xml")
    output_musicxml = str(RESULTS_DIR / f"{os.path.splitext(os.path.basename(midi_path))[0]}.musicxml")
    jams_path = j.save_jam(jam, output_name)

    #----------------------------------------------------------------------------------#

    ### FINAL TAB GENERATION

    tab_generator_final(jams_path, output_musicxml, bpm = bpm)

    # Clean exit to avoid double-free errors from conflicting C extension
    # destructors (TF + PyTorch + librosa/madmom cython) during interpreter shutdown.
    os._exit(0)
