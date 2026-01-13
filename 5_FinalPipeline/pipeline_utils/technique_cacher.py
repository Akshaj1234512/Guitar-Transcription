import pretty_midi
import py
import torch
import jams
import torch.nn as nn
import librosa
import subprocess
import os
import soundfile as sf
from pathlib import Path
import sys
import json
from typing import List, Tuple


import argparse
import shutil
import scipy.signal
import scipy.signal.windows

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


### Temporary caching script
def cached_peter_result(audio_path, midi_dict_peter, force_rerun=False):
    """
    Cache Peter's expressive technique predictions to avoid recomputing.
    
    Args:
        audio_path: Path to audio file
        midi_dict_peter: List of MIDI note dicts for Peter's model
        force_rerun: If True, ignore cache and recompute
        
    Returns:
        List of (technique, onset_ms, duration_ms) tuples
    """
    # Create cache directory
    cache_dir = Path("cache/peter_techniques")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache filename based on audio file
    audio_basename = Path(audio_path).stem
    cache_file = cache_dir / f"{audio_basename}.json"
    
    # Try to load from cache
    if not force_rerun and cache_file.exists():
        print(f"✓ Loading Peter's techniques from cache: {cache_file}")
        try:
            with open(cache_file, 'r') as f:
                exp_onset_dur_tuples = json.load(f)
            print(f"  Loaded {len(exp_onset_dur_tuples)} technique predictions from cache")
            return exp_onset_dur_tuples
        except Exception as e:
            print(f"  Cache load failed ({e}), recomputing...")
    
    # Cache miss or force rerun - compute the result
    print(f"⟳ Running Peter's model (no cache or force rerun)...")
    
    # Extract audio chunks and run Peter's model
    chunk_paths, midi_onsets, midi_durations = audio_midi_to_chunks(audio_path, midi_dict_peter)
    exp_onset_dur_tuples = run_peter_model_on_chunks(chunk_paths, midi_onsets, midi_durations)
    
    # Save to cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(exp_onset_dur_tuples, f, indent=2)
        print(f"✓ Saved {len(exp_onset_dur_tuples)} technique predictions to cache: {cache_file}")
    except Exception as e:
        print(f"⚠ Cache save failed ({e})")
    
    return exp_onset_dur_tuples
