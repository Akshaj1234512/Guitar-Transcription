#!/usr/bin/env python3
import os
import glob
import random
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve, butter, filtfilt

# --- Configuration ---
guitarset_dir = '/data/akshaj/MusicAI/GuitarSet/audio_hex_debleeded'
ir_dir = '/data/akshaj/MusicAI/EchoThief/noise' 
output_dir = '/data/akshaj/MusicAI/GuitarSet/acoustic_noisy'
TARGET_SR = 16000 

# Noise Level: Lower dB = More Noise
MIN_SNR = 25.0 
MAX_SNR = 45.0

os.makedirs(output_dir, exist_ok=True)
random.seed(42)
np.random.seed(42)

# --- Tone & Noise Functions ---

def acoustic_tone_shaping(signal, sr):
    """
    High Pass (>80Hz) to remove mud.
    Uses filtfilt (Zero-Phase) to ensure no timing shift.
    """
    b_high, a_high = butter(2, 80 / (0.5 * sr), btype='high')
    # filtfilt processes data forward and backward so there is zero phase delay
    signal = filtfilt(b_high, a_high, signal)
    return signal

def add_white_noise(signal, snr_db):
    rms_signal = np.sqrt(np.mean(signal**2))
    if rms_signal == 0: return signal
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, signal.shape)
    return signal + noise

def add_pink_noise(signal, snr_db):
    rms_signal = np.sqrt(np.mean(signal**2))
    if rms_signal == 0: return signal
    white = np.random.normal(0, 1, signal.shape)
    # Pink noise filter coefficients
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1, -2.494956002,   2.017265875,  -0.522189400])
    pink = filtfilt(b, a, white)
    rms_pink = np.sqrt(np.mean(pink**2))
    if rms_pink == 0: return signal
    rms_target = rms_signal / (10 ** (snr_db / 20))
    return signal + pink * (rms_target / rms_pink)

def add_mains_hum(signal, sr, snr_db, freq=60):
    t = np.arange(len(signal)) / sr
    hum_wave = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * (freq * 2) * t)
    rms_signal = np.sqrt(np.mean(signal**2))
    rms_hum = np.sqrt(np.mean(hum_wave**2))
    if rms_hum == 0: return signal
    rms_target = rms_signal / (10 ** (snr_db / 20))
    return signal + hum_wave * (rms_target / rms_hum)

# --- Main Pipeline ---

noise_combinations = [
    ['white'], ['pink'], ['hum'], 
    ['white', 'pink'], ['white', 'hum'], ['pink', 'hum'], 
    ['white', 'pink', 'hum']
]

guitar_files = sorted(glob.glob(os.path.join(guitarset_dir, '*.wav')))
ir_files = sorted(glob.glob(os.path.join(ir_dir, '*.wav')))

print(f"Processing {len(guitar_files)} files. Using Zero-Latency Alignment...")

for g_file in guitar_files:
    try:
        # 1. Load Hex Guitar
        hex_signal, _ = librosa.load(g_file, sr=TARGET_SR, mono=False)
        
        # 2. Sum to Mono 
        di_signal = np.sum(hex_signal, axis=0) if hex_signal.ndim > 1 else hex_signal
        
        # 3. Tone Shaping (Zero Phase)
        di_signal = acoustic_tone_shaping(di_signal, TARGET_SR)

        # 4. Convolve with Room IR (FIXED: Zero-Latency)
        ir_path = random.choice(ir_files)
        ir, _ = librosa.load(ir_path, sr=TARGET_SR, mono=True)
        ir = ir / (np.max(np.abs(ir)) + 1e-9) # Normalize IR energy

        # We take only the first len(di_signal) samples starting from index 0.
        # This keeps the direct guitar sound perfectly aligned with the MIDI labels.
        wet_signal = fftconvolve(di_signal, ir, mode='full')[:len(di_signal)]

        # 5. Add Noise Profile
        active_noises = random.choice(noise_combinations)
        current_snr = random.uniform(MIN_SNR, MAX_SNR)
        
        # Compensate SNR for multiple noise sources
        final_snr = current_snr + (3.0 if len(active_noises) > 1 else 0.0)
        noisy_signal = wet_signal.copy()

        for noise_type in active_noises:
            if noise_type == 'white':
                noisy_signal = add_white_noise(noisy_signal, final_snr)
            elif noise_type == 'pink':
                noisy_signal = add_pink_noise(noisy_signal, final_snr)
            elif noise_type == 'hum':
                noisy_signal = add_mains_hum(noisy_signal, TARGET_SR, final_snr)

        # 6. Final Normalization
        max_val = np.max(np.abs(noisy_signal))
        if max_val > 0:
            noisy_signal = (noisy_signal / max_val) * 0.9
        
        # 7. Save
        sf.write(os.path.join(output_dir, os.path.basename(g_file)), noisy_signal, TARGET_SR)

    except Exception as e:
        print(f"Error on {g_file}: {e}")

print("Done! All audio is now Zero-Latency aligned with labels.")