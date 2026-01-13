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
import argparse
import shutil
import scipy.signal
import scipy.signal.windows

### Not using this version anymore
def find_bpm_from_audio(audio_path):
    # Fix the 'hann' error by redirecting the old name to the new location
    scipy.signal.hann = scipy.signal.windows.hann
    # Fix the 'librosa.core' error if it appears
    if not hasattr(librosa, 'core'):
        librosa.core = librosa

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    # Estimate BPM using beat track function
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Estimated Audio BPM: {tempo.item(0):.2f}")
    return tempo.item(0)