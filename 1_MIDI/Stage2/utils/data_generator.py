import os
import sys
import glob
import numpy as np
import h5py
import csv
import time
import collections
import librosa
import sox
import logging
import scipy.signal
import random
from utilities import (create_folder, int16_to_float32, traverse_folder, 
    pad_truncate_sequence, TargetProcessor, write_events_to_midi, 
    plot_waveform_midi_targets)
import config


class MaestroDataset(object):
    def __init__(self, hdf5s_dir, segment_seconds, frames_per_second, 
        max_note_shift=0, max_timing_shift=0, augmentor=None):
        """This class takes the meta of an audio segment as input, and return 
        the waveform and targets of the audio segment. This class is used by 
        DataLoader. 
        
        Args:
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
          max_note_shift: int, number of semitone for pitch augmentation
          augmentor: object
        """
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.sample_rate = config.sample_rate
        self.max_note_shift = max_note_shift
        self.max_timing_shift = max_timing_shift
        self.begin_note = config.begin_note
        self.classes_num = config.classes_num
        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        self.augmentor = augmentor

        self.random_state = np.random.RandomState(1234)

        self.target_processor = TargetProcessor(self.segment_seconds, 
            self.frames_per_second, self.begin_note, self.classes_num)
        """Used for processing MIDI events to target."""

    def __getitem__(self, meta):
        """Prepare input and target of a segment for training.
        
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': '/full/path/to/file.h5', 
            'start_time': 65.0}

        Returns:
          data_dict: {
            'waveform_clean': (samples_num,)   <-- NEW (If augmentor exists)
            'waveform_aug':   (samples_num,)   <-- NEW (If augmentor exists)
            'waveform':       (samples_num,)   <-- (Only in Val/Test mode)
            'onset_roll': (frames_num, classes_num), 
            ... }
        """
        [year, hdf5_path, start_time] = meta
        
        data_dict = {}

        note_shift = self.random_state.randint(low=-self.max_note_shift, 
            high=self.max_note_shift + 1)

        # Load hdf5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + self.segment_samples

            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample : end_sample])

            # --- 1. Geometric Augmentation (Pitch & Time Shift) ---
            # We apply this FIRST so it affects both Clean and Augmented versions equally,
            # ensuring they both match the MIDI targets.

            if note_shift != 0:
                """Augment pitch"""
                waveform = librosa.effects.pitch_shift(waveform, sr=self.sample_rate, 
                    n_steps=note_shift, bins_per_octave=12)

            # Apply timing shift augmentation
            if self.max_timing_shift > 0:
                timing_shift = self.random_state.uniform(
                    low=-self.max_timing_shift, 
                    high=self.max_timing_shift)
                # Shift audio by timing_shift seconds
                shift_samples = int(timing_shift * self.sample_rate)
                waveform = np.roll(waveform, shift_samples)
            else:
                timing_shift = 0

            # --- 2. Consistency Branching (Clean vs Aug) ---
            if self.augmentor:
                # TRAINING MODE: Return both Clean and Augmented
                data_dict['waveform_clean'] = waveform.copy()
                data_dict['waveform_aug'] = self.augmentor.augment(waveform.copy())
            else:
                # VALIDATION/TEST MODE: Return standard waveform
                data_dict['waveform'] = waveform

            # --- 3. Process Targets ---
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            
            # Process MIDI events to target
            (target_dict, note_events, pedal_events) = \
                self.target_processor.process(start_time, midi_events_time, 
                    midi_events, extend_pedal=True, note_shift=note_shift, timing_shift=timing_shift)

        # Combine input and target
        for key in target_dict.keys():
            data_dict[key] = target_dict[key]

        debugging = False
        if debugging:
            plot_waveform_midi_targets(data_dict, start_time, note_events)
            exit()

        return data_dict

import os
import glob
import numpy as np
import scipy.signal
import scipy.io.wavfile
import sox

import os
import glob
import numpy as np
import scipy.signal
import scipy.io.wavfile
import sox

import os
import glob
import numpy as np
import scipy.signal
import librosa

# class Augmentor(object):
#     def __init__(self, ir_path=None, sample_rate=16000):
#         self.sample_rate = sample_rate
#         # Using a fixed seed for reproducibility in debug, 
#         # but the random_state ensures variety during training.
#         self.random_state = np.random.RandomState(42)
#         self.irs = []
        
#         if ir_path:
#             # We use sorted to ensure file order is consistent across environments
#             ir_files = sorted(glob.glob(os.path.join(ir_path, '*.wav')))
#             print(f"Augmentor: Loading {len(ir_files)} IR files...")
            
#             for ir_file in ir_files:
#                 try:
#                     # EXACT MATCH: librosa load at TARGET_SR, mono=True
#                     ir, _ = librosa.load(ir_file, sr=self.sample_rate, mono=True)
#                     # Normalize IR as in your logic
#                     ir = ir / (np.max(np.abs(ir)) + 1e-9)
#                     self.irs.append(ir)
#                 except Exception as e:
#                     print(f"Skipping IR {ir_file}: {e}")
#             print(f"Augmentor: Successfully loaded {len(self.irs)} IRs.")

#     def apply_clipping_distortion(self, signal, gain):
#         # EXACT MATCH: np.tanh(signal * gain)
#         return np.tanh(signal * gain)

#     def augment(self, x):
#         """
#         Direct port of your distortion script.
#         Guarantees 0ms alignment error.
#         """
#         # 1. Apply Distortion
#         gain = self.random_state.uniform(5.0, 15.0)
#         clipped_signal = self.apply_clipping_distortion(x, gain)

#         # 2. Convolution with Peak Alignment
#         if self.irs:
#             # random.choice(ir_files) equivalent
#             idx = self.random_state.randint(0, len(self.irs))
#             ir = self.irs[idx]
            
#             # EXACT MATCH: peak_idx = np.argmax(np.abs(ir))
#             peak_idx = np.argmax(np.abs(ir))
            
#             # EXACT MATCH: fftconvolve(..., mode='full')
#             full_conv = scipy.signal.fftconvolve(clipped_signal, ir, mode='full')
            
#             # EXACT MATCH: slice starting at PEAK_IDX for length of SIGNAL
#             x_aug = full_conv[peak_idx : peak_idx + len(x)]
#         else:
#             x_aug = clipped_signal

#         # 3. Peak Normalization
#         # EXACT MATCH: (distorted_signal / max_val) * 0.9
#         max_val = np.max(np.abs(x_aug))
#         if max_val > 0:
#             x_aug = (x_aug / max_val) * 0.9
                
#         return x_aug
    

import numpy as np
import scipy.signal
import librosa
import glob
import os

class Augmentor(object):
    def __init__(self, ir_path=None, sample_rate=16000):
        self.sample_rate = sample_rate
        # REMOVE FIXED SEED: Let numpy handle entropy, or seed via worker_init_fn
        # If you really need debug reproducibility, pass the seed as an argument, 
        # but default to None.
        self.random_state = np.random.RandomState(None) 
        
        self.irs = []
        
        if ir_path:
            ir_files = sorted(glob.glob(os.path.join(ir_path, '*.wav')))
            print(f"Augmentor: Loading {len(ir_files)} IR files...")
            
            for ir_file in ir_files:
                try:
                    ir, _ = librosa.load(ir_file, sr=self.sample_rate, mono=True)
                    ir = ir / (np.max(np.abs(ir)) + 1e-9)
                    self.irs.append(ir)
                except Exception as e:
                    print(f"Skipping IR {ir_file}: {e}")
            print(f"Augmentor: Successfully loaded {len(self.irs)} IRs.")

    def apply_clipping_distortion(self, signal, gain):
        return np.tanh(signal * gain)

    def augment(self, x):
        # 1. Apply Distortion
        # UPDATED RANGE: 2.0 (Crunch) to 15.0 (High Gain)
        # This helps the student learn the transition from clean to distorted
        gain = self.random_state.uniform(2.0, 15.0)
        clipped_signal = self.apply_clipping_distortion(x, gain)

        # 2. Convolution with Peak Alignment
        if self.irs:
            idx = self.random_state.randint(0, len(self.irs))
            ir = self.irs[idx]
            
            # This logic is PERFECT. Keep it exactly like this.
            # It ensures the "attack" of the IR lines up with the "attack" of the input.
            peak_idx = np.argmax(np.abs(ir))
            full_conv = scipy.signal.fftconvolve(clipped_signal, ir, mode='full')
            
            # Safety check: ensure we don't go out of bounds (rare edge case)
            if len(full_conv) >= peak_idx + len(x):
                x_aug = full_conv[peak_idx : peak_idx + len(x)]
            else:
                # Fallback if convolution output is unexpectedly short
                x_aug = full_conv[:len(x)] 
        else:
            x_aug = clipped_signal

        # 3. Peak Normalization
        max_val = np.max(np.abs(x_aug))
        if max_val > 1e-9: # slightly safer epsilon
            x_aug = (x_aug / max_val) * 0.9
                
        return x_aug
    
class Sampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data, random_seed=1234):
        """Sampler for training.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'val']
        self.hdf5s_dir = hdf5s_dir
        self.split = split
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            # Use folder structure instead of HDF5 split tags
            split_dir = os.path.join(hdf5s_dir, split)
            if hdf5_path.startswith(split_dir):
                audio_name = hdf5_path.split('/')[-1]
                try:
                    with h5py.File(hdf5_path, 'r') as hf:
                        year_attr = hf.attrs.get('year', 'unknown')
                        if isinstance(year_attr, bytes):
                            year = year_attr.decode()
                        else:
                            year = str(year_attr)
                        duration = hf.attrs['duration']
                except:
                    year = 'unknown'
                    duration = 10.0
                
                start_time = 0
                while (start_time + self.segment_seconds < duration):
                    self.segment_list.append([year, hdf5_path, start_time])
                    start_time += self.hop_seconds
                
                n += 1
                if mini_data and n == 10:
                    break

        logging.info('{} segments: {}'.format(split, len(self.segment_list)))

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def __len__(self):
        return -1
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


class TestSampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data, random_seed=1234):
        """Sampler for testing."""
        assert split in ['train', 'val', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.max_evaluate_iteration = 20    # Number of mini-batches to validate

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            split_dir = os.path.join(hdf5s_dir, split)
            if hdf5_path.startswith(split_dir):
                audio_name = hdf5_path.split('/')[-1]
                try:
                    with h5py.File(hdf5_path, 'r') as hf:
                        year_attr = hf.attrs.get('year', 'unknown')
                        if isinstance(year_attr, bytes):
                            year = year_attr.decode()
                        else:
                            year = str(year_attr)
                        duration = hf.attrs['duration']
                except:
                    year = 'unknown'
                    duration = 10.0
                
                start_time = 0
                while (start_time + self.segment_seconds < duration):
                    self.segment_list.append([year, hdf5_path, start_time])
                    start_time += self.hop_seconds
                
                n += 1
                if mini_data and n == 10:
                    break

        logging.info('Evaluate {} segments: {}'.format(split, len(self.segment_list)))

        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        pointer = 0
        iteration = 0

        while True:
            if iteration == self.max_evaluate_iteration:
                break

            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer]
                pointer += 1
                
                batch_segment_list.append(self.segment_list[index])
                i += 1

            iteration += 1

            yield batch_segment_list

    def __len__(self):
        return -1


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.
    Automatically handles any keys present in the dictionary.
    
    Returns:
      np_data_dict: {
        'waveform': (batch_size, samples) OR 'waveform_clean'/'waveform_aug',
        'onset_roll': (batch_size, frames, classes),
        ...
      }
    """
    np_data_dict = {}
    
    # We iterate over the keys of the first element to determine what to stack
    # This works for both standard training ('waveform') and consistency ('waveform_clean', 'waveform_aug')
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict