import os
import sys
import numpy as np
import h5py
import csv
import time
import collections
import librosa
import sox
import logging

from utilities import (create_folder, int16_to_float32, traverse_folder, 
    pad_truncate_sequence, TargetProcessor, write_events_to_midi, 
    plot_waveform_midi_targets)
import config

import os
import glob
import numpy as np
import librosa
import scipy.signal
from scipy.signal import butter, filtfilt

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
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        """
        [year, hdf5_path, start_time] = meta
        
        # The samplers now pass the full path to the HDF5 file
        # so we can use it directly
         
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

            if self.augmentor:
                waveform = self.augmentor.augment(waveform)

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

            data_dict['waveform'] = waveform

            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            
            # Note: Both GuitarSet and EGDB now store times in seconds directly
            # No conversion needed anymore

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


class Augmentor(object):
    """
    Augmentor that matches noisy GuitarSet generator:

    1) High-pass filter (>80Hz) with filtfilt (zero-phase, no timing shift)
    2) IR convolution
    3) Additive noise: white / pink / hum with random combinations
    4) Random SNR in [MIN_SNR, MAX_SNR], +3dB compensation if multiple noise sources
    5) Peak normalization to 0.9

    Notes:
    - Uses self.random_state for all randomness so you can control it via worker seeding.
    """

    def __init__(
        self,
        ir_path='/data/akshaj/MusicAI/consistency_IRs',
        sample_rate=16000,
        min_snr=25.0,
        max_snr=45.0,
        hum_freq=60.0,
        hpf_hz=80.0,
        max_ir_seconds=None,  # optionally cap IR length for speed (e.g., 2.0)
        prob=0.5,             
        seed=None
    ):
        self.sample_rate = int(sample_rate)
        self.min_snr = float(min_snr)
        self.max_snr = float(max_snr)
        self.hum_freq = float(hum_freq)
        self.hpf_hz = float(hpf_hz)
        self.max_ir_seconds = max_ir_seconds
        self.prob = float(prob)

        self.random_state = np.random.RandomState(seed)

        self.noise_combinations = [
            ['white'], ['pink'], ['hum'],
            ['white', 'pink'], ['white', 'hum'], ['pink', 'hum'],
            ['white', 'pink', 'hum']
        ]

        self.pink_b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float32)
        self.pink_a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float32)

        # Preload IRs
        self.irs = []
        if ir_path:
            ir_files = sorted(glob.glob(os.path.join(ir_path, '*.wav')))
            print(f"Augmentor: Loading {len(ir_files)} IR files from {ir_path} ...")

            for ir_file in ir_files:
                try:
                    ir, _ = librosa.load(ir_file, sr=self.sample_rate, mono=True)
                    if ir.size == 0:
                        continue

                    # Optional: cap IR length for speed
                    if self.max_ir_seconds is not None:
                        max_len = int(self.max_ir_seconds * self.sample_rate)
                        if ir.shape[0] > max_len:
                            ir = ir[:max_len]

                    ir = ir / (np.max(np.abs(ir)) + 1e-9)
                    self.irs.append(ir.astype(np.float32))
                except Exception as e:
                    print(f"Skipping IR {ir_file}: {e}")

            print(f"Augmentor: Successfully loaded {len(self.irs)} IRs.")


    def acoustic_tone_shaping(self, signal: np.ndarray) -> np.ndarray:
        """
        High Pass (>hpf_hz) to remove mud.
        Uses filtfilt (zero-phase) to ensure no timing shift.
        """
        sr = self.sample_rate
        b_high, a_high = butter(2, self.hpf_hz / (0.5 * sr), btype='high')
        return filtfilt(b_high, a_high, signal).astype(np.float32)


    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2))) if x.size else 0.0

    def add_white_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        rms_signal = self._rms(signal)
        if rms_signal == 0:
            return signal
        rms_noise = rms_signal / (10 ** (snr_db / 20))
        noise = self.random_state.normal(0.0, rms_noise, size=signal.shape).astype(np.float32)
        return (signal + noise).astype(np.float32)

    def add_pink_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        rms_signal = self._rms(signal)
        if rms_signal == 0:
            return signal

        white = self.random_state.normal(0.0, 1.0, size=signal.shape).astype(np.float32)
        pink = filtfilt(self.pink_b, self.pink_a, white).astype(np.float32)

        rms_pink = self._rms(pink)
        if rms_pink == 0:
            return signal

        rms_target = rms_signal / (10 ** (snr_db / 20))
        return (signal + pink * (rms_target / (rms_pink + 1e-9))).astype(np.float32)

    def add_mains_hum(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        sr = self.sample_rate
        t = (np.arange(len(signal), dtype=np.float32) / sr)
        hum_wave = np.sin(2 * np.pi * self.hum_freq * t) + 0.5 * np.sin(2 * np.pi * (2.0 * self.hum_freq) * t)

        rms_signal = self._rms(signal)
        rms_hum = self._rms(hum_wave)
        if rms_signal == 0 or rms_hum == 0:
            return signal

        rms_target = rms_signal / (10 ** (snr_db / 20))
        return (signal + hum_wave.astype(np.float32) * (rms_target / (rms_hum + 1e-9))).astype(np.float32)


    def convolve_ir_zero_latency(self, signal: np.ndarray) -> np.ndarray:
        """
        Matches your generator:
          wet = fftconvolve(signal, ir, mode='full')[:len(signal)]
        """
        if not self.irs:
            return signal
        ir = self.irs[self.random_state.randint(0, len(self.irs))]
        wet = scipy.signal.fftconvolve(signal, ir, mode='full')[:len(signal)]
        return wet.astype(np.float32)


    def augment(self, x: np.ndarray) -> np.ndarray:
        """
        x: 1D float waveform (already pitch/time shifted upstream in MaestroDataset)
        returns: augmented waveform, same length
        """
        x = x.astype(np.float32, copy=False)

        # Optional: sometimes skip augmentation
        if self.random_state.rand() > self.prob:
            return x

        # 1) Tone shaping (HPF, zero-phase)
        y = self.acoustic_tone_shaping(x)

        # 2) IR convolution (truncate => no global shift)
        y = self.convolve_ir_zero_latency(y)

        # 3) Add noise combination at random SNR
        active_noises = self.noise_combinations[self.random_state.randint(0, len(self.noise_combinations))]
        current_snr = float(self.random_state.uniform(self.min_snr, self.max_snr))
        final_snr = current_snr + (3.0 if len(active_noises) > 1 else 0.0)

        for noise_type in active_noises:
            if noise_type == 'white':
                y = self.add_white_noise(y, final_snr)
            elif noise_type == 'pink':
                y = self.add_pink_noise(y, final_snr)
            elif noise_type == 'hum':
                y = self.add_mains_hum(y, final_snr)

        # 4) Peak normalization to 0.9
        max_val = float(np.max(np.abs(y))) if y.size else 0.0
        if max_val > 0:
            y = (y / (max_val + 1e-9)) * 0.9

        return y.astype(np.float32)
    
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
            # Check if the file is in the correct split subdirectory
            split_dir = os.path.join(hdf5s_dir, split)
            if hdf5_path.startswith(split_dir):
                audio_name = hdf5_path.split('/')[-1]
                # Try to get year from HDF5 attributes, fallback to folder name if not available
                try:
                    with h5py.File(hdf5_path, 'r') as hf:
                        year_attr = hf.attrs.get('year', 'unknown')
                        if isinstance(year_attr, bytes):
                            year = year_attr.decode()
                        else:
                            year = str(year_attr)
                        duration = hf.attrs['duration']
                except:
                    # Fallback: try to extract year from path or use 'unknown'
                    year = 'unknown'
                    duration = 10.0  # Default duration if not available
                
                start_time = 0
                while (start_time + self.segment_seconds < duration):
                    self.segment_list.append([year, hdf5_path, start_time])
                    start_time += self.hop_seconds
                
                n += 1
                if mini_data and n == 10:
                    break
        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""

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
        """Sampler for testing.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool, sample from a small amount of data for debugging
        """
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
            # Use folder structure instead of HDF5 split tags
            # Check if the file is in the correct split subdirectory
            split_dir = os.path.join(hdf5s_dir, split)
            if hdf5_path.startswith(split_dir):
                audio_name = hdf5_path.split('/')[-1]
                # Try to get year from HDF5 attributes, fallback to folder name if not available
                try:
                    with h5py.File(hdf5_path, 'r') as hf:
                        year_attr = hf.attrs.get('year', 'unknown')
                        if isinstance(year_attr, bytes):
                            year = year_attr.decode()
                        else:
                            year = str(year_attr)
                        duration = hf.attrs['duration']
                except:
                    # Fallback: try to extract year from path or use 'unknown'
                    year = 'unknown'
                    duration = 10.0  # Default duration if not available
                
                start_time = 0
                while (start_time + self.segment_seconds < duration):
                    self.segment_list.append([year, hdf5_path, start_time])
                    start_time += self.hop_seconds
                
                n += 1
                if mini_data and n == 10:
                    break
        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""

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

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict