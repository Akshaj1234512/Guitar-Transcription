import numpy as np
import argparse
import os
import time
import logging
import h5py
import librosa
from utilities import (create_folder, float32_to_int16, create_logging, 
    get_filename, normalize_audio)
import config


def read_egdb_midi(midi_path):
    """Parse MIDI file of EGDB dataset using pretty_midi for correct timing.
    Combines all instruments/channels into a single channel for training.
    
    Args:
        midi_path: str, path to MIDI file
        
    Returns:
        dict: Dictionary containing MIDI events and timestamps in SECONDS
    """
    import pretty_midi
    
    midi_file = pretty_midi.PrettyMIDI(midi_path)
    
    midi_dict = {
        'midi_event': [],
        'midi_event_time': []
    }
    
    # Process ALL instruments and combine them into one channel
    for instrument in midi_file.instruments:
        for note in instrument.notes:
            # Note on event - combine all instruments into channel 0
            event_str = f"note_on channel=0 note={note.pitch} velocity={note.velocity} time={note.start}"
            midi_dict['midi_event'].append(event_str)
            midi_dict['midi_event_time'].append(note.start)
            
            # Note off event - combine all instruments into channel 0
            event_str = f"note_off channel=0 note={note.pitch} velocity=0 time={note.end}"
            midi_dict['midi_event'].append(event_str)
            midi_dict['midi_event_time'].append(note.end)
    
    # Sort events by time to ensure chronological order
    sorted_indices = np.argsort(midi_dict['midi_event_time'])
    midi_dict['midi_event'] = [midi_dict['midi_event'][i] for i in sorted_indices]
    midi_dict['midi_event_time'] = [midi_dict['midi_event_time'][i] for i in sorted_indices]
    
    return midi_dict


def pack_egdb_dataset_to_hdf5(args):
    """Load & resample EGDB audio files, then write to hdf5 files.
    
    This version creates a combined dataset using all available audio types
    with the same MIDI labels, effectively multiplying the training data.

    Args:
      dataset_dir: str, directory of EGDB dataset
      workspace: str, directory of your workspace
      audio_type: str, 'combined' to use all audio types, or specific type for single
    """
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    audio_type = args.audio_type
    
    sample_rate = config.sample_rate
    
    # Paths - EGDB has MIDI files in audio_label
    midi_dir = os.path.join(dataset_dir, 'audio_label')
    
    if audio_type == 'combined':
        # For combined dataset, we'll process all audio types
        # EGDB has these audio types based on the directory structure
        audio_types = ['audio_DI', 'audio_Mesa', 'audio_Marshall', 'audio_Ftwin', 'audio_Plexi', 'audio_JCjazz']
        waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'egdb', 'combined_train_val')
    else:
        # Single audio type
        audio_types = [audio_type]
        waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'egdb', audio_type)
    
    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    # Find all MIDI files
    midi_files = []
    for root, dirs, files in os.walk(midi_dir):
        for file in files:
            if file.endswith('.midi'):
                midi_files.append(file)
    
    midi_files.sort()
    audios_num = len(midi_files)
    logging.info('Total MIDI files: {}'.format(audios_num))
    
    # EGDB uses clips 193-240 for validation (80:20 split)
    # So clips 1-192 are for training, clips 193-240 are for validation
    train_indices = []
    validation_indices = []
    
    for i, midi_file in enumerate(midi_files):
        # Extract clip number from filename (e.g., "1.midi" -> 1)
        clip_name = os.path.splitext(midi_file)[0]  # Remove .midi extension
        try:
            clip_num = int(clip_name)  # Extract number
            if 1 <= clip_num <= 192:  # clips 1-192 are training (80%)
                train_indices.append(i)
            elif 193 <= clip_num <= 240:  # clips 193-240 are validation (20%)
                validation_indices.append(i)
            else:
                logging.warning(f'Unknown clip number: {clip_num} from {midi_file}')
        except ValueError:
            logging.warning(f'Could not parse clip number from {midi_file}')
    
    logging.info('Training clips: {}'.format(len(train_indices)))
    logging.info('Validation clips: {}'.format(len(validation_indices)))
    
    feature_time = time.time()
    
    # Load & resample each audio file to a hdf5 file
    for n in range(audios_num):
        midi_filename = midi_files[n]
        base_name = os.path.splitext(midi_filename)[0]  # Remove .midi extension

        # Process each audio type for this MIDI file
        for current_audio_type in audio_types:
            audio_dir = os.path.join(dataset_dir, current_audio_type)
            
            # Find corresponding audio file
            # EGDB naming convention: audio files match MIDI numbering (1.wav, 2.wav, etc.)
            audio_filename = base_name + '.wav'
            
            # Check if audio file exists
            if not os.path.exists(os.path.join(audio_dir, audio_filename)):
                logging.warning(f'No audio file found for {midi_filename} in {current_audio_type}: {audio_filename}')
                continue

            logging.info('{} {} ({}) -> {}'.format(n, midi_filename, current_audio_type, audio_filename))

            try:
                # Read MIDI (same for all audio types)
                midi_path = os.path.join(midi_dir, midi_filename)
                midi_dict = read_egdb_midi(midi_path)

                # Load audio
                audio_path = os.path.join(audio_dir, audio_filename)
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

                # Normalize audio
                audio = normalize_audio(audio)

                # Calculate duration
                duration = len(audio) / sample_rate

                # Determine split based on clip number
                if n in train_indices:
                    split = 'train'
                else:
                    split = 'val'

                # Create HDF5 filename with audio type
                hdf5_filename = base_name + '__' + current_audio_type + '.h5'
                
                # Create split-specific directory and place file there
                split_dir = os.path.join(waveform_hdf5s_dir, split)
                packed_hdf5_path = os.path.join(split_dir, hdf5_filename)

                create_folder(os.path.dirname(packed_hdf5_path))

                with h5py.File(packed_hdf5_path, 'w') as hf:
                    # Create attributes
                    hf.attrs.create('dataset', data='egdb'.encode(), dtype='S20')
                    hf.attrs.create('split', data=split.encode(), dtype='S20')
                    hf.attrs.create('year', data='2024'.encode(), dtype='S10')
                    hf.attrs.create('audio_type', data=current_audio_type.encode(), dtype='S30')
                    hf.attrs.create('midi_filename', data=midi_filename.encode(), dtype='S100')
                    hf.attrs.create('audio_filename', data=audio_filename.encode(), dtype='S100')
                    hf.attrs.create('duration', data=duration, dtype=np.float32)

                    # Create datasets
                    hf.create_dataset(name='midi_event',
                        data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                    hf.create_dataset(name='midi_event_time',
                        data=midi_dict['midi_event_time'], dtype=np.float32)
                    hf.create_dataset(name='waveform',
                        data=float32_to_int16(audio), dtype=np.int16)

            except Exception as e:
                logging.warning('Failed to process {} in {}: {}, skipping...'.format(midi_filename, current_audio_type, str(e)))
                continue
    
    logging.info('Write hdf5 to {}'.format(waveform_hdf5s_dir))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack EGDB dataset to HDF5 format')
    parser.add_argument('--dataset_dir', type=str, required=True, 
        help='Directory of EGDB dataset')
    parser.add_argument('--workspace', type=str, required=True, 
        help='Directory of your workspace')
    parser.add_argument('--audio_type', type=str, default='combined',
        choices=['combined', 'audio_DI', 'audio_Mesa', 'audio_Marshall', 'audio_Ftwin', 'audio_Plexi', 'audio_JCjazz'],
        help='Type of audio to process. "combined" uses all audio types (more training data!)')
    
    args = parser.parse_args()
    pack_egdb_dataset_to_hdf5(args)
