import numpy as np
import argparse
import os
import glob
import time
import logging
import h5py
import librosa
from utilities import (
    create_folder,
    float32_to_int16,
    create_logging,
    get_filename,
    normalize_audio
)
import config


def read_gtechs_midi(midi_path):
    """Parse MIDI file using pretty_midi for correct timing.
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


def pack_gtechs_dataset_to_hdf5(args):
    """Load & resample Guitar-TECHS audio files, then write to HDF5 files.

    Assumes:
        dataset_dir/
            audio/   -> .wav files
            midi/    -> .mid files

    One audio per MIDI file, 80/20 train/validation split.
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    sample_rate = config.sample_rate

    # Use your actual folder names here
    midi_dir = dataset_dir
    audio_dir = dataset_dir

    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'gtechs', 'combined', '2025')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # find all the different players' folders
    dataset_subfolders = glob.glob(os.path.join(midi_dir, 'P[1-3]_*'))

    midi_files = glob.glob(os.path.join(midi_dir, 'P[1-3]_*', 'midi', '*.mid'))

    midi_files.sort()

    audios_num = len(midi_files)
    logging.info('Total MIDI files: {}'.format(audios_num))

    if audios_num == 0:
        logging.warning('No MIDI files found in {}'.format(midi_dir))
        return

    # Create train/validation split (80/20)
    np.random.seed(1234)
    indices = np.random.permutation(audios_num)
    train_indices = indices[:int(0.8 * audios_num)]
    validation_indices = indices[int(0.8 * audios_num):]

    feature_time = time.time()
    channel_folders = ['directinput', 'micamp']

    # Load & resample each audio file to a HDF5 file
    for n in range(audios_num):
        midi_path = midi_files[n]
        midi_filename = os.path.basename(midi_path)
        base_name = os.path.splitext(midi_filename)[0].replace('midi_', '')

        for channel_folder in channel_folders:

            # Assume corresponding audio file is base_name + '.mp3' in audio_dir
            item_dir_path = os.path.dirname(midi_path)
            audio_path = os.path.join(item_dir_path.replace('midi', 'audio'), channel_folder, f"{channel_folder}_{base_name}.wav")
            current_base_name = f"{channel_folder}_{base_name}"

            if not os.path.exists(audio_path):
                logging.warning('No audio file found for MIDI {} (expected {})'.format(
                    midi_filename, audio_path))
                continue

            logging.info('{} {}'.format(n, midi_filename, channel_folder))

            try:
                # Read MIDI
                midi_dict = read_gtechs_midi(midi_path)

                # Load audio
                audio, _ = librosa.core.load(audio_path, sr=sample_rate, mono=True)

                # Normalize audio
                audio = normalize_audio(audio)

                # Calculate duration
                duration = len(audio) / sample_rate

                # Determine split
                if n in train_indices:
                    split = 'train'
                else:
                    split = 'validation'

                # HDF5 filename
                hdf5_filename = current_base_name + '.h5'

                # Create split-specific directory and place file there
                split_dir = os.path.join(waveform_hdf5s_dir, split)
                packed_hdf5_path = os.path.join(split_dir, hdf5_filename)

                create_folder(os.path.dirname(packed_hdf5_path))

                with h5py.File(packed_hdf5_path, 'w') as hf:
                    # Attributes
                    hf.attrs.create('dataset', data='gtechs'.encode(), dtype='S20')
                    hf.attrs.create('split', data=split.encode(), dtype='S20')
                    hf.attrs.create('year', data='2025'.encode(), dtype='S10')
                    hf.attrs.create('audio_type', data='mono'.encode(), dtype='S30')
                    hf.attrs.create('midi_filename', data=midi_filename.encode(), dtype='S100')
                    hf.attrs.create('audio_filename', data=os.path.basename(audio_path).encode(), dtype='S100')
                    hf.attrs.create('duration', data=duration, dtype=np.float32)

                    # Datasets
                    hf.create_dataset(
                        name='midi_event',
                        data=[e.encode() for e in midi_dict['midi_event']],
                        dtype='S100'
                    )
                    hf.create_dataset(
                        name='midi_event_time',
                        data=midi_dict['midi_event_time'],
                        dtype=np.float32
                    )
                    hf.create_dataset(
                        name='waveform',
                        data=float32_to_int16(audio),
                        dtype=np.int16
                    )

            except Exception as e:
                logging.warning('Failed to process {} {}: {}, skipping...'.format(
        midi_filename, channel_folder, str(e)))
                continue

    logging.info('Write hdf5 to {}'.format(waveform_hdf5s_dir))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack Guitar-TECHS dataset to HDF5 format')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory of Guitar-TECHS dataset (with audio/ and midi/ subfolders)')
    parser.add_argument('--workspace', type=str, required=True,
                        help='Directory of your workspace')

    args = parser.parse_args()
    pack_gtechs_dataset_to_hdf5(args)