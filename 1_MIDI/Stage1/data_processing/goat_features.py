import numpy as np
import argparse
import os
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


def read_goat_midi(midi_path):
    """Parse MIDI with pretty_midi, output event strings + times in seconds."""
    import pretty_midi

    midi_file = pretty_midi.PrettyMIDI(midi_path)

    midi_dict = {'midi_event': [], 'midi_event_time': []}

    for instrument in midi_file.instruments:
        for note in instrument.notes:
            event_str = f"note_on channel=0 note={note.pitch} velocity={note.velocity} time={note.start}"
            midi_dict['midi_event'].append(event_str)
            midi_dict['midi_event_time'].append(note.start)

            event_str = f"note_off channel=0 note={note.pitch} velocity=0 time={note.end}"
            midi_dict['midi_event'].append(event_str)
            midi_dict['midi_event_time'].append(note.end)

    # Sort by time
    sorted_indices = np.argsort(midi_dict['midi_event_time'])
    midi_dict['midi_event'] = [midi_dict['midi_event'][i] for i in sorted_indices]
    midi_dict['midi_event_time'] = [midi_dict['midi_event_time'][i] for i in sorted_indices]

    return midi_dict


def _parse_item_and_amp(audio_filename):
    """
    Examples:
      item_0__item_0_amp_1.wav -> (item_0, amp_1)
      item_7__item_7_amp_5.wav -> (item_7, amp_5)
    Returns (item_id, amp_type) or (None, None) if unexpected.
    """
    base = os.path.splitext(audio_filename)[0]  # remove .wav
    if "__" not in base:
        return None, None
    item_id, rest = base.split("__", 1)  # item_0, item_0_amp_1
    amp_type = "mono"
    if "_amp_" in rest:
        try:
            amp_num = int(rest.rsplit("_amp_", 1)[1])
            amp_type = f"amp_{amp_num}"
        except Exception:
            amp_type = "amp_unknown"
    return item_id, amp_type


def _pick_midi_path(midi_dir, item_id):
    """
    Prefer fine aligned:
      item_7__item_7_fine_aligned.mid/.midi
    Fallback:
      item_7__item_7.mid/.midi
    """
    # prefer fine aligned
    for ext in [".mid", ".midi"]:
        p = os.path.join(midi_dir, f"{item_id}__{item_id}_fine_aligned{ext}")
        if os.path.exists(p):
            return p

    # fallback regular
    for ext in [".mid", ".midi"]:
        p = os.path.join(midi_dir, f"{item_id}__{item_id}{ext}")
        if os.path.exists(p):
            return p

    # last resort: any midi starting with item_id__
    for ext in [".mid", ".midi"]:
        candidates = [f for f in os.listdir(midi_dir)
                      if f.startswith(item_id + "__") and f.lower().endswith(ext)]
        if candidates:
            candidates.sort()
            return os.path.join(midi_dir, candidates[0])

    return None


def pack_goat_dataset_to_hdf5(args):
    """
    GAPS-style packer.
    Args:
      dataset_dir: must contain audio/ and midi/
      workspace: output root
    """
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    sample_rate = config.sample_rate

    audio_dir = os.path.join(dataset_dir, "audio")
    midi_dir = os.path.join(dataset_dir, "midi")

    waveform_hdf5s_dir = os.path.join(workspace, "hdf5s", "goat", "combined", "2024")

    logs_dir = os.path.join(workspace, "logs", get_filename(__file__))
    create_logging(logs_dir, filemode="w")
    logging.info(args)

    # Collect audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
    audio_files.sort()
    if len(audio_files) == 0:
        logging.warning("No audio files found in {}".format(audio_dir))
        return

    # Determine unique items from audio filenames (so split is by item)
    item_ids = []
    for f in audio_files:
        item_id, _ = _parse_item_and_amp(f)
        if item_id is not None:
            item_ids.append(item_id)
    item_ids = sorted(list(set(item_ids)))

    logging.info("Total audio files: {}".format(len(audio_files)))
    logging.info("Total unique items: {}".format(len(item_ids)))

    # Train/val split by item id (80/20)
    np.random.seed(1234)
    perm = np.random.permutation(len(item_ids))
    train_items = set([item_ids[i] for i in perm[:int(0.8 * len(item_ids))]])
    val_items = set([item_ids[i] for i in perm[int(0.8 * len(item_ids)):]])

    feature_time = time.time()
    processed = 0
    skipped_no_midi = 0

    for n, audio_filename in enumerate(audio_files):
        item_id, amp_type = _parse_item_and_amp(audio_filename)
        if item_id is None:
            logging.warning("Skipping (unexpected name): {}".format(audio_filename))
            continue

        midi_path = _pick_midi_path(midi_dir, item_id)
        if midi_path is None:
            skipped_no_midi += 1
            logging.warning("No MIDI found for {}, skipping {}".format(item_id, audio_filename))
            continue

        audio_path = os.path.join(audio_dir, audio_filename)

        # Split based on item_id
        split = "train" if item_id in train_items else "validation"

        # HDF5 filename: match EGDB style: base__audio_type.h5
        base_name = os.path.splitext(audio_filename)[0]  # item_0__item_0_amp_1
        hdf5_filename = base_name + "__" + amp_type + ".h5"

        split_dir = os.path.join(waveform_hdf5s_dir, split)
        packed_hdf5_path = os.path.join(split_dir, hdf5_filename)
        create_folder(os.path.dirname(packed_hdf5_path))

        logging.info("{} {} -> {} ({})".format(n, audio_filename, os.path.basename(midi_path), amp_type))

        try:
            # Read MIDI
            midi_dict = read_goat_midi(midi_path)

            # Load audio
            audio, _ = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            audio = normalize_audio(audio)
            duration = len(audio) / sample_rate

            with h5py.File(packed_hdf5_path, "w") as hf:
                hf.attrs.create("dataset", data="goat".encode(), dtype="S20")
                hf.attrs.create("split", data=split.encode(), dtype="S20")
                hf.attrs.create("year", data="2024".encode(), dtype="S10")
                hf.attrs.create("audio_type", data=amp_type.encode(), dtype="S30")
                hf.attrs.create("item_id", data=item_id.encode(), dtype="S50")
                hf.attrs.create("midi_filename", data=os.path.basename(midi_path).encode(), dtype="S200")
                hf.attrs.create("audio_filename", data=audio_filename.encode(), dtype="S200")
                hf.attrs.create("duration", data=duration, dtype=np.float32)

                hf.create_dataset(
                    name="midi_event",
                    data=[e.encode() for e in midi_dict["midi_event"]],
                    dtype="S120"
                )
                hf.create_dataset(
                    name="midi_event_time",
                    data=midi_dict["midi_event_time"],
                    dtype=np.float32
                )
                hf.create_dataset(
                    name="waveform",
                    data=float32_to_int16(audio),
                    dtype=np.int16
                )

            processed += 1

        except Exception as e:
            logging.warning("Failed to process {}: {}, skipping...".format(audio_filename, str(e)))
            continue

    logging.info("Write hdf5 to {}".format(waveform_hdf5s_dir))
    logging.info("Processed: {}, skipped_no_midi: {}".format(processed, skipped_no_midi))
    logging.info("Time: {:.3f} s".format(time.time() - feature_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack GOAT dataset to HDF5 format")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory of GOAT dataset (with audio/ and midi/ subfolders)")
    parser.add_argument("--workspace", type=str, required=True,
                        help="Directory of your workspace")

    args = parser.parse_args()
    pack_goat_dataset_to_hdf5(args)
