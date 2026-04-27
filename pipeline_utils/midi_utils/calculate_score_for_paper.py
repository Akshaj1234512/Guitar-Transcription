import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../../autoth'))
import numpy as np
import argparse
import librosa
import mir_eval
import torch
import time
import h5py
import pickle 
from sklearn import metrics
from concurrent.futures import ProcessPoolExecutor
 
from utilities import (create_folder, get_filename, traverse_folder, 
    int16_to_float32, note_to_freq, TargetProcessor, RegressionPostProcessor, 
    OnsetsFramesPostProcessor)
import config
from inference import PianoTranscription


def infer_prob(args):
    """Inference the output probabilites on any dataset, and write out to
    disk. This will reduce duplicate computation for later evaluation.
    """

    # Arguments & parameters
    workspace = args.workspace
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    augmentation = args.augmentation
    dataset_name = args.dataset_name
    hdf5s_dir = args.hdf5s_dir
    split = args.split
    post_processor_type = args.post_processor_type
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    begin_note = config.begin_note

    # Paths
    model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    # --- MIDI generation commented out ---
    # predicted_midi_dir = os.path.join(workspace, 'probs', f'MIDI_{model_name}_{dataset_name}')
    # create_folder(predicted_midi_dir)
    
    probs_dir = os.path.join(workspace, 'probs', f'PKL_{model_name}_{dataset_name}')
    create_folder(probs_dir)

    # Transcriptor
    transcriptor = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples, 
        post_processor_type=post_processor_type)

    (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)

    for n, hdf5_path in enumerate(hdf5_paths):
        print(n, hdf5_path)

        # Load audio                
        with h5py.File(hdf5_path, 'r') as hf:
            audio = int16_to_float32(hf['waveform'][:])
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
        
        # Ground truths processor
        target_processor = TargetProcessor(
            segment_seconds=len(audio) / sample_rate, 
            frames_per_second=frames_per_second, begin_note=begin_note, 
            classes_num=classes_num)

        (target_dict, note_events, pedal_events) = \
            target_processor.process(start_time=0, 
                midi_events_time=midi_events_time, 
                midi_events=midi_events, extend_pedal=True)

        ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
        ref_midi_notes = np.array([event['midi_note'] for event in note_events])
        ref_velocity = np.array([event['velocity'] for event in note_events])

        # --- Bypassing the crashing transcriptor.transcribe() ---
        # Instead of calling transcribe, we do a raw forward pass to get the output_dict
        
        audio_input = audio[None, :]  # (1, audio_samples)
        
        # Pad audio
        audio_len = audio_input.shape[1]
        pad_len = int(np.ceil(audio_len / segment_samples)) * segment_samples - audio_len
        audio_padded = np.concatenate((audio_input, np.zeros((1, pad_len))), axis=1)

        # Forward Pass
        segments = transcriptor.enframe(audio_padded, segment_samples)
        from pytorch_utils import forward
        output_dict = forward(transcriptor.model, segments, batch_size=32)

        # Deframe
        for key in output_dict.keys():
            output_dict[key] = transcriptor.deframe(output_dict[key])[0 : audio_len]

        # Pack probabilities to dump (The .pkl generation)
        total_dict = {key: output_dict[key] for key in output_dict.keys()}
        total_dict['frame_roll'] = target_dict['frame_roll']
        total_dict['ref_on_off_pairs'] = ref_on_off_pairs
        total_dict['ref_midi_notes'] = ref_midi_notes
        total_dict['ref_velocity'] = ref_velocity

        if 'pedal_frame_output' in output_dict.keys():
            total_dict['ref_pedal_on_off_pairs'] = \
                np.array([[event['onset_time'], event['offset_time']] for event in pedal_events])
            total_dict['pedal_frame_roll'] = target_dict['pedal_frame_roll']
            
        prob_path = os.path.join(probs_dir, '{}.pkl'.format(get_filename(hdf5_path)))
        create_folder(os.path.dirname(prob_path))
        pickle.dump(total_dict, open(prob_path, 'wb'))


class ScoreCalculator(object):
    def __init__(self, hdf5s_dir, probs_dir, split, post_processor_type='regression'):
        """Evaluate piano transcription metrics of the post processed 
        pre-calculated system outputs.
        """
        self.split = split
        self.probs_dir = probs_dir
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.velocity_scale = config.velocity_scale
        self.velocity = True  # True | False
        self.pedal = False

        self.evaluate_frame = True
        self.onset_tolerance = 0.05
        self.offset_ratio = None  # None | 0.2
        self.offset_min_tolerance = None

        self.pedal_offset_threshold = 0.2
        self.pedal_offset_ratio = None  # None | 0.2
        self.pedal_offset_min_tolerance = None

        self.post_processor_type = post_processor_type
        
        (hdf5_names, self.hdf5_paths) = traverse_folder(hdf5s_dir)

    def __call__(self, params):
        """Calculate metrics of all songs.

        Args:
          params: list of float, thresholds
        """
        stats_dict = self.metrics(params)
        return np.mean(stats_dict['f1'])

    def metrics(self, params):
        """Calculate metrics of all songs.

        Args:
          params: list of float, thresholds
        """
        n = 0
        list_args = []

        for n, hdf5_path in enumerate(self.hdf5_paths):
            list_args.append([n, hdf5_path, params])
            """e.g., [0, 'xx.h5', [0.3, 0.3, 0.3]]"""
               
        debug = False
        if debug:
            list_args = list_args[0 :] 
            for i in range(len(list_args)):
                print(i, list_args[i][1])
                self.calculate_score_per_song(list_args[i])

        # Calculate metrics in parallel
        with ProcessPoolExecutor() as exector:
            results = exector.map(self.calculate_score_per_song, list_args)

        stats_list = list(results)
        stats_dict = {}
        for key in stats_list[0].keys():
            stats_dict[key] = [e[key] for e in stats_list if key in e.keys()]
        
        return stats_dict

    def calculate_score_per_song(self, args):
        """Calculate score per song.

        Args:
          args: [n, hdf5_path, params]
        """
        n = args[0]
        hdf5_path = args[1]
        [onset_threshold, offset_threshold, frame_threshold] = args[2]

        return_dict = {}

        # Load pre-calculated system outputs and ground truths
        prob_path = os.path.join(self.probs_dir, '{}.pkl'.format(get_filename(hdf5_path)))
        total_dict = pickle.load(open(prob_path, 'rb'))

        ref_on_off_pairs = total_dict['ref_on_off_pairs']
        ref_midi_notes = total_dict['ref_midi_notes']
        output_dict = total_dict

        # Calculate frame metric
        if self.evaluate_frame:
            y_pred = (np.sign(total_dict['frame_output'] - frame_threshold) + 1) / 2
            y_pred[np.where(y_pred==0.5)] = 0
            y_true = total_dict['frame_roll']
            y_pred = y_pred[0 : y_true.shape[0], :]
            y_true = y_true[0 : y_pred.shape[0], :]

            tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
            return_dict['frame_precision'] = tmp[0][1]
            return_dict['frame_recall'] = tmp[1][1]
            return_dict['frame_f1'] = tmp[2][1]

        # Post processor
        if self.post_processor_type == 'regression':
            post_processor = RegressionPostProcessor(self.frames_per_second, 
                classes_num=self.classes_num, onset_threshold=onset_threshold, 
                offset_threshold=offset_threshold, 
                frame_threshold=frame_threshold, 
                pedal_offset_threshold=False)

        elif self.post_processor_type == 'onsets_frames':
            post_processor = OnsetsFramesPostProcessor(self.frames_per_second, 
                classes_num=self.classes_num)

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = \
            post_processor.output_dict_to_note_pedal_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""

        # # Detect piano notes from output_dict
        est_on_offs = est_on_off_note_vels[:, 0 : 2]
        est_midi_notes = est_on_off_note_vels[:, 2]
        est_vels = est_on_off_note_vels[:, 3] * self.velocity_scale
        
        # Simple fix: ensure positive durations for mir_eval
        est_on_offs = np.maximum(est_on_offs, 0.0)  # No negative times
        est_on_offs[:, 1] = np.maximum(est_on_offs[:, 1], est_on_offs[:, 0] + 0.01)  # Min 10ms duration

        # Calculate P50, R50, F50 metrics like GuitarSet paper
        if len(est_on_offs) == 0 or len(ref_on_off_pairs) == 0:
            note_precision = 0.0
            note_recall = 0.0
            note_f1 = 0.0
        else:
            try:
                # Additional validation before mir_eval
                # Check for invalid intervals (negative durations, etc.)
                if np.any(est_on_offs[:, 1] <= est_on_offs[:, 0]):
                    # Fix invalid intervals by setting offset = onset + 0.01
                    invalid_mask = est_on_offs[:, 1] <= est_on_offs[:, 0]
                    est_on_offs[invalid_mask, 1] = est_on_offs[invalid_mask, 0] + 0.01
                
                if np.any(ref_on_off_pairs[:, 1] <= ref_on_off_pairs[:, 0]):
                    # Fix invalid reference intervals too
                    invalid_mask = ref_on_off_pairs[:, 1] <= ref_on_off_pairs[:, 0]
                    ref_on_off_pairs[invalid_mask, 1] = ref_on_off_pairs[invalid_mask, 0] + 0.01
                
                # P50, R50, F50: onset-only, 50ms tolerance, no offset matching
                note_precision, note_recall, note_f1, _ = \
                    mir_eval.transcription.precision_recall_f1_overlap(
                        ref_intervals=ref_on_off_pairs, 
                        ref_pitches=note_to_freq(ref_midi_notes), 
                        est_intervals=est_on_offs, 
                        est_pitches=note_to_freq(est_midi_notes), 
                        onset_tolerance=0.05,  # 50ms tolerance
                        offset_ratio=None,    # No offset matching
                        offset_min_tolerance=None)  # No offset matching
            except Exception as e:
                # Print the specific error for debugging
                print(f"mir_eval error in {get_filename(hdf5_path)}: {e}")
                # If mir_eval fails, return zero metrics
                note_precision = 0.0
                note_recall = 0.0
                note_f1 = 0.0

        if self.pedal:
            # Detect piano notes from output_dict
            ref_pedal_on_off_pairs = output_dict['ref_pedal_on_off_pairs']

            # Calculate pedal metrics
            if len(ref_pedal_on_off_pairs) > 0:
                pedal_precision, pedal_recall, pedal_f1, _ = \
                    mir_eval.transcription.precision_recall_f1_overlap(
                        ref_intervals=ref_pedal_on_off_pairs, 
                        ref_pitches=np.ones(ref_pedal_on_off_pairs.shape[0]), 
                        est_intervals=est_pedal_on_offs, 
                        est_pitches=np.ones(est_pedal_on_offs.shape[0]), 
                        onset_tolerance=0.2, 
                        offset_ratio=self.pedal_offset_ratio, 
                        offset_min_tolerance=self.pedal_offset_min_tolerance)

                return_dict['pedal_precision'] = pedal_precision
                return_dict['pedal_recall'] = pedal_recall
                return_dict['pedal_f1'] = pedal_f1

                y_pred = (np.sign(total_dict['pedal_frame_output'] - 0.5) + 1) / 2
                y_pred[np.where(y_pred==0.5)] = 0
                y_true = total_dict['pedal_frame_roll']
                y_pred = y_pred[0 : y_true.shape[0]]
                y_true = y_true[0 : y_pred.shape[0]]
                
                tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
                return_dict['pedal_frame_precision'] = tmp[0][1]
                return_dict['pedal_frame_recall'] = tmp[1][1]
                return_dict['pedal_frame_f1'] = tmp[2][1]

                print('pedal f1: {:.3f}, frame f1: {:.3f}'.format(pedal_f1, return_dict['pedal_frame_f1']))

        
        # Note: Individual song metrics are not saved to file here
        # The final averaged metrics are saved in the calculate_metrics function

        return_dict['note_precision'] = note_precision
        return_dict['note_recall'] = note_recall
        return_dict['note_f1'] = note_f1
        return return_dict


def calculate_metrics(args, thresholds=None):
    """Load pre-calculate probabilities, and apply thresholds to calculate 
    metrics. Users may adjust the hyper-parameters in ScoreCalculator to 
    evaluate with or without offset, velocity and pedals.

    Args:
      workspace: str, directory of your workspace
      model_type: str
      augmentation: str, e.g. 'none'
      dataset_name: str, name of the dataset (e.g., guitarset, maestro)
      hdf5s_dir: str, path to HDF5 files directory
      split: str, dataset split (e.g., 'val', 'test')
      post_processor_type: 'regression' | 'onsets_frames'. High-resolution 
        system should use 'regression'. 'onsets_frames' is only used to compare
        with Google's onsets and frames system.
      cuda: bool
    """

    # Arguments & parameters
    workspace = args.workspace
    model_type = args.model_type
    augmentation = args.augmentation
    dataset_name = args.dataset_name
    hdf5s_dir = args.hdf5s_dir
    split = args.split
    post_processor_type = args.post_processor_type

    # Paths
    if hasattr(args, 'model_name') and args.model_name:
        model_name = args.model_name
    else:
        model_name = 'default'
    
    probs_dir = os.path.join(workspace, 'probs', f'PKL_{model_name}_{dataset_name}')

    # Score calculator
    score_calculator = ScoreCalculator(hdf5s_dir, probs_dir, split=split, post_processor_type=post_processor_type)

    if not thresholds:
        # Use custom thresholds from command line if provided, otherwise defaults
        if hasattr(args, 'thresholds') and args.thresholds is not None:
            thresholds = args.thresholds
        else:
            # Use the same thresholds as training for consistency
            # These should match the thresholds used in guitar_test.py
            thresholds = [0.3, 0.3, 0.1]  # onset, offset, frame (matching training)
    else:
        # Use custom thresholds from command line
        thresholds = args.thresholds

    t1 = time.time()
    stats_dict = score_calculator.metrics(thresholds)
    print('Time: {:.3f}'.format(time.time() - t1))
    
    for key in stats_dict.keys():
        if key in ['note_precision', 'note_recall', 'note_f1']:
            # Print as percentages like GuitarSet paper (P50, R50, F50)
            print('{}: {:.1f}%'.format(key, np.mean(stats_dict[key]) * 100))
        else:
            print('{}: {:.4f}'.format(key, np.mean(stats_dict[key])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_infer_prob = subparsers.add_parser('infer_prob')
    parser_infer_prob.add_argument('--workspace', type=str, required=True)
    parser_infer_prob.add_argument('--model_type', type=str, required=True)
    parser_infer_prob.add_argument('--augmentation', type=str, required=True)
    parser_infer_prob.add_argument('--checkpoint_path', type=str, required=True)
    parser_infer_prob.add_argument('--dataset_name', type=str, required=True,
        help='Name of the dataset (e.g., egdb, guitarset, maestro)')
    parser_infer_prob.add_argument('--hdf5s_dir', type=str, required=True,
        help='Path to HDF5 files directory')
    parser_infer_prob.add_argument('--split', type=str, required=True)
    parser_infer_prob.add_argument('--post_processor_type', type=str, default='regression')
    parser_infer_prob.add_argument('--cuda', action='store_true', default=False)

    parser_metrics = subparsers.add_parser('calculate_metrics')
    parser_metrics.add_argument('--workspace', type=str, required=True)
    parser_metrics.add_argument('--model_type', type=str, required=True)
    parser_metrics.add_argument('--augmentation', type=str, required=True)
    parser_metrics.add_argument('--dataset_name', type=str, required=True,
        help='Name of the dataset (e.g., egdb, guitarset, maestro)')
    parser_metrics.add_argument('--hdf5s_dir', type=str, required=True,
        help='Path to HDF5 files directory')
    parser_metrics.add_argument('--split', type=str, required=True)
    parser_metrics.add_argument('--post_processor_type', type=str, default='regression')
    parser_metrics.add_argument('--thresholds', nargs=3, type=float, 
        help='Custom thresholds: onset offset frame (e.g., --thresholds 0.3 0.3 0.1)')
    parser_metrics.add_argument('--model_name', type=str, 
        help='Model name for folder structure (e.g., PT, guitarset_finetuned)')

    args = parser.parse_args()

    if args.mode == 'infer_prob':
        infer_prob(args)

    elif args.mode == 'calculate_metrics':
        calculate_metrics(args)

    else:
        raise Exception('Incorrct argument!')