import os
import sys
import numpy as np
import mir_eval
import librosa
from pathlib import Path

def load_midi_notes(midi_path):
    """Load MIDI file and return intervals and pitches like in calculate_score_for_paper.py"""
    import pretty_midi
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    intervals = []
    pitches = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)

    if not intervals:
        return np.array([]).reshape(0, 2), np.array([])

    return np.array(intervals), np.array(pitches)

def calculate_midi_f1(pred_midi_path, gt_midi_path):
    """Calculate F1 score using the exact method from calculate_score_for_paper.py"""

    # Load predicted notes
    est_on_offs, est_midi_notes = load_midi_notes(pred_midi_path)

    # Load ground truth notes
    ref_on_off_pairs, ref_midi_notes = load_midi_notes(gt_midi_path)

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
            if np.any(est_on_offs[:, 1] <= est_on_offs[:, 0]):
                invalid_mask = est_on_offs[:, 1] <= est_on_offs[:, 0]
                est_on_offs[invalid_mask, 1] = est_on_offs[invalid_mask, 0] + 0.01

            if np.any(ref_on_off_pairs[:, 1] <= ref_on_off_pairs[:, 0]):
                invalid_mask = ref_on_off_pairs[:, 1] <= ref_on_off_pairs[:, 0]
                ref_on_off_pairs[invalid_mask, 1] = ref_on_off_pairs[invalid_mask, 0] + 0.01

            # P50, R50, F50: onset-only, 50ms tolerance, no offset matching
            note_precision, note_recall, note_f1, _ = \
                mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals=ref_on_off_pairs,
                    ref_pitches=librosa.midi_to_hz(ref_midi_notes),
                    est_intervals=est_on_offs,
                    est_pitches=librosa.midi_to_hz(est_midi_notes),
                    onset_tolerance=0.05,  # 50ms tolerance
                    offset_ratio=None,    # No offset matching
                    offset_min_tolerance=None)  # No offset matching
        except Exception as e:
            print(f"mir_eval error: {e}")
            note_precision = 0.0
            note_recall = 0.0
            note_f1 = 0.0

    return note_precision, note_recall, note_f1

def evaluate_folder(pred_dir, gt_dir, suffix_to_remove):
    """Evaluates all MIDI files in a single folder and returns the statistics."""
    print(f"\n" + "="*50)
    print(f"Evaluating Directory: {pred_dir.name}")
    print(f"="*50)
    
    # Get all predicted MIDI files
    pred_files = list(pred_dir.glob('*.mid'))
    print(f'Found {len(pred_files)} predicted MIDI files')

    # Create mapping from predicted to ground truth
    pred_to_gt = {}
    for pred_file in pred_files:
        # Remove specific suffix from filename to match GT
        gt_name = pred_file.name.replace(suffix_to_remove, '')
        gt_path = gt_dir / gt_name
        if gt_path.exists():
            pred_to_gt[pred_file] = gt_path
        else:
            print(f'No GT match for {pred_file.name}')

    print(f'Matched {len(pred_to_gt)} file pairs')

    # Compute F1 for each pair
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for pred_path, gt_path in pred_to_gt.items():
        try:
            precision, recall, f1 = calculate_midi_f1(pred_path, gt_path)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            # Optional: Uncomment if you want to print per-file stats again
            # print(f'{pred_path.name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}')

        except Exception as e:
            print(f'Error processing {pred_path.name}: {e}')
            continue

    # Compute overall statistics
    if all_f1s:
        mean_precision = np.mean(all_precisions)
        mean_recall = np.mean(all_recalls)
        mean_f1 = np.mean(all_f1s)
        std_f1 = np.std(all_f1s)

        print(f'\nResults for {pred_dir.name}:')
        print(f'Files processed: {len(all_f1s)}')
        print(f'note_precision: {mean_precision * 100:.1f}%')
        print(f'note_recall: {mean_recall * 100:.1f}%')
        print(f'note_f1: {mean_f1 * 100:.1f}%')
        print(f'F1 std: ±{std_f1:.4f}')
        
        return mean_precision, mean_recall, mean_f1
    else:
        print(f"No valid files processed in {pred_dir.name}")
        return 0, 0, 0

def main():
    # Base paths
    base_dir = Path(os.environ.get('RESULTS_DIR', './results'))
    gt_dir = Path('../GuitarSet/MIDIAnnotations') # Adjust if necessary based on where you run the script
    
    # Define the target folders and the specific string suffixes to strip from their files
    evaluation_targets = [
        {'dir': base_dir / 'audio_hex_debleeded', 'suffix': '_hex_cln'},
        {'dir': base_dir / 'audio_hex_original',  'suffix': '_hex'},
        {'dir': base_dir / 'audio_mono-mic',      'suffix': '_mic'},
        {'dir': base_dir / 'audio_mix',           'suffix': '_mix'}
    ]

    # Store aggregate results for a final comparison printout
    summary_results = {}

    for target in evaluation_targets:
        if not target['dir'].exists():
            print(f"\nWarning: Directory not found -> {target['dir']}")
            continue
            
        p, r, f = evaluate_folder(target['dir'], gt_dir, target['suffix'])
        summary_results[target['dir'].name] = {'P': p, 'R': r, 'F1': f}
        
    # Print a final summary comparison table
    print("\n" + "#"*50)
    print("FINAL SUMMARY ACROSS ALL FOLDERS")
    print("#"*50)
    for folder_name, metrics in summary_results.items():
        print(f"{folder_name.ljust(25)} | P: {metrics['P']*100:>5.1f}% | R: {metrics['R']*100:>5.1f}% | F1: {metrics['F1']*100:>5.1f}%")

    # Print a final summary comparison table
    print("\n" + "#"*50)
    print("FINAL SUMMARY ACROSS ALL FOLDERS")
    print("#"*50)
    
    total_p, total_r, total_f1 = 0, 0, 0
    valid_folders = len(summary_results)

    for folder_name, metrics in summary_results.items():
        print(f"{folder_name.ljust(25)} | P: {metrics['P']*100:>5.1f}% | R: {metrics['R']*100:>5.1f}% | F1: {metrics['F1']*100:>5.1f}%")
        
        # Add to running totals
        total_p += metrics['P']
        total_r += metrics['R']
        total_f1 += metrics['F1']

    # Calculate and print the grand average
    if valid_folders > 0:
        print("-" * 50)
        print(f"{'GRAND AVERAGE'.ljust(25)} | P: {(total_p/valid_folders)*100:>5.1f}% | R: {(total_r/valid_folders)*100:>5.1f}% | F1: {(total_f1/valid_folders)*100:>5.1f}%")
        print("#"*50)

if __name__ == '__main__':
    main()