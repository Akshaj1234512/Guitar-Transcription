#!/usr/bin/env python3
"""
Inference from Andrea
"""

from pathlib import Path
import sys
from typing import Tuple
import pretty_midi

# fix for system environment
sys.path.insert(0, str(Path(__file__).parent))

import t5_fretting_transformer.src.fret_t5 as fret_t5
sys.modules['fret_t5'] = fret_t5
####

from t5_fretting_transformer.src.fret_t5.inference import FretT5Inference


# Standard acoustic guitar tuning (string 1 == high E) (taken from Andrea's util scripts)
STANDARD_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 40)
HALF_STEP_DOWN_TUNING: Tuple[int, ...] = tuple(pitch - 1 for pitch in STANDARD_TUNING)
FULL_STEP_DOWN_TUNING: Tuple[int, ...] = tuple(pitch - 2 for pitch in STANDARD_TUNING)
DROP_D_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 38)


def run_tab_generation(midi_path, capo=0, tuning=STANDARD_TUNING):
    ''' run inference on andrea's model '''
    inference = FretT5Inference(
        checkpoint_path="/data/shamakg/music_ai_pipeline/Music-AI/5_FinalPipeline/src_backup/fret_t5/best_model.pt",
        tokenizer_path="/data/shamakg/music_ai_pipeline/Music-AI/5_FinalPipeline/t5_fretting_transformer/universal_tokenizer"
    )
    midi_notes = load_midi_notes(midi_path)
    print("Andrea Capo", capo)
    print("Andrea Tuning",tuning)
    tab_events = inference.predict_with_timing(midi_notes, capo=capo, tuning=tuning)
    
    return tab_events

def load_midi_notes(midi_path):
    '''converts midi notes into a format readable by above function'''
    pm = pretty_midi.PrettyMIDI(midi_path)

    notes = []
        
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'duration': note.end - note.start
                })
    
    notes.sort(key=lambda x: (round(x['start'], 4), x['pitch']))
    
    print(f"  Extracted {len(notes)} notes")
    
    return notes

### for testing
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Tab generation")
#     parser.add_argument("--midi_path", type=str, required=True, help="Path to midi file")
#     args = parser.parse_args()
#     MIDI_PATH = args.midi_path
#     run_tab_generation(MIDI_PATH) ### using 100 because it was predicted by the model in the pipeline for the path (using Leduc 3yC4c.mp3 for testing)
