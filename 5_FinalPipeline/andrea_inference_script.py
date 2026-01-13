#!/usr/bin/env python3
"""
Inference from Andrea
"""

from pathlib import Path
import sys
import pretty_midi
import importlib


### system path fix, you may need to change below based on where your src folder is (and inference model)
from fret_t5.inference import FretT5Inference
###



def run_tab_generation(midi_path):
    ''' run inference on andrea's model '''
    inference = FretT5Inference(
        checkpoint_path="/data/shamakg/music_ai_pipeline/Music-AI/5_FinalPipeline/src/fret_t5/best_model.pt",
        tokenizer_path="/data/shamakg/music_ai_pipeline/Music-AI/5_FinalPipeline/src/fret_t5/universal_tokenizer"
    )
    midi_notes = load_midi_notes(midi_path)
    tab_events = inference.predict_with_timing(midi_notes)
    
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
