#!/usr/bin/env python3
"""
Batch inference for fusion (AudioConditionedFretT5) models.
Takes Stage 1 MIDI output + original audio, runs the fusion model,
and saves predictions as JAMS files for evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pretty_midi
import torch
from transformers import LogitsProcessorList, T5Config, T5ForConditionalGeneration

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "t5_fretting_transformer" / "src"))

from fret_t5.audio_features import AudioFeatureConfig, AudioFeatureExtractor
from fret_t5.audio_conditioned_model import AudioConditionedFretT5
from fret_t5.tokenization import MidiTabTokenizerV3, STANDARD_TUNING, DEFAULT_CONDITIONING_TUNINGS
from fret_t5.constrained_generation import V3ConstrainedProcessor

STANDARD_TUNING_DICT = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}


def load_fusion_model(ckpt_path: str, tokenizer, device='cuda'):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    t5_sd = ck['model_state_dict']
    vocab_size = t5_sd['shared.weight'].shape[0] if 'shared.weight' in t5_sd else t5_sd['encoder.embed_tokens.weight'].shape[0]
    # Detect FFN type from state_dict (gated-gelu has wi_0/wi_1, relu has wi)
    has_gated = any('wi_0' in k for k in t5_sd.keys())
    ffn_type = "gated-gelu" if has_gated else "relu"
    hf_config = T5Config(
        vocab_size=vocab_size, d_model=256, d_ff=1024, num_layers=6, num_heads=8,
        dropout_rate=0.1,
        feed_forward_proj=ffn_type,
        is_encoder_decoder=True,
        decoder_start_token_id=tokenizer.shared_token_to_id.get('<sos>', 0),
        eos_token_id=tokenizer.shared_token_to_id['<eos>'],
        pad_token_id=tokenizer.shared_token_to_id['<pad>'],
    )
    t5 = T5ForConditionalGeneration(hf_config)
    t5.load_state_dict(t5_sd, strict=False)
    audio_config = AudioFeatureConfig(embedding_dim=256)
    model = AudioConditionedFretT5(t5, audio_config)
    if 'audio_extractor_state_dict' in ck:
        model.audio_extractor.load_state_dict(ck['audio_extractor_state_dict'])
    model.to(device).eval()
    return model, audio_config


def load_midi_notes(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            notes.append({
                'pitch': int(n.pitch),
                'start': float(n.start),
                'duration': float(n.end - n.start),
            })
    notes.sort(key=lambda x: (x['start'], x['pitch']))
    return notes


def predict_notes(model, tokenizer, audio_config, midi_notes, audio_path, device='cuda',
                  max_notes_per_chunk=100, hand_position_bias=0.0):
    """Run fusion inference on a list of notes. Chunks long sequences."""
    extractor = AudioFeatureExtractor(audio_config)
    try:
        audio, sr = librosa.load(audio_path, sr=audio_config.sample_rate)
    except Exception:
        return []

    processor = V3ConstrainedProcessor(tokenizer, hand_position_bias=hand_position_bias)
    all_preds = []  # (string, fret) per note in order

    for chunk_start in range(0, len(midi_notes), max_notes_per_chunk):
        chunk = midi_notes[chunk_start:chunk_start + max_notes_per_chunk]
        if len(chunk) == 0:
            continue

        # Build encoder tokens
        enc_tokens = []
        for i, note in enumerate(chunk):
            dur = max(100, min(5000, int(round(note['duration']*1000/100))*100))
            is_chord = i < len(chunk)-1 and abs(chunk[i+1]['start'] - note['start']) < 0.02
            token_dur = 0 if is_chord else dur
            enc_tokens.extend([
                f"NOTE_ON<{note['pitch']}>",
                f"TIME_SHIFT<{token_dur}>",
                f"NOTE_OFF<{note['pitch']}>",
            ])

        prefix = tokenizer.build_conditioning_prefix(0, STANDARD_TUNING)
        full = prefix + enc_tokens
        enc_ids = tokenizer.encode_encoder_tokens_shared(full)
        inp = torch.tensor([enc_ids], dtype=torch.long, device=device)

        mel_batch, pitch_batch = extractor.extract_batch_mels(audio, sr, chunk)
        mel_batch = mel_batch.to(device)
        pitch_batch = pitch_batch.to(device)

        with torch.no_grad():
            audio_emb = model.audio_extractor(mel_batch, pitch_batch).unsqueeze(0)
            out = model.generate(
                input_ids=inp, audio_embeddings=audio_emb,
                max_length=512, num_beams=4, do_sample=False,
                eos_token_id=tokenizer.shared_token_to_id['<eos>'],
                pad_token_id=tokenizer.shared_token_to_id['<pad>'],
                logits_processor=LogitsProcessorList([processor]),
            )
        dec_tokens = tokenizer.shared_to_decoder_tokens(out[0].cpu().tolist())

        # Parse TAB tokens in order
        tab_preds = []
        for tok in dec_tokens:
            if tok.startswith('TAB<'):
                s, f = tok[4:-1].split(',')
                tab_preds.append((int(s), int(f)))

        # Map chunk notes to predictions (positional)
        for i, note in enumerate(chunk):
            if i < len(tab_preds):
                all_preds.append(tab_preds[i])
            else:
                all_preds.append((1, 0))  # Fallback

    return all_preds


def save_jams(midi_notes, preds, out_path, duration=None):
    data = []
    for note, (s, f) in zip(midi_notes, preds):
        # Correct the pitch if string/fret is valid
        if 1 <= s <= 6:
            correct_pitch = STANDARD_TUNING_DICT[s] + f
        else:
            correct_pitch = note['pitch']
        data.append({
            "time": float(note['start']),
            "duration": float(note['duration']),
            "value": {
                "pitch": int(note['pitch']),  # Use Stage 1 pitch
                "string": int(s),
                "fret": int(f),
                "techniques": [],
            },
            "confidence": 1.0,
        })
    dur = duration or (max(n['start'] + n['duration'] for n in midi_notes) if midi_notes else 1.0)
    jams_dict = {
        "annotations": [{
            "namespace": "tab_note",
            "data": data,
            "annotation_metadata": {"curator":{},"annotator":{},"version":"","corpus":"fusion",
                                    "annotation_tools":"","annotation_rules":"","validation":"","data_source":""},
            "sandbox": {}
        }],
        "file_metadata": {"duration": float(dur), "identifiers": {}, "jams_version": "0.3.4"},
        "sandbox": {}
    }
    with open(out_path, 'w') as f:
        json.dump(jams_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--midi-dirs', nargs='+', required=True)
    parser.add_argument('--audio-dirs', nargs='+', required=True)
    parser.add_argument('--output-base', required=True)
    parser.add_argument('--hand-position-bias', type=float, default=0.0,
                        help='Soft hand-position bias magnitude in V3ConstrainedProcessor (0 disables, default)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = MidiTabTokenizerV3.load(str(SCRIPT_DIR / 't5_fretting_transformer' / 'universal_tokenizer'))
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS,
    )

    print(f"Loading fusion model: {args.checkpoint}")
    model, audio_config = load_fusion_model(args.checkpoint, tokenizer, device)

    output_base = Path(args.output_base)

    for midi_dir_str, audio_dir_str in zip(args.midi_dirs, args.audio_dirs):
        midi_dir = Path(midi_dir_str)
        audio_dir = Path(audio_dir_str)
        out_dir = output_base / midi_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        midi_files = sorted(midi_dir.glob('*.mid'))
        print(f"\n[{midi_dir.name}] {len(midi_files)} files")

        ok, fail = 0, 0
        for i, midi_path in enumerate(midi_files):
            stem = midi_path.stem
            audio_path = audio_dir / f'{stem}.wav'
            if not audio_path.exists():
                fail += 1
                continue

            try:
                notes = load_midi_notes(str(midi_path))
                if not notes:
                    fail += 1
                    continue
                preds = predict_notes(model, tokenizer, audio_config, notes, str(audio_path), device,
                                      hand_position_bias=args.hand_position_bias)
                if not preds:
                    fail += 1
                    continue
                save_jams(notes, preds, out_dir / f'{stem}.jams')
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 3:
                    print(f"  Error {midi_path.name}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(midi_files)}] ok={ok} fail={fail}")

        print(f"  Done: {ok} ok, {fail} failed")


if __name__ == '__main__':
    main()