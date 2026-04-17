#!/usr/bin/env python3
"""Oracle inference: AudioFret fed ground-truth MIDI instead of Stage 1 output.

For each audio file:
- GuitarSet: strip suffix to find matching GT MIDI in MIDIAnnotations/
- EGDB: load notes from JAMS annotations (merge all string tracks)

Saves JAMS predictions identical format to batch_fusion_inference.py.
"""

import argparse
import json
import sys
from pathlib import Path

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


def load_fusion_model(ckpt_path, tokenizer, device='cuda'):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    t5_sd = ck['model_state_dict']
    vocab_size = t5_sd['shared.weight'].shape[0] if 'shared.weight' in t5_sd else t5_sd['encoder.embed_tokens.weight'].shape[0]
    has_gated = any('wi_0' in k for k in t5_sd.keys())
    ffn_type = "gated-gelu" if has_gated else "relu"
    hf_config = T5Config(
        vocab_size=vocab_size, d_model=256, d_ff=1024, num_layers=6, num_heads=8,
        dropout_rate=0.1, feed_forward_proj=ffn_type, is_encoder_decoder=True,
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


def load_gt_notes_from_midi(midi_path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            notes.append({'pitch': int(n.pitch), 'start': float(n.start),
                          'duration': float(n.end - n.start)})
    notes.sort(key=lambda x: (x['start'], x['pitch']))
    return notes


def load_gt_notes_from_jams(jams_path):
    with open(jams_path) as f:
        data = json.load(f)
    notes = []
    for ann in data.get('annotations', []):
        if ann.get('namespace') != 'note_midi':
            continue
        for n in ann.get('data', []):
            val = n.get('value')
            if val is None: continue
            pitch = int(round(float(val)))
            t = float(n.get('time', 0))
            d = float(n.get('duration', 0.1))
            notes.append({'pitch': pitch, 'start': t, 'duration': d})
    notes.sort(key=lambda x: (x['start'], x['pitch']))
    return notes


def predict_notes(model, tokenizer, audio_config, midi_notes, audio_path, device='cuda',
                  max_notes_per_chunk=100, hand_position_bias=0.0):
    extractor = AudioFeatureExtractor(audio_config)
    try:
        audio, sr = librosa.load(audio_path, sr=audio_config.sample_rate)
    except Exception:
        return []

    processor = V3ConstrainedProcessor(tokenizer, hand_position_bias=hand_position_bias)
    all_preds = []

    for chunk_start in range(0, len(midi_notes), max_notes_per_chunk):
        chunk = midi_notes[chunk_start:chunk_start + max_notes_per_chunk]
        if len(chunk) == 0:
            continue

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

        tab_preds = []
        for tok in dec_tokens:
            if tok.startswith('TAB<'):
                s, f = tok[4:-1].split(',')
                tab_preds.append((int(s), int(f)))

        for i, note in enumerate(chunk):
            if i < len(tab_preds):
                all_preds.append(tab_preds[i])
            else:
                all_preds.append((1, 0))

    return all_preds


def save_jams(midi_notes, preds, out_path, duration=None):
    data = []
    for note, (s, f) in zip(midi_notes, preds):
        data.append({
            "time": float(note['start']),
            "duration": float(note['duration']),
            "value": {"pitch": int(note['pitch']), "string": int(s), "fret": int(f), "techniques": []},
            "confidence": 1.0,
        })
    dur = duration or (max(n['start'] + n['duration'] for n in midi_notes) if midi_notes else 1.0)
    jams_dict = {
        "annotations": [{
            "namespace": "tab_note", "data": data,
            "annotation_metadata": {"curator":{},"annotator":{},"version":"","corpus":"oracle",
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
    parser.add_argument('--dataset', choices=['guitarset', 'egdb'], required=True)
    parser.add_argument('--audio-dirs', nargs='+', required=True)
    parser.add_argument('--suffixes', nargs='+', default=None,
                        help='Suffix to strip from audio stem to match GT MIDI stem (GuitarSet only)')
    parser.add_argument('--gt-dir', required=True,
                        help='GT annotation directory (MIDIAnnotations for GS, annotation_jams for EGDB)')
    parser.add_argument('--output-base', required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = MidiTabTokenizerV3.load(str(SCRIPT_DIR / 't5_fretting_transformer' / 'universal_tokenizer'))
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS,
    )

    print(f"Loading fusion model: {args.checkpoint}")
    model, audio_config = load_fusion_model(args.checkpoint, tokenizer, device)

    gt_dir = Path(args.gt_dir)
    output_base = Path(args.output_base)

    suffixes = args.suffixes or [''] * len(args.audio_dirs)
    if len(suffixes) == 1 and len(args.audio_dirs) > 1:
        suffixes = suffixes * len(args.audio_dirs)

    for audio_dir_str, suffix in zip(args.audio_dirs, suffixes):
        audio_dir = Path(audio_dir_str)
        out_dir = output_base / audio_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_files = sorted(audio_dir.glob('*.wav'))
        print(f"\n[{audio_dir.name}] {len(audio_files)} files (suffix='{suffix}')")

        ok, fail = 0, 0
        for i, audio_path in enumerate(audio_files):
            stem = audio_path.stem
            if args.dataset == 'guitarset':
                gt_stem = stem[:-len(suffix)] if suffix and stem.endswith(suffix) else stem
                gt_path = gt_dir / f'{gt_stem}.mid'
                if not gt_path.exists():
                    fail += 1
                    continue
                try:
                    notes = load_gt_notes_from_midi(gt_path)
                except Exception:
                    fail += 1
                    continue
            else:  # egdb
                gt_path = gt_dir / f'{stem}.jams'
                if not gt_path.exists():
                    fail += 1
                    continue
                try:
                    notes = load_gt_notes_from_jams(gt_path)
                except Exception:
                    fail += 1
                    continue

            if not notes:
                fail += 1
                continue

            try:
                preds = predict_notes(model, tokenizer, audio_config, notes, str(audio_path), device)
                if not preds:
                    fail += 1
                    continue
                save_jams(notes, preds, out_dir / f'{stem}.jams')
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 3:
                    print(f"  Error {audio_path.name}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(audio_files)}] ok={ok} fail={fail}")

        print(f"  Done: {ok} ok, {fail} failed")

    print("\nAll done.")


if __name__ == '__main__':
    main()
