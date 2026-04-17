#!/usr/bin/env python3
"""String/Fret inference using AudioFret (audio-conditioned FretT5).

Takes Stage 1 predicted MIDI + original audio and produces per-note (string, fret)
assignments. AudioFret fuses a per-note audio encoder (3-block 2D CNN over a 200 ms
mel-spectrogram patch around each note onset) with a symbolic T5 backbone (d=256,
6 layers, 8 heads, gated-GELU). The audio embeddings are prepended before each
NOTE_ON token in the encoder input, letting self-attention mix timbral and symbolic
cues to resolve string/fret ambiguity.

Returns a list of TabEvent-like objects with onset_sec, duration_sec, string, fret,
pitch -- matching the interface expected by the rest of the pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import librosa
import pretty_midi
import torch
from transformers import LogitsProcessorList, T5Config, T5ForConditionalGeneration

# Make t5_fretting_transformer importable under the module path the checkpoint expects
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "t5_fretting_transformer" / "src"))

import t5_fretting_transformer.src.fret_t5 as fret_t5
sys.modules['fret_t5'] = fret_t5

from fret_t5.audio_features import AudioFeatureConfig, AudioFeatureExtractor
from fret_t5.audio_conditioned_model import AudioConditionedFretT5
from fret_t5.tokenization import MidiTabTokenizerV3, DEFAULT_CONDITIONING_TUNINGS
from fret_t5.constrained_generation import V3ConstrainedProcessor


STANDARD_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 40)
HALF_STEP_DOWN_TUNING: Tuple[int, ...] = tuple(p - 1 for p in STANDARD_TUNING)
FULL_STEP_DOWN_TUNING: Tuple[int, ...] = tuple(p - 2 for p in STANDARD_TUNING)
DROP_D_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 38)


@dataclass
class TabEvent:
    onset_sec: float
    duration_sec: float
    string: int
    fret: int
    pitch: int


_MODEL_CACHE: dict = {}


def _load_audiofret(checkpoint_path: str, tokenizer, device: str = "cuda"):
    """Load AudioConditionedFretT5 from a checkpoint dict, with FFN auto-detection."""
    if checkpoint_path in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint_path]

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    t5_sd = ck["model_state_dict"]
    vocab_size = (
        t5_sd["shared.weight"].shape[0]
        if "shared.weight" in t5_sd
        else t5_sd["encoder.embed_tokens.weight"].shape[0]
    )
    has_gated = any("wi_0" in k for k in t5_sd.keys())
    ffn_type = "gated-gelu" if has_gated else "relu"

    hf_config = T5Config(
        vocab_size=vocab_size, d_model=256, d_ff=1024, num_layers=6, num_heads=8,
        dropout_rate=0.1, feed_forward_proj=ffn_type, is_encoder_decoder=True,
        decoder_start_token_id=tokenizer.shared_token_to_id.get("<sos>", 0),
        eos_token_id=tokenizer.shared_token_to_id["<eos>"],
        pad_token_id=tokenizer.shared_token_to_id["<pad>"],
    )
    t5 = T5ForConditionalGeneration(hf_config)
    t5.load_state_dict(t5_sd, strict=False)

    audio_config = AudioFeatureConfig(embedding_dim=256)
    model = AudioConditionedFretT5(t5, audio_config)
    if "audio_extractor_state_dict" in ck:
        model.audio_extractor.load_state_dict(ck["audio_extractor_state_dict"])

    model.to(device).eval()
    _MODEL_CACHE[checkpoint_path] = (model, audio_config)
    return model, audio_config


def load_midi_notes(midi_path: str) -> List[dict]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({
                "pitch": int(n.pitch),
                "start": float(n.start),
                "duration": float(n.end - n.start),
            })
    notes.sort(key=lambda x: (round(x["start"], 4), x["pitch"]))
    return notes


def _predict_tabs(model, tokenizer, audio_config, midi_notes: List[dict],
                  audio_path: str, tuning: Tuple[int, ...] = STANDARD_TUNING,
                  capo: int = 0, device: str = "cuda",
                  max_notes_per_chunk: int = 100) -> List[Tuple[int, int]]:
    extractor = AudioFeatureExtractor(audio_config)
    audio, sr = librosa.load(audio_path, sr=audio_config.sample_rate)

    processor = V3ConstrainedProcessor(tokenizer)  # bias defaults to 0 in our cleaned-up version
    all_preds: List[Tuple[int, int]] = []

    for chunk_start in range(0, len(midi_notes), max_notes_per_chunk):
        chunk = midi_notes[chunk_start : chunk_start + max_notes_per_chunk]
        if not chunk:
            continue

        enc_tokens = []
        for i, note in enumerate(chunk):
            dur = max(100, min(5000, int(round(note["duration"] * 1000 / 100)) * 100))
            is_chord = i < len(chunk) - 1 and abs(chunk[i + 1]["start"] - note["start"]) < 0.02
            token_dur = 0 if is_chord else dur
            enc_tokens.extend([
                f"NOTE_ON<{note['pitch']}>",
                f"TIME_SHIFT<{token_dur}>",
                f"NOTE_OFF<{note['pitch']}>",
            ])

        prefix = tokenizer.build_conditioning_prefix(capo, tuning)
        enc_ids = tokenizer.encode_encoder_tokens_shared(prefix + enc_tokens)
        inp = torch.tensor([enc_ids], dtype=torch.long, device=device)

        mel_batch, pitch_batch = extractor.extract_batch_mels(audio, sr, chunk)
        mel_batch = mel_batch.to(device)
        pitch_batch = pitch_batch.to(device)

        with torch.no_grad():
            audio_emb = model.audio_extractor(mel_batch, pitch_batch).unsqueeze(0)
            out = model.generate(
                input_ids=inp, audio_embeddings=audio_emb,
                max_length=512, num_beams=4, do_sample=False,
                eos_token_id=tokenizer.shared_token_to_id["<eos>"],
                pad_token_id=tokenizer.shared_token_to_id["<pad>"],
                logits_processor=LogitsProcessorList([processor]),
            )
        dec_tokens = tokenizer.shared_to_decoder_tokens(out[0].cpu().tolist())

        tab_preds: List[Tuple[int, int]] = []
        for tok in dec_tokens:
            if tok.startswith("TAB<"):
                s, f = tok[4:-1].split(",")
                tab_preds.append((int(s), int(f)))

        for i, _ in enumerate(chunk):
            all_preds.append(tab_preds[i] if i < len(tab_preds) else (1, 0))

    return all_preds


def run_tab_generation(midi_path: str, audio_path: Optional[str] = None,
                       capo: int = 0, tuning: Tuple[int, ...] = STANDARD_TUNING) -> List[TabEvent]:
    """Run AudioFret string/fret inference on a MIDI file + corresponding audio.

    Parameters
    ----------
    midi_path : str
        Path to the Stage-1 predicted MIDI file.
    audio_path : str, optional
        Path to the original audio file. AudioFret requires audio for timbral
        disambiguation; if None, raises an error.
    capo, tuning : int, tuple of int
        Instrument conditioning tokens passed to the encoder.
    """
    if audio_path is None:
        raise ValueError("run_tab_generation requires audio_path for AudioFret inference.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = str(ROOT / "models" / "string-fret" / "audiofret.pt")
    TOKENIZER_PATH = str(ROOT / "t5_fretting_transformer" / "universal_tokenizer")

    tokenizer = MidiTabTokenizerV3.load(TOKENIZER_PATH)
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS,
    )

    model, audio_config = _load_audiofret(CHECKPOINT_PATH, tokenizer, device)

    midi_notes = load_midi_notes(midi_path)
    print(f"  Extracted {len(midi_notes)} notes  (capo={capo}, tuning={tuning})")
    if not midi_notes:
        return []

    preds = _predict_tabs(
        model, tokenizer, audio_config, midi_notes, audio_path,
        tuning=tuning, capo=capo, device=device,
    )

    tab_events = [
        TabEvent(onset_sec=n["start"], duration_sec=n["duration"],
                 string=s, fret=f, pitch=n["pitch"])
        for n, (s, f) in zip(midi_notes, preds)
    ]
    return tab_events
