"""
Audio-Conditioned FretT5 Model.

Wraps the standard T5 encoder-decoder with an audio feature extractor
that injects per-note audio embeddings into the encoder input sequence.

Architecture (Audio-as-Encoder-Prefix):
    For each MIDI note, we:
    1. Extract a mel spectrogram patch from the audio
    2. Run it through a CNN to produce a d_model-dimensional embedding
    3. Insert this embedding as a special token before the NOTE_ON token

    The T5 architecture is unchanged — the audio embeddings are projected
    to the same d_model dimension and concatenated with the standard token
    embeddings in the encoder input.

    Encoder input sequence becomes:
    [CAPO, TUNING, audio_emb_1, NOTE_ON<p1>, TIME_SHIFT<d1>, NOTE_OFF<p1>,
     audio_emb_2, NOTE_ON<p2>, TIME_SHIFT<d2>, NOTE_OFF<p2>, ...]
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import LogitsProcessorList, T5Config, T5ForConditionalGeneration

from .audio_features import AudioFeatureConfig, AudioFeatureExtractor
from .constrained_generation import V3ConstrainedProcessor
from .postprocess import (
    TabEvent,
    TimingContext,
    midi_notes_to_encoder_tokens_with_timing,
    postprocess_with_timing,
    tab_events_to_dict_list,
)
from .tokenization import MidiTabTokenizerV3, STANDARD_TUNING, DEFAULT_CONDITIONING_TUNINGS
from .training import ModelConfig

# Constants for chunking
MAX_ENCODER_LENGTH = 512
CONDITIONING_TOKENS = 2
TOKENS_PER_NOTE = 4  # audio_emb + NOTE_ON + TIME_SHIFT + NOTE_OFF (was 3, now 4 with audio)
MAX_NOTES_PER_CHUNK = (MAX_ENCODER_LENGTH - CONDITIONING_TOKENS) // TOKENS_PER_NOTE  # ~127
OVERLAP_NOTES = 4


class AudioConditionedFretT5(nn.Module):
    """T5 model with audio feature injection for string/fret assignment.

    The audio features are injected by:
    1. Computing per-note audio embeddings via CNN on mel spectrograms
    2. Adding these embeddings to the encoder input at positions before each note's tokens
    3. The T5 self-attention naturally learns to use timbral cues for string disambiguation
    """

    def __init__(
        self,
        t5_model: T5ForConditionalGeneration,
        audio_config: AudioFeatureConfig,
    ):
        super().__init__()
        self.t5_model = t5_model
        self.audio_extractor = AudioFeatureExtractor(audio_config)

        # Ensure audio embedding dim matches T5 d_model
        assert audio_config.embedding_dim == t5_model.config.d_model, \
            f"Audio embedding dim ({audio_config.embedding_dim}) must match T5 d_model ({t5_model.config.d_model})"

    def create_audio_augmented_inputs(
        self,
        input_ids: torch.Tensor,
        audio_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create encoder inputs with audio embeddings interleaved.

        For training, audio_embeddings are pre-computed and passed in.
        For inference, they are computed on-the-fly.

        Args:
            input_ids: (batch, seq_len) standard token IDs
            audio_embeddings: (batch, num_notes, d_model) per-note audio features
            attention_mask: (batch, seq_len) attention mask

        Returns:
            inputs_embeds: (batch, new_seq_len, d_model) with audio tokens interleaved
            attention_mask: (batch, new_seq_len) updated mask
        """
        # Get token embeddings from T5
        token_embeddings = self.t5_model.encoder.embed_tokens(input_ids)

        if audio_embeddings is None:
            return token_embeddings, attention_mask

        batch_size, seq_len, d_model = token_embeddings.shape
        num_notes = audio_embeddings.shape[1]

        # Find NOTE_ON token positions to insert audio embeddings before them
        # We insert one audio embedding before each group of (NOTE_ON, TIME_SHIFT, NOTE_OFF)
        # For simplicity, we prepend all audio embeddings after conditioning tokens

        # Strategy: prepend audio embeddings as a block after conditioning prefix
        # [CAPO_emb, TUNING_emb, audio_1, audio_2, ..., audio_N, NOTE_ON<p1>, ...]
        # This is simpler and lets self-attention figure out the alignment

        augmented = torch.cat([
            token_embeddings[:, :CONDITIONING_TOKENS, :],  # Conditioning tokens
            audio_embeddings,  # All audio embeddings
            token_embeddings[:, CONDITIONING_TOKENS:, :],  # MIDI tokens
        ], dim=1)

        # Update attention mask
        if attention_mask is not None:
            audio_mask = torch.ones(
                batch_size, num_notes,
                dtype=attention_mask.dtype, device=attention_mask.device
            )
            augmented_mask = torch.cat([
                attention_mask[:, :CONDITIONING_TOKENS],
                audio_mask,
                attention_mask[:, CONDITIONING_TOKENS:],
            ], dim=1)
        else:
            augmented_mask = None

        return augmented, augmented_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with optional audio conditioning.

        If audio_embeddings is provided, they are interleaved into the encoder input.
        Otherwise, falls back to standard T5 behavior.
        """
        if audio_embeddings is not None:
            inputs_embeds, attention_mask = self.create_audio_augmented_inputs(
                input_ids, audio_embeddings, attention_mask
            )
            return self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                **kwargs,
            )
        else:
            return self.t5_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                **kwargs,
            )

    def generate(self, input_ids=None, audio_embeddings=None, attention_mask=None, **kwargs):
        """Generate with optional audio conditioning."""
        if audio_embeddings is not None:
            inputs_embeds, attention_mask = self.create_audio_augmented_inputs(
                input_ids, audio_embeddings, attention_mask
            )
            return self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            return self.t5_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )


class AudioConditionedInference:
    """Inference pipeline for audio-conditioned FretT5.

    Extends FretT5Inference to incorporate audio features during prediction.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str = "universal_tokenizer",
        audio_checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        max_fret_span: int = 5,
        d_model: int = 256,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_fret_span = max_fret_span

        # Load tokenizer
        self.tokenizer = MidiTabTokenizerV3.load(tokenizer_path)
        self.tokenizer.ensure_conditioning_tokens(
            capo_values=tuple(range(8)),
            tuning_options=DEFAULT_CONDITIONING_TUNINGS,
        )

        # Load T5 model
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]

        model_config = checkpoint.get("model_config", None)
        if model_config:
            t5_d_model = model_config.tiny_dims.get("d_model", d_model)
        else:
            t5_d_model = d_model

        hf_config = T5Config(
            vocab_size=vocab_size,
            d_model=t5_d_model,
            d_ff=model_config.tiny_dims.get("d_ff", 1024) if model_config else 1024,
            num_layers=model_config.tiny_dims.get("num_layers", 6) if model_config else 6,
            num_heads=model_config.tiny_dims.get("num_heads", 8) if model_config else 8,
            dropout_rate=0.1,
            is_encoder_decoder=True,
            decoder_start_token_id=self.tokenizer.shared_token_to_id.get("<sos>", 0),
            eos_token_id=self.tokenizer.shared_token_to_id["<eos>"],
            pad_token_id=self.tokenizer.shared_token_to_id["<pad>"],
        )
        t5_model = T5ForConditionalGeneration(hf_config)
        t5_model.load_state_dict(state_dict, strict=False)

        # Audio feature extractor
        audio_config = AudioFeatureConfig(embedding_dim=t5_d_model)
        self.model = AudioConditionedFretT5(t5_model, audio_config)

        # Load audio weights if provided
        if audio_checkpoint_path and os.path.exists(audio_checkpoint_path):
            audio_state = torch.load(audio_checkpoint_path, map_location="cpu", weights_only=False)
            self.model.audio_extractor.load_state_dict(
                audio_state.get("audio_extractor_state_dict", audio_state), strict=False
            )

        self.model.to(self.device)
        self.model.eval()

    def predict_with_timing(
        self,
        midi_notes: List[Dict],
        audio_path: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
        sr: int = 22050,
        capo: int = 0,
        tuning: tuple = STANDARD_TUNING,
        pitch_window: int = 5,
        alignment_window: int = 5,
        return_dict: bool = False,
    ) -> List[TabEvent]:
        """Generate tablature with audio conditioning.

        Args:
            midi_notes: List of dicts with 'pitch', 'start', 'duration'
            audio_path: Path to audio file (loads if audio not provided)
            audio: Pre-loaded audio waveform (1D numpy array)
            sr: Sample rate of the audio
            capo, tuning: Conditioning parameters
        """
        sorted_notes = sorted(midi_notes, key=lambda x: (x.get('start', 0), x['pitch']))

        # Load audio if path provided
        if audio is None and audio_path is not None:
            audio, sr = librosa.load(audio_path, sr=self.model.audio_extractor.config.sample_rate)

        # Extract audio features if audio available
        audio_embeddings = None
        if audio is not None:
            mel_batch, pitch_batch = self.model.audio_extractor.extract_batch_mels(
                audio, sr, sorted_notes
            )
            mel_batch = mel_batch.to(self.device)
            pitch_batch = pitch_batch.to(self.device)
            with torch.no_grad():
                audio_embeddings = self.model.audio_extractor(mel_batch, pitch_batch)
                audio_embeddings = audio_embeddings.unsqueeze(0)  # Add batch dim

        # Create encoder tokens
        encoder_tokens, timing_context = midi_notes_to_encoder_tokens_with_timing(
            sorted_notes,
            time_shift_quantum_ms=self.tokenizer.config.time_shift_quantum_ms,
            max_duration_ms=self.tokenizer.config.max_duration_ms,
        )
        prefix = self.tokenizer.build_conditioning_prefix(capo, tuning)
        full_tokens = prefix + encoder_tokens

        input_ids = self.tokenizer.encode_encoder_tokens_shared(full_tokens)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        logits_processors = [V3ConstrainedProcessor(self.tokenizer, max_fret_span=self.max_fret_span)]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_tensor,
                audio_embeddings=audio_embeddings,
                max_length=512,
                num_beams=4,
                do_sample=False,
                eos_token_id=self.tokenizer.shared_token_to_id["<eos>"],
                pad_token_id=self.tokenizer.shared_token_to_id["<pad>"],
                logits_processor=LogitsProcessorList(logits_processors),
            )

        decoder_tokens = self.tokenizer.shared_to_decoder_tokens(outputs[0].cpu().tolist())

        tab_events = postprocess_with_timing(
            encoder_tokens=full_tokens,
            decoder_tokens=decoder_tokens,
            timing_context=timing_context,
            capo=capo,
            tuning=tuning,
            pitch_window=pitch_window,
            alignment_window=alignment_window,
        )

        if return_dict:
            return tab_events_to_dict_list(tab_events)
        return tab_events
