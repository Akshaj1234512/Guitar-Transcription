"""
Audio feature extraction for guitar string disambiguation.

Extracts per-note audio embeddings from mel spectrograms using a lightweight CNN.
These embeddings can be injected into the T5 encoder as additional tokens
to provide timbral cues for string/fret assignment.

Each guitar string has distinct spectral characteristics due to differences in
string gauge, tension, and overtone structure. Even the same pitch played on
different strings produces audibly different timbres that this module learns
to distinguish.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioFeatureConfig:
    """Configuration for audio feature extraction."""
    sample_rate: int = 22050
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    segment_duration_sec: float = 0.2  # 200ms per note
    embedding_dim: int = 256  # Must match d_model of the T5 model
    # CNN architecture
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    cnn_kernel_size: int = 3
    dropout: float = 0.1


class AudioFeatureExtractor(nn.Module):
    """Lightweight CNN that extracts per-note audio embeddings from mel spectrograms.

    Architecture:
    - Input: mel spectrogram patch (n_mels x time_frames) for each note
    - 3 conv layers with batch norm and ReLU
    - Global average pooling
    - Linear projection to embedding_dim

    The output embedding can be inserted into the T5 encoder sequence as a
    special "audio token" before each NOTE_ON token.
    """

    def __init__(self, config: AudioFeatureConfig):
        super().__init__()
        self.config = config

        # Number of time frames for the fixed segment duration
        self.n_frames = int(config.segment_duration_sec * config.sample_rate / config.hop_length) + 1

        # CNN layers
        layers = []
        in_channels = 1  # Mono mel spectrogram
        for out_channels in config.cnn_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, config.cnn_kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Compute output size after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, 1, config.n_mels, self.n_frames)
            cnn_out = self.cnn(dummy)
            self.cnn_output_size = cnn_out.view(1, -1).shape[1]

        # Projection to T5 embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.cnn_output_size, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.Dropout(config.dropout),
        )

        # Pitch embedding (helps disambiguate same-pitch-different-string)
        self.pitch_embedding = nn.Embedding(128, config.embedding_dim)

        # Fusion: combine audio embedding + pitch embedding
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

    def extract_mel_segment(
        self,
        audio: np.ndarray,
        sr: int,
        onset_sec: float,
        duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Extract a mel spectrogram patch for a single note.

        Args:
            audio: Full audio waveform (1D numpy array)
            sr: Sample rate
            onset_sec: Note onset time in seconds
            duration_sec: Note duration (uses config default if None)

        Returns:
            Mel spectrogram patch of shape (n_mels, n_frames)
        """
        seg_dur = duration_sec or self.config.segment_duration_sec
        seg_dur = min(seg_dur, self.config.segment_duration_sec)  # Cap at max

        start_sample = max(0, int(onset_sec * sr))
        end_sample = min(len(audio), int((onset_sec + seg_dur) * sr))

        segment = audio[start_sample:end_sample]

        # Pad if too short
        expected_samples = int(seg_dur * sr)
        if len(segment) < expected_samples:
            segment = np.pad(segment, (0, expected_samples - len(segment)))

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=segment.astype(np.float32),
            sr=sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to [-1, 1]
        mel_db = (mel_db + 80) / 80  # Typical range is [-80, 0] dB

        # Ensure fixed time dimension
        if mel_db.shape[1] < self.n_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.n_frames - mel_db.shape[1])))
        elif mel_db.shape[1] > self.n_frames:
            mel_db = mel_db[:, :self.n_frames]

        return mel_db

    def extract_batch_mels(
        self,
        audio: np.ndarray,
        sr: int,
        notes: List[dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract mel spectrograms and pitches for a batch of notes.

        Args:
            audio: Full audio waveform
            sr: Sample rate
            notes: List of dicts with 'pitch', 'start'/'onset', 'duration' keys

        Returns:
            mel_batch: Tensor of shape (num_notes, 1, n_mels, n_frames)
            pitch_batch: Tensor of shape (num_notes,) with MIDI pitch values
        """
        mels = []
        pitches = []

        for note in notes:
            onset = note.get('start', note.get('onset', 0.0))
            duration = note.get('duration', self.config.segment_duration_sec)
            pitch = note['pitch']

            mel = self.extract_mel_segment(audio, sr, onset, duration)
            mels.append(mel)
            pitches.append(min(pitch, 127))  # Clamp to valid MIDI range

        mel_batch = torch.tensor(np.array(mels), dtype=torch.float32).unsqueeze(1)  # Add channel dim
        pitch_batch = torch.tensor(pitches, dtype=torch.long)

        return mel_batch, pitch_batch

    def forward(self, mel_batch: torch.Tensor, pitch_batch: torch.Tensor) -> torch.Tensor:
        """Compute audio embeddings from mel spectrograms and pitches.

        Args:
            mel_batch: (batch, 1, n_mels, n_frames)
            pitch_batch: (batch,) MIDI pitch values

        Returns:
            embeddings: (batch, embedding_dim) audio-informed note embeddings
        """
        # CNN features from mel spectrogram
        cnn_out = self.cnn(mel_batch)
        cnn_flat = cnn_out.view(cnn_out.size(0), -1)
        audio_emb = self.projection(cnn_flat)

        # Pitch embedding
        pitch_emb = self.pitch_embedding(pitch_batch)

        # Fuse audio + pitch
        combined = torch.cat([audio_emb, pitch_emb], dim=-1)
        fused = self.fusion(combined)

        return fused


class StringClassifier(nn.Module):
    """Standalone 6-way string classifier for audio notes.

    Can be used independently to predict string from audio, or as a
    component to provide STRING_HINT conditioning tokens to the T5 model.
    """

    def __init__(self, config: AudioFeatureConfig, num_strings: int = 6):
        super().__init__()
        self.feature_extractor = AudioFeatureExtractor(config)
        self.classifier = nn.Linear(config.embedding_dim, num_strings)

    def forward(self, mel_batch: torch.Tensor, pitch_batch: torch.Tensor) -> torch.Tensor:
        """Predict string probabilities.

        Returns:
            logits: (batch, num_strings) unnormalized string predictions
        """
        embeddings = self.feature_extractor(mel_batch, pitch_batch)
        return self.classifier(embeddings)

    def predict_strings(self, mel_batch: torch.Tensor, pitch_batch: torch.Tensor) -> torch.Tensor:
        """Predict most likely string for each note.

        Returns:
            strings: (batch,) predicted string numbers (1-6)
        """
        logits = self.forward(mel_batch, pitch_batch)
        return logits.argmax(dim=-1) + 1  # 0-indexed to 1-indexed
