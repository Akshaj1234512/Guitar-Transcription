#!/usr/bin/env python3
"""
Finetune audio-conditioned FretT5 on real guitar recordings (GAPS + IDMT).

Phase 1: Train audio feature extractor + string classifier on GAPS+IDMT
Phase 2: Finetune full model (T5 + audio CNN) end-to-end

Usage:
    # Phase 1: String classifier pre-training
    CUDA_VISIBLE_DEVICES=1 python train_audio_conditioned.py --phase 1

    # Phase 2: Full audio-conditioned finetuning
    CUDA_VISIBLE_DEVICES=1 python train_audio_conditioned.py --phase 2 \
        --pretrained-checkpoint checkpoints_scaled/best_model.pt
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "t5_fretting_transformer" / "src"))
sys.path.insert(0, str(SCRIPT_DIR / "data_loaders"))
sys.path.insert(0, str(SCRIPT_DIR / "pipeline_utils" / "midi_utils"))

from fret_t5.audio_features import AudioFeatureConfig, AudioFeatureExtractor, StringClassifier
from gaps_loader import load_gaps_dataset
from idmt_loader import load_idmt_dataset
from goat_loader import load_goat_dataset
from guitartechs_loader import load_guitartechs_dataset

# Import the same Augmentor used for Stage 1 audio-to-MIDI training
from data_generator import Augmentor


# ─── Dataset for string classification (Phase 1) ─────────────────────────

class StringClassificationDataset(Dataset):
    """Per-note audio dataset for string classification pre-training.

    Each item: (mel_spectrogram_patch, pitch, string_label)

    If augment=True, loads the full audio file once per sample and applies the
    same Augmentor used in Stage 1 training (HPF + IR + white/pink/hum noise)
    BEFORE extracting the mel segment, so the CNN sees the same distribution
    that Stage 1 was trained on.
    """

    def __init__(self, entries: List[Dict], config: AudioFeatureConfig,
                 max_notes_per_file: int = 50, augment: bool = False,
                 ir_path: str = None):  # Match Stage 1 training: no IR convolution
        self.config = config
        self.augment = augment
        self.samples = []  # List of (audio_path, onset, duration, pitch, string)
        self.augmentor = None
        if augment:
            # Stage 1 training used: HPF + white/pink/hum noise, NO IRs
            # prob=0.5 → 50% clean / 50% augmented, balancing robustness with clean performance
            self.augmentor = Augmentor(
                ir_path=ir_path,  # None = skip IR convolution
                sample_rate=config.sample_rate,
                min_snr=25.0,
                max_snr=45.0,
                prob=0.5,  # 50/50 clean vs augmented
            )

        for entry in entries:
            audio_path = entry['audio_path']
            notes = entry['notes']

            # Sample notes from each file
            if len(notes) > max_notes_per_file:
                sampled = random.sample(notes, max_notes_per_file)
            else:
                sampled = notes

            for note in sampled:
                self.samples.append({
                    'audio_path': audio_path,
                    'onset': note['start'],
                    'duration': note['duration'],
                    'pitch': note['pitch'],
                    'string': note['string'],  # 1-6
                })

        print(f"StringClassificationDataset: {len(self.samples)} samples from {len(entries)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # Load a slightly larger context so augmentation has room for IR convolution
            load_pad = 0.5 if self.augment else 0.02
            audio, sr = librosa.load(
                sample['audio_path'],
                sr=self.config.sample_rate,
                offset=max(0, sample['onset'] - load_pad),
                duration=self.config.segment_duration_sec + 2 * load_pad,
            )
        except Exception:
            audio = np.zeros(int(self.config.segment_duration_sec * self.config.sample_rate))
            sr = self.config.sample_rate

        # Apply Stage 1-style augmentation: HPF + IR + noise
        if self.augment and self.augmentor is not None and audio.size > 0:
            audio = self.augmentor.augment(audio)

        # The segment we want is now at offset=load_pad in the loaded clip
        extractor = AudioFeatureExtractor(self.config)
        mel = extractor.extract_mel_segment(audio, sr, load_pad if self.augment else 0.02, sample['duration'])

        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, n_frames)
        pitch_tensor = torch.tensor(min(sample['pitch'], 127), dtype=torch.long)
        string_label = torch.tensor(sample['string'] - 1, dtype=torch.long)  # 0-indexed

        return mel_tensor, pitch_tensor, string_label


# ─── Phase 1: Train string classifier ────────────────────────────────────

def train_string_classifier(
    gaps_dir: str,
    idmt_dir: str,
    output_dir: str,
    config: AudioFeatureConfig,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Train a standalone string classifier on GAPS + IDMT data."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading GAPS dataset...")
    gaps_data = load_gaps_dataset(gaps_dir)
    print("Loading GOAT dataset...")
    goat_data = load_goat_dataset('./data/datasets/GOAT/data')
    print("Loading GuitarTechs dataset...")
    guitartechs_data = load_guitartechs_dataset()

    all_data = gaps_data + goat_data + guitartechs_data
    print(f"\nCombined: {len(all_data)} files "
          f"(gaps={len(gaps_data)}, goat={len(goat_data)}, guitartechs={len(guitartechs_data)})")
    random.shuffle(all_data)

    # Split 85/15
    split_idx = int(0.85 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    train_dataset = StringClassificationDataset(train_data, config, augment=True)
    val_dataset = StringClassificationDataset(val_data, config, max_notes_per_file=20, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = StringClassifier(config, num_strings=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining string classifier: {sum(p.numel() for p in model.parameters())} params")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for mel, pitch, label in train_loader:
            mel, pitch, label = mel.to(device), pitch.to(device), label.to(device)

            logits = model(mel, pitch)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mel.size(0)
            train_correct += (logits.argmax(1) == label).sum().item()
            train_total += mel.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for mel, pitch, label in val_loader:
                mel, pitch, label = mel.to(device), pitch.to(device), label.to(device)
                logits = model(mel, pitch)
                val_correct += (logits.argmax(1) == label).sum().item()
                val_total += mel.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss/train_total:.4f} "
                  f"train_acc={train_acc:.1f}% val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'audio_extractor_state_dict': model.feature_extractor.state_dict(),
                'config': config,
                'best_val_acc': best_val_acc,
                'epoch': epoch,
            }, str(output_dir / 'best_string_classifier.pt'))

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Saved to: {output_dir / 'best_string_classifier.pt'}")

    return best_val_acc


# ─── Phase 2: Full audio-conditioned finetuning ──────────────────────────

class AudioConditionedTabDataset(Dataset):
    """Dataset for audio-conditioned tablature training.

    Each item provides:
    - encoder_tokens (MIDI representation)
    - decoder_tokens (TAB representation)
    - audio features (mel spectrograms per note)
    """

    def __init__(
        self,
        entries: List[Dict],
        tokenizer,
        config: AudioFeatureConfig,
        max_notes: int = 127,  # Reduced from 170 due to audio tokens
    ):
        self.entries = entries
        self.tokenizer = tokenizer
        self.config = config
        self.max_notes = max_notes
        self.extractor = AudioFeatureExtractor(config)

        # Precompute tokenized examples
        self.examples = []
        for entry in entries:
            notes = entry['notes'][:max_notes]
            if len(notes) < 3:
                continue
            self.examples.append({
                'audio_path': entry['audio_path'],
                'notes': notes,
            })

        print(f"AudioConditionedTabDataset: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        notes = example['notes']
        audio_path = example['audio_path']

        # Build encoder tokens (MIDI)
        encoder_tokens = []
        for i, note in enumerate(notes):
            dur_ms = int(round(note['duration'] * 1000 / 100)) * 100
            dur_ms = max(100, min(dur_ms, 5000))

            # Check if next note is simultaneous (chord)
            is_chord = False
            if i < len(notes) - 1:
                if abs(notes[i+1]['start'] - note['start']) < 0.02:
                    is_chord = True

            token_dur = 0 if is_chord else dur_ms

            encoder_tokens.extend([
                f"NOTE_ON<{note['pitch']}>",
                f"TIME_SHIFT<{token_dur}>",
                f"NOTE_OFF<{note['pitch']}>",
            ])

        # Build decoder tokens (TAB)
        decoder_tokens = []
        for i, note in enumerate(notes):
            dur_ms = int(round(note['duration'] * 1000 / 100)) * 100
            dur_ms = max(100, min(dur_ms, 5000))

            is_chord = False
            if i < len(notes) - 1:
                if abs(notes[i+1]['start'] - note['start']) < 0.02:
                    is_chord = True

            token_dur = 0 if is_chord else dur_ms

            decoder_tokens.extend([
                f"TAB<{note['string']},{note['fret']}>",
                f"TIME_SHIFT<{token_dur}>",
            ])

        # Encode to IDs
        prefix = self.tokenizer.build_conditioning_prefix(0, (64, 59, 55, 50, 45, 40))
        full_encoder = prefix + encoder_tokens

        enc_ids = self.tokenizer.encode_encoder_tokens_shared(full_encoder)
        dec_ids = self.tokenizer.encode_decoder_tokens_shared(decoder_tokens)

        # Pad/truncate to max length
        max_enc = 512
        max_dec = 512
        enc_ids = enc_ids[:max_enc]
        dec_ids = dec_ids[:max_dec]

        pad_id = self.tokenizer.shared_token_to_id["<pad>"]
        enc_ids += [pad_id] * (max_enc - len(enc_ids))
        dec_ids += [pad_id] * (max_dec - len(dec_ids))

        # Extract audio features
        try:
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            mel_batch, pitch_batch = self.extractor.extract_batch_mels(audio, sr, notes)
        except Exception:
            # Fallback: zero features
            n_notes = len(notes)
            mel_batch = torch.zeros(n_notes, 1, self.config.n_mels, self.extractor.n_frames)
            pitch_batch = torch.tensor([n['pitch'] for n in notes], dtype=torch.long)

        return {
            'input_ids': torch.tensor(enc_ids, dtype=torch.long),
            'labels': torch.tensor(dec_ids, dtype=torch.long),
            'mel_batch': mel_batch,
            'pitch_batch': pitch_batch,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="1=string classifier, 2=full audio-conditioned")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None,
                        help="Pretrained T5 checkpoint for Phase 2")
    parser.add_argument("--string-classifier-checkpoint", type=str, default=None,
                        help="String classifier checkpoint to initialize audio extractor")
    parser.add_argument("--gaps-dir", type=str, default="./data/gaps_v1")
    parser.add_argument("--idmt-dir", type=str,
                        default="./data/IDMT-SMT-Guitar/IDMT-SMT-GUITAR_V2")
    parser.add_argument("--output-dir", type=str, default="checkpoints_audio_conditioned")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    args = parser.parse_args()

    audio_config = AudioFeatureConfig(embedding_dim=args.d_model)

    if args.phase == 1:
        train_string_classifier(
            gaps_dir=args.gaps_dir,
            idmt_dir=args.idmt_dir,
            output_dir=args.output_dir,
            config=audio_config,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    elif args.phase == 2:
        if not args.pretrained_checkpoint:
            print("Phase 2 requires --pretrained-checkpoint")
            sys.exit(1)

        print("Phase 2: Audio-conditioned finetuning")
        print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
        print(f"This will finetune the T5 model with audio features on GAPS+IDMT")
        print("Implementation: use AudioConditionedTabDataset + custom training loop")
        # Full implementation would use the AudioConditionedFretT5 model
        # with a custom HF Trainer or manual training loop
        # This is the framework — fill in with actual training once Phase 1 validates audio features


if __name__ == "__main__":
    main()
