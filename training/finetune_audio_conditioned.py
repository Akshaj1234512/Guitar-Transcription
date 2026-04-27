#!/usr/bin/env python3
"""
Finetune AudioConditionedFretT5 end-to-end on GAPS+IDMT with audio features.

Takes:
- Pretrained scaled T5 checkpoint (from DadaGP+SynthTab training)
- Pretrained audio CNN checkpoint (from GAPS+IDMT string classifier)
- Training dataset: either GT-MIDI or noisy Stage 1-MIDI

And trains the combined model (T5 encoder sees audio embeddings interleaved
with MIDI tokens) with a lower learning rate since both components are pretrained.

Usage:
    # GT-MIDI training
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune_audio_conditioned.py \
        --manifest results/finetune_gt.jsonl \
        --output-dir checkpoints_fusion_gt

    # Noisy-MIDI training
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 finetune_audio_conditioned.py \
        --manifest results/finetune_noisy.jsonl \
        --output-dir checkpoints_fusion_noisy
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "t5_fretting_transformer" / "src"))

from transformers import T5Config, T5ForConditionalGeneration
from fret_t5.audio_features import AudioFeatureConfig, AudioFeatureExtractor
from fret_t5.audio_conditioned_model import AudioConditionedFretT5
from fret_t5.tokenization import MidiTabTokenizerV3, STANDARD_TUNING, DEFAULT_CONDITIONING_TUNINGS


STANDARD_TUNING_DICT = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}


# ─── Dataset ─────────────────────────────────────────────────────────

class AudioTabDataset(Dataset):
    """Each item: MIDI-token encoder input + audio mel batch + TAB-token decoder target."""

    def __init__(self, manifest_path: str, tokenizer: MidiTabTokenizerV3,
                 audio_config: AudioFeatureConfig, max_notes: int = 100):
        self.tokenizer = tokenizer
        self.audio_config = audio_config
        self.max_notes = max_notes
        self.examples = []

        with open(manifest_path) as f:
            for line in f:
                ex = json.loads(line)
                notes = ex.get('notes', [])
                if len(notes) < 3:
                    continue
                # Split long recordings into chunks of max_notes with small overlap
                for start in range(0, len(notes), max_notes - 5):
                    chunk = notes[start:start + max_notes]
                    if len(chunk) < 3:
                        continue
                    self.examples.append({
                        'audio_path': ex['audio_path'],
                        'notes': chunk,
                    })

        self._extractor = AudioFeatureExtractor(audio_config)
        print(f"AudioTabDataset: {len(self.examples)} chunks from {manifest_path}")

    def __len__(self):
        return len(self.examples)

    def _build_token_sequences(self, notes):
        enc_tokens, dec_tokens = [], []
        for i, note in enumerate(notes):
            dur_ms = int(round(note['duration'] * 1000 / 100)) * 100
            dur_ms = max(100, min(dur_ms, 5000))
            is_chord = (i < len(notes) - 1 and
                        abs(notes[i + 1]['start'] - note['start']) < 0.02)
            token_dur = 0 if is_chord else dur_ms

            enc_tokens.extend([
                f"NOTE_ON<{note['pitch']}>",
                f"TIME_SHIFT<{token_dur}>",
                f"NOTE_OFF<{note['pitch']}>",
            ])
            # Decoder target needs valid string/fret; clamp fret to [0, 24]
            s = max(1, min(6, int(note['string'])))
            fret = max(0, min(24, int(note['fret'])))
            dec_tokens.extend([
                f"TAB<{s},{fret}>",
                f"TIME_SHIFT<{token_dur}>",
            ])
        return enc_tokens, dec_tokens

    def __getitem__(self, idx):
        ex = self.examples[idx]
        notes = ex['notes']

        # Build tokens
        enc_tokens, dec_tokens = self._build_token_sequences(notes)
        prefix = self.tokenizer.build_conditioning_prefix(0, STANDARD_TUNING)
        full_enc = prefix + enc_tokens

        enc_ids = self.tokenizer.encode_encoder_tokens_shared(full_enc)
        dec_ids = self.tokenizer.encode_decoder_tokens_shared(dec_tokens)

        pad_id = self.tokenizer.shared_token_to_id['<pad>']
        max_len = 512
        enc_ids = (enc_ids[:max_len] + [pad_id] * max(0, max_len - len(enc_ids)))
        dec_ids_padded = (dec_ids[:max_len] + [pad_id] * max(0, max_len - len(dec_ids)))

        enc_mask = [1 if t != pad_id else 0 for t in enc_ids]
        # Labels: -100 on pad
        labels = [t if t != pad_id else -100 for t in dec_ids_padded]

        # Extract audio mels for each note
        try:
            audio, sr = librosa.load(ex['audio_path'], sr=self.audio_config.sample_rate)
            mel_batch, pitch_batch = self._extractor.extract_batch_mels(audio, sr, notes)
        except Exception:
            n = len(notes)
            mel_batch = torch.zeros(n, 1, self.audio_config.n_mels, self._extractor.n_frames)
            pitch_batch = torch.tensor([min(nt['pitch'], 127) for nt in notes], dtype=torch.long)

        return {
            'input_ids': torch.tensor(enc_ids, dtype=torch.long),
            'attention_mask': torch.tensor(enc_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'mel_batch': mel_batch,
            'pitch_batch': pitch_batch,
            'num_notes': len(notes),
        }


def collate(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])

    max_notes = max(b['num_notes'] for b in batch)
    n_mels = batch[0]['mel_batch'].shape[2]
    n_frames = batch[0]['mel_batch'].shape[3]

    bsz = len(batch)
    mel_padded = torch.zeros(bsz, max_notes, 1, n_mels, n_frames)
    pitch_padded = torch.zeros(bsz, max_notes, dtype=torch.long)
    audio_mask = torch.zeros(bsz, max_notes, dtype=torch.long)

    for i, b in enumerate(batch):
        n = b['num_notes']
        mel_padded[i, :n] = b['mel_batch']
        pitch_padded[i, :n] = b['pitch_batch']
        audio_mask[i, :n] = 1

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'mel_padded': mel_padded,
        'pitch_padded': pitch_padded,
        'audio_mask': audio_mask,
    }


# ─── Model loading ──────────────────────────────────────────────────

def build_model(t5_ckpt_path: str, cnn_ckpt_path: str, tokenizer: MidiTabTokenizerV3,
                d_model: int = 256) -> AudioConditionedFretT5:
    """Load pretrained T5 + CNN into the combined AudioConditionedFretT5."""
    ckpt = torch.load(t5_ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    # T5 uses either 'shared.weight' or 'encoder.embed_tokens.weight' depending on format
    if 'shared.weight' in state_dict:
        vocab_size = state_dict['shared.weight'].shape[0]
    elif 'encoder.embed_tokens.weight' in state_dict:
        vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]
    else:
        raise KeyError(f"Can't find embedding weight. Keys: {list(state_dict.keys())[:10]}")

    mc = ckpt.get('model_config')
    if mc is not None:
        dims = mc.tiny_dims
    else:
        dims = {'d_model': d_model, 'd_ff': 1024, 'num_layers': 6, 'num_heads': 8}

    # Detect FFN type from state_dict (gated-gelu has wi_0/wi_1, relu has wi)
    has_gated = any('wi_0' in k for k in state_dict.keys())
    ffn_type = "gated-gelu" if has_gated else dims.get('feed_forward_proj', 'relu')

    hf_config = T5Config(
        vocab_size=vocab_size,
        d_model=int(dims.get('d_model', d_model)),
        d_ff=int(dims.get('d_ff', 1024)),
        num_layers=int(dims.get('num_layers', 6)),
        num_heads=int(dims.get('num_heads', 8)),
        dropout_rate=float(dims.get('dropout_rate', 0.1)),
        feed_forward_proj=ffn_type,
        layer_norm_epsilon=float(dims.get('layer_norm_epsilon', 1e-6)),
        relative_attention_num_buckets=int(dims.get('relative_attention_num_buckets', 32)),
        is_encoder_decoder=True,
        decoder_start_token_id=tokenizer.shared_token_to_id.get('<sos>', 0),
        eos_token_id=tokenizer.shared_token_to_id['<eos>'],
        pad_token_id=tokenizer.shared_token_to_id['<pad>'],
    )
    t5 = T5ForConditionalGeneration(hf_config)
    missing, unexpected = t5.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  WARNING: T5 load has {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing[:3]: print(f"    missing sample: {missing[:3]}")
        if unexpected[:3]: print(f"    unexpected sample: {unexpected[:3]}")
    else:
        print(f"  T5 loaded cleanly ({ffn_type} FFN)")

    audio_config = AudioFeatureConfig(embedding_dim=int(dims.get('d_model', d_model)))
    model = AudioConditionedFretT5(t5, audio_config)

    # Load CNN weights
    if cnn_ckpt_path and os.path.exists(cnn_ckpt_path):
        cnn_ckpt = torch.load(cnn_ckpt_path, map_location='cpu', weights_only=False)
        cnn_sd = cnn_ckpt.get('audio_extractor_state_dict') or cnn_ckpt.get('model_state_dict', {})
        if cnn_sd:
            # Strip 'feature_extractor.' prefix if present
            cleaned = {k.replace('feature_extractor.', ''): v for k, v in cnn_sd.items()
                       if 'classifier' not in k}
            missing, unexpected = model.audio_extractor.load_state_dict(cleaned, strict=False)
            print(f"  CNN loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    return model, audio_config


# ─── Training loop ──────────────────────────────────────────────────

def train_loop(model, train_loader, val_loader, output_dir: Path, args, rank: int = 0,
               world_size: int = 1, device: str = 'cuda'):
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    base = model.module if world_size > 1 else model
    t5_params = list(base.t5_model.parameters())
    cnn_params = list(base.audio_extractor.parameters())
    optimizer = torch.optim.AdamW([
        {'params': t5_params, 'lr': args.lr},
        {'params': cnn_params, 'lr': args.lr * 5},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(args.epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss_sum, steps = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            mel_padded = batch['mel_padded'].to(device)
            pitch_padded = batch['pitch_padded'].to(device)

            bsz, max_notes = mel_padded.shape[:2]
            mel_flat = mel_padded.view(-1, *mel_padded.shape[2:])
            pitch_flat = pitch_padded.view(-1)

            if args.amp:
                with torch.amp.autocast('cuda'):
                    audio_emb_flat = base.audio_extractor(mel_flat, pitch_flat)
                    audio_emb = audio_emb_flat.view(bsz, max_notes, -1)
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_embeddings=audio_emb,
                    )
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                audio_emb_flat = base.audio_extractor(mel_flat, pitch_flat)
                audio_emb = audio_emb_flat.view(bsz, max_notes, -1)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_embeddings=audio_emb,
                )
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()
            train_loss_sum += loss.item()
            steps += 1

            if rank == 0 and (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1} step {batch_idx+1}: loss={loss.item():.4f}", flush=True)

        scheduler.step()
        avg_train = train_loss_sum / max(steps, 1)

        # Validation
        model.eval()
        val_loss_sum, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                mel_padded = batch['mel_padded'].to(device)
                pitch_padded = batch['pitch_padded'].to(device)

                bsz, max_notes = mel_padded.shape[:2]
                mel_flat = mel_padded.view(-1, *mel_padded.shape[2:])
                pitch_flat = pitch_padded.view(-1)

                audio_emb_flat = base.audio_extractor(mel_flat, pitch_flat)
                audio_emb = audio_emb_flat.view(bsz, max_notes, -1)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_embeddings=audio_emb,
                )
                val_loss_sum += out.loss.item()
                val_steps += 1

        avg_val = val_loss_sum / max(val_steps, 1)
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: train={avg_train:.4f} val={avg_val:.4f}", flush=True)

            if avg_val < best_val:
                best_val = avg_val
                patience_ctr = 0
                save = {
                    'model_state_dict': base.t5_model.state_dict(),
                    'audio_extractor_state_dict': base.audio_extractor.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': avg_val,
                    'model_config': None,
                }
                torch.save(save, str(output_dir / 'best_model.pt'))
                print(f"  ✓ Saved best model (val={avg_val:.4f})", flush=True)
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}", flush=True)
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5-checkpoint', default='best_model_scaled.pt')
    parser.add_argument('--cnn-checkpoint', default='checkpoints_audio_conditioned/best_string_classifier.pt')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--val-frac', type=float, default=0.15)
    args = parser.parse_args()

    # DDP setup
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = MidiTabTokenizerV3.load(str(SCRIPT_DIR / 't5_fretting_transformer' / 'universal_tokenizer'))
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS,
    )

    # Model
    if rank == 0:
        print(f"Loading T5 from {args.t5_checkpoint}")
        print(f"Loading CNN from {args.cnn_checkpoint}")
    model, audio_config = build_model(args.t5_checkpoint, args.cnn_checkpoint, tokenizer)

    # Dataset
    full_dataset = AudioTabDataset(args.manifest, tokenizer, audio_config)

    # Split
    random.seed(42)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    val_n = int(len(indices) * args.val_frac)
    val_idx, train_idx = indices[:val_n], indices[val_n:]
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler, val_sampler = None, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=2, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=2, collate_fn=collate)

    if rank == 0:
        print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")
        print(f"World size: {world_size}, LR: {args.lr}, Batch: {args.batch_size}")

    train_loop(model, train_loader, val_loader, output_dir, args,
               rank=rank, world_size=world_size, device=device)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
