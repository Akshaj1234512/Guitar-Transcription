#!/usr/bin/env python3
"""
Unified Finetuning Script (NOTE-ONLY) - MEAN TEACHER
(GRADIENT ACCUMULATION + FROZEN BN + SUPERVISED BURN-IN)

Strategy:
1. Burn-in Phase (0 -> 20k): Supervised Only. Teacher follows Student quickly.
2. Ramp-up Phase (20k -> 27.5k): Consistency turns on slowly (Non-linear).
3. Stable Phase (27.5k+): Full Mean Teacher consistency.
"""

import os
import sys
import numpy as np
import copy

sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))

import argparse
import logging

import torch
import torch.optim as optim
import torch.utils.data

from utilities import (create_folder, StatisticsContainer)
from data_generator import Augmentor, collate_fn
from models import Regress_onset_offset_frame_velocity_CRNN
from pytorch_utils import move_data_to_device
from losses import get_loss_func, consistency_loss
from evaluate import SegmentEvaluator
import config

CONSISTENCY_WEIGHT = 80.0

# Phase 1: Burn-in (Supervised Only, No Consistency)
CONSISTENCY_START_ITER = 0 

# Phase 2: Ramp-up Duration (Consistency grows 0 -> 100)
CONSISTENCY_RAMP_LEN = 1600

# Alpha Schedule: 0.99 (Fast) during burn-in/ramp, 0.999 (Stable) afterwards
# Fast alpha during burn-in ensures the teacher isn't "cold" when consistency starts.
ALPHA_FAST = 0.99
ALPHA_STABLE = 0.999

# --- Gradient Accumulation Config ---
ACCUMULATION_STEPS = 16


def update_ema_variables(model, ema_model, iteration):
    # Keep Teacher updating quickly (0.99) until the ramp-up is fully finished
    # This prevents the Teacher from "lagging" while the Student learns the new domain
    ramp_end = CONSISTENCY_START_ITER + CONSISTENCY_RAMP_LEN
    
    alpha = ALPHA_FAST if iteration < ramp_end else ALPHA_STABLE

    # EMA for parameters
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    # Copy buffers (BN running stats, etc.)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)

    return alpha


def get_current_consistency_weight(iteration):
    """
    Returns 0.0 during Burn-in.
    Ramps up gaussianly during Ramp-up.
    Returns max weight during Stable phase.
    """
    # Phase 1: Supervised Burn-in
    if iteration < CONSISTENCY_START_ITER:
        return 0.0
    
    # Calculate progress through the ramp phase (0.0 to 1.0)
    progress = (iteration - CONSISTENCY_START_ITER) / CONSISTENCY_RAMP_LEN
    
    # Phase 3: Stable Mean Teacher
    if progress >= 1.0:
        return CONSISTENCY_WEIGHT
    
    # Phase 2: Gaussian Ramp-up (Standard Mean Teacher Sigmoid-like curve)
    return CONSISTENCY_WEIGHT * np.exp(-5.0 * (1.0 - progress) ** 2)


def _load_pt_note_only(model, pretrained_path, logger=logging):
    logger.info(f'Loading pretrained model from {pretrained_path}')
    ckpt = torch.load(pretrained_path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if isinstance(state, dict) and 'note_model' in state:
        state = state['note_model']
    if isinstance(state, dict) and any(k.startswith('note_model.') for k in state.keys()):
        state = {k.replace('note_model.', '', 1): v for k, v in state.items() if k.startswith('note_model.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f'Pretrained loaded. Missing={len(missing)} Unexpected={len(unexpected)}')


def finetune(args):
    workspace = args.workspace
    dataset_name = args.dataset_name
    hdf5s_dir = args.hdf5s_dir
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    max_timing_shift = args.max_timing_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    mini_data = args.mini_data

    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8

    loss_func = get_loss_func(loss_type)

    # Paths
    checkpoints_dir = os.path.join(workspace, 'checkpoints')
    create_folder(checkpoints_dir)
    statistics_path = os.path.join(workspace, 'statistics', 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))
    logs_dir = os.path.join(workspace, 'logs')
    create_folder(logs_dir)

    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    current_log_number = max([int(f.split('.')[0]) for f in log_files]) + 1 if log_files else 0

    log_folder_name = f"log_{current_log_number:04d}"
    log_checkpoints_dir = os.path.join(checkpoints_dir, log_folder_name)
    create_folder(log_checkpoints_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=os.path.join(logs_dir, f'{current_log_number:04d}.log'),
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info(args)
    logging.info("--- CONFIGURATION ---")
    logging.info(f"Hardware Batch Size: {batch_size}")
    logging.info(f"Accumulation Steps: {ACCUMULATION_STEPS}")
    logging.info(f"Effective Batch Size: {batch_size * ACCUMULATION_STEPS}")
    logging.info(f"Consistency Weight: {CONSISTENCY_WEIGHT}")
    logging.info(f"Burn-in Length: {CONSISTENCY_START_ITER} iters")
    logging.info(f"Ramp-up Length: {CONSISTENCY_RAMP_LEN} iters")
    logging.info(f"Alpha Strategy: {ALPHA_FAST} -> {ALPHA_STABLE}")

    # --- Initialize Student ---
    model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=frames_per_second, classes_num=classes_num)
    if args.pretrained_path and args.pretrained_path != 'none':
        _load_pt_note_only(model, args.pretrained_path, logger=logging)
    model.to(device)

    # --- Initialize Teacher ---
    teacher_model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=frames_per_second, classes_num=classes_num)
    teacher_model.load_state_dict(model.state_dict())
    teacher_model.to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # Augmentor
    if augmentation == 'aug':
        augmentor = Augmentor(ir_path=args.ir_path)
    else:
        logging.error("Augmentation must be enabled ('aug') for Mean Teacher training.")
        return

    from data_generator import MaestroDataset, Sampler, TestSampler

    train_dataset = MaestroDataset(
        hdf5s_dir=hdf5s_dir,
        segment_seconds=segment_seconds,
        frames_per_second=frames_per_second,
        max_note_shift=max_note_shift,
        max_timing_shift=max_timing_shift,
        augmentor=augmentor
    )
    evaluate_dataset = MaestroDataset(
        hdf5s_dir=hdf5s_dir,
        segment_seconds=segment_seconds,
        frames_per_second=frames_per_second,
        max_note_shift=0,
        max_timing_shift=0
    )

    train_sampler = Sampler(
        hdf5s_dir=hdf5s_dir,
        split='train',
        segment_seconds=segment_seconds,
        hop_seconds=hop_seconds,
        batch_size=batch_size,
        mini_data=mini_data
    )
    evaluate_validate_sampler = TestSampler(
        hdf5s_dir=hdf5s_dir,
        split='val',
        segment_seconds=segment_seconds,
        hop_seconds=hop_seconds,
        batch_size=batch_size,
        mini_data=mini_data
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    validate_loader = torch.utils.data.DataLoader(
        dataset=evaluate_dataset,
        batch_sampler=evaluate_validate_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    teacher_evaluator = SegmentEvaluator(teacher_model, batch_size)
    statistics_container = StatisticsContainer(statistics_path)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    iteration = resume_iteration
    best_validation_metric = float('inf')

    # Accum trackers (store per-microbatch values scaled by 1/ACCUMULATION_STEPS)
    accum_total = 0.0
    accum_sup = 0.0
    accum_cons = 0.0

    # Store last complete update loss for checkpoint metadata
    last_real_total = 0.0

    optimizer.zero_grad()

    for batch_data_dict in train_loader:
        # Student train mode, but freeze BN stats for small batch stability
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

        teacher_model.eval()

        # Move non-waveform tensors to device
        for key in list(batch_data_dict.keys()):
            if 'waveform' not in key:
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        if 'waveform_clean' not in batch_data_dict or 'waveform_aug' not in batch_data_dict:
            raise RuntimeError("Batch missing waveforms. Expected waveform_clean and waveform_aug.")

        # Waveforms are numpy -> torch
        x_clean = torch.from_numpy(batch_data_dict['waveform_clean']).to(device, dtype=torch.float32)
        x_aug = torch.from_numpy(batch_data_dict['waveform_aug']).to(device, dtype=torch.float32)

        # --- Forward passes ---
        # Teacher on CLEAN (no grad)
        with torch.no_grad():
            out_t_clean = teacher_model(x_clean)

        # Student on AUG (one forward only)
        out_s_aug = model(x_aug)
        out_s_clean = model(x_clean)

        loss_sup = loss_func(model, out_s_clean, batch_data_dict)

        loss_cons = consistency_loss(out_s_aug, out_t_clean, batch_data_dict)

        # --- Ramp Up Logic ---
        current_consistency_weight = get_current_consistency_weight(iteration)
        
        # Weighted Consistency Loss
        weighted_cons = current_consistency_weight * loss_cons
        
        # Total Loss
        loss_batch = loss_sup + weighted_cons

        # Scale for gradient accumulation
        loss_backward = loss_batch / ACCUMULATION_STEPS
        loss_backward.backward()

        # Accumulate (scaled) for exact logging later
        accum_total += loss_backward.item()
        accum_sup += (loss_sup.item() / ACCUMULATION_STEPS)
        accum_cons += (weighted_cons.item() / ACCUMULATION_STEPS)

        # --- UPDATE STEP ---
        if (iteration + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            current_alpha = update_ema_variables(model, teacher_model, iteration)

            # Convert back to full (unscaled) totals for this update window
            real_total = accum_total * ACCUMULATION_STEPS
            real_sup = accum_sup * ACCUMULATION_STEPS
            real_cons = accum_cons * ACCUMULATION_STEPS
            last_real_total = real_total

            statistics_container.append(iteration, {'loss': real_total}, 'train')

            logging.info(
                f'Iter: {iteration}, Alpha: {current_alpha:.4f}, '
                f'Total: {real_total:.3f} (Sup: {real_sup:.3f} + Cons: {real_cons:.3f}), '
                f'ConsWt: {current_consistency_weight:.2f}, '
                f'lr: {optimizer.param_groups[0]["lr"]:.2e}'
            )

            accum_total = 0.0
            accum_sup = 0.0
            accum_cons = 0.0

            if iteration % reduce_iteration == 0 and iteration > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9

        # --- VALIDATION / CHECKPOINT ---
        if (iteration + 1) % 320 == 0:
            checkpoint_path = os.path.join(log_checkpoints_dir, f'{dataset_name}_teacher_iter{iteration}.pth')
            torch.save({
                'iteration': iteration,
                'model': teacher_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': float(last_real_total)
            }, checkpoint_path)

            logging.info('Evaluating Teacher...')
            val_stats = teacher_evaluator.evaluate(validate_loader)
            logging.info(f'Teacher Validation statistics: {val_stats}')

            if val_stats and 'reg_onset_mae' in val_stats:
                if val_stats['reg_onset_mae'] < best_validation_metric and val_stats['reg_onset_mae'] < 0.2:
                    best_validation_metric = float(val_stats['reg_onset_mae'])
                    torch.save(teacher_model.state_dict(), os.path.join(log_checkpoints_dir, 'best_teacher_model.pth'))
                    logging.info(f'New Best Teacher Model: {best_validation_metric:.4f}')

        iteration += 1
        if iteration >= early_stop:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--hdf5s_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='Regress_onset_offset_frame_velocity_CRNN')
    parser.add_argument('--loss_type', type=str, default='regress_onset_offset_frame_velocity_bce')
    parser.add_argument('--augmentation', type=str, default='aug')
    parser.add_argument('--max_note_shift', type=int, default=0)
    parser.add_argument('--max_timing_shift', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--reduce_iteration', type=int, default=10000)
    parser.add_argument('--resume_iteration', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=200000)
    parser.add_argument('--mini_data', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--pretrained_path', type=str, default='none')
    parser.add_argument('--ir_path', type=str, default=None)

    args = parser.parse_args()
    finetune(args)