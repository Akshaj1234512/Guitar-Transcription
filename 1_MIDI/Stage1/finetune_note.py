#!/usr/bin/env python3

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))

import argparse
import time
import logging

import torch
import torch.optim as optim
import torch.utils.data

from utilities import (create_folder, StatisticsContainer)
from data_generator import Augmentor, collate_fn
from models import Regress_onset_offset_frame_velocity_CRNN  # NOTE-ONLY
from pytorch_utils import move_data_to_device
from losses import get_loss_func
from evaluate import SegmentEvaluator
import config


def _load_pt_note_only(model, pretrained_path, logger=logging):

    logger.info(f'Loading pretrained model from {pretrained_path}')
    ckpt = torch.load(pretrained_path, map_location='cpu')

    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    if isinstance(state, dict) and 'note_model' in state and isinstance(state['note_model'], dict):
        logger.info("Detected split checkpoint with 'note_model' and 'pedal_model'. Using state['note_model'].")
        state = state['note_model']

    if isinstance(state, dict) and any(k.startswith('note_model.') for k in state.keys()):
        logger.info("Detected flat Note_pedal checkpoint with 'note_model.' prefix. Stripping prefix.")
        state = {k.replace('note_model.', '', 1): v for k, v in state.items() if k.startswith('note_model.')}

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint did not resolve to a state_dict (dict of tensors).")

    missing, unexpected = model.load_state_dict(state, strict=False)

    logger.info(f'Pretrained loaded with strict=False. Missing={len(missing)} Unexpected={len(unexpected)}')
    if len(missing) > 0:
        logger.info("Example missing keys: " + ", ".join(missing[:20]))
    if len(unexpected) > 0:
        logger.info("Example unexpected keys: " + ", ".join(unexpected[:20]))

    if len(unexpected) > 0 and len(missing) > 0 and len(unexpected) > 1000:
        logger.warning("WARNING: Huge number of unexpected keys. Likely wrong checkpoint format or wrong extraction.")



def finetune(args):
    # Arguments & parameters
    workspace = args.workspace
    dataset_name = args.dataset_name
    hdf5s_dir = args.hdf5s_dir

    model_type = args.model_type
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

    checkpoints_dir = os.path.join(workspace, 'checkpoints')
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs')
    create_folder(logs_dir)

    # Log number
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    if log_files:
        log_numbers = [int(f.split('.')[0]) for f in log_files]
        current_log_number = max(log_numbers) + 1
    else:
        current_log_number = 0

    log_folder_name = f"log_{current_log_number:04d}"
    log_checkpoints_dir = os.path.join(checkpoints_dir, log_folder_name)
    create_folder(log_checkpoints_dir)

    log_path = os.path.join(logs_dir, f'{current_log_number:04d}.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.info(f'Using log number: {current_log_number}')
    logging.info(f'Checkpoints will be saved to: {log_checkpoints_dir}')
    logging.info(f'Dataset: {dataset_name}')
    logging.info(f'HDF5 directory: {hdf5s_dir}')
    logging.info(args)

    if model_type != 'Regress_onset_offset_frame_velocity_CRNN':
        logging.warning(f'Overriding model_type={model_type} -> Regress_onset_offset_frame_velocity_CRNN (note-only).')
    model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=frames_per_second, classes_num=classes_num)

    # Augmentation
    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    elif augmentation == 'teacher':
        augmentor = TeacherAugmentor()
    else:
        raise Exception('Incorrect augmentation!')

    # Dataset / Samplers
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

    evaluate_train_sampler = TestSampler(
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

    evaluate_train_loader = torch.utils.data.DataLoader(
        dataset=evaluate_dataset,
        batch_sampler=evaluate_train_sampler,
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

    evaluator = SegmentEvaluator(model, batch_size)
    statistics_container = StatisticsContainer(statistics_path)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    if args.pretrained_path and args.pretrained_path.lower() != 'none':
        _load_pt_note_only(model, args.pretrained_path, logger=logging)
    else:
        logging.info('No pretrained_path provided; training from scratch.')

    model.to(device)

    torch.autograd.set_detect_anomaly(True)

    logging.info("===== PRE-TRAIN VALIDATION (sanity check) =====")
    model.eval()
    with torch.no_grad():
        pretrain_val_stats = evaluator.evaluate(validate_loader)

    logging.info(f"Pre-train validation statistics: {pretrain_val_stats}")
    logging.info("===== END PRE-TRAIN VALIDATION =====")


    train_bgn_time = time.time()
    iteration = resume_iteration

    best_validation_metric = float('inf')
    best_validation_iteration = 0

    for batch_data_dict in train_loader:
        model.train()

        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        batch_output_dict = model(batch_data_dict['waveform'])
        loss = loss_func(model, batch_output_dict, batch_data_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        statistics_container.append(iteration, {'loss': loss_value}, 'train')

        if iteration % 100 == 0:
            logging.info(f'Iteration: {iteration}, loss: {loss_value:.3f}, lr: {optimizer.param_groups[0]["lr"]:.2e}')

        if iteration % reduce_iteration == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
                logging.info(f'Reduce learning rate to {param_group["lr"]:.3e}')

        if iteration % 1000 == 0 and iteration > 0:
            checkpoint_filename = f'{dataset_name}_{loss_type}_log{current_log_number}_iter{iteration}_lr{learning_rate:.0e}_bs{batch_size}.pth'
            checkpoint_path = os.path.join(log_checkpoints_dir, checkpoint_filename)
            torch.save({
                'iteration': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss_value,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'dataset_name': dataset_name,
                'model_type': 'Regress_onset_offset_frame_velocity_CRNN',
                'batch_size': batch_size,
                'augmentation': augmentation,
                'max_note_shift': max_note_shift
            }, checkpoint_path)
            logging.info(f'Save checkpoint to {checkpoint_path}')

        if iteration % 1000 == 0 and iteration > 0:
            logging.info('------------------------------------')
            logging.info(f'Iteration: {iteration}')

            train_fin_time = time.time()

            evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
            validate_statistics = evaluator.evaluate(validate_loader)

            logging.info(f'    Train statistics: {evaluate_train_statistics}')
            logging.info(f'    Validation statistics: {validate_statistics}')

            statistics_container.append(iteration, evaluate_train_statistics, 'train')
            statistics_container.append(iteration, validate_statistics, 'validation')
            statistics_container.dump()

            # choose best by lowest reg_onset_mae (good proxy for P50/F50)
            if validate_statistics and 'reg_onset_mae' in validate_statistics:
                metric = float(validate_statistics['reg_onset_mae'])
                if metric < best_validation_metric:
                    best_validation_metric = metric
                    best_validation_iteration = iteration

                    best_checkpoint_path = os.path.join(
                        log_checkpoints_dir,
                        f'best_model_note_only_{dataset_name}_log{current_log_number}.pth'
                    )
                    torch.save({
                        'iteration': iteration,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': loss_value,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'dataset_name': dataset_name,
                        'model_type': 'Regress_onset_offset_frame_velocity_CRNN',
                        'batch_size': batch_size,
                        'augmentation': augmentation,
                        'max_note_shift': max_note_shift,
                        'validation_metric_reg_onset_mae': metric,
                        'validation_statistics': validate_statistics
                    }, best_checkpoint_path)
                    logging.info(f'New best model saved! reg_onset_mae={metric:.6f} at iter={iteration}')

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info(f'Train time: {train_time:.3f} s, validate time: {validate_time:.3f} s')
            train_bgn_time = time.time()

        iteration += 1
        if iteration >= early_stop:
            break

    logging.info(f'Training done. Best reg_onset_mae={best_validation_metric:.6f} at iter={best_validation_iteration}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune NOTE-ONLY transcription model on any dataset')
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--hdf5s_dir', type=str, required=True)

    # Force note-only defaults but keep args for compatibility
    parser.add_argument('--model_type', type=str, default='Regress_onset_offset_frame_velocity_CRNN')
    parser.add_argument('--loss_type', type=str, default='regress_onset_offset_frame_velocity_bce')

    parser.add_argument('--augmentation', type=str, default='aug')
    parser.add_argument('--max_note_shift', type=int, default=0)
    parser.add_argument('--max_timing_shift', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--reduce_iteration', type=int, default=10000)
    parser.add_argument('--resume_iteration', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=100000)
    parser.add_argument('--mini_data', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)

    # can be PT.pth, or "none"
    parser.add_argument('--pretrained_path', type=str, default='none')

    args = parser.parse_args()
    finetune(args)
