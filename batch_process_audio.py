#!/usr/bin/env python3
"""
Batch processing script for TART guitar transcription.
Processes all audio files in multiple directories using the predict.py script.

example usage

cd /data/user/MusicAI/Music-AI
python batch_process_audio.py \
  /data/user/MusicAI/EGDB/audio_DI \
  /data/user/MusicAI/EGDB/audio_Ftwin \
  /data/user/MusicAI/EGDB/audio_JCjazz \
  /data/user/MusicAI/EGDB/audio_Marshall \
  /data/user/MusicAI/EGDB/audio_Mesa \
  /data/user/MusicAI/EGDB/audio_Plexi \
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import shutil

def process_audio_directory(audio_dir, output_dir=None, extensions=['.wav', '.mp3', '.flac']):
    """
    Process all audio files in a directory using predict.py.
    
    Args:
        audio_dir: Path to directory containing audio files
        output_dir: Optional directory to organize results (default: ./results)
        extensions: List of audio file extensions to process
    
    Returns:
        results dict with processing status
    """
    
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        print(f"Error: Audio directory does not exist: {audio_dir}")
        return None
    
    # Get list of audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(sorted(audio_path.glob(f'*{ext}')))
    
    if not audio_files:
        print(f"Warning: No audio files found in {audio_dir}")
        return None
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Setup output directory with source directory name
    if output_dir is None:
        output_dir = Path('./results')
    else:
        output_dir = Path(output_dir)
    
    # Create subdirectory for this audio source
    source_name = audio_path.name
    source_output_dir = output_dir / source_name
    source_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results log
    log_file = source_output_dir / f'batch_process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    results = {
        'source_directory': str(audio_dir),
        'source_name': source_name,
        'start_time': datetime.now().isoformat(),
        'total_files': len(audio_files),
        'processed': [],
        'failed': []
    }
    
    # Process each audio file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"  [{i}/{len(audio_files)}] {audio_file.name}", end=" ... ", flush=True)
        
        try:
            # Run predict.py
            cmd = ['python', 'predict.py', '--audio_path', str(audio_file)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            
            if result.returncode == 0:
                print("✓", end="")
                
                # Organize output files into subdirectory
                file_stem = audio_file.stem
                results_root = Path("results")
                output_files = []
                
                # Move generated files (.mid, .musicxml, .jams) to the source-specific subdirectory
                for ext in ['.mid', '.musicxml', '.jams']:
                    src_file = results_root / f"{file_stem}{ext}"
                    if src_file.exists():
                        dst_file = source_output_dir / f"{file_stem}{ext}"
                        shutil.move(str(src_file), str(dst_file))
                        output_files.append(dst_file.name)
                
                print(f" (moved {len(output_files)} file(s))")
                results['processed'].append({
                    'file': audio_file.name,
                    'path': str(audio_file),
                    'output_files': output_files,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                print(f"✗ (code {result.returncode})")
                results['failed'].append({
                    'file': audio_file.name,
                    'path': str(audio_file),
                    'error': result.stderr[:500],
                    'timestamp': datetime.now().isoformat()
                })
        
        except subprocess.TimeoutExpired:
            print("✗ (timeout)")
            results['failed'].append({
                'file': audio_file.name,
                'path': str(audio_file),
                'error': 'Process timeout (5 minutes)',
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            print(f"✗ (exception)")
            results['failed'].append({
                'file': audio_file.name,
                'path': str(audio_file),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # Save results log
    results['end_time'] = datetime.now().isoformat()
    results['successful_count'] = len(results['processed'])
    results['failed_count'] = len(results['failed'])
    
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary for this directory
    print(f"\n  Results: {results['successful_count']} successful, {results['failed_count']} failed")
    print(f"  Log: {log_file}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch process audio files with TART transcription from multiple directories'
    )
    parser.add_argument(
        'audio_dirs',
        nargs='+',
        help='One or more directories containing audio files to process'
    )
    parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Output directory for results (default: ./results)'
    )
    parser.add_argument(
        '--extensions',
        '-e',
        nargs='+',
        default=['.wav', '.mp3', '.flac'],
        help='Audio file extensions to process (default: .wav .mp3 .flac)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output is None:
        output_base = Path('./results')
    else:
        output_base = Path(args.output)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process all directories
    all_results = {
        'batch_start_time': datetime.now().isoformat(),
        'total_directories': len(args.audio_dirs),
        'directories': []
    }
    
    print("="*70)
    print("BATCH PROCESSING MULTIPLE DIRECTORIES")
    print("="*70)
    
    for idx, audio_dir in enumerate(args.audio_dirs, 1):
        print(f"\n[{idx}/{len(args.audio_dirs)}] Processing: {audio_dir}")
        result = process_audio_directory(audio_dir, output_base, args.extensions)
        if result:
            all_results['directories'].append(result)
    
    # Save overall results
    overall_log = output_base / f'batch_overall_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    all_results['batch_end_time'] = datetime.now().isoformat()
    
    with open(overall_log, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("OVERALL BATCH PROCESSING COMPLETE")
    print("="*70)
    total_successful = sum(d.get('successful_count', 0) for d in all_results['directories'])
    total_failed = sum(d.get('failed_count', 0) for d in all_results['directories'])
    print(f"Directories processed: {len(args.audio_dirs)}")
    print(f"Total files successful: {total_successful}")
    print(f"Total files failed: {total_failed}")
    print(f"Overall log: {overall_log}")
    print(f"Results directory: {output_base}")
