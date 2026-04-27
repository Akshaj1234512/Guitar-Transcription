#!/usr/bin/env python3
"""
Batch Stage 1 only (audio → MIDI) for evaluation purposes.
Skips BeatNet, technique detection, and tab generation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Batch audio-to-MIDI (Stage 1 only)")
    parser.add_argument("audio_dirs", nargs="+", help="Directories containing audio files")
    parser.add_argument("--output", "-o", default="results", help="Output base directory")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"])
    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).parent.resolve()
    MODEL_PATH = str(
        SCRIPT_DIR / "models" / "audio_to_midi" /
        "gaps_goat_guitartechs_leduc_limited_regress_onset_offset_frame_velocity_bce_log332_iter2000_lr1e-05_bs4.pth"
    )
    INFERENCE_SCRIPT = str(SCRIPT_DIR / "pipeline_utils" / "midi_utils" / "inference.py")
    output_base = Path(args.output)

    for audio_dir_str in args.audio_dirs:
        audio_dir = Path(audio_dir_str)
        source_name = audio_dir.name
        out_dir = output_base / source_name
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        for ext in args.extensions:
            audio_files.extend(sorted(audio_dir.glob(f"*{ext}")))

        print(f"\n[{source_name}] {len(audio_files)} audio files -> {out_dir}")
        ok, fail = 0, 0

        for i, audio_file in enumerate(audio_files, 1):
            midi_out = out_dir / f"{audio_file.stem}.mid"
            if midi_out.exists():
                print(f"  [{i}/{len(audio_files)}] {audio_file.name} ... skip (exists)")
                ok += 1
                continue

            print(f"  [{i}/{len(audio_files)}] {audio_file.name}", end=" ... ", flush=True)
            try:
                cmd = [
                    sys.executable,
                    INFERENCE_SCRIPT,
                    "--model_type", "Regress_onset_offset_frame_velocity_CRNN",
                    "--checkpoint_path", MODEL_PATH,
                    "--post_processor_type", "regression",
                    "--audio_path", str(audio_file),
                    "--cuda"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    print(f"✗ (exit {result.returncode})")
                    fail += 1
                    continue

                # The inference script writes to results/<stem>.mid
                default_out = Path("results") / f"{audio_file.stem}.mid"
                if default_out.exists():
                    default_out.rename(midi_out)
                    print("✓")
                    ok += 1
                else:
                    print("✗ (no output)")
                    fail += 1
            except subprocess.TimeoutExpired:
                print("✗ (timeout)")
                fail += 1
            except Exception as e:
                print(f"✗ ({e})")
                fail += 1

        print(f"  Done: {ok} ok, {fail} failed")

    print("\nAll done.")


if __name__ == "__main__":
    main()
