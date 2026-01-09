#!/usr/bin/env python3
import argparse
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

NOTE_F1_RE = re.compile(r"note_f1:\s*([0-9.]+)%", re.IGNORECASE)

def run_cmd(cmd: list) -> Tuple[int, str]:
    """Run command, return (returncode, combined stdout/stderr)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def parse_note_f1(output: str) -> Optional[float]:
    """Return note_f1 as percent (e.g. 78.2), or None if not found."""
    m = NOTE_F1_RE.search(output)
    return float(m.group(1)) if m else None

def frange(spec: str):
    """
    Parse "start:stop:step" into a list of floats inclusive-ish.
    Example: "0.1:0.31:0.05" -> [0.1, 0.15, 0.2, 0.25, 0.3]
    """
    try:
        start_s, stop_s, step_s = spec.split(":")
        start, stop, step = float(start_s), float(stop_s), float(step_s)
    except Exception:
        raise ValueError(f"Bad range spec '{spec}'. Use start:stop:step")

    if step <= 0:
        raise ValueError("step must be > 0")

    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 6))
        x += step
    return vals

def main():
    ap = argparse.ArgumentParser(
        description="Grid search onset/offset/frame thresholds by calling calculate_metrics and parsing note_f1."
    )

    ap.add_argument("--workspace", required=True)
    ap.add_argument("--model_type", default="Note_pedal")
    ap.add_argument("--augmentation", default="none")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--hdf5s_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model_name", required=True)

    ap.add_argument("--onset", default="0.10:0.35:0.05",
                    help='Range "start:stop:step" for onset_threshold (default: 0.10:0.35:0.05)')
    ap.add_argument("--offset", default="0.10:0.40:0.05",
                    help='Range "start:stop:step" for offset_threshold (default: 0.10:0.40:0.05)')
    ap.add_argument("--frame", default="0.05:0.35:0.05",
                    help='Range "start:stop:step" for frame_threshold (default: 0.05:0.35:0.05)')

    ap.add_argument("--max_runs", type=int, default=0,
                    help="Optional cap on number of combinations (0 = no cap). Useful to test script quickly.")
    ap.add_argument("--out_csv", default="threshold_grid_results.csv",
                    help="CSV path to write all results (default: threshold_grid_results.csv)")

    args = ap.parse_args()

    onset_vals = frange(args.onset)
    offset_vals = frange(args.offset)
    frame_vals = frange(args.frame)

    combos = list(itertools.product(onset_vals, offset_vals, frame_vals))
    if args.max_runs and args.max_runs > 0:
        combos = combos[: args.max_runs]

    out_csv = Path(args.out_csv)

    print(f"Grid sizes: onset={len(onset_vals)}, offset={len(offset_vals)}, frame={len(frame_vals)}")
    print(f"Total combinations: {len(combos)}")
    print(f"Writing results to: {out_csv.resolve()}")
    print()

    best_f1 = -1.0
    best_thr = None
    rows = []

    for i, (on, off, fr) in enumerate(combos, start=1):
        cmd = [
            sys.executable, "pytorch/calculate_score_for_paper.py", "calculate_metrics",
            "--workspace", args.workspace,
            "--model_type", args.model_type,
            "--augmentation", args.augmentation,
            "--dataset_name", args.dataset_name,
            "--hdf5s_dir", args.hdf5s_dir,
            "--split", args.split,
            "--model_name", args.model_name,
            "--thresholds", str(on), str(off), str(fr),
        ]

        rc, out = run_cmd(cmd)
        f1 = parse_note_f1(out)

        status = "OK"
        if rc != 0:
            status = f"FAIL(rc={rc})"
        if f1 is None:
            status = status + "|NO_F1"

        f1_str = f"{f1:.2f}%" if f1 is not None else "NA"
        print(f"[{i:4d}/{len(combos)}] onset={on:.3f} offset={off:.3f} frame={fr:.3f} -> note_f1={f1_str}  {status}")

        rows.append({
            "onset": on,
            "offset": off,
            "frame": fr,
            "note_f1_percent": f1 if f1 is not None else "",
            "returncode": rc,
            "status": status,
        })

        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_thr = (on, off, fr)

    # Write CSV
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["onset", "offset", "frame", "note_f1_percent", "returncode", "status"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== BEST ===")
    if best_thr is None:
        print("No successful runs with parsable note_f1.")
    else:
        on, off, fr = best_thr
        print(f"best note_f1 = {best_f1:.2f}%")
        print(f"thresholds   = onset {on:.3f}   offset {off:.3f}   frame {fr:.3f}")
        print(f"results csv  = {out_csv.resolve()}")

if __name__ == "__main__":
    main()

'''
python test.py   --workspace /data/akshaj/MusicAI/workspace   --dataset_name gaps_goat_guitartechs_leduc   --hdf5s_dir /data/akshaj/MusicAI/workspace/hdf5s/guitarset_full/2024   --split train   --model_name gaps_goat_guitartechs_leduc_regress_onset_offset_frame_velocity_bce_log87_iter54000_lr1e-05_bs4   --onset 0.05:0.40:0.01   --offset 0.10:0.10:0.05   --frame 0.10:0.10:0.05
'''