#!/usr/bin/env python3
"""Import HW1 real up/down/left/right CSVs into the final dataset.

The HW1 files are one recording per direction. To make them usable with the
existing train/test pipeline, this script creates small jittered variants of
each real recording while preserving the measured sensor baseline/noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import CSV_COLUMNS, RAW_DATA_DIR, SAMPLE_RATE, SAMPLES_PER_WINDOW

DEFAULT_SOURCE_DIR = Path("/Users/tungngo/Downloads/528tungngohw1/gesture_data")
GESTURE_MAP = {
    "gesture_left.csv": "swipe_left",
    "gesture_right.csv": "swipe_right",
    "gesture_up.csv": "swipe_up",
    "gesture_down.csv": "swipe_down",
}


def clean_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[CSV_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.iloc[:SAMPLES_PER_WINDOW].copy()
    if len(df) < SAMPLES_PER_WINDOW:
        raise ValueError(f"{path} has only {len(df)} rows")
    df["Time(ms)"] = np.arange(SAMPLES_PER_WINDOW) * int(1000 / SAMPLE_RATE)
    return df.reset_index(drop=True)


def augment(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    shift = int(rng.integers(-8, 9))
    for col in CSV_COLUMNS[1:]:
        values = np.roll(out[col].to_numpy(dtype=float), shift)
        baseline = float(np.mean(values[:40]))
        dynamic = values - baseline
        scale = float(rng.normal(1.0, 0.08))
        noise_scale = 0.002 if "Accel" in col else 0.12
        bias_scale = 0.003 if "Accel" in col else 0.18
        values = baseline + dynamic * scale + rng.normal(0.0, noise_scale, len(values)) + rng.normal(0.0, bias_scale)
        out[col] = values
    out["Time(ms)"] = np.arange(len(out)) * int(1000 / SAMPLE_RATE)
    return out.round({col: 4 for col in CSV_COLUMNS[1:]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5281)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for filename, gesture in GESTURE_MAP.items():
        src = args.source_dir / filename
        df = clean_source(src)
        gesture_dir = args.out_dir / gesture
        gesture_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for old_file in gesture_dir.glob(f"{gesture}_*.csv"):
                old_file.unlink()
        elif list(gesture_dir.glob(f"{gesture}_*.csv")):
            print(f"Skipping {gesture}; files already exist. Use --overwrite to replace.")
            continue

        for idx in range(args.count):
            sample = augment(df, rng)
            sample.to_csv(gesture_dir / f"{gesture}_{idx:03d}.csv", index=False)
        print(f"Imported {src.name} -> {gesture} ({args.count} augmented real-derived samples)")


if __name__ == "__main__":
    main()
