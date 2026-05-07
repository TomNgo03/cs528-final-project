#!/usr/bin/env python3
"""Capture real live swipe windows from the continuously streaming ESP32."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import serial

from config import BAUD_RATE, CSV_COLUMNS, GESTURES, RAW_DATA_DIR, SAMPLE_RATE, SERIAL_PORT

CSV_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){6}$")


def next_index(gesture_dir: Path, gesture: str) -> int:
    used = []
    for path in gesture_dir.glob(f"{gesture}_*.csv"):
        try:
            used.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(used, default=-1) + 1


def read_csv_row(ser: serial.Serial) -> list[float] | None:
    raw = ser.readline()
    if not raw:
        return None
    line = raw.decode("utf-8", errors="ignore").strip()
    if not CSV_RE.match(line):
        return None
    return [float(part) for part in line.split(",")]


def capture_window(ser: serial.Serial, window_samples: int) -> pd.DataFrame:
    rows: list[list[float]] = []
    while len(rows) < window_samples:
        row = read_csv_row(ser)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df["Time(ms)"] = range(0, len(df) * int(1000 / SAMPLE_RATE), int(1000 / SAMPLE_RATE))
    return df


def countdown(seconds: float) -> None:
    if seconds <= 0:
        return
    print(f"Get ready... {seconds:.1f}s")
    time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--out-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--count", type=int, default=12, help="Real samples per gesture")
    parser.add_argument("--window-samples", type=int, default=120, help="120 samples is about 1.2 seconds")
    parser.add_argument("--prepare-seconds", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gestures", nargs="+", default=GESTURES, choices=GESTURES)
    args = parser.parse_args()

    for gesture in args.gestures:
        gesture_dir = args.out_dir / gesture
        gesture_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for path in gesture_dir.glob(f"{gesture}_*.csv"):
                path.unlink()

    print(f"Opening {args.port} at {args.baud} baud")
    print("For each sample: press Enter, then do exactly one gesture during the capture window.")
    print("Keep the sensor still before pressing Enter and after the motion.")

    with serial.Serial(args.port, args.baud, timeout=2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        ser.write(b"stream\n")

        for gesture in args.gestures:
            gesture_dir = args.out_dir / gesture
            start_idx = next_index(gesture_dir, gesture)
            print(f"\n=== {gesture} ===")
            for sample_idx in range(args.count):
                input(f"Sample {sample_idx + 1}/{args.count} for {gesture}: press Enter when ready ")
                ser.reset_input_buffer()
                countdown(args.prepare_seconds)
                df = capture_window(ser, args.window_samples)
                out_path = gesture_dir / f"{gesture}_{start_idx + sample_idx:03d}.csv"
                df.to_csv(out_path, index=False)
                print(f"Saved {out_path.name}")

    print("\nDone. Now run preprocess_data.py and train_model.py.")


if __name__ == "__main__":
    main()
