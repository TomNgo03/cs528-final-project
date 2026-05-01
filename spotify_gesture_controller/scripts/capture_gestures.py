#!/usr/bin/env python3
"""Capture labeled 3 second gesture windows from ESP32 serial output."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import serial

from config import (
    BAUD_RATE,
    CSV_COLUMNS,
    GESTURES,
    MAX_VALID_ROWS,
    MIN_VALID_ROWS,
    RAW_DATA_DIR,
    SAMPLES_PER_WINDOW,
    SERIAL_PORT,
)

START_RE = re.compile(r"=+ START_GESTURE_([A-Za-z0-9_]+) =+")
END_RE = re.compile(r"=+ END_GESTURE_([A-Za-z0-9_]+) =+")
CSV_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){6}$")


def next_index(gesture_dir: Path, gesture: str) -> int:
    used = []
    for path in gesture_dir.glob(f"{gesture}_*.csv"):
        try:
            used.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(used, default=-1) + 1


def validate_rows(lines: list[str]) -> pd.DataFrame | None:
    if len(lines) < MIN_VALID_ROWS or len(lines) > MAX_VALID_ROWS:
        print(f"Warning: expected about {SAMPLES_PER_WINDOW} rows, got {len(lines)}")
        return None
    rows = [line.split(",") for line in lines]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS).apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    if len(df) < MIN_VALID_ROWS:
        print(f"Warning: sample has too many malformed rows after cleaning: {len(df)}")
        return None
    return df


def send_collection_command(ser: serial.Serial, gesture: str) -> None:
    ser.write(f"label {gesture}\n".encode("utf-8"))
    time.sleep(0.15)
    ser.write(b"collect\n")


def collect_samples(port: str, baud: int, gesture: str, target_count: int, out_dir: Path) -> None:
    gesture_dir = out_dir / gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_index(gesture_dir, gesture)
    collected = 0

    print(f"Connecting to {port} at {baud} baud")
    with serial.Serial(port, baud, timeout=2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        while collected < target_count:
            print(f"\nArming sample {collected + 1}/{target_count} for {gesture}")
            send_collection_command(ser, gesture)

            active = False
            rows: list[str] = []
            marker_gesture = gesture
            deadline = time.monotonic() + 12

            while time.monotonic() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                start_match = START_RE.match(line)
                if start_match:
                    marker_gesture = start_match.group(1)
                    active = marker_gesture == gesture
                    rows = []
                    continue

                end_match = END_RE.match(line)
                if end_match and active:
                    df = validate_rows(rows)
                    if df is None:
                        print("Skipping corrupted sample; retrying this count.")
                    else:
                        file_idx = start_idx + collected
                        path = gesture_dir / f"{gesture}_{file_idx:03d}.csv"
                        df.to_csv(path, index=False)
                        collected += 1
                        print(f"Collected {collected}/{target_count} for {gesture}: {path.name}")
                    active = False
                    break

                if active and line == ",".join(CSV_COLUMNS):
                    continue
                if active and CSV_RE.match(line):
                    rows.append(line)
            else:
                print("Timed out waiting for a complete gesture window; retrying.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--gesture", choices=GESTURES)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()

    gesture = args.gesture
    if gesture is None:
        print("Available gestures:")
        for idx, name in enumerate(GESTURES, start=1):
            print(f"  {idx}. {name}")
        choice = input("Gesture to collect: ").strip()
        gesture = GESTURES[int(choice) - 1] if choice.isdigit() else choice
        if gesture not in GESTURES:
            raise SystemExit(f"Unknown gesture: {gesture}")

    collect_samples(args.port, args.baud, gesture, args.count, args.out_dir)


if __name__ == "__main__":
    main()
