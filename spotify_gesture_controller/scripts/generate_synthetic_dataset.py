#!/usr/bin/env python3
"""Generate a synthetic MPU6050 gesture dataset for pipeline testing.

This is not a replacement for real glove recordings. It creates realistic-looking
CSV windows so the preprocessing, training, and real-time code can be tested
before hardware data collection is complete. This script intentionally uses only
the Python standard library so it can run before installing ML dependencies.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from config import CSV_COLUMNS, GESTURES, RAW_DATA_DIR, SAMPLE_RATE, SAMPLES_PER_WINDOW


def gaussian(t: float, center: float, width: float, amplitude: float) -> float:
    return amplitude * math.exp(-0.5 * ((t - center) / width) ** 2)


def damped_ring(t: float, start: float, freq: float, amplitude: float, decay: float) -> float:
    if t < start:
        return 0.0
    tm = t - start
    return amplitude * math.sin(2 * math.pi * freq * tm) * math.exp(-decay * tm)


def base_row(i: int, rng: random.Random, phase: float, drift_freq: float) -> dict[str, float]:
    t = i / SAMPLE_RATE
    drift = 0.015 * math.sin(2 * math.pi * drift_freq * t + phase)
    return {
        "Time(ms)": i * int(1000 / SAMPLE_RATE),
        "AccelX(g)": drift + rng.gauss(0.0, 0.018),
        "AccelY(g)": -0.01 * drift + rng.gauss(0.0, 0.018),
        "AccelZ(g)": 1.0 + 0.5 * drift + rng.gauss(0.0, 0.015),
        "GyroX(dps)": rng.gauss(0.0, 1.2),
        "GyroY(dps)": rng.gauss(0.0, 1.2),
        "GyroZ(dps)": rng.gauss(0.0, 1.2),
    }


def add_tap(row: dict[str, float], t: float, center: float, strength: float) -> None:
    impact = gaussian(t, center, 0.025, strength)
    rebound = gaussian(t, center + 0.07, 0.04, -0.45 * strength)
    ring = damped_ring(t, center, 17.0, 25.0 * strength, 8.0)
    row["AccelZ(g)"] += impact + rebound
    row["AccelX(g)"] += 0.25 * impact
    row["GyroY(dps)"] += ring
    row["GyroZ(dps)"] += -0.45 * ring


def add_swipe(
    row: dict[str, float],
    t: float,
    axis: str,
    sign: float,
    strength: float,
    center: float,
) -> None:
    main = gaussian(t, center, 0.22, sign * strength)
    counter = gaussian(t, center + 0.34, 0.20, -0.65 * sign * strength)
    row[axis] += main + counter
    if axis == "AccelX(g)":
        row["GyroZ(dps)"] += 135.0 * (main + 0.55 * counter)
        row["GyroY(dps)"] += 28.0 * gaussian(t, center, 0.28, sign * strength)
    else:
        row["GyroX(dps)"] += 135.0 * (main + 0.55 * counter)
        row["GyroZ(dps)"] += 24.0 * gaussian(t, center, 0.26, -sign * strength)


def synthesize_window(gesture: str, rng: random.Random) -> list[dict[str, float]]:
    phase = rng.uniform(0, 2 * math.pi)
    drift_freq = rng.uniform(0.15, 0.45)
    strength = rng.gauss(1.0, 0.12)
    tap_center = rng.gauss(1.45, 0.08)
    swipe_center = rng.gauss(1.45, 0.04)
    double_gap = rng.gauss(0.34, 0.035)

    rows = []
    for i in range(SAMPLES_PER_WINDOW):
        t = i / SAMPLE_RATE
        row = base_row(i, rng, phase, drift_freq)

        if gesture == "tap_index":
            add_tap(row, t, tap_center, strength)
        elif gesture == "double_tap":
            add_tap(row, t, tap_center - 0.18, strength)
            add_tap(row, t, tap_center - 0.18 + double_gap, rng.gauss(0.9, 0.08))
        elif gesture == "swipe_right":
            add_swipe(row, t, "AccelX(g)", 1.0, strength, swipe_center)
        elif gesture == "swipe_left":
            add_swipe(row, t, "AccelX(g)", -1.0, strength, swipe_center)
        elif gesture == "swipe_up":
            add_swipe(row, t, "AccelY(g)", 1.0, strength, swipe_center)
        elif gesture == "swipe_down":
            add_swipe(row, t, "AccelY(g)", -1.0, strength, swipe_center)
        else:
            raise ValueError(f"Unknown gesture: {gesture}")

        for col in CSV_COLUMNS[1:]:
            row[col] = round(row[col], 4)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=528)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for gesture in GESTURES:
        gesture_dir = args.out_dir / gesture
        gesture_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(gesture_dir.glob(f"{gesture}_*.csv"))
        if existing and not args.overwrite:
            print(f"Skipping {gesture}: {len(existing)} files already exist. Use --overwrite to replace them.")
            continue

        if args.overwrite:
            for path in existing:
                path.unlink()

        for idx in range(args.count):
            rows = synthesize_window(gesture, rng)
            write_csv(gesture_dir / f"{gesture}_{idx:03d}.csv", rows)
            written += 1
        print(f"Generated {args.count} synthetic samples for {gesture}")

    print(f"Done. Wrote {written} files under {args.out_dir}")


if __name__ == "__main__":
    main()
