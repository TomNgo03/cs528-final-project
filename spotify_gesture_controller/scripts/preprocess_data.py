#!/usr/bin/env python3
"""Clean raw MPU6050 gesture CSV files and extract ML features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CSV_COLUMNS,
    GESTURES,
    IMU_AXES,
    MIN_VALID_ROWS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLE_RATE,
)


def estimate_sample_rate(time_ms: np.ndarray) -> float:
    time_s = (time_ms - time_ms[0]) / 1000.0
    dt = np.diff(time_s)
    dt = dt[dt > 0]
    if dt.size == 0:
        return float(SAMPLE_RATE)
    return float(1.0 / np.median(dt))


def clean_dataframe(
    path: Path,
    smooth: bool = False,
    smooth_window: int = 3,
    min_rows: int = MIN_VALID_ROWS,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[c for c in CSV_COLUMNS if c in df.columns]].copy()
    if list(df.columns) != CSV_COLUMNS:
        raise ValueError(f"{path} is missing required CSV columns")

    for col in CSV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["Time(ms)"]).sort_values("Time(ms)")
    if len(df) < min_rows:
        raise ValueError(f"{path} has too few valid rows: {len(df)}")

    df["Time(ms)"] = df["Time(ms)"] - df["Time(ms)"].iloc[0]
    if smooth:
        for col in IMU_AXES:
            df[col] = df[col].rolling(smooth_window, center=True, min_periods=1).mean()
    return df.reset_index(drop=True)


def zero_crossing_rate(x: np.ndarray) -> float:
    x0 = x - np.mean(x)
    signs = np.signbit(x0)
    return float(np.mean(signs[1:] != signs[:-1])) if len(x0) > 1 else 0.0


def axis_features(x: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    peak_abs = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    p2p = float(max_v - min_v)
    energy = float(np.sum(x**2) / len(x))
    zcr = zero_crossing_rate(x)

    x0 = x - mean
    window = np.hanning(len(x0))
    mag = np.abs(np.fft.rfft(x0 * window))
    freqs = np.fft.rfftfreq(len(x0), d=1.0 / fs)

    mag = mag[1:]
    freqs = freqs[1:]
    total_mag = float(np.sum(mag))
    if mag.size == 0 or total_mag == 0.0:
        dom_freq = 0.0
        spectral_centroid = 0.0
        low_band_ratio = 0.0
    else:
        dom_freq = float(freqs[int(np.argmax(mag))])
        spectral_centroid = float(np.sum(freqs * mag) / total_mag)
        low_band_ratio = float(np.sum(mag[freqs <= 5.0]) / total_mag)

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": min_v,
        f"{prefix}_max": max_v,
        f"{prefix}_peak_abs": peak_abs,
        f"{prefix}_rms": rms,
        f"{prefix}_peak_to_peak": p2p,
        f"{prefix}_energy": energy,
        f"{prefix}_zero_crossing_rate": zcr,
        f"{prefix}_dominant_frequency": dom_freq,
        f"{prefix}_spectral_centroid": spectral_centroid,
        f"{prefix}_low_band_ratio_0_5hz": low_band_ratio,
    }


def directional_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    """Capture signed motion shape so opposite swipes do not look identical."""
    x = np.asarray(x, dtype=float)
    x0 = x - np.mean(x)
    half = max(1, len(x0) // 2)
    max_idx = int(np.argmax(x0))
    min_idx = int(np.argmin(x0))
    return {
        f"{prefix}_signed_area": float(np.sum(x0) / len(x0)),
        f"{prefix}_signed_abs_area": float(np.sum(np.sign(x0) * np.abs(x0)) / len(x0)),
        f"{prefix}_first_half_mean": float(np.mean(x0[:half])),
        f"{prefix}_second_half_mean": float(np.mean(x0[half:])),
        f"{prefix}_half_delta": float(np.mean(x0[:half]) - np.mean(x0[half:])),
        f"{prefix}_max_before_min": float(max_idx < min_idx),
        f"{prefix}_peak_order_delta": float((max_idx - min_idx) / max(1, len(x0) - 1)),
        f"{prefix}_signed_peak": float(x0[max_idx] if abs(x0[max_idx]) >= abs(x0[min_idx]) else x0[min_idx]),
    }


def extract_features(df: pd.DataFrame, sample_rate: float | None = None) -> dict[str, float]:
    fs = sample_rate or estimate_sample_rate(df["Time(ms)"].to_numpy())
    features: dict[str, float] = {"sample_rate_estimate": fs, "row_count": float(len(df))}

    for col in IMU_AXES:
        prefix = (
            col.replace("(g)", "")
            .replace("(dps)", "")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "")
        )
        features.update(axis_features(df[col].to_numpy(), fs, prefix))
        features.update(directional_features(df[col].to_numpy(), prefix))

    accel_mag = np.sqrt(
        df["AccelX(g)"].to_numpy() ** 2
        + df["AccelY(g)"].to_numpy() ** 2
        + df["AccelZ(g)"].to_numpy() ** 2
    )
    gyro_mag = np.sqrt(
        df["GyroX(dps)"].to_numpy() ** 2
        + df["GyroY(dps)"].to_numpy() ** 2
        + df["GyroZ(dps)"].to_numpy() ** 2
    )
    features.update(axis_features(accel_mag, fs, "AccelMagnitude"))
    features.update(axis_features(gyro_mag, fs, "GyroMagnitude"))
    return features


def build_feature_table(
    raw_dir: Path,
    smooth: bool = False,
    min_rows: int = MIN_VALID_ROWS,
) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    for gesture in GESTURES:
        gesture_dir = raw_dir / gesture
        if not gesture_dir.exists():
            continue
        for csv_path in sorted(gesture_dir.glob("*.csv")):
            try:
                df = clean_dataframe(csv_path, smooth=smooth, min_rows=min_rows)
                rows.append(extract_features(df))
                labels.append(gesture)
            except Exception as exc:
                print(f"Skipping {csv_path}: {exc}")

    if not rows:
        raise RuntimeError(f"No usable gesture CSV files found under {raw_dir}")
    return pd.DataFrame(rows).fillna(0.0), pd.Series(labels, name="label")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--smooth", action="store_true", help="Apply a small moving average filter")
    parser.add_argument("--min-rows", type=int, default=MIN_VALID_ROWS)
    args = parser.parse_args()

    features, labels = build_feature_table(args.raw_dir, smooth=args.smooth, min_rows=args.min_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_dir / "features.csv", index=False)
    labels.to_csv(args.out_dir / "labels.csv", index=False)
    print(f"Saved {len(labels)} samples to {args.out_dir}")
    print(labels.value_counts().to_string())


if __name__ == "__main__":
    main()
