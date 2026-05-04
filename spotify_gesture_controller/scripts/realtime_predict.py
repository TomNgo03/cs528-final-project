#!/usr/bin/env python3
"""Predict gestures from live ESP32 IMU serial data and control Spotify."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import serial

from config import (
    BAUD_RATE,
    CSV_COLUMNS,
    MODEL_DIR,
    RAW_DATA_DIR,
    SAMPLE_RATE,
    SAMPLES_PER_WINDOW,
    SERIAL_PORT,
)
from preprocess_data import extract_features
from spotify_control import execute_gesture, get_client

CSV_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){6}$")


def prediction_confidence(model, row: pd.DataFrame) -> tuple[int, float]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row)[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])
    pred = int(model.predict(row)[0])
    return pred, 1.0


def load_model(model_dir: Path):
    model = joblib.load(model_dir / "gesture_model.pkl")
    encoder = joblib.load(model_dir / "label_encoder.pkl")
    metadata_path = model_dir / "model_metadata.json"
    feature_columns = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = list(metadata.get("feature_columns", [])) or None
    return model, encoder, feature_columns


def predict_dataframe(model, encoder, feature_columns: list[str] | None, df: pd.DataFrame) -> tuple[str, float]:
    features = extract_features(df, sample_rate=SAMPLE_RATE)
    row = pd.DataFrame([features])
    if feature_columns is not None:
        row = row.reindex(columns=feature_columns, fill_value=0.0)
    pred_idx, conf = prediction_confidence(model, row)
    return encoder.inverse_transform([pred_idx])[0], conf


def motion_score(df: pd.DataFrame) -> float:
    accel = df[["AccelX(g)", "AccelY(g)", "AccelZ(g)"]].to_numpy(dtype=float)
    gyro = df[["GyroX(dps)", "GyroY(dps)", "GyroZ(dps)"]].to_numpy(dtype=float)
    accel_dynamic = accel - accel.mean(axis=0)
    accel_peak = float(np.max(np.linalg.norm(accel_dynamic, axis=1)))
    gyro_peak = float(np.max(np.linalg.norm(gyro, axis=1)))
    return accel_peak + gyro_peak / 250.0


def replay_raw_dataset(
    model,
    encoder,
    feature_columns: list[str] | None,
    raw_dir: Path,
    delay: float,
    execute: bool,
    threshold: float,
    limit: int | None,
    gesture_filter: str | None,
    no_spotify: bool,
    replay_sequence: list[str] | None,
    samples_per_gesture: int,
    replay_start_index: int,
) -> None:
    if replay_sequence:
        files = []
        for gesture in replay_sequence:
            gesture_files = sorted((raw_dir / gesture).glob("*.csv"))
            if not gesture_files:
                raise SystemExit(f"No CSV files found for gesture '{gesture}' under {raw_dir}")
            files.extend(gesture_files[replay_start_index : replay_start_index + samples_per_gesture])
    else:
        files = sorted(raw_dir.glob("*/*.csv"))
        if gesture_filter:
            files = [path for path in files if path.parent.name == gesture_filter]
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No raw CSV files found under {raw_dir}")

    sp = None
    if execute and not no_spotify:
        sp = get_client()

    print(f"Replaying {len(files)} generated/raw CSV windows from {raw_dir}")
    for path in files:
        df = pd.read_csv(path)
        gesture, conf = predict_dataframe(model, encoder, feature_columns, df)
        actual = path.parent.name
        print(f"File: {path.name}, actual: {actual}, predicted: {gesture}, confidence: {conf:.2f}")
        if execute and conf >= threshold:
            if no_spotify:
                print(f"Trigger skipped (--no-spotify): {gesture}")
            else:
                execute_gesture(gesture, sp)
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=50, help="Rows between predictions")
    parser.add_argument("--motion-threshold", type=float, default=0.35)
    parser.add_argument("--no-spotify", action="store_true", help="Print predictions without API calls")
    parser.add_argument("--debug-serial", action="store_true", help="Print non-CSV serial lines from ESP32")
    parser.add_argument("--replay-raw", action="store_true", help="Replay saved raw CSV windows instead of using serial")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--replay-delay", type=float, default=0.15)
    parser.add_argument("--execute-replay", action="store_true", help="Execute Spotify commands during --replay-raw")
    parser.add_argument("--replay-limit", type=int, help="Limit number of replayed CSV windows")
    parser.add_argument("--replay-gesture", help="Replay only one actual gesture folder, such as swipe_right")
    parser.add_argument(
        "--replay-sequence",
        help="Comma-separated gesture sequence to replay, such as tap_index,swipe_right,swipe_up",
    )
    parser.add_argument("--samples-per-gesture", type=int, default=1)
    parser.add_argument("--replay-start-index", type=int, default=0)
    args = parser.parse_args()

    model, encoder, feature_columns = load_model(args.model_dir)

    if args.replay_raw:
        replay_sequence = None
        if args.replay_sequence:
            replay_sequence = [gesture.strip() for gesture in args.replay_sequence.split(",") if gesture.strip()]
        replay_raw_dataset(
            model,
            encoder,
            feature_columns,
            args.raw_dir,
            args.replay_delay,
            args.execute_replay,
            args.threshold,
            args.replay_limit,
            args.replay_gesture,
            args.no_spotify,
            replay_sequence,
            args.samples_per_gesture,
            args.replay_start_index,
        )
        return

    sp = None if args.no_spotify else get_client()
    window: deque[list[float]] = deque(maxlen=SAMPLES_PER_WINDOW)
    last_trigger_time = 0.0
    rows_since_prediction = 0
    csv_rows_seen = 0
    last_status_time = time.monotonic()

    print(f"Opening {args.port} at {args.baud} baud")
    print("Put firmware in streaming mode by sending 'stream' if needed.")
    with serial.Serial(args.port, args.baud, timeout=2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        ser.write(b"stream\n")

        while True:
            raw = ser.readline()
            if not raw:
                now = time.monotonic()
                if now - last_status_time >= 5 and csv_rows_seen == 0:
                    print("Still waiting for CSV IMU rows. If this keeps happening, flash firmware or type 'stream' in monitor.")
                    last_status_time = now
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not CSV_RE.match(line):
                if args.debug_serial:
                    print(f"Serial: {line}")
                continue

            values = [float(part) for part in line.split(",")]
            window.append(values)
            csv_rows_seen += 1
            rows_since_prediction += 1
            if csv_rows_seen == 1:
                print("Receiving IMU CSV rows. First prediction appears after about 3 seconds of data.")
            if len(window) < SAMPLES_PER_WINDOW or rows_since_prediction < args.stride:
                continue
            rows_since_prediction = 0

            df = pd.DataFrame(list(window), columns=CSV_COLUMNS)
            score = motion_score(df)
            if score < args.motion_threshold:
                if args.debug_serial:
                    print(f"Waiting for gesture motion, score: {score:.2f}")
                continue

            gesture, conf = predict_dataframe(model, encoder, feature_columns, df)
            print(f"Predicted: {gesture}, confidence: {conf:.2f}, motion: {score:.2f}")

            now = time.monotonic()
            if conf >= args.threshold and now - last_trigger_time >= args.cooldown:
                if args.no_spotify:
                    print(f"Trigger skipped (--no-spotify): {gesture}")
                else:
                    execute_gesture(gesture, sp)
                last_trigger_time = now


if __name__ == "__main__":
    main()
