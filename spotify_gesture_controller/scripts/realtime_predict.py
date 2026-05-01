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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=50, help="Rows between predictions")
    parser.add_argument("--no-spotify", action="store_true", help="Print predictions without API calls")
    args = parser.parse_args()

    model = joblib.load(args.model_dir / "gesture_model.pkl")
    encoder = joblib.load(args.model_dir / "label_encoder.pkl")
    metadata_path = args.model_dir / "model_metadata.json"
    feature_columns = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = list(metadata.get("feature_columns", [])) or None

    sp = None if args.no_spotify else get_client()
    window: deque[list[float]] = deque(maxlen=SAMPLES_PER_WINDOW)
    last_trigger_time = 0.0
    rows_since_prediction = 0

    print(f"Opening {args.port} at {args.baud} baud")
    print("Put firmware in streaming mode by sending 'stream' if needed.")
    with serial.Serial(args.port, args.baud, timeout=2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        ser.write(b"stream\n")

        while True:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not CSV_RE.match(line):
                continue

            values = [float(part) for part in line.split(",")]
            window.append(values)
            rows_since_prediction += 1
            if len(window) < SAMPLES_PER_WINDOW or rows_since_prediction < args.stride:
                continue
            rows_since_prediction = 0

            df = pd.DataFrame(list(window), columns=CSV_COLUMNS)
            features = extract_features(df, sample_rate=SAMPLE_RATE)
            row = pd.DataFrame([features])
            if feature_columns is not None:
                row = row.reindex(columns=feature_columns, fill_value=0.0)

            pred_idx, conf = prediction_confidence(model, row)
            gesture = encoder.inverse_transform([pred_idx])[0]
            print(f"Predicted: {gesture}, confidence: {conf:.2f}")

            now = time.monotonic()
            if conf >= args.threshold and now - last_trigger_time >= args.cooldown:
                if args.no_spotify:
                    print(f"Trigger skipped (--no-spotify): {gesture}")
                else:
                    execute_gesture(gesture, sp)
                last_trigger_time = now


if __name__ == "__main__":
    main()
