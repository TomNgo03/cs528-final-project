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
    metadata = {}
    feature_columns = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = list(metadata.get("feature_columns", [])) or None
    return model, encoder, feature_columns, metadata


def predict_dataframe(model, encoder, feature_columns: list[str] | None, df: pd.DataFrame) -> tuple[str, float]:
    features = extract_features(df, sample_rate=SAMPLE_RATE)
    row = pd.DataFrame([features])
    if feature_columns is not None:
        row = row.reindex(columns=feature_columns, fill_value=0.0)
    pred_idx, conf = prediction_confidence(model, row)
    return encoder.inverse_transform([pred_idx])[0], conf


def resample_for_model(df: pd.DataFrame, target_rows: int = SAMPLES_PER_WINDOW) -> pd.DataFrame:
    if len(df) == target_rows:
        out = df.copy()
        out["Time(ms)"] = np.arange(target_rows) * int(1000 / SAMPLE_RATE)
        return out

    source_x = np.linspace(0.0, 1.0, len(df))
    target_x = np.linspace(0.0, 1.0, target_rows)
    out = pd.DataFrame({"Time(ms)": np.arange(target_rows) * int(1000 / SAMPLE_RATE)})
    for col in CSV_COLUMNS[1:]:
        out[col] = np.interp(target_x, source_x, df[col].to_numpy(dtype=float))
    return out


def motion_score(df: pd.DataFrame) -> float:
    accel = df[["AccelX(g)", "AccelY(g)", "AccelZ(g)"]].to_numpy(dtype=float)
    gyro = df[["GyroX(dps)", "GyroY(dps)", "GyroZ(dps)"]].to_numpy(dtype=float)
    accel_dynamic = accel - accel.mean(axis=0)
    gyro_dynamic = gyro - gyro.mean(axis=0)
    accel_peak = float(np.max(np.linalg.norm(accel_dynamic, axis=1)))
    gyro_peak = float(np.max(np.linalg.norm(gyro_dynamic, axis=1)))
    return accel_peak + gyro_peak / 350.0


def heuristic_predict(df: pd.DataFrame) -> tuple[str, float]:
    accel = df[["AccelX(g)", "AccelY(g)", "AccelZ(g)"]].to_numpy(dtype=float)
    gyro = df[["GyroX(dps)", "GyroY(dps)", "GyroZ(dps)"]].to_numpy(dtype=float)
    accel_dynamic = accel - accel.mean(axis=0)
    gyro_dynamic = gyro - gyro.mean(axis=0)

    gyro_ranges = np.ptp(gyro_dynamic, axis=0)
    accel_ranges = np.ptp(accel_dynamic, axis=0)
    horizontal_strength = float(gyro_ranges[2] + 45.0 * accel_ranges[0])
    vertical_strength = float(gyro_ranges[0] + 45.0 * accel_ranges[1])
    if horizontal_strength >= vertical_strength:
        x = accel_dynamic[:, 0]
        half = max(1, len(x) // 2)
        direction = float(np.mean(x[:half]) - np.mean(x[half:]))
        return ("swipe_right" if direction > 0 else "swipe_left"), min(0.98, 0.70 + abs(direction) * 2.0)

    y = accel_dynamic[:, 1]
    half = max(1, len(y) // 2)
    direction = float(np.mean(y[:half]) - np.mean(y[half:]))
    return ("swipe_up" if direction > 0 else "swipe_down"), min(0.98, 0.70 + abs(direction) * 2.0)


def horizontal_direction_override(
    df: pd.DataFrame,
    gesture: str,
    confidence: float,
    min_delta: float = 60.0,
    min_range: float = 180.0,
) -> tuple[str, float, bool]:
    """Use real-data GyroZ sign to stabilize left/right during live demos."""
    gyro = df[["GyroX(dps)", "GyroY(dps)", "GyroZ(dps)"]].to_numpy(dtype=float)
    gyro_dynamic = gyro - gyro.mean(axis=0)
    gyro_ranges = np.ptp(gyro_dynamic, axis=0)
    gyro_z = gyro_dynamic[:, 2]
    half = max(1, len(gyro_z) // 2)
    gyro_z_delta = float(np.mean(gyro_z[:half]) - np.mean(gyro_z[half:]))

    is_horizontal = gyro_ranges[2] >= max(gyro_ranges[0], gyro_ranges[1]) * 0.85
    if not is_horizontal or abs(gyro_z_delta) < min_delta or gyro_ranges[2] < min_range:
        return gesture, confidence, False

    corrected = "swipe_right" if gyro_z_delta < 0 else "swipe_left"
    corrected_confidence = max(confidence, min(0.95, 0.72 + abs(gyro_z_delta) / 450.0))
    return corrected, corrected_confidence, corrected != gesture


def vertical_direction_override(
    df: pd.DataFrame,
    gesture: str,
    confidence: float,
    min_delta: float = 55.0,
    min_range: float = 250.0,
) -> tuple[str, float, bool]:
    """Use real-data GyroY sign to stabilize up/down during live demos."""
    gyro = df[["GyroX(dps)", "GyroY(dps)", "GyroZ(dps)"]].to_numpy(dtype=float)
    gyro_dynamic = gyro - gyro.mean(axis=0)
    gyro_ranges = np.ptp(gyro_dynamic, axis=0)
    gyro_y = gyro_dynamic[:, 1]
    half = max(1, len(gyro_y) // 2)
    gyro_y_delta = float(np.mean(gyro_y[:half]) - np.mean(gyro_y[half:]))

    is_vertical = gyro_ranges[1] >= max(gyro_ranges[0], gyro_ranges[2]) * 0.85
    if not is_vertical or abs(gyro_y_delta) < min_delta or gyro_ranges[1] < min_range:
        return gesture, confidence, False

    corrected = "swipe_up" if gyro_y_delta < 0 else "swipe_down"
    corrected_confidence = max(confidence, min(0.95, 0.72 + abs(gyro_y_delta) / 450.0))
    return corrected, corrected_confidence, corrected != gesture


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
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--cooldown", type=float)
    parser.add_argument("--stride", type=int, help="Rows between predictions")
    parser.add_argument("--motion-threshold", type=float)
    parser.add_argument("--reset-motion-threshold", type=float)
    parser.add_argument("--live-window-samples", type=int)
    parser.add_argument(
        "--demo-preset",
        choices=["instant", "fast", "balanced", "stable"],
        default="balanced",
        help="Convenience tuning for live sensor demo.",
    )
    parser.add_argument("--live-classifier", choices=["heuristic", "model"], default="model")
    parser.add_argument(
        "--no-direction-override",
        action="store_true",
        help="Disable Gyro sign correction for live directional swipes.",
    )
    parser.add_argument(
        "--swap-left-right",
        action="store_true",
        help="Swap swipe_left and swipe_right after prediction for sensor orientation differences.",
    )
    parser.add_argument(
        "--swap-up-down",
        action="store_true",
        help="Swap swipe_up and swipe_down after prediction for sensor orientation differences.",
    )
    parser.add_argument(
        "--trigger-mode",
        choices=["continuous", "one-shot"],
        default="one-shot",
        help="continuous triggers as soon as cooldown passes; one-shot waits for stillness before rearming.",
    )
    parser.add_argument(
        "--stable-votes",
        type=int,
        default=2,
        help="Require this many recent matching predictions before triggering.",
    )
    parser.add_argument(
        "--stable-window",
        type=int,
        default=3,
        help="Number of recent predictions used for vote confirmation.",
    )
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
        help="Comma-separated gesture sequence to replay, such as swipe_left,swipe_right,swipe_up,swipe_down",
    )
    parser.add_argument("--samples-per-gesture", type=int, default=1)
    parser.add_argument("--replay-start-index", type=int, default=0)
    args = parser.parse_args()

    presets = {
        "instant": {
            "threshold": 0.55,
            "cooldown": 0.45,
            "stride": 3,
            "motion_threshold": 0.08,
            "reset_motion_threshold": 0.03,
            "live_window_samples": 30,
        },
        "fast": {
            "threshold": 0.78,
            "cooldown": 2.0,
            "stride": 15,
            "motion_threshold": 0.35,
            "reset_motion_threshold": 0.12,
            "live_window_samples": 90,
        },
        "balanced": {
            "threshold": 0.82,
            "cooldown": 2.5,
            "stride": 20,
            "motion_threshold": 0.45,
            "reset_motion_threshold": 0.16,
            "live_window_samples": 120,
        },
        "stable": {
            "threshold": 0.88,
            "cooldown": 3.5,
            "stride": 30,
            "motion_threshold": 0.65,
            "reset_motion_threshold": 0.22,
            "live_window_samples": 160,
        },
    }
    preset = presets[args.demo_preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    model, encoder, feature_columns, metadata = load_model(args.model_dir)
    model_window_samples = int(metadata.get("training_window_samples") or SAMPLES_PER_WINDOW)

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
    window: deque[list[float]] = deque(maxlen=args.live_window_samples)
    last_trigger_time = 0.0
    rows_since_prediction = 0
    csv_rows_seen = 0
    last_status_time = time.monotonic()
    gesture_armed = True
    recent_predictions: deque[tuple[str, float]] = deque(maxlen=max(1, args.stable_window))

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
                seconds = args.live_window_samples / SAMPLE_RATE
                print(f"Receiving IMU CSV rows. First prediction appears after about {seconds:.1f} seconds of data.")
            if len(window) < args.live_window_samples or rows_since_prediction < args.stride:
                continue
            rows_since_prediction = 0

            df = pd.DataFrame(list(window), columns=CSV_COLUMNS)
            score = motion_score(df)
            if score < args.motion_threshold:
                if args.trigger_mode == "one-shot" and score < args.reset_motion_threshold and not gesture_armed:
                    gesture_armed = True
                    recent_predictions.clear()
                    if args.debug_serial:
                        print(f"Ready for next gesture, motion settled: {score:.2f}")
                if args.debug_serial:
                    print(f"Waiting for gesture motion, score: {score:.2f}")
                continue
            if args.trigger_mode == "one-shot" and not gesture_armed:
                if args.debug_serial:
                    print(f"Gesture already triggered; wait for stillness, score: {score:.2f}")
                continue

            if args.live_classifier == "heuristic":
                gesture, conf = heuristic_predict(df)
            else:
                model_df = resample_for_model(df, target_rows=model_window_samples)
                gesture, conf = predict_dataframe(model, encoder, feature_columns, model_df)
            if not args.no_direction_override:
                original_gesture = gesture
                gesture, conf, changed = horizontal_direction_override(df, gesture, conf)
                if changed and args.debug_serial:
                    print(f"Direction override: {original_gesture} -> {gesture}")
                original_gesture = gesture
                gesture, conf, changed = vertical_direction_override(df, gesture, conf)
                if changed and args.debug_serial:
                    print(f"Direction override: {original_gesture} -> {gesture}")
            if args.swap_left_right and gesture in {"swipe_left", "swipe_right"}:
                gesture = "swipe_right" if gesture == "swipe_left" else "swipe_left"
            if args.swap_up_down and gesture in {"swipe_up", "swipe_down"}:
                gesture = "swipe_down" if gesture == "swipe_up" else "swipe_up"
            recent_predictions.append((gesture, conf))

            matching_confidences = [
                recent_conf for recent_gesture, recent_conf in recent_predictions if recent_gesture == gesture
            ]
            stable_count = len(matching_confidences)
            stable_conf = float(np.mean(matching_confidences)) if matching_confidences else conf
            print(
                f"Predicted: {gesture}, confidence: {conf:.2f}, "
                f"stable: {stable_count}/{args.stable_votes}, motion: {score:.2f}"
            )

            now = time.monotonic()
            if (
                stable_count >= args.stable_votes
                and stable_conf >= args.threshold
                and now - last_trigger_time >= args.cooldown
            ):
                if args.no_spotify:
                    print(f"Trigger skipped (--no-spotify): {gesture}")
                else:
                    execute_gesture(gesture, sp)
                last_trigger_time = now
                recent_predictions.clear()
                if args.trigger_mode == "one-shot":
                    gesture_armed = False


if __name__ == "__main__":
    main()
