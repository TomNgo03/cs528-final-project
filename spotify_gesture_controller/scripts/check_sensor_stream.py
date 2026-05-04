#!/usr/bin/env python3
"""Check whether the ESP32 + MPU6050 stream is alive and changing."""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd
import serial

from config import BAUD_RATE, CSV_COLUMNS, SERIAL_PORT

CSV_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){6}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--print-rows", type=int, default=5)
    args = parser.parse_args()

    rows: list[list[float]] = []
    non_csv = 0
    deadline = time.monotonic() + args.seconds

    print(f"Reading {args.seconds:.1f}s from {args.port} at {args.baud} baud")
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        time.sleep(1.0)
        ser.reset_input_buffer()
        while time.monotonic() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if CSV_RE.match(line):
                values = [float(part) for part in line.split(",")]
                rows.append(values)
                if len(rows) <= args.print_rows:
                    print("CSV:", line)
            else:
                non_csv += 1
                if non_csv <= args.print_rows:
                    print("Non-CSV:", line)

    if not rows:
        print("\nResult: no CSV rows received.")
        print("Likely causes: firmware not flashed/running, wrong port, or ESP32 reset loop.")
        return

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    print(f"\nReceived {len(df)} CSV rows and {non_csv} non-CSV lines.")
    print("\nPer-axis stats:")
    stats = df[CSV_COLUMNS[1:]].agg(["mean", "std", "min", "max"]).T
    print(stats.to_string(float_format=lambda x: f"{x: .5f}"))

    accel_dynamic = df[["AccelX(g)", "AccelY(g)", "AccelZ(g)"]].to_numpy()
    gyro = df[["GyroX(dps)", "GyroY(dps)", "GyroZ(dps)"]].to_numpy()
    accel_range = float(np.max(np.ptp(accel_dynamic, axis=0)))
    gyro_range = float(np.max(np.ptp(gyro, axis=0)))

    print("\nHealth check:")
    print(f"- Max accel axis range: {accel_range:.5f} g")
    print(f"- Max gyro axis range: {gyro_range:.5f} dps")

    if accel_range < 0.005 and gyro_range < 0.5:
        print("Result: stream is almost flat. Move the sensor while running this test.")
        print("If it stays flat while moving, check MPU6050 wiring: VCC, GND, SDA GPIO0, SCL GPIO1.")
    else:
        print("Result: sensor stream is changing. Hardware is probably alive.")


if __name__ == "__main__":
    main()
