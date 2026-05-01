# Invisible Spotify Controller Using Finger Gestures

Glove-based invisible controller using an ESP32S3 and MPU6050 IMU to recognize finger/hand gestures and map them to Spotify playback commands.

The project extends the HW1 ESP32S3 + MPU6050 I2C firmware and the HW2 Python serial capture plus SVM feature pipeline into a complete demo system.

## Gestures

| Gesture | Spotify command |
| --- | --- |
| `tap_index` | Play / pause |
| `swipe_right` | Next track |
| `swipe_left` | Previous track |
| `swipe_up` | Volume up |
| `swipe_down` | Volume down |
| `double_tap` | Add current track to Liked Songs |

## Hardware

- ESP32S3
- MPU6050 IMU
- Glove-mounted sensor setup
- USB cable for serial data collection and live prediction

Start with one MPU6050 mounted consistently on the hand or finger area. The firmware uses a sensor table so later sensors can be added with additional addresses or an I2C mux.

## Wiring

| ESP32S3 | MPU6050 |
| --- | --- |
| GPIO 0 | SDA |
| GPIO 1 | SCL |
| 3V3 | VCC |
| GND | GND |

Firmware settings inherited from HW1:

- I2C address: `0x68`
- I2C frequency: `100 kHz`
- Sample rate: `100 Hz`
- Window length: `3 seconds`
- Accelerometer range: `+/-4g`
- Gyroscope range: `+/-500 dps`

## Setup

```bash
cd spotify_gesture_controller
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `scripts/config.py` if your serial port differs from the default. On macOS it is usually `/dev/cu.usbmodem*`; on Windows it may be `COM18`; on Linux it may be `/dev/ttyACM0`.

## Flash Firmware

```bash
cd spotify_gesture_controller/firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbmodem1101 flash monitor
```

Serial commands supported by the firmware:

```text
label tap_index
collect
auto
stream
help
```

`collect` records one 3 second labeled window and prints:

```text
========== START_GESTURE_tap_index ==========
Time(ms),AccelX(g),AccelY(g),AccelZ(g),GyroX(dps),GyroY(dps),GyroZ(dps)
...
========== END_GESTURE_tap_index ==========
```

`stream` continuously prints CSV rows for real-time prediction.
`auto` continuously records 3 second windows with the current label; reset the board to stop it.

## Collect Gesture Data

Collect 100 samples per gesture:

```bash
cd spotify_gesture_controller
python scripts/capture_gestures.py --gesture tap_index --count 100 --port /dev/cu.usbmodem1101
python scripts/capture_gestures.py --gesture swipe_right --count 100 --port /dev/cu.usbmodem1101
python scripts/capture_gestures.py --gesture swipe_left --count 100 --port /dev/cu.usbmodem1101
python scripts/capture_gestures.py --gesture swipe_up --count 100 --port /dev/cu.usbmodem1101
python scripts/capture_gestures.py --gesture swipe_down --count 100 --port /dev/cu.usbmodem1101
python scripts/capture_gestures.py --gesture double_tap --count 100 --port /dev/cu.usbmodem1101
```

Files are saved as:

```text
data/raw/{gesture_name}/{gesture_name}_{index:03d}.csv
```

Each CSV contains:

```text
Time(ms),AccelX(g),AccelY(g),AccelZ(g),GyroX(dps),GyroY(dps),GyroZ(dps)
```

## Preprocess

```bash
python scripts/preprocess_data.py
```

Outputs:

- `data/processed/features.csv`
- `data/processed/labels.csv`

Features are extracted from all six IMU axes plus acceleration magnitude and gyro magnitude. Each signal gets mean, standard deviation, min, max, peak absolute value, RMS, peak-to-peak range, energy, zero crossing rate, dominant FFT frequency, spectral centroid, and 0-5 Hz low-band frequency ratio.

## Train Model

```bash
python scripts/train_model.py
```

Training uses:

- SVM with `StandardScaler` and `GridSearchCV`
- `RandomForestClassifier`
- KNN baseline

Outputs:

- `models/gesture_model.pkl`
- `models/label_encoder.pkl`
- `models/model_metadata.json`
- `results/confusion_matrix.png`
- `results/classification_report.txt`
- `results/training_summary.json`

## Spotify Credentials

Create a Spotify Developer app and set the redirect URI, for example `http://127.0.0.1:8080/callback`.

```bash
export SPOTIPY_CLIENT_ID="your_client_id"
export SPOTIPY_CLIENT_SECRET="your_client_secret"
export SPOTIPY_REDIRECT_URI="http://127.0.0.1:8080/callback"
```

The first run opens an authorization flow. Spotify playback commands require an active device, so start Spotify on your phone or laptop before the demo. Double tap uses Spotify's library permission to add the current track to Liked Songs.

## Real-Time Prediction

```bash
python scripts/realtime_predict.py --port /dev/cu.usbmodem1101 --threshold 0.70 --cooldown 2.0
```

For a dry run without controlling Spotify:

```bash
python scripts/realtime_predict.py --port /dev/cu.usbmodem1101 --no-spotify
```

The script keeps a sliding 300 sample window, predicts every 50 rows by default, applies a confidence threshold, and uses a cooldown period to avoid repeated triggers.

## Folder Structure

```text
spotify_gesture_controller/
├── firmware/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── models/
├── results/
├── requirements.txt
└── README.md
```

## Troubleshooting

- No serial connection: check the port in `scripts/config.py` or pass `--port`.
- Capture times out: make sure the firmware is flashed and the board accepts serial commands.
- Too few rows: avoid unplugging the board mid-window and keep the gesture window still before/after the finger motion.
- Low accuracy: collect more consistent samples, mount the sensor in the same orientation every time, and inspect the confusion matrix.
- Spotify does nothing: open Spotify on an active device, verify environment variables, and confirm your app redirect URI matches exactly.
