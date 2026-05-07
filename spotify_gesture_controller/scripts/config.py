from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERIAL_PORT = "/dev/cu.usbmodem1101"
BAUD_RATE = 115200
SAMPLE_RATE = 100
WINDOW_SECONDS = 3
SAMPLES_PER_WINDOW = SAMPLE_RATE * WINDOW_SECONDS
MIN_VALID_ROWS = int(SAMPLES_PER_WINDOW * 0.85)
MAX_VALID_ROWS = int(SAMPLES_PER_WINDOW * 1.20)

GESTURES = [
    "swipe_right",
    "swipe_left",
    "swipe_up",
    "swipe_down",
]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CSV_COLUMNS = [
    "Time(ms)",
    "AccelX(g)",
    "AccelY(g)",
    "AccelZ(g)",
    "GyroX(dps)",
    "GyroY(dps)",
    "GyroZ(dps)",
]

IMU_AXES = CSV_COLUMNS[1:]

SPOTIFY_COMMANDS = {
    "swipe_right": "next_track",
    "swipe_left": "previous_track",
    "swipe_up": "volume_up",
    "swipe_down": "volume_down",
}
