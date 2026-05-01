# Firmware

ESP-IDF firmware for the ESP32S3 + MPU6050 glove controller.

## Build and Flash

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbmodem1101 flash monitor
```

## Serial Commands

- `label <gesture>` sets the label for collection mode.
- `collect` records one 3 second window.
- `auto` repeatedly records 3 second windows with the current label; reset the board to stop.
- `stream` continuously prints CSV rows for live prediction.
- `help` prints command help.

The data format is:

```text
Time(ms),AccelX(g),AccelY(g),AccelZ(g),GyroX(dps),GyroY(dps),GyroZ(dps)
```
