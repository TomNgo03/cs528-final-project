#!/usr/bin/env python3
"""Spotify Web API commands used by real-time gesture prediction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth

from config import SPOTIFY_COMMANDS

SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "user-library-modify"
)


def load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_client() -> spotipy.Spotify:
    load_dotenv()
    return spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES))


def print_devices(sp: spotipy.Spotify | None = None) -> None:
    sp = sp or get_client()
    devices = sp.devices().get("devices", [])
    if not devices:
        print("Spotify: no available devices. Open Spotify on your laptop/phone and start a song once.")
        return

    print("Spotify devices:")
    for device in devices:
        active = "active" if device.get("is_active") else "inactive"
        volume = device.get("volume_percent")
        print(f"- {device.get('name')} ({device.get('type')}, {active}, volume={volume}%)")


def handle_spotify_error(exc: SpotifyException) -> None:
    message = str(exc)
    if "NO_ACTIVE_DEVICE" in message or "No active device" in message:
        print("Spotify: no active device found.")
        print("Open Spotify on your laptop or phone, start playing any song once, then rerun the command.")
        try:
            print_devices()
        except Exception:
            pass
        return
    raise exc


def play_pause(sp: spotipy.Spotify | None = None) -> None:
    sp = sp or get_client()
    playback = sp.current_playback()
    try:
        if playback and playback.get("is_playing"):
            sp.pause_playback()
            print("Spotify: pause")
        else:
            sp.start_playback()
            print("Spotify: play")
    except SpotifyException as exc:
        handle_spotify_error(exc)


def next_track(sp: spotipy.Spotify | None = None) -> None:
    try:
        (sp or get_client()).next_track()
        print("Spotify: next track")
    except SpotifyException as exc:
        handle_spotify_error(exc)


def previous_track(sp: spotipy.Spotify | None = None) -> None:
    try:
        (sp or get_client()).previous_track()
        print("Spotify: previous track")
    except SpotifyException as exc:
        handle_spotify_error(exc)


def _current_volume(sp: spotipy.Spotify) -> int:
    playback = sp.current_playback()
    if playback and playback.get("device"):
        volume = playback["device"].get("volume_percent")
        if volume is not None:
            return int(volume)
    return 50


def volume_up(sp: spotipy.Spotify | None = None, step: int = 10) -> None:
    sp = sp or get_client()
    volume = min(100, _current_volume(sp) + step)
    try:
        sp.volume(volume)
        print(f"Spotify: volume {volume}%")
    except SpotifyException as exc:
        handle_spotify_error(exc)


def volume_down(sp: spotipy.Spotify | None = None, step: int = 10) -> None:
    sp = sp or get_client()
    volume = max(0, _current_volume(sp) - step)
    try:
        sp.volume(volume)
        print(f"Spotify: volume {volume}%")
    except SpotifyException as exc:
        handle_spotify_error(exc)


def add_to_liked_songs(sp: spotipy.Spotify | None = None) -> None:
    sp = sp or get_client()
    playback = sp.current_playback()
    if not playback or not playback.get("item"):
        print("Spotify: no active track to like")
        return

    track = playback["item"]
    track_id = track.get("id")
    if not track_id:
        print("Spotify: current item cannot be added to Liked Songs")
        return

    sp.current_user_saved_tracks_add(tracks=[track_id])
    artists = ", ".join(artist["name"] for artist in track.get("artists", []))
    title = track.get("name", "current track")
    print(f"Spotify: added to Liked Songs: {title} - {artists}")


def execute_gesture(gesture: str, sp: spotipy.Spotify | None = None) -> None:
    command = SPOTIFY_COMMANDS.get(gesture)
    if not command:
        print(f"No Spotify command mapped for gesture: {gesture}")
        return
    globals()[command](sp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = sorted(set(SPOTIFY_COMMANDS.values()) | {"devices"})
    parser.add_argument("command", choices=commands)
    args = parser.parse_args()
    if args.command == "devices":
        print_devices(get_client())
    else:
        globals()[args.command](get_client())


if __name__ == "__main__":
    main()
