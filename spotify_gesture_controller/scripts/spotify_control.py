#!/usr/bin/env python3
"""Spotify Web API commands used by real-time gesture prediction."""

from __future__ import annotations

import argparse

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from config import SPOTIFY_COMMANDS

SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "user-library-modify"
)


def get_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES))


def play_pause(sp: spotipy.Spotify | None = None) -> None:
    sp = sp or get_client()
    playback = sp.current_playback()
    if playback and playback.get("is_playing"):
        sp.pause_playback()
        print("Spotify: pause")
    else:
        sp.start_playback()
        print("Spotify: play")


def next_track(sp: spotipy.Spotify | None = None) -> None:
    (sp or get_client()).next_track()
    print("Spotify: next track")


def previous_track(sp: spotipy.Spotify | None = None) -> None:
    (sp or get_client()).previous_track()
    print("Spotify: previous track")


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
    sp.volume(volume)
    print(f"Spotify: volume {volume}%")


def volume_down(sp: spotipy.Spotify | None = None, step: int = 10) -> None:
    sp = sp or get_client()
    volume = max(0, _current_volume(sp) - step)
    sp.volume(volume)
    print(f"Spotify: volume {volume}%")


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
    parser.add_argument("command", choices=sorted(set(SPOTIFY_COMMANDS.values())))
    args = parser.parse_args()
    globals()[args.command](get_client())


if __name__ == "__main__":
    main()
