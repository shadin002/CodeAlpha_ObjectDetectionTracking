from __future__ import annotations

from pathlib import Path


def parse_source(source: str):
    stripped = str(source).strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def validate_file_source(source) -> None:
    if isinstance(source, int):
        return

    text = str(source)
    if text.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return

    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(
            f"Video source was not found: {path}\n"
            "Use --source 0 for your webcam, or provide a valid video file path."
        )


def build_label(track_id, class_name: str, confidence: float) -> str:
    id_text = f"ID {track_id}" if track_id is not None else "ID ?"
    return f"{id_text} | {class_name} | {confidence:.2f}"
