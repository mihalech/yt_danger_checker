from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yt_dlp


@dataclass
class VideoFormatResult:
    width: Optional[int]
    height: Optional[int]
    is_vertical: bool


def probe_video_format(video_url: str) -> VideoFormatResult:
    """
    Получает разрешение видео и определяет вертикальное ли оно.
    """

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        formats = info.get("formats", [])

        best_format = max(
            (
                fmt
                for fmt in formats
                if fmt.get("width") is not None and fmt.get("height") is not None
            ),
            key=lambda fmt: fmt["width"] * fmt["height"],
            default=None,
        )

        if best_format is not None:
            width = best_format["width"]
            height = best_format["height"]
        else:
            width = info.get("width")
            height = info.get("height")

        is_vertical = (
            width is not None
            and height is not None
            and height > width
        )

        return VideoFormatResult(
            width=width,
            height=height,
            is_vertical=is_vertical,
        )

    except Exception:
        return VideoFormatResult(
            width=None,
            height=None,
            is_vertical=False,
        )