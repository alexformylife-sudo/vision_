"""Tag 与颜色独立防抖；两路都稳定且有效才 ALIGN=1。"""
from __future__ import annotations

import runpy
import time
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")), run_name="termux_vision_bootstrap")

from termux_vision.config import ALLOWED_COLORS, ALLOWED_SIDES, Config
from termux_vision.models import ColorObservation, TagObservation, VisionResult


class _ChannelFilter:
    def __init__(self, min_frames: int, min_seconds: float) -> None:
        self._min_frames = min_frames
        self._min_seconds = min_seconds
        self._candidate = None
        self._count = 0
        self._started_at = 0.0

    def update(self, value):
        now = time.monotonic()
        if value != self._candidate:
            self._candidate = value
            self._count = 1
            self._started_at = now
        else:
            self._count += 1
        by_frames = self._count >= self._min_frames
        elapsed = now - self._started_at
        by_time = self._min_seconds > 0.0 and elapsed >= self._min_seconds
        return by_frames or by_time, self._candidate

    def reset(self) -> None:
        self._candidate = None
        self._count = 0
        self._started_at = 0.0


class TemporalFilter:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._tag_filter = _ChannelFilter(config.stable_frame_count, config.stable_seconds)
        self._color_filter = _ChannelFilter(config.stable_frame_count, config.stable_seconds)

    def reset(self) -> None:
        self._tag_filter.reset()
        self._color_filter.reset()

    def update(self, tag: TagObservation, color: ColorObservation) -> VisionResult:
        tag_stable, tag_value = self._tag_filter.update((tag.side, tag.tag_id, tag.known))
        color_stable, color_value = self._color_filter.update(color.color)
        side, tag_id, known = tag_value if tag_stable else (
            self._config.invalid_side,
            self._config.invalid_tag_id,
            False,
        )
        out_color = color_value if color_stable else self._config.invalid_color
        tag_valid = bool(known and side in ALLOWED_SIDES and tag_id >= 0)
        color_valid = out_color in ALLOWED_COLORS
        align = 1 if (tag_stable and color_stable and tag_valid and color_valid) else 0
        if not tag_stable or not known:
            side, tag_id = self._config.invalid_side, self._config.invalid_tag_id
        if not color_stable or out_color not in ALLOWED_COLORS:
            out_color = self._config.invalid_color
        return VisionResult(side=side, tag_id=tag_id, color=out_color, align=align)
