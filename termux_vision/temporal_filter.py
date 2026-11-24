"""Tag 与颜色各自防抖，并锁存首个稳定有效结果到进程结束。

ALIGN=1 表示本次任务两路都已确认，不是当前帧同时有效。
"""
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
        self._latched = None

    def update(self, value, valid: bool):
        """返回已锁存值。还没锁存时返回 None，无效值不会被锁上。"""
        if self._latched is not None:
            return self._latched
        if self._reached_stable(value) and valid:
            self._latched = value
        return self._latched

    def reset(self) -> None:
        """只清未确认连续计数。已锁存的值保留到进程结束。"""
        self._candidate = None
        self._count = 0
        self._started_at = 0.0

    def _reached_stable(self, value) -> bool:
        now = time.monotonic()
        if value != self._candidate:
            self._candidate = value
            self._count = 1
            self._started_at = now
        else:
            self._count += 1
        elapsed = now - self._started_at
        by_frames = self._count >= self._min_frames
        by_time = self._min_seconds > 0.0 and elapsed >= self._min_seconds
        return by_frames or by_time


class TemporalFilter:
    def __init__(self, config: Config) -> None:
        frames = config.stable_frame_count
        seconds = config.stable_seconds
        self._config = config
        self._tag_filter = _ChannelFilter(frames, seconds)
        self._color_filter = _ChannelFilter(frames, seconds)

    def reset(self) -> None:
        """断流时断开未确认连续计数，不清除已锁存结果。"""
        self._tag_filter.reset()
        self._color_filter.reset()

    def update(self, tag: TagObservation, color: ColorObservation) -> VisionResult:
        tag_key = (tag.side, tag.tag_id, tag.known)
        tag_latched = self._tag_filter.update(tag_key, _tag_valid(tag))
        color_latched = self._color_filter.update(color.color, color.color in ALLOWED_COLORS)
        return _vision_result(self._config, tag_latched, color_latched)


def _tag_valid(tag: TagObservation) -> bool:
    return bool(tag.known and tag.side in ALLOWED_SIDES and tag.tag_id >= 0)


def _vision_result(config: Config, tag_latched, color_latched) -> VisionResult:
    side = config.invalid_side
    tag_id = config.invalid_tag_id
    if tag_latched is not None:
        side, tag_id, _known = tag_latched
    if color_latched is None:
        out_color = config.invalid_color
    else:
        out_color = color_latched
    align = 1 if tag_latched is not None and color_latched is not None else 0
    return VisionResult(side=side, tag_id=tag_id, color=out_color, align=align)
