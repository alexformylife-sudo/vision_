"""检测与输出所用的不可变数据结构。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameRead:
    frame: np.ndarray | None
    ended: bool
    seq: int = -1  # 新帧序号；frame 为 None 时无意义


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    side: str
    known: bool
    corners: np.ndarray | None
    center: tuple[float, float] | None
    side_px: float


@dataclass(frozen=True)
class ColorObservation:
    color: str
    contour_area: float
    bounding_box: tuple[int, int, int, int] | None
    center: tuple[float, float] | None


@dataclass(frozen=True)
class VisionResult:
    side: str
    tag_id: int
    color: str
    align: int

    def to_message(self) -> str:
        return f"SIDE={self.side};TAG={self.tag_id};COLOR={self.color};ALIGN={self.align}\n"


def invalid_tag(invalid_side: str, invalid_tag_id: int) -> TagObservation:
    return TagObservation(
        tag_id=invalid_tag_id,
        side=invalid_side,
        known=False,
        corners=None,
        center=None,
        side_px=0.0,
    )


def invalid_color(invalid_code: str) -> ColorObservation:
    return ColorObservation(
        color=invalid_code,
        contour_area=0.0,
        bounding_box=None,
        center=None,
    )
