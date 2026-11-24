"""AprilTag 与红黄蓝颜色检测，互相独立，结果不互相伪装。"""
from __future__ import annotations

import logging
import runpy
from dataclasses import dataclass
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")), run_name="termux_vision_bootstrap")

import cv2
import numpy as np

from termux_vision.config import ALLOWED_COLORS, Config, CroppedRoi, HsvRange
from termux_vision.models import (
    ColorObservation,
    TagObservation,
    invalid_color,
    invalid_tag,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarkerPick:
    tag_id: int
    corners: np.ndarray
    center: tuple[float, float]
    side_px: float
    area: float


@dataclass(frozen=True)
class ColorSearch:
    hsv: np.ndarray
    origin_x: int
    origin_y: int
    scale: float


@dataclass(frozen=True)
class RawMarkers:
    corners_list: object
    ids: object


class TagDetector:
    def __init__(self, config: Config) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.minMarkerPerimeterRate = config.min_marker_perimeter_rate
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._config = config

    def detect(self, frame: np.ndarray) -> TagObservation:
        cropped = self._config.tag_roi.crop(frame)
        if cropped.image.size == 0:
            return invalid_tag(self._config.invalid_side, self._config.invalid_tag_id)
        gray = _as_gray(cropped.image)
        try:
            corners_list, ids, _rejected = self._detector.detectMarkers(gray)
        except cv2.error as exc:
            LOGGER.warning("AprilTag 检测失败: %s", exc)
            return invalid_tag(self._config.invalid_side, self._config.invalid_tag_id)
        raw = RawMarkers(corners_list=corners_list, ids=ids)
        picked = _select_marker(raw, cropped, self._config, frame.shape)
        return _tag_from_pick(picked, self._config)


class ColorDetector:
    def __init__(self, config: Config) -> None:
        self._config = config

    def detect(self, frame: np.ndarray) -> ColorObservation:
        cropped = self._config.color_roi.crop(frame)
        if cropped.image.size == 0:
            return invalid_color(self._config.invalid_color)
        working, scale = _scaled_roi(cropped.image, self._config.color_roi_scale)
        try:
            hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        except cv2.error as exc:
            LOGGER.warning("颜色空间转换失败: %s", exc)
            return invalid_color(self._config.invalid_color)
        search = ColorSearch(hsv=hsv, origin_x=cropped.x0, origin_y=cropped.y0, scale=scale)
        return _pick_dominant_color(self._config, search)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _scaled_roi(image: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    if scale == 1.0:
        return image, 1.0
    height, width = image.shape[:2]
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_w, new_h)), scale


def _select_marker(raw: RawMarkers, cropped: CroppedRoi, config: Config, frame_shape) -> MarkerPick | None:
    if raw.ids is None or len(raw.ids) == 0:
        return None
    picks = _collect_markers(raw.corners_list, raw.ids, cropped, config.min_tag_side_px)
    if not picks:
        return None
    if len(picks) == 1:
        return picks[0]
    return _choose_marker(picks, config.multi_tag_policy, frame_shape)


def _collect_markers(corners_list, ids, cropped: CroppedRoi, min_side_px: float) -> list[MarkerPick]:
    offset = np.array([cropped.x0, cropped.y0], dtype=np.float32)
    flat_ids = np.asarray(ids).flatten()
    picks: list[MarkerPick] = []
    for index, marker_corners in enumerate(corners_list):
        corners = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2) + offset
        pick = _measure_marker(int(flat_ids[index]), corners)
        if pick.side_px < min_side_px:
            LOGGER.info("忽略过小 Tag 噪点: id=%s side_px=%.1f", pick.tag_id, pick.side_px)
            continue
        picks.append(pick)
    return picks


def _measure_marker(tag_id: int, corners: np.ndarray) -> MarkerPick:
    side_lengths = [
        float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]))
        for i in range(4)
    ]
    center = (float(corners[:, 0].mean()), float(corners[:, 1].mean()))
    return MarkerPick(
        tag_id=tag_id,
        corners=corners,
        center=center,
        side_px=float(sum(side_lengths) / 4.0),
        area=float(cv2.contourArea(corners)),
    )


def _choose_marker(picks: list[MarkerPick], policy: str, frame_shape) -> MarkerPick:
    if policy == "nearest_center":
        height, width = frame_shape[:2]
        mid = (width / 2.0, height / 2.0)
        chosen = min(picks, key=lambda pick: (pick.center[0] - mid[0]) ** 2 + (pick.center[1] - mid[1]) ** 2)
    else:
        chosen = max(picks, key=lambda pick: pick.area)
    LOGGER.info("同框多 Tag，按 %s 选取 id=%s", policy, chosen.tag_id)
    return chosen


def _tag_from_pick(pick: MarkerPick | None, config: Config) -> TagObservation:
    if pick is None:
        return invalid_tag(config.invalid_side, config.invalid_tag_id)
    LOGGER.info(
        "检出 Tag id=%s 像素边长=%.1f 中心=(%.1f, %.1f) TAG_SIZE_M=%s",
        pick.tag_id,
        pick.side_px,
        pick.center[0],
        pick.center[1],
        config.TAG_SIZE_M,
    )
    side = config.tag_id_to_side.get(pick.tag_id)
    if side is None:
        LOGGER.info("未知 Tag ID=%s，输出 SIDE=N TAG=-1，不伪装有效", pick.tag_id)
        return invalid_tag(config.invalid_side, config.invalid_tag_id)
    return TagObservation(
        tag_id=pick.tag_id,
        side=side,
        known=True,
        corners=pick.corners,
        center=pick.center,
        side_px=pick.side_px,
    )


def _pick_dominant_color(config: Config, search: ColorSearch) -> ColorObservation:
    ranges = {
        "R": (config.red_hsv_low, config.red_hsv_high),
        "Y": (config.yellow_hsv,),
        "B": (config.blue_hsv,),
    }
    candidates: list[ColorObservation] = []
    for color in ALLOWED_COLORS:
        candidates.extend(_color_blobs(search, config, color, ranges[color]))
    valid = [obs for obs in candidates if _shape_ok(obs, config)]
    if not valid:
        return invalid_color(config.invalid_color)
    return max(valid, key=lambda obs: obs.contour_area)


def _color_blobs(search: ColorSearch, config: Config, color_name: str, ranges: tuple[HsvRange, ...]) -> list[ColorObservation]:
    mask = _cleaned_mask(search.hsv, ranges, config.color_morph_kernel)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [_blob_from_contour(contour, search, color_name) for contour in contours]


def _blob_from_contour(contour, search: ColorSearch, color_name: str) -> ColorObservation:
    area = float(cv2.contourArea(contour)) / (search.scale ** 2)
    box, center = _box_on_frame(cv2.boundingRect(contour), search)
    return ColorObservation(color=color_name, contour_area=area, bounding_box=box, center=center)


def _cleaned_mask(hsv: np.ndarray, ranges: tuple[HsvRange, ...], kernel_size: int) -> np.ndarray:
    merged = cv2.inRange(hsv, np.array(ranges[0].lower(), np.uint8), np.array(ranges[0].upper(), np.uint8))
    for hsv_range in ranges[1:]:
        extra = cv2.inRange(hsv, np.array(hsv_range.lower(), np.uint8), np.array(hsv_range.upper(), np.uint8))
        merged = cv2.bitwise_or(merged, extra)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)


def _box_on_frame(rect: tuple[int, int, int, int], search: ColorSearch) -> tuple[tuple[int, int, int, int], tuple[float, float]]:
    x, y, w, h = rect
    inv = 1.0 / search.scale
    left = int(round(x * inv + search.origin_x))
    top = int(round(y * inv + search.origin_y))
    width = int(round(w * inv))
    height = int(round(h * inv))
    center = (left + width / 2.0, top + height / 2.0)
    return (left, top, width, height), center


def _shape_ok(observation: ColorObservation, config: Config) -> bool:
    if observation.bounding_box is None:
        return False
    if observation.contour_area < config.color_min_area:
        return False
    _x, _y, width, height = observation.bounding_box
    if width <= 0 or height <= 0:
        return False
    if height / width < config.color_min_height_width:
        return False
    extent = observation.contour_area / float(width * height)
    return extent >= config.color_min_extent
