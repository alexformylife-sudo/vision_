"""无 GUI 调试：标注 ROI 后 imwrite，并输出颜色 ROI 中心区 HSV 统计。"""
from __future__ import annotations

import logging
import runpy
from dataclasses import dataclass
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")), run_name="termux_vision_bootstrap")

import cv2
import numpy as np

from termux_vision.config import ALLOWED_COLORS, Config, RatioRoi
from termux_vision.models import ColorObservation, TagObservation, VisionResult
from termux_vision.util import freeze_frame, resolve_debug_dir

LOGGER = logging.getLogger(__name__)

TAG_ROI_BGR = (255, 128, 0)
COLOR_ROI_BGR = (0, 200, 0)
CENTER_ROI_BGR = (0, 255, 255)
TAG_BOX_BGR = (0, 255, 255)
COLOR_BOX_BGR = (0, 0, 255)
DET_LABEL_BGR = {"R": (0, 0, 255), "Y": (0, 255, 255), "B": (255, 0, 0), "N": (180, 180, 180)}
TEXT_BGR = (255, 255, 255)
TEXT_ORIGIN = (8, 24)
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
BOX_THICKNESS = 2


@dataclass(frozen=True)
class DebugSnapshot:
    frame: np.ndarray
    tag: TagObservation
    color: ColorObservation
    result: VisionResult
    frame_index: int


def save_debug_frame(snapshot: DebugSnapshot, config: Config) -> bool:
    """返回 False 表示调试目录不可用，调用方应关闭存图。"""
    output_dir = _ensure_dir(config)
    if output_dir is None:
        return False
    vis = freeze_frame(annotate_frame(snapshot, config))
    path = output_dir / f"frame_{snapshot.frame_index:04d}.png"
    return _write_image(path, vis)


def save_hsv_calib_frame(frame: np.ndarray, config: Config, label: str) -> bool:
    """返回 False 表示调试目录不可用，调用方应关闭存图。"""
    output_dir = _ensure_dir(config)
    if output_dir is None:
        return False
    vis = _draw_calib_overlays(frame.copy(), config)
    path = output_dir / f"hsv_calib_{label}.png"
    return _write_image(path, vis)


def log_center_hsv_stats(frame: np.ndarray, config: Config, label: str) -> None:
    box = _center_box(config.color_roi, frame.shape, config.hsv_center_ratio)
    if box is None:
        LOGGER.info("HSV 调参 %s: 中心区域为空", label)
        return
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    LOGGER.info("HSV 调参 %s 中心区(%s,%s)-(%s,%s): %s", label, x0, y0, x1, y1, _hsv_summary(hsv, config))


def annotate_frame(snapshot: DebugSnapshot, config: Config) -> np.ndarray:
    vis = snapshot.frame.copy()
    _draw_roi(vis, config.tag_roi, TAG_ROI_BGR)
    _draw_roi(vis, config.color_roi, COLOR_ROI_BGR)
    if snapshot.tag.corners is not None:
        points = np.int32(np.round(snapshot.tag.corners)).reshape(-1, 1, 2)
        cv2.polylines(vis, [points], True, TAG_BOX_BGR, BOX_THICKNESS)
    _draw_color_detection(vis, snapshot.color)
    cv2.putText(
        vis,
        f"LAT {snapshot.result.to_message().strip()}",
        TEXT_ORIGIN,
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        TEXT_BGR,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    return vis


def _draw_calib_overlays(vis: np.ndarray, config: Config) -> np.ndarray:
    _draw_roi(vis, config.tag_roi, TAG_ROI_BGR)
    _draw_roi(vis, config.color_roi, COLOR_ROI_BGR)
    box = _center_box(config.color_roi, vis.shape, config.hsv_center_ratio)
    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(vis, (x0, y0), (x1, y1), CENTER_ROI_BGR, BOX_THICKNESS)
    return vis


def _center_box(roi: RatioRoi, frame_shape, ratio: float) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    x0, y0, x1, y1 = roi.pixel_box(width, height)
    roi_w = x1 - x0
    roi_h = y1 - y0
    if roi_w <= 0 or roi_h <= 0:
        return None
    pad_x = (1.0 - ratio) / 2.0 * roi_w
    pad_y = (1.0 - ratio) / 2.0 * roi_h
    cx0 = int(round(x0 + pad_x))
    cy0 = int(round(y0 + pad_y))
    cx1 = int(round(x1 - pad_x))
    cy1 = int(round(y1 - pad_y))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return cx0, cy0, cx1, cy1


def _hsv_summary(hsv: np.ndarray, config: Config) -> str:
    names = ("H", "S", "V")
    parts = []
    for index, name in enumerate(names):
        channel = hsv[:, :, index].reshape(-1)
        mean = float(channel.mean())
        pct = np.percentile(channel, config.hsv_percentiles)
        pct_text = " ".join(
            f"p{int(level)}={float(value):.1f}"
            for level, value in zip(config.hsv_percentiles, pct)
        )
        parts.append(f"{name} mean={mean:.1f} {pct_text}")
    return " | ".join(parts)


def _draw_color_detection(vis: np.ndarray, color: ColorObservation) -> None:
    """标当帧颜色检测框；框旁 DET=R/Y/B/N 与锁存报文无关。"""
    if color.bounding_box is None:
        return
    x, y, w, h = color.bounding_box
    cv2.rectangle(vis, (x, y), (x + w, y + h), COLOR_BOX_BGR, BOX_THICKNESS)
    code = color.color if color.color in ALLOWED_COLORS else "N"
    label = f"DET={code}"
    label_bgr = DET_LABEL_BGR.get(code, DET_LABEL_BGR["N"])
    baseline = max(14, y - 4)
    cv2.putText(
        vis,
        label,
        (x, baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        label_bgr,
        TEXT_THICKNESS + 1,
        cv2.LINE_AA,
    )


def _draw_roi(vis: np.ndarray, roi: RatioRoi, color: tuple[int, int, int]) -> None:
    height, width = vis.shape[:2]
    x0, y0, x1, y1 = roi.pixel_box(width, height)
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, BOX_THICKNESS)


def _ensure_dir(config: Config) -> Path | None:
    output_dir = resolve_debug_dir(config)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("调试目录不可用 %s: %s", output_dir, exc)
        return None
    return output_dir


def _write_image(path: Path, image: np.ndarray) -> bool:
    if _try_write_png_or_jpeg(path, image):
        return True
    if path.suffix.lower() != ".png":
        return False
    fallback = path.with_suffix(".jpg")
    LOGGER.warning("PNG 写入不可用，改用 JPEG: %s", fallback)
    return _try_write_png_or_jpeg(fallback, image)


def _try_write_png_or_jpeg(path: Path, image: np.ndarray) -> bool:
    tmp_path = path.with_name(f"{path.stem}.part{path.suffix}")
    try:
        ok = cv2.imwrite(str(tmp_path), image)
    except (cv2.error, OSError) as exc:
        LOGGER.warning("imwrite 异常 %s: %s", path, exc)
        _remove_if_exists(tmp_path)
        return _write_image_encoded(path, image, tmp_path)
    if not ok:
        LOGGER.warning("imwrite 失败 %s，尝试 imencode", path)
        _remove_if_exists(tmp_path)
        return _write_image_encoded(path, image, tmp_path)
    return _finalize_atomic_write(tmp_path, path)


def _write_image_encoded(path: Path, image: np.ndarray, tmp_path: Path) -> bool:
    ext = path.suffix if path.suffix else ".png"
    try:
        ok, buf = cv2.imencode(ext, image)
    except cv2.error as exc:
        LOGGER.warning("imencode 异常 %s: %s", path, exc)
        return False
    if not ok:
        return False
    try:
        tmp_path.write_bytes(buf.tobytes())
    except OSError as exc:
        LOGGER.warning("写调试图字节失败 %s: %s", path, exc)
        _remove_if_exists(tmp_path)
        return False
    return _finalize_atomic_write(tmp_path, path)


def _finalize_atomic_write(tmp_path: Path, path: Path) -> bool:
    try:
        tmp_path.replace(path)
    except OSError as exc:
        LOGGER.warning("调试图重命名失败 %s: %s", path, exc)
        _remove_if_exists(tmp_path)
        return False
    LOGGER.info("已保存调试图: %s", path)
    return True


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        LOGGER.debug("删除临时调试图跳过 %s: %s", path, exc)
