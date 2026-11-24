"""帧预处理与可中断休眠。"""
from __future__ import annotations

import runpy
import time
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")), run_name="termux_vision_bootstrap")

import cv2
import numpy as np

from termux_vision.config import Config


def freeze_frame(frame: np.ndarray) -> np.ndarray:
    """深拷贝为连续内存，避免与抓帧线程或 OpenCV 内缓冲共享导致撕裂。"""
    return np.ascontiguousarray(frame).copy()


def interruptible_sleep(is_running, duration_s: float, slice_s: float) -> None:
    deadline = time.monotonic() + duration_s
    while is_running() and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        time.sleep(min(slice_s, remaining))


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def resize_to_config(frame: np.ndarray, config: Config) -> np.ndarray:
    if config.no_resize:
        return frame
    height, width = frame.shape[:2]
    if width == config.frame_width and height == config.frame_height:
        return frame
    return cv2.resize(frame, (config.frame_width, config.frame_height))


def prepare_frame(frame: np.ndarray, config: Config) -> np.ndarray:
    return freeze_frame(resize_to_config(ensure_bgr(frame), config))


def resolve_debug_dir(config: Config) -> Path:
    path = Path(config.debug_dir)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path
