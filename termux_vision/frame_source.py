"""画面输入：图片/文件夹、本地视频、HTTP/RTSP 低延迟抓帧。"""
from __future__ import annotations

import logging
import runpy
import threading
from pathlib import Path
from typing import Protocol

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")), run_name="termux_vision_bootstrap")

import cv2
import numpy as np

from termux_vision.config import Config
from termux_vision.models import FrameRead
from termux_vision.util import interruptible_sleep

LOGGER = logging.getLogger(__name__)


class FrameSource(Protocol):
    def open(self) -> None: ...

    def read(self) -> FrameRead: ...

    def close(self) -> None: ...


class ImageDirSource:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._paths: list[Path] = []
        self._index = 0
        self._repeats_left = 0
        self._current_frame: np.ndarray | None = None

    def open(self) -> None:
        self._paths = _list_image_paths(Path(self._config.source_path), self._config.image_suffixes)
        self._index = 0
        self._repeats_left = 0
        self._current_frame = None
        if not self._paths:
            raise FileNotFoundError(f"未找到图片: {self._config.source_path}")
        LOGGER.info("图片源: %s 张，路径 %s", len(self._paths), self._config.source_path)

    def read(self) -> FrameRead:
        if self._repeats_left > 0 and self._current_frame is not None:
            self._repeats_left -= 1
            return FrameRead(frame=self._current_frame.copy(), ended=False)
        return self._load_next_image()

    def close(self) -> None:
        self._current_frame = None

    def _load_next_image(self) -> FrameRead:
        while self._index < len(self._paths):
            path = self._paths[self._index]
            self._index += 1
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                LOGGER.warning("读图失败，跳过: %s", path)
                continue
            self._current_frame = frame
            self._repeats_left = self._config.image_emit_each - 1
            return FrameRead(frame=frame.copy(), ended=False)
        return FrameRead(frame=None, ended=True)


class VideoFileSource:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._config.source_path)
        if not capture.isOpened():
            capture.release()
            raise FileNotFoundError(f"无法打开视频: {self._config.source_path}")
        self._capture = capture
        LOGGER.info("视频源已打开: %s", self._config.source_path)

    def read(self) -> FrameRead:
        if self._capture is None:
            return FrameRead(frame=None, ended=True)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return FrameRead(frame=None, ended=True)
        return FrameRead(frame=frame.copy(), ended=False)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class StreamSource:
    """抓帧线程独占 capture；主线程只经 _lock 读最新帧，阻塞的 open/read 不持锁。"""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._delivered_seq = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        with self._lock:
            self._running = True
            self._latest = None
        self._thread = threading.Thread(target=self._grab_loop, name="stream-grab", daemon=True)
        self._thread.start()
        LOGGER.info("流媒体源已启动: %s", self._config.source_path)

    def read(self) -> FrameRead:
        with self._lock:
            if self._latest is None or self._seq == self._delivered_seq:
                return FrameRead(frame=None, ended=False)
            self._delivered_seq = self._seq
            return FrameRead(frame=self._latest.copy(), ended=False, seq=self._seq)

    def close(self) -> None:
        with self._lock:
            self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._config.thread_join_timeout_s)
        if thread is not None and thread.is_alive():
            LOGGER.warning("抓帧线程仍阻塞在取流调用中，后台线程随进程退出")
        self._thread = None

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    def _store_latest(self, frame: np.ndarray | None) -> None:
        with self._lock:
            self._latest = None if frame is None else frame.copy()
            if frame is not None:
                self._seq += 1

    def _grab_loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            while self._is_running():
                capture = self._grab_once(capture)
        finally:
            if capture is not None:
                capture.release()

    def _grab_once(self, capture: cv2.VideoCapture | None) -> cv2.VideoCapture | None:
        if capture is None:
            capture = _open_stream_capture(self._config)
        if capture is None:
            self._store_latest(None)
            self._wait_reconnect()
            return None
        ok, frame = capture.read()
        if ok and frame is not None:
            self._store_latest(frame)
            return capture
        LOGGER.warning("读流失败，清空最新帧并准备重连")
        self._store_latest(None)
        capture.release()
        self._wait_reconnect()
        return None

    def _wait_reconnect(self) -> None:
        interruptible_sleep(self._is_running, self._config.reconnect_delay_s, self._config.idle_poll_s)


def build_source(config: Config) -> FrameSource:
    if config.source_type == "images":
        return ImageDirSource(config)
    if config.source_type == "video":
        return VideoFileSource(config)
    return StreamSource(config)


def _list_image_paths(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"图片路径不存在: {root}")
    allowed = {item.lower() for item in suffixes}
    paths = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in allowed]
    return sorted(paths)


def _open_stream_capture(config: Config) -> cv2.VideoCapture | None:
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.open_timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.read_timeout_ms,
    ]
    try:
        capture = cv2.VideoCapture(config.source_path, cv2.CAP_FFMPEG, params)
    except cv2.error as exc:
        LOGGER.warning("打开视频流异常: %s", exc)
        return None
    if not capture.isOpened():
        LOGGER.warning("打开视频流失败: %s", config.source_path)
        capture.release()
        return None
    return capture
