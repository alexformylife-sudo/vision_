"""Termux 无头视觉主程序：FrameSource -> Detectors -> TemporalFilter -> ResultSender。"""
from __future__ import annotations

import argparse
import logging
import runpy
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")), run_name="termux_vision_bootstrap")

import cv2

from termux_vision.config import Config, RatioRoi, make_config
from termux_vision.debug_view import (
    DebugSnapshot,
    log_center_hsv_stats,
    save_debug_frame,
    save_hsv_calib_frame,
)
from termux_vision.detectors import ColorDetector, TagDetector
from termux_vision.frame_source import FrameSource, build_source
from termux_vision.models import invalid_color, invalid_tag
from termux_vision.result_sender import ResultSender, build_sender
from termux_vision.temporal_filter import TemporalFilter
from termux_vision.util import prepare_frame

LOGGER = logging.getLogger(__name__)

SOURCE_ALIASES = {"http": "stream", "rtsp": "stream"}


@dataclass
class Pipeline:
    config: Config
    source: FrameSource
    tag_detector: TagDetector
    color_detector: ColorDetector
    temporal: TemporalFilter
    sender: ResultSender
    save_images: bool
    last_frame_at: float = 0.0
    stale: bool = False


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    try:
        config = config_from_args(argv)
        sender = build_sender(config)
    except ValueError as exc:
        LOGGER.error("参数无效: %s", exc)
        return 2
    _log_startup(config)
    return _run_with_pipeline(config, sender)


def _run_with_pipeline(config: Config, sender: ResultSender) -> int:
    source = build_source(config)
    pipeline = Pipeline(
        config=config,
        source=source,
        tag_detector=TagDetector(config),
        color_detector=ColorDetector(config),
        temporal=TemporalFilter(config),
        sender=sender,
        save_images=config.save_debug or config.hsv_calib,
    )
    try:
        if not _open_source(source):
            return 1
        return _run_mode(pipeline)
    except KeyboardInterrupt:
        LOGGER.info("收到 Ctrl+C，正在退出")
        return 0
    finally:
        sender.close()
        source.close()


def _open_source(source: FrameSource) -> bool:
    try:
        source.open()
    except FileNotFoundError as exc:
        LOGGER.error("输入源不可用: %s", exc)
        return False
    return True


def _run_mode(pipeline: Pipeline) -> int:
    if pipeline.config.hsv_calib:
        _run_hsv_calib(pipeline)
        return 0
    processed = _run_loop(pipeline)
    if processed > 0:
        return 0
    LOGGER.error("没有任何可用帧，输出无效报文")
    _send_invalid(pipeline)
    return 1


def _run_loop(pipeline: Pipeline) -> int:
    processed = 0
    pipeline.last_frame_at = time.monotonic()
    while True:
        started = time.monotonic()
        retrieved = pipeline.source.read()
        if retrieved.ended:
            return processed
        if retrieved.frame is None:
            _handle_no_new_frame(pipeline, started)
            _idle_until(started, pipeline.config, got_frame=False)
            continue
        pipeline.last_frame_at = started
        pipeline.stale = False
        _process_frame(pipeline, retrieved.frame, processed, retrieved.seq)
        processed += 1
        _idle_until(started, pipeline.config, got_frame=True)


def _handle_no_new_frame(pipeline: Pipeline, now: float) -> None:
    if now - pipeline.last_frame_at < pipeline.config.stale_frame_timeout_s:
        return
    if not pipeline.stale:
        LOGGER.warning("超过 %.1fs 无新帧，未确认连续计数清零，已锁存结果保留", pipeline.config.stale_frame_timeout_s)
        pipeline.stale = True
    pipeline.temporal.reset()
    _send_invalid(pipeline)


def _send_invalid(pipeline: Pipeline) -> None:
    """无效观测仍走 TemporalFilter。已锁存的路不会被这次观测覆盖。"""
    tag = invalid_tag(pipeline.config.invalid_side, pipeline.config.invalid_tag_id)
    color = invalid_color(pipeline.config.invalid_color)
    result = pipeline.temporal.update(tag, color)
    pipeline.sender.send(result.to_message())


def _run_hsv_calib(pipeline: Pipeline) -> None:
    frame_index = 0
    while True:
        retrieved = pipeline.source.read()
        if retrieved.ended:
            break
        if retrieved.frame is None:
            time.sleep(pipeline.config.idle_poll_s)
            continue
        frame = _prepare_or_none(retrieved.frame, pipeline.config)
        if frame is None:
            continue
        label = f"{frame_index:04d}"
        _save_calib_image(pipeline, frame, label)
        log_center_hsv_stats(frame, pipeline.config, label)
        frame_index += 1
        if pipeline.config.source_type != "images":
            break


def _save_calib_image(pipeline: Pipeline, frame, label: str) -> None:
    if not pipeline.save_images:
        return
    if not save_hsv_calib_frame(frame, pipeline.config, label):
        LOGGER.warning("调参截图保存失败，已关闭存图，只打印 HSV 统计")
        pipeline.save_images = False


def _process_frame(pipeline: Pipeline, raw_frame, frame_index: int, source_seq: int = -1) -> None:
    frame = _prepare_or_none(raw_frame, pipeline.config)
    if frame is None:
        _send_invalid(pipeline)
        return
    tag = pipeline.tag_detector.detect(frame)
    color = pipeline.color_detector.detect(frame)
    result = pipeline.temporal.update(tag, color)
    pipeline.sender.send(result.to_message())
    debug_index = source_seq if source_seq >= 0 else frame_index + 1
    snap = DebugSnapshot(frame, tag, color, result, debug_index)
    _maybe_debug(pipeline, snap)


def _prepare_or_none(raw_frame, config: Config):
    try:
        return prepare_frame(raw_frame, config)
    except cv2.error as exc:
        LOGGER.warning("预处理帧失败: %s", exc)
        return None


def _maybe_debug(pipeline: Pipeline, snapshot: DebugSnapshot) -> None:
    if pipeline.config.print_hsv_stats:
        log_center_hsv_stats(snapshot.frame, pipeline.config, f"frame_{snapshot.frame_index:04d}")
    if not (pipeline.config.save_debug and pipeline.save_images):
        return
    if not save_debug_frame(snapshot, pipeline.config):
        LOGGER.warning("调试存图失败，已关闭调试存图，主循环继续")
        pipeline.save_images = False


def _idle_until(started: float, config: Config, got_frame: bool) -> None:
    if config.target_fps <= 0:
        if not got_frame:
            time.sleep(config.idle_poll_s)
        return
    remaining = (1.0 / config.target_fps) - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _log_startup(config: Config) -> None:
    LOGGER.info(
        "启动 source=%s path=%s size=%sx%s TAG_SIZE_M=%s 离地=%sm 防抖=%s帧/%ss 映射=%s 发送=%s",
        config.source_type,
        config.source_path,
        config.frame_width,
        config.frame_height,
        config.TAG_SIZE_M,
        config.tag_bottom_height_m,
        config.stable_frame_count,
        config.stable_seconds,
        dict(config.tag_id_to_side),
        config.send_policy,
    )


def config_from_args(argv: list[str] | None = None) -> Config:
    args = _build_parser().parse_args(argv)
    if not args.path:
        raise ValueError("必须提供 --path（图片/文件夹、视频或视频流地址）")
    source_type = SOURCE_ALIASES.get(args.source, args.source)
    config = make_config(source_type=source_type, source_path=args.path)
    config = _apply_optional_overrides(config, args)
    if config.hsv_calib and config.image_emit_each != 1:
        return replace(config, image_emit_each=1)
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Termux 无头视觉：AprilTag 判左右 + 红黄蓝，输出一行报文。")
    _add_source_args(parser)
    _add_output_args(parser)
    _add_debug_args(parser)
    return parser


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=("images", "video", "stream", "http", "rtsp"), default="images")
    parser.add_argument("--path", default="", help="图片文件/目录、视频文件，或 HTTP/RTSP 地址")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="目标帧率，0 表示不休眠")
    parser.add_argument("--no-resize", action="store_true", help="不强制缩放到工作分辨率")
    parser.add_argument("--stable-frames", type=int, default=None)
    parser.add_argument("--stable-seconds", type=float, default=None)
    parser.add_argument("--emit-each", type=int, default=None, help="每张静图重复送入的次数，便于离线看防抖")
    parser.add_argument("--reconnect-delay", type=float, default=None)
    parser.add_argument("--tag-map", default=None, help="例如 0:L,1:R")
    parser.add_argument("--tag-roi", default=None, help="x0,y0,x1,y1 比例")
    parser.add_argument("--color-roi", default=None, help="x0,y0,x1,y1 比例")
    parser.add_argument("--min-color-area", type=int, default=None)
    parser.add_argument("--multi-tag", default=None, choices=("largest_area", "nearest_center"))


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tcp", default=None, help="HOST:PORT，启用非阻塞 TCP 输出")
    parser.add_argument("--no-print", action="store_true", help="关闭终端报文")
    parser.add_argument("--send-policy", default=None, choices=("on_change", "heartbeat"))
    parser.add_argument("--heartbeat", type=float, default=None, help="心跳间隔秒，0 表示每帧都发")


def _add_debug_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--hsv-stats", action="store_true")
    parser.add_argument("--hsv-calib", action="store_true", help="保存 ROI 截图并打印颜色 ROI 中心区 HSV")


def _apply_optional_overrides(config: Config, args) -> Config:
    updates = _simple_overrides(args)
    updates.update(_parse_complex_overrides(args))
    cleaned = {key: value for key, value in updates.items() if value is not None}
    return replace(config, **cleaned)


def _simple_overrides(args) -> dict:
    return {
        "frame_width": args.width,
        "frame_height": args.height,
        "target_fps": args.fps,
        "stable_frame_count": args.stable_frames,
        "stable_seconds": args.stable_seconds,
        "image_emit_each": args.emit_each,
        "reconnect_delay_s": args.reconnect_delay,
        "color_min_area": args.min_color_area,
        "multi_tag_policy": args.multi_tag,
        "debug_dir": args.debug_dir,
        "heartbeat_interval_s": args.heartbeat,
        "send_policy": args.send_policy,
        "save_debug": True if args.save_debug else None,
        "print_hsv_stats": True if args.hsv_stats else None,
        "hsv_calib": True if args.hsv_calib else None,
        "print_enabled": False if args.no_print else None,
        "no_resize": True if args.no_resize else None,
    }


def _parse_complex_overrides(args) -> dict:
    updates = {}
    if args.tag_map is not None:
        updates["tag_id_to_side"] = _parse_tag_map(args.tag_map)
    if args.tag_roi is not None:
        updates["tag_roi"] = _parse_roi(args.tag_roi)
    if args.color_roi is not None:
        updates["color_roi"] = _parse_roi(args.color_roi)
    if args.tcp is not None:
        host, port = _parse_tcp(args.tcp)
        updates["tcp_enabled"] = True
        updates["tcp_host"] = host
        updates["tcp_port"] = port
    return updates


def _parse_tag_map(text: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for chunk in text.split(","):
        if ":" not in chunk:
            raise ValueError(f"无效的 tag 映射项: {chunk}")
        raw_id, raw_side = chunk.split(":", 1)
        side = raw_side.strip().upper()
        if side not in {"L", "R"}:
            raise ValueError(f"侧别必须是 L 或 R: {chunk}")
        mapping[int(raw_id.strip())] = side
    if not mapping:
        raise ValueError("tag 映射不能为空")
    return mapping


def _parse_roi(text: str) -> RatioRoi:
    parts = [float(item.strip()) for item in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI 必须是 x0,y0,x1,y1")
    return RatioRoi(*parts)


def _parse_tcp(text: str) -> tuple[str, int]:
    if ":" not in text:
        raise ValueError("TCP 必须是 HOST:PORT")
    host, raw_port = text.rsplit(":", 1)
    if not host:
        raise ValueError("TCP 主机名不能为空")
    return host, int(raw_port)


if __name__ == "__main__":
    raise SystemExit(main())
