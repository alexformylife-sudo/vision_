"""离线核对锁存：真实检测器、打印入口，以及 stale/reset/_send_invalid。"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    print(f"缺少依赖，无法跑离线 OpenCV 流程: {exc}", file=sys.stderr)
    raise SystemExit(2)

from termux_vision.app import Pipeline, _handle_no_new_frame, _send_invalid
from termux_vision.config import make_config
from termux_vision.detectors import ColorDetector, TagDetector
from termux_vision.make_smoke import BLOCK_COLORS, _blank, _draw_block, _paste_tag
from termux_vision.models import VisionResult, invalid_color, invalid_tag
from termux_vision.temporal_filter import TemporalFilter
from termux_vision.util import prepare_frame

INVALID = ("N", -1, "N", 0)
MSG_INVALID = "SIDE=N;TAG=-1;COLOR=N;ALIGN=0"
MSG_TAG = "SIDE=L;TAG=0;COLOR=N;ALIGN=0"
MSG_BOTH = "SIDE=L;TAG=0;COLOR=R;ALIGN=1"
MSG_COLOR = "SIDE=N;TAG=-1;COLOR=Y;ALIGN=0"
MSG_COLOR_TAG = "SIDE=R;TAG=1;COLOR=Y;ALIGN=1"
MSG_BLUE = "SIDE=L;TAG=0;COLOR=B;ALIGN=1"


@dataclass
class _Rig:
    config: object
    temporal: TemporalFilter
    tag_detector: TagDetector
    color_detector: ColorDetector


class _ListSender:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)

    def close(self) -> None:
        return None


def _dictionary():
    cached = getattr(_dictionary, "value", None)
    if cached is None:
        cached = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        _dictionary.value = cached
    return cached


def _frame(tag_id: int | None, color_name: str | None):
    canvas = _blank()
    if tag_id is not None:
        _paste_tag(canvas, _dictionary(), tag_id)
    if color_name is not None:
        _draw_block(canvas, BLOCK_COLORS[color_name])
    return canvas


def _rig(frames: int = 5, seconds: float = 0.0) -> _Rig:
    config = make_config(
        source_type="images",
        source_path=".",
        stable_frame_count=frames,
        stable_seconds=seconds,
        target_fps=0,
    )
    temporal = TemporalFilter(config)
    return _Rig(config, temporal, TagDetector(config), ColorDetector(config))


def _once(rig: _Rig, frame) -> VisionResult:
    prepared = prepare_frame(frame, rig.config)
    tag = rig.tag_detector.detect(prepared)
    color = rig.color_detector.detect(prepared)
    return rig.temporal.update(tag, color)


def _times(rig: _Rig, frame, count: int) -> list[VisionResult]:
    return [_once(rig, frame) for _ in range(count)]


def _expect(result: VisionResult, wanted: tuple) -> None:
    actual = (result.side, result.tag_id, result.color, result.align)
    text = f"SIDE={wanted[0]};TAG={wanted[1]};COLOR={wanted[2]};ALIGN={wanted[3]}\n"
    if actual != wanted or result.to_message() != text:
        raise AssertionError(f"{actual} {result.to_message()!r} != {wanted}")


def _expect_last(results: list[VisionResult], wanted: tuple) -> None:
    _expect(results[-1], wanted)


def _pipeline(rig: _Rig, sender: _ListSender) -> Pipeline:
    return Pipeline(
        config=rig.config,
        source=object(),
        tag_detector=rig.tag_detector,
        color_detector=rig.color_detector,
        temporal=rig.temporal,
        sender=sender,
        save_images=False,
        last_frame_at=time.monotonic(),
    )


def _print_env() -> None:
    version = sys.version.split()[0]
    print(f"env python={version} cv2={cv2.__version__} numpy={np.__version__}")
    print("note this run is a cloud VM, not an Android Termux phone")


def _check_both_same_frame() -> None:
    rig = _rig()
    frame = _frame(0, "red")
    _expect_last(_times(rig, frame, 4), INVALID)
    done = _once(rig, frame)
    _expect(done, ("L", 0, "R", 1))
    print("both", done.to_message().strip())


def _check_tag_then_color() -> None:
    rig = _rig()
    _expect_last(_times(rig, _frame(0, None), 4), INVALID)
    _expect(_once(rig, _frame(0, None)), ("L", 0, "N", 0))
    _expect_last(_times(rig, _frame(None, None), 3), ("L", 0, "N", 0))
    _expect_last(_times(rig, _frame(None, "red"), 4), ("L", 0, "N", 0))
    done = _once(rig, _frame(None, "red"))
    _expect(done, ("L", 0, "R", 1))
    print("tag_then_color", done.to_message().strip())


def _check_color_then_tag() -> None:
    rig = _rig()
    _expect_last(_times(rig, _frame(None, "yellow"), 4), INVALID)
    _expect(_once(rig, _frame(None, "yellow")), ("N", -1, "Y", 0))
    _expect_last(_times(rig, _frame(1, None), 4), ("N", -1, "Y", 0))
    done = _once(rig, _frame(1, None))
    _expect(done, ("R", 1, "Y", 1))
    print("color_then_tag", done.to_message().strip())


def _check_jitter_does_not_latch() -> None:
    rig = _rig()
    frames = (_frame(0, "red"), _frame(1, "yellow"), _frame(None, None))
    last = _once(rig, frames[0])
    _expect(last, INVALID)
    for index in range(1, 9):
        last = _once(rig, frames[index % 3])
        _expect(last, INVALID)
    print("jitter", last.to_message().strip())


def _check_invalid_does_not_latch() -> None:
    rig = _rig(1)
    _expect_last(_times(rig, _frame(None, None), 4), INVALID)
    direct = TemporalFilter(rig.config)
    tag = invalid_tag(rig.config.invalid_side, rig.config.invalid_tag_id)
    color = invalid_color(rig.config.invalid_color)
    _expect(direct.update(tag, color), INVALID)
    print("invalid", MSG_INVALID)


def _check_no_overwrite() -> None:
    rig = _rig()
    _expect_last(_times(rig, _frame(0, "red"), 5), ("L", 0, "R", 1))
    changed = _times(rig, _frame(1, "blue"), 8)[-1]
    _expect(changed, ("L", 0, "R", 1))
    _expect_last(_times(rig, _frame(None, None), 4), ("L", 0, "R", 1))
    print("no_overwrite", changed.to_message().strip())


def _check_unknown_tag_not_parity() -> None:
    rig = _rig()
    last = _times(rig, _frame(2, "blue"), 6)[-1]
    _expect(last, ("N", -1, "B", 0))
    print("tag2_not_parity", last.to_message().strip())


def _check_reset_breaks_open_streak() -> None:
    rig = _rig()
    _expect_last(_times(rig, _frame(None, "red"), 5), ("N", -1, "R", 0))
    _times(rig, _frame(0, None), 4)
    rig.temporal.reset()
    held = _once(rig, _frame(0, None))
    _expect(held, ("N", -1, "R", 0))
    print("reset_streak", held.to_message().strip())


def _check_short_gap_keeps_streak() -> None:
    rig = _rig()
    sender = _ListSender()
    pipeline = _pipeline(rig, sender)
    color = _frame(None, "red")
    _times(rig, color, 4)
    pipeline.last_frame_at = time.monotonic()
    _handle_no_new_frame(pipeline, time.monotonic())
    if sender.messages:
        raise AssertionError(sender.messages)
    done = _once(rig, color)
    _expect(done, ("N", -1, "R", 0))
    print("short_gap", done.to_message().strip())


def _check_stale_keeps_latch() -> None:
    rig = _rig()
    sender = _ListSender()
    pipeline = _pipeline(rig, sender)
    _expect_last(_times(rig, _frame(0, None), 5), ("L", 0, "N", 0))
    _times(rig, _frame(None, "red"), 4)
    pipeline.last_frame_at = time.monotonic() - 10
    _handle_no_new_frame(pipeline, time.monotonic())
    _handle_no_new_frame(pipeline, time.monotonic())
    _expect_stale_messages(sender.messages)
    _expect_last(_times(rig, _frame(None, "red"), 4), ("L", 0, "N", 0))
    done = _once(rig, _frame(None, "red"))
    _expect(done, ("L", 0, "R", 1))
    print("stale", sender.messages[-1].strip(), "then", done.to_message().strip())


def _expect_stale_messages(messages: list[str]) -> None:
    if messages != [MSG_TAG + "\n", MSG_TAG + "\n"]:
        raise AssertionError(messages)


def _check_send_invalid_keeps_latch() -> None:
    rig = _rig()
    sender = _ListSender()
    pipeline = _pipeline(rig, sender)
    _expect_last(_times(rig, _frame(0, "red"), 5), ("L", 0, "R", 1))
    _send_invalid(pipeline)
    rig.temporal.reset()
    _send_invalid(pipeline)
    if sender.messages != [MSG_BOTH + "\n", MSG_BOTH + "\n"]:
        raise AssertionError(sender.messages)
    _expect(_once(rig, _frame(1, "yellow")), ("L", 0, "R", 1))
    print("send_invalid", sender.messages[-1].strip())


def _check_new_instance() -> None:
    rig = _rig()
    _expect_last(_times(rig, _frame(0, "red"), 5), ("L", 0, "R", 1))
    _expect(_once(rig, _frame(None, None)), ("L", 0, "R", 1))
    fresh = _rig()
    _expect(_once(fresh, _frame(0, "red")), INVALID)
    print("new_instance", MSG_INVALID)


def _check_time_latch() -> None:
    rig = _rig(frames=30, seconds=0.05)
    frame = _frame(0, "red")
    _expect(_once(rig, frame), INVALID)
    started = time.monotonic()
    while time.monotonic() - started < 0.06:
        time.sleep(0.01)
    done = _once(rig, frame)
    _expect(done, ("L", 0, "R", 1))
    print("time_latch", done.to_message().strip())


def _write_frames(folder: Path, frames: list) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        path = folder / f"{index:02d}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"写入失败: {path}")


def _cli(folder: Path, emit: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "termux_vision",
        "--source", "images", "--path", str(folder),
        "--stable-frames", "5", "--emit-each", str(emit),
        "--fps", "0", "--send-policy", "heartbeat", "--heartbeat", "0",
    ]
    print("RUN", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(f"rc={proc.returncode} {(proc.stderr or '')[-1500:]}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith("SIDE=")]
    for line in lines:
        print(line)
    return lines


def _assert_lines(lines: list[str], expected: list[str]) -> None:
    if lines != expected:
        raise AssertionError(f"{lines} != {expected}")


def _sequences() -> dict[str, list]:
    both = [_frame(0, "red")] * 5
    tag_then = [_frame(0, None)] * 5 + [_frame(None, "red")] * 5
    color_then = [_frame(None, "yellow")] * 5 + [_frame(1, None)] * 5
    overwrite = [_frame(0, "red")] * 5 + [_frame(1, "blue")] * 5
    return {"both": both, "tag_then": tag_then, "color_then": color_then, "overwrite": overwrite}


def _expected_cli() -> dict[str, list[str]]:
    tag_hold = [MSG_INVALID] * 4 + [MSG_TAG] * 5 + [MSG_BOTH]
    color_hold = [MSG_INVALID] * 4 + [MSG_COLOR] * 5 + [MSG_COLOR_TAG]
    return {
        "both": [MSG_INVALID] * 4 + [MSG_BOTH],
        "tag_then": tag_hold,
        "color_then": color_hold,
        "overwrite": [MSG_INVALID] * 4 + [MSG_BOTH] * 6,
    }


def _check_cli(root: Path) -> None:
    expected = _expected_cli()
    for name, frames in _sequences().items():
        _write_frames(root / name, frames)
        _assert_lines(_cli(root / name, 1), expected[name])
        print("cli", name, "ok")


def _check_smoke_print(root: Path) -> None:
    smoke = root / "smoke"
    cmd = [sys.executable, "-m", "termux_vision.make_smoke", "--out", str(smoke)]
    print("RUN", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    print(proc.stdout.strip())
    expected = [MSG_INVALID] * 9 + [MSG_BLUE] * 16
    _assert_lines(_cli(smoke, 5), expected)
    print("cli_smoke ok")


def main() -> int:
    _print_env()
    root = Path(tempfile.mkdtemp(prefix="termux_latch_"))
    _check_both_same_frame()
    _check_tag_then_color()
    _check_color_then_tag()
    _check_jitter_does_not_latch()
    _check_invalid_does_not_latch()
    _check_no_overwrite()
    _check_unknown_tag_not_parity()
    _check_reset_breaks_open_streak()
    _check_short_gap_keeps_streak()
    _check_stale_keeps_latch()
    _check_send_invalid_keeps_latch()
    _check_new_instance()
    _check_time_latch()
    _check_cli(root)
    _check_smoke_print(root)
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
