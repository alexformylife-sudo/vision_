"""生成合成烟雾测试图（非赛场样本）：tag36h11 + 红黄蓝色块。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CANVAS_BGR = (220, 220, 220)
TAG_PIXELS = 120
TAG_ORIGIN = (80, 260)
BLOCK_SIZE = (90, 70)  # 高, 宽，与赛场物块高大于底面直径一致
BLOCK_ORIGIN = (300, 280)
BLOCK_COLORS = {
    "red": (40, 40, 200),
    "yellow": (30, 210, 210),
    "blue": (200, 40, 40),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 Termux 视觉合成烟雾测试图")
    parser.add_argument("--out", required=True, help="输出目录")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_all(out_dir)
    print(f"已写入合成图到 {out_dir}")
    return 0


def _write_all(out_dir: Path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    _save(out_dir, "smoke_tag0_red.png", _canvas_with(dictionary, 0, "red"))
    _save(out_dir, "smoke_tag1_yellow.png", _canvas_with(dictionary, 1, "yellow"))
    _save(out_dir, "smoke_tag0_blue.png", _canvas_with(dictionary, 0, "blue"))
    _save(out_dir, "smoke_unknown_tag7.png", _canvas_with(dictionary, 7, None))
    _save(out_dir, "smoke_empty.png", _blank())


def _canvas_with(dictionary, tag_id: int, block_name: str | None) -> np.ndarray:
    canvas = _blank()
    _paste_tag(canvas, dictionary, tag_id)
    if block_name is not None:
        _draw_block(canvas, BLOCK_COLORS[block_name])
    return canvas


def _blank() -> np.ndarray:
    return np.full((480, 640, 3), CANVAS_BGR, dtype=np.uint8)


def _paste_tag(canvas: np.ndarray, dictionary, tag_id: int) -> None:
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, TAG_PIXELS, borderBits=1)
    y, x = TAG_ORIGIN
    # 周围留白边，避免标签贴到画布边缘。
    pad = 16
    y0, x0 = y - pad, x - pad
    canvas[y0:y + TAG_PIXELS + pad, x0:x + TAG_PIXELS + pad] = 255
    canvas[y:y + TAG_PIXELS, x:x + TAG_PIXELS] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def _draw_block(canvas: np.ndarray, bgr: tuple[int, int, int]) -> None:
    y, x = BLOCK_ORIGIN
    height, width = BLOCK_SIZE
    canvas[y:y + height, x:x + width] = bgr


def _save(out_dir: Path, name: str, image: np.ndarray) -> None:
    path = out_dir / name
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"写入失败: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
