"""根据 assets 赛道实拍 jpg 粗测默认 HSV 是否过宽（非正式验收）。"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from termux_vision.config import make_config
from termux_vision.detectors import ColorDetector
from termux_vision.util import prepare_frame


def main() -> int:
    paths = sorted(ASSETS.glob("*.jpg"))
    if not paths:
        print("no field jpg in", ASSETS)
        return 0
    cfg = make_config()
    det = ColorDetector(cfg)
    counts: Counter[str] = Counter()
    for path in paths:
        raw = cv2.imread(str(path))
        if raw is None:
            continue
        frame = prepare_frame(cv2.resize(raw, (640, 480)), cfg)
        counts[det.detect(frame).color] += 1
    print("field samples", len(paths), dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())