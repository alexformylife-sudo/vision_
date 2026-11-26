"""离线回归：低饱和背景、空心放置圈、同色干扰与现场照片。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from termux_vision.config import Config, RatioRoi
from termux_vision.detectors import ColorDetector
from termux_vision.util import prepare_frame


def expect(detector, image, expected):
    result = detector.detect(image)
    assert result.color == expected, (expected, result)


def synthetic():
    config = Config(color_roi=RatioRoi(0, 0, 1, 1))
    detector = ColorDetector(config)
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[:] = (30, 80, 190)  # 模拟低饱和黄色地板，并非现场木地板样本
    def check(expected):
        expect(detector, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), expected)

    check("N")
    for color, hue in (("R", 153), ("Y", 35), ("B", 116)):
        hsv[:] = (30, 80, 190)
        cv2.circle(hsv, (150, 220), 100, (hue, 220, 180), -1)
        cv2.circle(hsv, (150, 220), 48, (30, 80, 190), -1)
        check("N")
        cv2.rectangle(hsv, (400, 230), (460, 310), (hue, 220, 180), -1)
        check(color)  # 更大的同色环不得遮蔽实心候选
    print("synthetic color filters PASSED")


def photos(folder: Path):
    config = Config()
    detector = ColorDetector(config)
    expected = {
        "IMG_2344.jpeg": "Y", "IMG_2345.jpeg": "Y",
        "IMG_2349.jpeg": "R", "IMG_2349(1).jpeg": "R",
        "IMG_2350.jpeg": "R", "IMG_2350(1).jpeg": "R",
        "IMG_2353.jpeg": "R", "IMG_2354.jpeg": "N",
    }
    for name, color in expected.items():
        image = cv2.imread(str(folder / name))
        assert image is not None, name
        expect(detector, prepare_frame(image, config), color)
        print(name, color, "PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=Path)
    args = parser.parse_args()
    synthetic()
    if args.photos is not None:
        photos(args.photos)
