#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np

from visual_pick_stack import (
    ColorBlockDetector,
    Detection,
    MappedBlock,
    cluster_mapped_candidates,
    load_config,
    pixel_to_global_workspace,
    pixel_to_workspace,
)


ROOT = Path(__file__).resolve().parent


def main() -> int:
    config = load_config(ROOT / "config.json")
    detector = ColorBlockDetector(config)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    expected = [
        ("yellow", (150, 320)),
        ("red", (490, 320)),
        ("green", (490, 150)),
        ("blue", (150, 150)),
        ("blue", (320, 150)),
    ]
    bgr = {
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
    }
    for name, center in expected:
        x, y = center
        cv2.rectangle(frame, (x - 25, y - 25), (x + 25, y + 25), bgr[name], -1)

    detections = detector.detect(frame, bgr.keys())
    actual = [(item.color, item.center) for item in detections]
    assert len(actual) == len(expected), actual
    for name, center in expected:
        matches = [item for item in detections if item.color == name]
        nearest = min(
            matches,
            key=lambda item: np.linalg.norm(np.array(item.center) - np.array(center)),
        )
        assert np.linalg.norm(np.array(nearest.center) - np.array(center)) <= 2, actual
        assert nearest.grasp_center == nearest.center

    dummy_box = np.zeros((4, 2), dtype=np.int32)
    mapped = [
        MappedBlock(Detection("blue", (100, 100), 1000, dummy_box), (0.00, 0.24), 80, "a", 80),
        MappedBlock(Detection("blue", (300, 100), 1000, dummy_box), (0.06, 0.24), 80, "a", 80),
        MappedBlock(Detection("blue", (120, 100), 1000, dummy_box), (0.004, 0.242), 85, "b", 85),
        MappedBlock(Detection("blue", (320, 100), 1000, dummy_box), (0.064, 0.242), 85, "b", 85),
    ]
    clusters = cluster_mapped_candidates(mapped, 0.03)
    assert sorted(len(cluster) for cluster in clusters) == [2, 2], clusters
    assert pixel_to_workspace((320, 480)) == (0.0, 0.19)
    assert pixel_to_global_workspace((320, 480), 90) == (0.0, 0.19)
    rotated = pixel_to_global_workspace((320, 480), 60)
    assert np.allclose(rotated, (0.095, 0.16454), atol=1e-5), rotated
    print("PASS: detected", actual, "clusters=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
