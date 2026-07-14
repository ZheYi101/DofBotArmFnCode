#!/usr/bin/env python3
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from visual_pick_stack import (
    ColorBlockDetector,
    Detection,
    MappedBlock,
    apply_workspace_corrections,
    cluster_mapped_candidates,
    fuse_cluster_observations,
    load_config,
    load_workspace_corrections,
    pixel_to_global_workspace,
    pixel_to_workspace,
    suppress_cross_color_conflicts,
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
    fused, inliers = fuse_cluster_observations(clusters[0], config, 0.03)
    assert fused.supporting_views == 2, fused
    assert fused.outlier_views == 0, fused
    assert len(inliers) == 2, inliers
    mapped_with_outlier = [
        MappedBlock(Detection("red", (320, 100), 3000, dummy_box), (0.040, 0.280), 81, "a", 81),
        MappedBlock(Detection("red", (322, 101), 2900, dummy_box), (0.041, 0.279), 82, "b", 82),
        MappedBlock(Detection("red", (330, 110), 2500, dummy_box), (0.070, 0.260), 95, "c", 95),
    ]
    fused_outlier, inliers_outlier = fuse_cluster_observations(mapped_with_outlier, config, 0.03)
    assert fused_outlier.supporting_views == 2, fused_outlier
    assert fused_outlier.raw_supporting_views == 3, fused_outlier
    assert fused_outlier.outlier_views == 1, fused_outlier
    assert len(inliers_outlier) == 2, inliers_outlier
    assert np.allclose(fused_outlier.workspace, (0.0405, 0.2795), atol=0.002), fused_outlier.workspace
    assert pixel_to_workspace((320, 480)) == (0.0, 0.19)
    assert pixel_to_global_workspace((320, 480), 90) == (0.0, 0.19)
    rotated = pixel_to_global_workspace((320, 480), 60)
    assert np.allclose(rotated, (0.095, 0.16454), atol=1e-5), rotated

    with TemporaryDirectory() as tmpdir:
        feedback_path = Path(tmpdir) / "feedback_compensation.json"
        feedback_path.write_text(
            json.dumps({"version": 1, "rules": []}),
            encoding="utf-8",
        )
        rules = load_workspace_corrections(feedback_path)
        assert len(rules) == 0, rules
        far_target = MappedBlock(
            Detection("red", (323, 75), 8254, dummy_box),
            (0.15108, 0.20943),
            60,
            "scan.jpg",
        )
        corrected = apply_workspace_corrections(far_target, rules)
        assert corrected.raw_workspace == (0.15108, 0.20943), corrected
        assert corrected.workspace == far_target.workspace, corrected.workspace
        assert corrected.workspace_correction_rule == "", corrected
        near_target = MappedBlock(
            Detection("green", (320, 220), 12000, dummy_box),
            (0.02, 0.215),
            90,
            "near.jpg",
        )
        near_corrected = apply_workspace_corrections(near_target, rules)
        assert near_corrected.workspace == near_target.workspace, near_corrected
        assert near_corrected.workspace_correction_rule == "", near_corrected

    conflict_config = load_config(ROOT / "config.json")
    strong_red = replace(
        far_target,
        detection=Detection("red", (322, 178), 10695, dummy_box, 0.45, 0.14),
        workspace=(0.06948, 0.26107),
        raw_workspace=(0.06948, 0.26107),
        supporting_views=7,
        target_id="red-01",
    )
    weak_blue = MappedBlock(
        detection=Detection("blue", (313, 253), 1243, dummy_box, 0.06, 0.07),
        workspace=(0.06276, 0.24242),
        raw_workspace=(0.06276, 0.24242),
        scan_angle=75,
        snapshot="blue-near-red.jpg",
        supporting_views=6,
        target_id="blue-01",
    )
    kept, suppressed = suppress_cross_color_conflicts(
        [(strong_red, [], []), (weak_blue, [], [])],
        conflict_config,
    )
    assert len(kept) == 1, kept
    assert kept[0][0].target_id == "red-01", kept
    assert len(suppressed) == 1, suppressed
    assert suppressed[0]["target_id"] == "blue-01", suppressed
    assert suppressed[0]["suppressed_by_target_id"] == "red-01", suppressed

    print("PASS: detected", actual, "clusters=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
