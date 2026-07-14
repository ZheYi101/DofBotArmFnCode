#!/usr/bin/env python3
# coding: utf-8
"""OpenCV color recognition and DOFBOT block pick/stack control.

The default mode is detection-only.  Servo motion is enabled only with
``--execute``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"


@dataclass
class Detection:
    color: str
    center: Tuple[int, int]
    area: float
    box: np.ndarray
    white_ring_ratio: float = 0.0
    body_edge_ratio: float = 0.0
    grasp_center: Optional[Tuple[int, int]] = None


@dataclass
class MappedBlock:
    detection: Detection
    workspace: Tuple[float, float]
    scan_angle: float
    snapshot: str
    commanded_scan_angle: Optional[int] = None
    coordinate_spread_m: float = 0.0
    supporting_views: int = 1
    target_id: str = ""
    raw_coordinate_spread_m: float = 0.0
    raw_supporting_views: int = 1
    outlier_views: int = 0
    fusion_method: str = "best_view"
    raw_workspace: Tuple[float, float] = (0.0, 0.0)
    workspace_correction_dx_m: float = 0.0
    workspace_correction_dy_m: float = 0.0
    workspace_correction_rule: str = ""
    workspace_correction_note: str = ""


@dataclass
class WorkspaceCorrectionRule:
    rule_id: str
    enabled: bool = True
    color: Optional[str] = None
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    radius_min: Optional[float] = None
    radius_max: Optional[float] = None
    radius_origin_x: float = 0.0
    radius_origin_y: float = 0.19
    scan_angle_min: Optional[float] = None
    scan_angle_max: Optional[float] = None
    dx_m: float = 0.0
    dy_m: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    y_shear_m_per_m: float = 0.0
    pivot_x: float = 0.0
    pivot_y: float = 0.19
    note: str = ""


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_workspace_corrections(path: Path) -> List[WorkspaceCorrectionRule]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    rules: List[WorkspaceCorrectionRule] = []
    for index, raw_rule in enumerate(payload.get("rules", []), start=1):
        rule_id = str(raw_rule.get("id") or "rule-%02d" % index)
        rules.append(
            WorkspaceCorrectionRule(
                rule_id=rule_id,
                enabled=bool(raw_rule.get("enabled", True)),
                color=raw_rule.get("color"),
                x_min=raw_rule.get("workspace_x_min"),
                x_max=raw_rule.get("workspace_x_max"),
                y_min=raw_rule.get("workspace_y_min"),
                y_max=raw_rule.get("workspace_y_max"),
                radius_min=raw_rule.get("workspace_radius_min"),
                radius_max=raw_rule.get("workspace_radius_max"),
                radius_origin_x=float(raw_rule.get("radius_origin_x", 0.0)),
                radius_origin_y=float(raw_rule.get("radius_origin_y", 0.19)),
                scan_angle_min=raw_rule.get("scan_angle_min"),
                scan_angle_max=raw_rule.get("scan_angle_max"),
                dx_m=float(raw_rule.get("dx_m", 0.0)),
                dy_m=float(raw_rule.get("dy_m", 0.0)),
                scale_x=float(raw_rule.get("scale_x", 1.0)),
                scale_y=float(raw_rule.get("scale_y", 1.0)),
                y_shear_m_per_m=float(raw_rule.get("y_shear_m_per_m", 0.0)),
                pivot_x=float(raw_rule.get("pivot_x", 0.0)),
                pivot_y=float(raw_rule.get("pivot_y", 0.19)),
                note=str(raw_rule.get("note", "")),
            )
        )
    return rules


class ColorBlockDetector:
    def __init__(self, config: dict):
        vision = config["vision"]
        self.ranges = vision["hsv_ranges"]
        self.min_area = float(vision["min_area"])
        self.max_area = float(vision["max_area"])
        self.min_aspect = float(vision["min_aspect_ratio"])
        self.max_aspect = float(vision["max_aspect_ratio"])
        self.kernel_size = int(vision["morph_kernel"])
        self.edge_margin = int(vision["edge_margin"])
        self.ring_padding = int(vision["surround_padding"])
        self.white_saturation_max = int(vision["white_saturation_max"])
        self.white_value_min = int(vision["white_value_min"])
        self.max_white_ring_ratio = float(vision["max_white_ring_ratio"])
        self.min_body_edge_ratio = float(vision["min_body_edge_ratio"])
        self.canny_low = int(vision["canny_low"])
        self.canny_high = int(vision["canny_high"])
        self.track_radius_px = float(vision["multi_object_track_radius_px"])

    @staticmethod
    def _find_contours(mask: np.ndarray) -> Sequence[np.ndarray]:
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return result[-2]

    def _mask_for_color(self, hsv: np.ndarray, color: str) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.ranges[color]:
            part = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask, part)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.kernel_size, self.kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def detect(self, frame: np.ndarray, colors: Iterable[str]) -> List[Detection]:
        frame = cv2.resize(frame, (640, 480))
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        detections: List[Detection] = []

        for color in colors:
            mask = self._mask_for_color(hsv, color)
            candidates: List[Detection] = []
            for contour in self._find_contours(mask):
                area = float(cv2.contourArea(contour))
                if not self.min_area <= area <= self.max_area:
                    continue
                x, y, width_px, height_px = cv2.boundingRect(contour)
                image_height, image_width = mask.shape
                if (
                    x <= self.edge_margin
                    or y <= self.edge_margin
                    or x + width_px >= image_width - self.edge_margin
                    or y + height_px >= image_height - self.edge_margin
                ):
                    # A cropped contour has an unreliable center and must never
                    # be sent to inverse kinematics.
                    continue
                pad = self.ring_padding
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(image_width, x + width_px + pad)
                y1 = min(image_height, y + height_px + pad)
                surround = hsv[y0:y1, x0:x1]
                ring = np.ones(surround.shape[:2], dtype=bool)
                ring[y - y0 : y - y0 + height_px, x - x0 : x - x0 + width_px] = False
                white = (
                    (surround[:, :, 1] < self.white_saturation_max)
                    & (surround[:, :, 2] > self.white_value_min)
                    & ring
                )
                white_ring_ratio = float(white.sum()) / max(1, int(ring.sum()))
                body_height = max(25, int(height_px * 1.1))
                body = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[
                    max(0, y + height_px - 2) : min(
                        image_height, y + height_px + body_height
                    ),
                    max(0, x - 5) : min(image_width, x + width_px + 5),
                ]
                if body.size:
                    body_edges = cv2.Canny(body, self.canny_low, self.canny_high)
                    body_edge_ratio = float((body_edges > 0).mean())
                else:
                    body_edge_ratio = 0.0
                if (
                    white_ring_ratio > self.max_white_ring_ratio
                    and body_edge_ratio < self.min_body_edge_ratio
                ):
                    # A printed marker has a white ring and no box-side texture.
                    # Requiring both conditions avoids rejecting cubes on a pale
                    # wooden or white table.
                    continue
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width <= 0 or height <= 0:
                    continue
                aspect = max(width, height) / min(width, height)
                if not self.min_aspect <= aspect <= self.max_aspect:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                center = (
                    int(round(moments["m10"] / moments["m00"])),
                    int(round(moments["m01"] / moments["m00"])),
                )
                # This mapping was calibrated by Yahboom against the centre of
                # the coloured top contour.  It already compensates for the
                # camera pitch and cube height; projecting to the visible base
                # would apply that correction twice.
                grasp_center = center
                box = cv2.boxPoints(rect).astype(np.int32)
                candidates.append(
                    Detection(
                        color,
                        center,
                        area,
                        box,
                        white_ring_ratio,
                        body_edge_ratio,
                        grasp_center,
                    )
                )

            # Keep every valid contour.  Temporal tracking and cross-view map
            # clustering later decide which observations belong to each block.
            detections.extend(
                sorted(
                    candidates,
                    key=lambda item: (
                        item.body_edge_ratio,
                        item.area,
                        -item.white_ring_ratio,
                    ),
                    reverse=True,
                )
            )

        return detections

    @staticmethod
    def annotate(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        output = cv2.resize(frame, (640, 480)).copy()
        palette = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 200, 0),
            "blue": (255, 0, 0),
        }
        for item in detections:
            color = palette.get(item.color, (255, 255, 255))
            cv2.drawContours(output, [item.box], 0, color, 2)
            cv2.circle(output, item.center, 5, (255, 255, 255), -1)
            if item.grasp_center is not None:
                cv2.line(output, item.center, item.grasp_center, (255, 255, 255), 2)
                cv2.circle(output, item.grasp_center, 7, (0, 0, 0), 2)
            label = "%s (%d,%d) A=%d W=%.2f E=%.2f" % (
                item.color,
                item.center[0],
                item.center[1],
                item.area,
                item.white_ring_ratio,
                item.body_edge_ratio,
            )
            cv2.putText(
                output,
                label,
                (max(0, item.center[0] - 70), max(22, item.center[1] - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        return output


def pixel_to_workspace(center: Tuple[int, int]) -> Tuple[float, float]:
    """Use the mapping supplied by the installed DOFBOT vision examples."""

    point_x, point_y = center
    x = round((point_x - 320.0) / 4000.0, 5)
    y = round(((480.0 - point_y) / 3000.0) * 0.8 + 0.19, 5)
    return x, y


def pixel_to_global_workspace(
    center: Tuple[int, int], scan_angle: float
) -> Tuple[float, float]:
    """Rotate a scan-view coordinate into the base-90 global map frame."""

    local_x, local_y = pixel_to_workspace(center)
    theta = math.radians(float(scan_angle) - 90.0)
    global_x = math.cos(theta) * local_x - math.sin(theta) * local_y
    global_y = math.sin(theta) * local_x + math.cos(theta) * local_y
    return round(global_x, 5), round(global_y, 5)


def open_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open /dev/video%d. Check whether a Jupyter notebook or another "
            "program is using the camera (fuser -v /dev/video%d)." % (index, index)
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def capture_stable(
    cap: cv2.VideoCapture,
    detector: ColorBlockDetector,
    colors: Sequence[str],
    sample_frames: int,
    min_hits: int,
    max_jitter_px: float,
) -> Tuple[np.ndarray, List[Detection]]:
    tracks: Dict[str, List[List[Detection]]] = {color: [] for color in colors}
    last_frame: Optional[np.ndarray] = None

    # Discard stale auto-exposure frames.
    for _ in range(8):
        cap.read()

    for _ in range(sample_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        last_frame = frame
        frame_items: Dict[str, List[Detection]] = {color: [] for color in colors}
        for item in detector.detect(frame, colors):
            frame_items[item.color].append(item)
        for color in colors:
            used_tracks = set()
            for item in frame_items[color]:
                best_index = None
                best_distance = detector.track_radius_px
                for index, track in enumerate(tracks[color]):
                    if index in used_tracks:
                        continue
                    recent = np.array(track[-1].center, dtype=np.float32)
                    distance = float(
                        np.linalg.norm(np.array(item.center, dtype=np.float32) - recent)
                    )
                    if distance < best_distance:
                        best_index = index
                        best_distance = distance
                if best_index is None:
                    tracks[color].append([item])
                    used_tracks.add(len(tracks[color]) - 1)
                else:
                    tracks[color][best_index].append(item)
                    used_tracks.add(best_index)
        time.sleep(0.03)

    if last_frame is None:
        raise RuntimeError("Camera opened but did not return a valid frame.")

    stable: List[Detection] = []
    for color in colors:
        for track_index, items in enumerate(tracks[color], start=1):
            if len(items) < min_hits:
                continue
            centers = np.array([item.center for item in items], dtype=np.float32)
            median = np.median(centers, axis=0)
            distances = np.linalg.norm(centers - median, axis=1)
            jitter80 = float(np.percentile(distances, 80))
            if jitter80 > max_jitter_px:
                print(
                    "Ignore unstable %s track %d: hits=%d, jitter80=%.1fpx"
                    % (color, track_index, len(items), jitter80)
                )
                continue
            representative = min(
                items,
                key=lambda item: float(np.linalg.norm(np.array(item.center) - median)),
            )
            center = (int(round(median[0])), int(round(median[1])))
            stable.append(
                Detection(
                    color=color,
                    center=center,
                    area=float(np.median([item.area for item in items])),
                    box=representative.box,
                    white_ring_ratio=float(
                        np.median([item.white_ring_ratio for item in items])
                    ),
                    body_edge_ratio=float(
                        np.median([item.body_edge_ratio for item in items])
                    ),
                    grasp_center=center,
                )
            )
    return last_frame, stable


def cluster_mapped_candidates(
    items: Sequence[MappedBlock], merge_distance_m: float
) -> List[List[MappedBlock]]:
    """Associate observations of multiple same-colour blocks across scan views."""

    clusters: List[List[MappedBlock]] = []
    ordered = sorted(
        items,
        key=lambda item: (
            item.commanded_scan_angle if item.commanded_scan_angle is not None else 999,
            item.workspace[0],
            item.workspace[1],
        ),
    )
    for item in ordered:
        best_cluster = None
        best_distance = merge_distance_m
        for cluster in clusters:
            commands = {candidate.commanded_scan_angle for candidate in cluster}
            if item.commanded_scan_angle in commands:
                # One physical block can contribute at most once per frame/view.
                continue
            centroid = np.mean(
                np.array([candidate.workspace for candidate in cluster]), axis=0
            )
            distance = float(
                np.linalg.norm(np.array(item.workspace, dtype=float) - centroid)
            )
            if distance < best_distance:
                best_cluster = cluster
                best_distance = distance
        if best_cluster is None:
            clusters.append([item])
        else:
            best_cluster.append(item)
    return clusters


def anchor_view_sort_key(item: MappedBlock) -> Tuple[float, float, float]:
    """Prefer views near the optical centre with stronger non-marker evidence."""

    return (
        abs(item.detection.center[0] - 320),
        item.detection.white_ring_ratio,
        -item.detection.area,
    )


def candidate_workspace_weight(item: MappedBlock, config: dict) -> float:
    """Score one observation for fusion; higher means more trustworthy."""

    fusion = config.get("vision", {}).get("fusion", {})
    center_scale_px = max(1.0, float(fusion.get("center_scale_px", 120.0)))
    area_reference_px = max(1.0, float(fusion.get("area_reference_px", 3000.0)))
    area_exponent = float(fusion.get("area_exponent", 0.5))
    min_ring_factor = max(0.05, float(fusion.get("min_ring_factor", 0.25)))
    edge_reference = max(1e-6, float(fusion.get("edge_reference", 0.08)))
    min_edge_factor = float(fusion.get("min_edge_factor", 0.75))
    max_edge_factor = max(
        min_edge_factor, float(fusion.get("max_edge_factor", 1.5))
    )

    center_distance = abs(item.detection.center[0] - 320.0)
    center_factor = 1.0 / (1.0 + center_distance / center_scale_px)

    area_ratio = max(item.detection.area, 1.0) / area_reference_px
    area_factor = min(2.5, max(0.35, area_ratio ** area_exponent))

    ring_factor = max(min_ring_factor, 1.0 - item.detection.white_ring_ratio)

    edge_factor = item.detection.body_edge_ratio / edge_reference
    edge_factor = min(max_edge_factor, max(min_edge_factor, edge_factor))

    return center_factor * area_factor * ring_factor * edge_factor


def distance_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _rule_matches_workspace(
    rule: WorkspaceCorrectionRule, item: MappedBlock, workspace: Tuple[float, float]
) -> bool:
    if not rule.enabled:
        return False
    if rule.color is not None and item.detection.color != rule.color:
        return False
    x, y = workspace
    if rule.x_min is not None and x < float(rule.x_min):
        return False
    if rule.x_max is not None and x > float(rule.x_max):
        return False
    if rule.y_min is not None and y < float(rule.y_min):
        return False
    if rule.y_max is not None and y > float(rule.y_max):
        return False
    if rule.radius_min is not None or rule.radius_max is not None:
        radius = distance_m(
            workspace, (float(rule.radius_origin_x), float(rule.radius_origin_y))
        )
        if rule.radius_min is not None and radius < float(rule.radius_min):
            return False
        if rule.radius_max is not None and radius > float(rule.radius_max):
            return False
    if rule.scan_angle_min is not None and item.scan_angle < float(rule.scan_angle_min):
        return False
    if rule.scan_angle_max is not None and item.scan_angle > float(rule.scan_angle_max):
        return False
    return True


def apply_workspace_corrections(
    item: MappedBlock, rules: Sequence[WorkspaceCorrectionRule]
) -> MappedBlock:
    raw_workspace = item.workspace
    corrected = raw_workspace
    applied_rules: List[str] = []
    correction_note: List[str] = []
    total_dx = 0.0
    total_dy = 0.0

    for rule in rules:
        if not _rule_matches_workspace(rule, item, raw_workspace):
            continue
        next_x = rule.pivot_x + (corrected[0] - rule.pivot_x) * rule.scale_x + rule.dx_m
        next_y = (
            rule.pivot_y
            + (corrected[1] - rule.pivot_y) * rule.scale_y
            + rule.dy_m
            + (corrected[0] - rule.pivot_x) * rule.y_shear_m_per_m
        )
        total_dx += next_x - corrected[0]
        total_dy += next_y - corrected[1]
        corrected = (round(next_x, 5), round(next_y, 5))
        applied_rules.append(rule.rule_id)
        if rule.note:
            correction_note.append(rule.note)

    return replace(
        item,
        raw_workspace=raw_workspace,
        workspace=corrected,
        workspace_correction_dx_m=round(total_dx, 5),
        workspace_correction_dy_m=round(total_dy, 5),
        workspace_correction_rule=",".join(applied_rules),
        workspace_correction_note="; ".join(correction_note),
    )


def mapped_conflict_score(item: MappedBlock, config: dict) -> float:
    """Score one fused target for cross-colour conflict suppression."""

    return candidate_workspace_weight(item, config) * max(
        1.0, float(item.supporting_views)
    )


def suppress_cross_color_conflicts(
    selected_with_views: Sequence[Tuple[MappedBlock, List[MappedBlock], List[MappedBlock]]],
    config: dict,
) -> Tuple[
    List[Tuple[MappedBlock, List[MappedBlock], List[MappedBlock]]],
    List[dict],
]:
    vision = config.get("vision", {})
    conflict_distance = float(vision.get("cross_color_conflict_distance_m", 0.0))
    if conflict_distance <= 0.0:
        return list(selected_with_views), []
    max_area_ratio = float(vision.get("cross_color_conflict_max_area_ratio", 0.35))
    max_score_ratio = float(vision.get("cross_color_conflict_max_score_ratio", 0.65))

    suppressed_indexes = set()
    suppressed_records: List[dict] = []

    for left_index, (left_item, _, _) in enumerate(selected_with_views):
        if left_index in suppressed_indexes:
            continue
        for right_index in range(left_index + 1, len(selected_with_views)):
            if right_index in suppressed_indexes:
                continue
            right_item = selected_with_views[right_index][0]
            if left_item.detection.color == right_item.detection.color:
                continue
            distance = distance_m(left_item.raw_workspace, right_item.raw_workspace)
            if distance > conflict_distance:
                continue

            left_score = mapped_conflict_score(left_item, config)
            right_score = mapped_conflict_score(right_item, config)
            left_key = (
                left_score,
                left_item.detection.area,
                left_item.supporting_views,
            )
            right_key = (
                right_score,
                right_item.detection.area,
                right_item.supporting_views,
            )
            if left_key >= right_key:
                winner_index, loser_index = left_index, right_index
                winner_item, loser_item = left_item, right_item
                winner_score, loser_score = left_score, right_score
            else:
                winner_index, loser_index = right_index, left_index
                winner_item, loser_item = right_item, left_item
                winner_score, loser_score = right_score, left_score

            area_ratio = min(left_item.detection.area, right_item.detection.area) / max(
                left_item.detection.area, right_item.detection.area, 1.0
            )
            score_ratio = min(left_score, right_score) / max(
                left_score, right_score, 1e-6
            )
            if area_ratio > max_area_ratio and score_ratio > max_score_ratio:
                continue

            suppressed_indexes.add(loser_index)
            suppressed_records.append(
                {
                    "target_id": loser_item.target_id,
                    "color": loser_item.detection.color,
                    "raw_workspace_x": loser_item.raw_workspace[0],
                    "raw_workspace_y": loser_item.raw_workspace[1],
                    "workspace_x": loser_item.workspace[0],
                    "workspace_y": loser_item.workspace[1],
                    "area": round(loser_item.detection.area, 1),
                    "supporting_views": loser_item.supporting_views,
                    "suppressed_by_target_id": winner_item.target_id,
                    "suppressed_by_color": winner_item.detection.color,
                    "distance_to_winner_m": round(distance, 5),
                    "area_ratio": round(area_ratio, 3),
                    "score_ratio": round(score_ratio, 3),
                }
            )

    kept = [
        entry
        for index, entry in enumerate(selected_with_views)
        if index not in suppressed_indexes
    ]
    return kept, suppressed_records


def weighted_workspace_average(
    items: Sequence[MappedBlock], config: dict
) -> Tuple[float, float]:
    points = np.array([item.workspace for item in items], dtype=np.float64)
    weights = np.array(
        [candidate_workspace_weight(item, config) for item in items], dtype=np.float64
    )
    total = float(weights.sum())
    if total <= 0.0:
        averaged = points.mean(axis=0)
    else:
        averaged = np.average(points, axis=0, weights=weights)
    return round(float(averaged[0]), 5), round(float(averaged[1]), 5)


def fuse_cluster_observations(
    items: Sequence[MappedBlock], config: dict, merge_distance_m: float
) -> Tuple[MappedBlock, List[MappedBlock]]:
    """Fuse one object cluster into a more stable global workspace coordinate."""

    if not items:
        raise ValueError("Cannot fuse an empty cluster.")

    if len(items) == 1:
        single = replace(
            items[0],
            coordinate_spread_m=0.0,
            raw_coordinate_spread_m=0.0,
            supporting_views=1,
            raw_supporting_views=1,
            outlier_views=0,
            fusion_method="single_view",
            raw_workspace=items[0].workspace,
        )
        return single, [items[0]]

    fusion = config.get("vision", {}).get("fusion", {})
    inlier_radius_m = float(
        fusion.get("inlier_radius_m", min(merge_distance_m * 0.5, 0.015))
    )
    minimum_inlier_views = int(fusion.get("minimum_inlier_views", 2))

    support_scores: List[float] = []
    for centre in items:
        score = 0.0
        for candidate in items:
            if distance_m(centre.workspace, candidate.workspace) <= inlier_radius_m:
                score += candidate_workspace_weight(candidate, config)
        support_scores.append(score)

    consensus_index = int(np.argmax(np.array(support_scores, dtype=np.float64)))
    consensus_item = items[consensus_index]
    inliers = [
        item
        for item in items
        if distance_m(consensus_item.workspace, item.workspace) <= inlier_radius_m
    ]
    if len(inliers) < minimum_inlier_views:
        inliers = list(items)

    fused_workspace = weighted_workspace_average(inliers, config)
    anchor = min(inliers, key=anchor_view_sort_key)

    inlier_spread = max(
        distance_m(item.workspace, fused_workspace) for item in inliers
    )
    raw_spread = max(distance_m(item.workspace, fused_workspace) for item in items)

    fused = replace(
        anchor,
        workspace=fused_workspace,
        coordinate_spread_m=round(inlier_spread, 5),
        raw_coordinate_spread_m=round(raw_spread, 5),
        supporting_views=len(inliers),
        raw_supporting_views=len(items),
        outlier_views=len(items) - len(inliers),
        fusion_method="weighted_inlier_mean",
        raw_workspace=fused_workspace,
    )
    return fused, inliers


def scan_workspace(
    cap: cv2.VideoCapture,
    arm: "DofbotArm",
    detector: ColorBlockDetector,
    config: dict,
    feedback_rules: Sequence[WorkspaceCorrectionRule],
    feedback_config: Path,
    colors: Sequence[str],
    sample_frames: int,
    min_hits: int,
    max_jitter_px: float,
    scan_dir: Path,
    map_output: Path,
) -> List[MappedBlock]:
    """Run a coarse sweep, re-centre targets, and build a global target map."""

    scan_dir.mkdir(parents=True, exist_ok=True)
    candidates: Dict[str, List[MappedBlock]] = {color: [] for color in colors}
    scan_poses = config["arm"]["scan_poses"]
    settle_seconds = float(config["arm"]["scan_settle_seconds"])
    merge_distance_m = float(config["vision"]["map_merge_distance_m"])
    captured_angles = set()

    def object_clusters(color: str) -> List[List[MappedBlock]]:
        return cluster_mapped_candidates(candidates[color], merge_distance_m)

    def capture_view(angle: int, pose: Sequence[float], phase: str) -> None:
        if angle in captured_angles:
            return
        captured_angles.add(angle)
        print("%s scan view %d degrees: pose=%r" % (phase, angle, list(pose)))
        arm.move(pose, 1000)
        time.sleep(settle_seconds)
        measured_angle = arm.read_joint(1)
        map_angle = float(measured_angle if measured_angle is not None else angle)
        if measured_angle is not None and measured_angle != angle:
            print(
                "  base feedback: commanded=%d actual=%d" % (angle, measured_angle)
            )
        frame, detections = capture_stable(
            cap,
            detector,
            colors,
            sample_frames,
            min_hits,
            max_jitter_px,
        )
        raw_snapshot = scan_dir / ("raw_%03d.jpg" % angle)
        snapshot = scan_dir / ("scan_%03d.jpg" % angle)
        cv2.imwrite(str(raw_snapshot), cv2.resize(frame, (640, 480)))
        cv2.imwrite(str(snapshot), detector.annotate(frame, detections))
        for item in detections:
            assert item.grasp_center is not None
            mapped = MappedBlock(
                detection=item,
                workspace=pixel_to_global_workspace(item.grasp_center, map_angle),
                scan_angle=map_angle,
                snapshot=str(snapshot),
                commanded_scan_angle=angle,
            )
            candidates[item.color].append(mapped)
            print(
                "  candidate %-6s top/grasp=%r global=%r angle=%.1f white_ring=%.2f"
                % (
                    item.color,
                    item.grasp_center,
                    mapped.workspace,
                    mapped.scan_angle,
                    item.white_ring_ratio,
                )
            )

    try:
        # First locate targets over a wide field of view.
        for view in scan_poses:
            angle = int(view["angle"])
            capture_view(angle, view["pose"], "Coarse")

        # A wide sweep often leaves a target near an image edge.  Re-scan close
        # to its best coarse angle so at least one measurement is near the
        # optical centre, where the vendor calibration is most accurate.
        refine_angles = set()
        offsets = [int(value) for value in config["arm"]["refine_scan_offsets"]]
        min_angle = int(config["arm"]["refine_min_angle"])
        max_angle = int(config["arm"]["refine_max_angle"])
        for color in colors:
            for cluster in object_clusters(color):
                source = min(
                    cluster, key=lambda item: abs(item.detection.center[0] - 320)
                )
                source_command = int(
                    source.commanded_scan_angle or round(source.scan_angle)
                )
                for offset in offsets:
                    angle = source_command + offset
                    if min_angle <= angle <= max_angle:
                        refine_angles.add(angle)

        template_pose = list(scan_poses[0]["pose"])
        for angle in sorted(refine_angles):
            pose = list(template_pose)
            pose[0] = angle
            capture_view(angle, pose, "Refine")

        # If two observations bracket the optical centre, interpolate the base
        # command that should put the target at x=320 and capture one final view.
        centre_angles = set()
        for color in colors:
            for cluster in object_clusters(color):
                items = sorted(cluster, key=lambda item: item.detection.center[0])
                for left, right in zip(items, items[1:]):
                    x0 = left.detection.center[0]
                    x1 = right.detection.center[0]
                    if not (x0 <= 320 <= x1) or x1 == x0:
                        continue
                    a0 = float(left.commanded_scan_angle or left.scan_angle)
                    a1 = float(right.commanded_scan_angle or right.scan_angle)
                    estimate = a0 + (320.0 - x0) * (a1 - a0) / float(x1 - x0)
                    # Servo 1 accepts integer degrees. Capture both sides of a
                    # fractional estimate and let the measured pixel choose.
                    for angle in {
                        int(math.floor(estimate)),
                        int(math.ceil(estimate)),
                    }:
                        if min_angle <= angle <= max_angle:
                            centre_angles.add(angle)
                    break
        for angle in sorted(centre_angles):
            pose = list(template_pose)
            pose[0] = angle
            capture_view(angle, pose, "Centred")
    finally:
        # Always leave the camera and arm at the calibrated 90-degree map view.
        arm.move(config["arm"]["poses"]["observe"], 1000)

    selected_with_views: List[Tuple[MappedBlock, List[MappedBlock], List[MappedBlock]]] = []
    for color in colors:
        clusters = object_clusters(color)
        clusters.sort(
            key=lambda cluster: tuple(
                np.mean(np.array([item.workspace for item in cluster]), axis=0)
            )
        )
        for object_index, items in enumerate(clusters, start=1):
            fused, inliers = fuse_cluster_observations(items, config, merge_distance_m)
            fused.target_id = "%s-%02d" % (color, object_index)
            selected_with_views.append(
                (apply_workspace_corrections(fused, feedback_rules), items, inliers)
            )

    selected_with_views, suppressed_blocks = suppress_cross_color_conflicts(
        selected_with_views, config
    )
    selected = [item for item, _, _ in selected_with_views]

    data = {
        "coordinate_frame": "base_90_global_xy_meters",
        "coordinate_source": (
            "vendor_top_contour_center_weighted_inlier_mean+feedback_compensation"
            if feedback_rules
            else "vendor_top_contour_center_weighted_inlier_mean"
        ),
        "feedback_compensation_file": str(feedback_config),
        "feedback_compensation_rules": [rule.rule_id for rule in feedback_rules],
        "scan_angles": sorted(captured_angles),
        "suppressed_blocks": suppressed_blocks,
        "blocks": [
            {
                "target_id": item.target_id,
                "color": item.detection.color,
                "raw_workspace_x": item.raw_workspace[0],
                "raw_workspace_y": item.raw_workspace[1],
                "workspace_x": item.workspace[0],
                "workspace_y": item.workspace[1],
                "scan_angle": item.scan_angle,
                "commanded_scan_angle": item.commanded_scan_angle,
                "pixel_center": list(item.detection.center),
                "grasp_pixel_center": list(item.detection.grasp_center),
                "area": round(item.detection.area, 1),
                "white_ring_ratio": round(item.detection.white_ring_ratio, 3),
                "body_edge_ratio": round(item.detection.body_edge_ratio, 3),
                "coordinate_spread_m": round(item.coordinate_spread_m, 5),
                "raw_coordinate_spread_m": round(item.raw_coordinate_spread_m, 5),
                "supporting_views": item.supporting_views,
                "raw_supporting_views": item.raw_supporting_views,
                "outlier_views": item.outlier_views,
                "fusion_method": item.fusion_method,
                "workspace_correction_dx_m": round(item.workspace_correction_dx_m, 5),
                "workspace_correction_dy_m": round(item.workspace_correction_dy_m, 5),
                "workspace_correction_rule": item.workspace_correction_rule,
                "workspace_correction_note": item.workspace_correction_note,
                "anchor_snapshot": item.snapshot,
                "view_candidates": [
                    {
                        "scan_angle": candidate.scan_angle,
                        "commanded_scan_angle": candidate.commanded_scan_angle,
                        "pixel_center": list(candidate.detection.center),
                        "workspace": list(candidate.workspace),
                        "weight": round(candidate_workspace_weight(candidate, config), 4),
                        "inlier": any(candidate is inlier for inlier in inliers),
                    }
                    for candidate in cluster
                ],
                "snapshot": item.snapshot,
            }
            for item, cluster, inliers in selected_with_views
        ],
    }
    map_output.parent.mkdir(parents=True, exist_ok=True)
    with map_output.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    print("Scan map saved: %s" % map_output)
    if suppressed_blocks:
        print("Suppressed %d conflicting target(s)." % len(suppressed_blocks))
        for entry in suppressed_blocks:
            print(
                "  suppress %-10s near %-10s distance=%.1fmm area_ratio=%.2f score_ratio=%.2f"
                % (
                    entry["target_id"],
                    entry["suppressed_by_target_id"],
                    entry["distance_to_winner_m"] * 1000.0,
                    entry["area_ratio"],
                    entry["score_ratio"],
                )
            )
    return selected


class DofbotArm:
    def __init__(self, config: dict, ik_service: str, enable_ik: bool):
        try:
            from Arm_Lib import Arm_Device
        except ImportError as exc:
            raise RuntimeError("Arm_Lib is unavailable.") from exc

        self.arm = Arm_Device()
        self.request_type = None
        self.ik_client = None
        self.poses = config["arm"]["poses"]
        self.open_angle = int(config["arm"]["gripper_open"])
        self.close_angle = int(config["arm"]["gripper_close"])
        self.open_time = int(config["arm"]["gripper_open_time_ms"])
        self.close_time = int(config["arm"]["gripper_close_time_ms"])
        self.move_time = int(config["arm"]["move_time_ms"])
        self.pick_descend_time = int(config["arm"]["pick_descend_time_ms"])
        self.pick_correction_time = int(config["arm"]["pick_correction_time_ms"])
        self.pick_settle_seconds = float(config["arm"]["pick_settle_seconds"])
        self.pick_joint_tolerance = float(config["arm"]["pick_joint_tolerance_deg"])
        self.pick_joint_abort = float(config["arm"]["pick_joint_abort_deg"])
        if enable_ik:
            try:
                import rospy
                from arm_info.srv import kinemarics, kinemaricsRequest
            except ImportError as exc:
                raise RuntimeError(
                    "ROS modules are unavailable. Source /opt/ros/noetic/setup.bash and "
                    "~/catkin_ws/devel/setup.bash before using --execute."
                ) from exc
            if not rospy.core.is_initialized():
                rospy.init_node(
                    "ai_visual_pick_stack", anonymous=True, disable_signals=True
                )
            self.request_type = kinemaricsRequest
            self.ik_client = rospy.ServiceProxy(ik_service, kinemarics)
            self.ik_client.wait_for_service(timeout=5.0)
        time.sleep(0.1)

    def clamp(self, close: bool) -> None:
        angle = self.close_angle if close else self.open_angle
        duration = self.close_time if close else self.open_time
        self.arm.Arm_serial_servo_write(6, angle, duration)
        time.sleep(duration / 1000.0 + 0.2)
        if close:
            feedback = self.read_joint(6)
            print("Gripper close feedback: commanded=%d actual=%r" % (angle, feedback))

    def move(self, pose: Sequence[float], move_time_ms: Optional[int] = None) -> None:
        self.validate_pose(pose)
        duration = int(move_time_ms or self.move_time)
        for index, raw_angle in enumerate(pose):
            servo_id = index + 1
            angle = int(round(raw_angle))
            if servo_id == 5:
                time.sleep(0.1)
                servo_time = int(duration * 1.2)
            elif servo_id == 1:
                servo_time = int(duration * 0.75)
            else:
                servo_time = duration
            self.arm.Arm_serial_servo_write(servo_id, angle, servo_time)
            time.sleep(0.01)
        time.sleep(duration / 1000.0)

    def read_joint(self, servo_id: int, attempts: int = 3) -> Optional[int]:
        for _ in range(attempts):
            value = self.arm.Arm_serial_servo_read(servo_id)
            if value is not None:
                return int(value)
            time.sleep(0.03)
        return None

    def read_pose(self) -> Optional[List[int]]:
        pose = [self.read_joint(servo_id) for servo_id in range(1, 6)]
        if any(value is None for value in pose):
            return None
        return [int(value) for value in pose if value is not None]

    def move_verified(self, pose: Sequence[float]) -> None:
        """Move to the pickup pose and correct servo backlash once."""

        target = [int(round(value)) for value in pose]
        self.move(target, self.pick_descend_time)
        actual = self.read_pose()
        if actual is None:
            print("Pickup pose feedback unavailable; continuing after timed settle.")
            time.sleep(self.pick_settle_seconds)
            return
        errors = [abs(actual[i] - target[i]) for i in range(5)]
        print("Pickup pose feedback target=%r actual=%r error=%r" % (target, actual, errors))
        if max(errors) > self.pick_joint_tolerance:
            print("Correcting pickup pose once to reduce servo backlash.")
            self.move(target, self.pick_correction_time)
            actual = self.read_pose()
            if actual is not None:
                errors = [abs(actual[i] - target[i]) for i in range(5)]
                print("Corrected pickup feedback actual=%r error=%r" % (actual, errors))
        if max(errors) > self.pick_joint_abort:
            raise RuntimeError(
                "Pickup pose did not converge; gripper was not closed: "
                "target=%r actual=%r error=%r" % (target, actual, errors)
            )
        time.sleep(self.pick_settle_seconds)

    @staticmethod
    def validate_pose(pose: Sequence[float]) -> None:
        if len(pose) != 5:
            raise ValueError("A pose must contain servo 1-5 angles.")
        for index, raw_angle in enumerate(pose):
            servo_id = index + 1
            angle = int(round(raw_angle))
            maximum = 270 if servo_id == 5 else 180
            if not 0 <= angle <= maximum:
                raise ValueError(
                    "Unsafe servo angle: id=%d angle=%d range=0..%d"
                    % (servo_id, angle, maximum)
                )

    def inverse_kinematics_xy(
        self, workspace: Tuple[float, float], source: str = ""
    ) -> List[int]:
        if self.request_type is None or self.ik_client is None:
            raise RuntimeError("Inverse kinematics was not initialized.")
        x, y = workspace
        request = self.request_type()
        request.tar_x = x
        request.tar_y = y
        request.kin_name = "ik"
        response = self.ik_client.call(request)
        joints = [
            float(response.joint1),
            float(response.joint2),
            float(response.joint3),
            float(response.joint4),
        ]
        if not all(np.isfinite(joints)):
            raise RuntimeError("IK returned a non-finite joint value for %r" % (workspace,))
        if joints[2] < 0:
            joints[1] += joints[2] * 3.0 / 5.0
            joints[3] += joints[2] * 3.0 / 5.0
            joints[2] = 0.0
        pose = [int(round(value)) for value in joints] + [270]
        # Validate before the arm starts moving toward the object.
        for index, angle in enumerate(pose, start=1):
            maximum = 270 if index == 5 else 180
            if not 0 <= angle <= maximum:
                raise RuntimeError("IK pose outside servo limits: %r" % pose)
        print(
            "IK %s workspace=(%.5f, %.5f) pose=%r" % (source, x, y, pose)
        )
        return pose

    def inverse_kinematics(self, center: Tuple[int, int]) -> List[int]:
        return self.inverse_kinematics_xy(
            pixel_to_workspace(center), source="pixel=%r" % (center,)
        )

    def observe(self, open_gripper: bool = True) -> None:
        if open_gripper:
            self.clamp(False)
        self.move(self.poses["observe"], 1000)
        time.sleep(1.0)

    def pick_and_place(self, pick_pose: Sequence[int], layer_index: int) -> None:
        layers = self.poses["layers"]
        if not 0 <= layer_index < len(layers):
            raise ValueError("No configured target layer %d" % (layer_index + 1))
        self.move(self.poses["top"], 1000)
        # Align the base while the gripper is high, then descend without a
        # simultaneous large base sweep.  This reduces lateral overshoot.
        approach_pose = [pick_pose[0], 80, 50, 50, pick_pose[4]]
        self.move(approach_pose, 900)
        self.clamp(False)
        self.move_verified(pick_pose)
        self.clamp(True)
        self.move(approach_pose, 1000)
        self.move(self.poses["top"], 1000)
        self.move(layers[layer_index], 1000)
        self.clamp(False)
        time.sleep(0.1)
        self.move(self.poses["observe"], 1100)

    def close(self) -> None:
        del self.arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize colored blocks with OpenCV and stack them with DOFBOT."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=Path, help="Detect from an image instead of a camera.")
    parser.add_argument(
        "--colors",
        default="yellow,red,green,blue",
        help="Recognition and stacking order, comma-separated.",
    )
    parser.add_argument("--sample-frames", type=int, default=20)
    parser.add_argument("--min-hits", type=int, default=12)
    parser.add_argument("--max-jitter", type=float, default=12.0)
    parser.add_argument(
        "--snapshot", type=Path, default=ROOT / "last_detection.jpg"
    )
    parser.add_argument("--scan-dir", type=Path, default=ROOT / "scan")
    parser.add_argument("--map-output", type=Path, default=ROOT / "scan_map.json")
    parser.add_argument(
        "--feedback-config",
        type=Path,
        default=ROOT / "feedback_compensation.json",
        help="Optional workspace correction rules used after multi-view fusion.",
    )
    parser.add_argument(
        "--scan-map",
        action="store_true",
        help="Move the arm-mounted camera through all views and build a map; do not grip.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Scan the map, then enable real pick-and-stack motion.",
    )
    parser.add_argument("--ik-service", default="/get_kinemarics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    feedback_rules = load_workspace_corrections(args.feedback_config)
    colors = [item.strip().lower() for item in args.colors.split(",") if item.strip()]
    supported = set(config["vision"]["hsv_ranges"])
    unknown = [color for color in colors if color not in supported]
    if unknown:
        raise ValueError("Unknown colors: %s" % ", ".join(unknown))
    if (args.execute or args.scan_map) and args.image:
        raise ValueError("--scan-map/--execute cannot be combined with --image.")
    if args.sample_frames < 1 or args.min_hits < 1 or args.min_hits > args.sample_frames:
        raise ValueError("Require 1 <= min-hits <= sample-frames.")

    detector = ColorBlockDetector(config)
    arm: Optional[DofbotArm] = None
    cap: Optional[cv2.VideoCapture] = None
    scan_enabled = bool(args.scan_map or args.execute)
    try:
        if scan_enabled:
            arm = DofbotArm(
                config,
                args.ik_service,
                enable_ik=args.execute,
            )
            arm.observe(open_gripper=args.execute)

        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise RuntimeError("Cannot read image: %s" % args.image)
            detections = detector.detect(frame, colors)
            mapped_blocks: List[MappedBlock] = []
        else:
            cap = open_camera(args.camera)
            if scan_enabled:
                assert arm is not None
                mapped_blocks = scan_workspace(
                    cap,
                    arm,
                    detector,
                    config,
                    feedback_rules,
                    args.feedback_config,
                    colors,
                    args.sample_frames,
                    args.min_hits,
                    args.max_jitter,
                    args.scan_dir,
                    args.map_output,
                )
                detections = [item.detection for item in mapped_blocks]
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                mapped_blocks = []
                frame, detections = capture_stable(
                    cap,
                    detector,
                    colors,
                    args.sample_frames,
                    args.min_hits,
                    args.max_jitter,
                )

        if not scan_enabled:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            annotated = detector.annotate(frame, detections)
            if not cv2.imwrite(str(args.snapshot), annotated):
                raise RuntimeError("Failed to write snapshot: %s" % args.snapshot)

        color_rank = {color: index for index, color in enumerate(colors)}
        ordered = sorted(
            detections,
            key=lambda item: (
                color_rank[item.color],
                item.center[1],
                item.center[0],
            ),
        )
        if scan_enabled:
            ordered_mapped = sorted(
                mapped_blocks,
                key=lambda item: (
                    color_rank[item.detection.color],
                    item.workspace[1],
                    item.workspace[0],
                ),
            )
            ordered = [item.detection for item in ordered_mapped]
            print("Mapped %d grippable block(s)." % len(ordered))
            for mapped in ordered_mapped:
                item = mapped.detection
                print(
                    "  %-10s global=%r raw=%r anchor_scan=%.1f pixel=%r area=%.0f"
                    % (
                        mapped.target_id,
                        mapped.workspace,
                        mapped.raw_workspace,
                        mapped.scan_angle,
                        item.center,
                        item.area,
                    )
                )
                print(
                    "         views=%d/%d outliers=%d coordinate_spread=%.1fmm raw_spread=%.1fmm correction=(%.1fmm,%.1fmm) rule=%s"
                    % (
                        mapped.supporting_views,
                        mapped.raw_supporting_views,
                        mapped.outlier_views,
                        mapped.coordinate_spread_m * 1000.0,
                        mapped.raw_coordinate_spread_m * 1000.0,
                        mapped.workspace_correction_dx_m * 1000.0,
                        mapped.workspace_correction_dy_m * 1000.0,
                        mapped.workspace_correction_rule or "-",
                    )
                )
        else:
            print(
                "Detected %d block(s); snapshot: %s" % (len(ordered), args.snapshot)
            )
            for item in ordered:
                print(
                    "  %-6s pixel=%r workspace=%r area=%.0f"
                    % (
                        item.color,
                        item.center,
                        pixel_to_workspace(item.center),
                        item.area,
                    )
                )

        if not args.execute:
            if scan_enabled:
                print("Map-only mode: camera servos moved; gripper was not operated.")
            else:
                print(
                    "Detection-only mode: arm was not moved. Add --scan-map to scan."
                )
            return 0 if ordered else 2
        if not ordered:
            print("No stable colored block detected; no pickup motion executed.")
            return 2

        assert arm is not None
        if len(ordered_mapped) > len(config["arm"]["poses"]["layers"]):
            raise RuntimeError(
                "Detected %d blocks but only %d stack layers are configured; "
                "pickup was not started."
                % (len(ordered_mapped), len(config["arm"]["poses"]["layers"]))
            )
        maximum_spread = float(config["arm"]["max_coordinate_spread_m"])
        minimum_views = int(config["arm"]["min_execute_supporting_views"])
        unsafe = [
            item
            for item in mapped_blocks
            if item.supporting_views < minimum_views
            or item.coordinate_spread_m > maximum_spread
        ]
        if unsafe:
            details = ", ".join(
                "%s(views=%d,spread=%.1fmm)"
                % (
                    item.target_id,
                    item.supporting_views,
                    item.coordinate_spread_m * 1000.0,
                )
                for item in unsafe
            )
            raise RuntimeError(
                "Coordinate validation failed; pickup was not started: %s" % details
            )
        # Compute and validate every IK pose before performing the first pickup.
        planned = [
            (
                mapped,
                arm.inverse_kinematics_xy(
                    mapped.workspace,
                    source="%s fused anchor=%.1f views=%d"
                    % (mapped.target_id, mapped.scan_angle, mapped.supporting_views),
                ),
            )
            for mapped in ordered_mapped
        ]
        for _, pick_pose in planned:
            arm.validate_pose(pick_pose)
        arm.arm.Arm_Buzzer_On(1)
        time.sleep(0.5)
        for layer_index, (mapped, pick_pose) in enumerate(planned):
            print(
                "Pick %s -> stack layer %d"
                % (mapped.target_id, layer_index + 1)
            )
            arm.pick_and_place(pick_pose, layer_index)
        print("Completed %d pick-and-place operation(s)." % len(planned))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; stopping after the current servo command.", file=sys.stderr)
        return 130
    finally:
        if cap is not None:
            cap.release()
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
