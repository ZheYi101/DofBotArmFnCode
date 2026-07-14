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
from dataclasses import dataclass
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


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


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


def scan_workspace(
    cap: cv2.VideoCapture,
    arm: "DofbotArm",
    detector: ColorBlockDetector,
    config: dict,
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

    selected_with_views: List[Tuple[MappedBlock, List[MappedBlock]]] = []
    for color in colors:
        clusters = object_clusters(color)
        clusters.sort(
            key=lambda cluster: tuple(
                np.mean(np.array([item.workspace for item in cluster]), axis=0)
            )
        )
        for object_index, items in enumerate(clusters, start=1):
            # Use the most horizontally centred view. The arm sweep changes x
            # strongly but leaves y mostly fixed, so x is the useful term.
            best = min(
                items,
                key=lambda item: (
                    abs(item.detection.center[0] - 320),
                    item.detection.white_ring_ratio,
                    -item.detection.area,
                ),
            )
            distances = [
                math.hypot(
                    item.workspace[0] - best.workspace[0],
                    item.workspace[1] - best.workspace[1],
                )
                for item in items
            ]
            best.coordinate_spread_m = max(distances, default=0.0)
            best.supporting_views = len(items)
            best.target_id = "%s-%02d" % (color, object_index)
            selected_with_views.append((best, items))

    selected = [item for item, _ in selected_with_views]

    data = {
        "coordinate_frame": "base_90_global_xy_meters",
        "coordinate_source": "vendor_top_contour_center_best_centered_view",
        "scan_angles": sorted(captured_angles),
        "blocks": [
            {
                "target_id": item.target_id,
                "color": item.detection.color,
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
                "supporting_views": item.supporting_views,
                "view_candidates": [
                    {
                        "scan_angle": candidate.scan_angle,
                        "commanded_scan_angle": candidate.commanded_scan_angle,
                        "pixel_center": list(candidate.detection.center),
                        "workspace": list(candidate.workspace),
                    }
                    for candidate in cluster
                ],
                "snapshot": item.snapshot,
            }
            for item, cluster in selected_with_views
        ],
    }
    map_output.parent.mkdir(parents=True, exist_ok=True)
    with map_output.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    print("Scan map saved: %s" % map_output)
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
                    "  %-10s global=%r from_scan=%.1f pixel=%r area=%.0f"
                    % (
                        mapped.target_id,
                        mapped.workspace,
                        mapped.scan_angle,
                        item.center,
                        item.area,
                    )
                )
                print(
                    "         views=%d coordinate_spread=%.1fmm"
                    % (mapped.supporting_views, mapped.coordinate_spread_m * 1000.0)
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
                    source="%s scan=%.1f"
                    % (mapped.target_id, mapped.scan_angle),
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
