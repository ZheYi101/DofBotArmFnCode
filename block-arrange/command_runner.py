#!/usr/bin/env python3
"""Arrange colored blocks with relative commands and undo history."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_STATE = ROOT / "scene_state.json"
DEFAULT_HISTORY = ROOT / "undo_history.json"
DEFAULT_AI_VISUAL_DIR = PROJECT_ROOT / "functions" / "ai-visual"

COLOR_ALIASES = {
    "红": "red",
    "红色": "red",
    "red": "red",
    "yellow": "yellow",
    "黄": "yellow",
    "黄色": "yellow",
    "green": "green",
    "绿": "green",
    "绿色": "green",
    "blue": "blue",
    "蓝": "blue",
    "蓝色": "blue",
}

RELATION_ALIASES = {
    "左": "left",
    "左边": "left",
    "left": "left",
    "右": "right",
    "右边": "right",
    "right": "right",
    "前": "front",
    "前面": "front",
    "front": "front",
    "后": "back",
    "后面": "back",
    "back": "back",
    "上": "above",
    "上面": "above",
    "above": "above",
}

COMMAND_RE = re.compile(
    r"把\s*(?P<source>红色|红|黄色|黄|绿色|绿|蓝色|蓝|red|yellow|green|blue)\s*方块\s*"
    r"放到\s*(?P<target>红色|红|黄色|黄|绿色|绿|蓝色|蓝|red|yellow|green|blue)\s*方块\s*"
    r"(?P<relation>左边|左|右边|右|前面|前|后面|后|上面|上|left|right|front|back|above)\s*"
    r"(?:(?P<distance>[0-9]+(?:\.[0-9]+)?)\s*cm)?\s*$",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_alias(raw: str, table: Dict[str, str], kind: str) -> str:
    key = raw.strip().lower()
    if key not in table:
        raise ValueError("Unsupported %s: %s" % (kind, raw))
    return table[key]


def load_visual_module(ai_visual_dir: Path):
    module_path = ai_visual_dir / "visual_pick_stack.py"
    if not module_path.exists():
        raise FileNotFoundError("Cannot find ai-visual runner: %s" % module_path)
    spec = importlib.util.spec_from_file_location("ai_visual_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load ai-visual module from %s" % module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def empty_state() -> dict:
    return {
        "version": 1,
        "updated_at": "",
        "coordinate_frame": "base_90_global_xy_meters",
        "blocks": [],
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return empty_state()
    return load_json(path)


def empty_history() -> dict:
    return {"version": 1, "updated_at": "", "entries": []}


def load_history(path: Path) -> dict:
    if not path.exists():
        return empty_history()
    return load_json(path)


def color_order(colors: Iterable[str]) -> Dict[str, int]:
    return {color: index for index, color in enumerate(colors)}


def get_block(state: dict, color: str) -> dict:
    for item in state.get("blocks", []):
        if item["color"] == color:
            return item
    raise ValueError("Block not found in scene state: %s" % color)


def blocks_in_stack(state: dict, stack_id: str) -> List[dict]:
    return sorted(
        [item for item in state.get("blocks", []) if item["stack_id"] == stack_id],
        key=lambda item: item["level"],
    )


def top_level(state: dict, stack_id: str) -> int:
    stack = blocks_in_stack(state, stack_id)
    if not stack:
        raise ValueError("Stack not found: %s" % stack_id)
    return int(stack[-1]["level"])


def top_block(state: dict, stack_id: str) -> dict:
    stack = blocks_in_stack(state, stack_id)
    if not stack:
        raise ValueError("Stack not found: %s" % stack_id)
    return stack[-1]


def ensure_topmost(state: dict, color: str) -> dict:
    block = get_block(state, color)
    stack = blocks_in_stack(state, block["stack_id"])
    if not stack or stack[-1]["color"] != color:
        raise RuntimeError(
            "Cannot move %s: it is not the top block of its stack." % color
        )
    return block


def normalize_colors(raw: str) -> List[str]:
    return [resolve_alias(item, COLOR_ALIASES, "color") for item in raw.split(",") if item.strip()]


def parse_command_text(text: str) -> dict:
    match = COMMAND_RE.match(text.strip())
    if not match:
        raise ValueError(
            "Unsupported command text. Example: 把红色方块放到蓝色方块左边3cm"
        )
    relation = resolve_alias(match.group("relation"), RELATION_ALIASES, "relation")
    distance_cm = match.group("distance")
    return {
        "source": resolve_alias(match.group("source"), COLOR_ALIASES, "color"),
        "target": resolve_alias(match.group("target"), COLOR_ALIASES, "color"),
        "relation": relation,
        "distance_cm": float(distance_cm) if distance_cm else None,
        "command_text": text.strip(),
    }


def resolve_move_request(args: argparse.Namespace, config: dict) -> dict:
    if args.command:
        request = parse_command_text(args.command)
    else:
        if not args.source or not args.target or not args.relation:
            raise ValueError(
                "Move requires --command or the full structured form: "
                "--source/--target/--relation."
            )
        request = {
            "source": resolve_alias(args.source, COLOR_ALIASES, "color"),
            "target": resolve_alias(args.target, COLOR_ALIASES, "color"),
            "relation": resolve_alias(args.relation, RELATION_ALIASES, "relation"),
            "distance_cm": args.distance_cm,
            "command_text": "",
        }
    if request["source"] == request["target"]:
        raise ValueError("Source and target blocks must be different.")
    if request["relation"] != "above" and request["distance_cm"] is None:
        request["distance_cm"] = float(config["scene"]["default_distance_cm"])
    if request["relation"] == "above":
        request["distance_cm"] = None
    return request


def xy_to_polar(workspace: Tuple[float, float]) -> Tuple[float, float]:
    x, y = float(workspace[0]), float(workspace[1])
    return math.hypot(x, y), math.atan2(x, y)


def polar_to_xy(radius: float, theta: float) -> Tuple[float, float]:
    return round(radius * math.sin(theta), 5), round(radius * math.cos(theta), 5)


def offset_polar(
    workspace: Tuple[float, float], radial_m: float = 0.0, tangent_m: float = 0.0
) -> Tuple[float, float]:
    radius, theta = xy_to_polar(workspace)
    new_radius = radius + radial_m
    if new_radius <= 0.0:
        raise ValueError("Polar offset would move target behind the arm base.")
    new_theta = theta + (tangent_m / max(radius, 1e-6))
    return polar_to_xy(new_radius, new_theta)


def parse_level_polar_offsets(motion_config: dict) -> Dict[int, Tuple[float, float]]:
    raw_offsets = motion_config.get("level_polar_offsets")
    if raw_offsets is None:
        raw_offsets = motion_config.get("level_workspace_offsets_m", {"1": [0.0, 0.0]})

    parsed: Dict[int, Tuple[float, float]] = {}
    for level, rule in raw_offsets.items():
        if isinstance(rule, dict):
            radial_m = float(rule.get("radial_m", 0.0))
            tangent_m = float(rule.get("tangent_m", 0.0))
        else:
            radial_m = float(rule[0])
            tangent_m = float(rule[1])
        parsed[int(level)] = (radial_m, tangent_m)
    parsed.setdefault(1, (0.0, 0.0))
    return parsed


def relation_to_polar_offset(relation: str, distance_m: float) -> Tuple[float, float]:
    if relation == "left":
        return 0.0, -distance_m
    if relation == "right":
        return 0.0, distance_m
    if relation == "front":
        return distance_m, 0.0
    if relation == "back":
        return -distance_m, 0.0
    raise ValueError("Unsupported relation: %s" % relation)


class ArrangementArm:
    def __init__(self, visual_module, ai_config: dict, motion_config: dict, ik_service: str, enable_ik: bool):
        self.visual = visual_module
        self.inner = visual_module.DofbotArm(ai_config, ik_service, enable_ik)
        self.motion = motion_config["motion"]
        self.observe_move_time = int(self.motion.get("observe_move_time_ms", 1000))
        self.top_move_time = int(self.motion.get("top_move_time_ms", 1000))
        self.approach_move_time = int(self.motion.get("approach_move_time_ms", 900))
        self.place_move_time = int(self.motion.get("place_move_time_ms", 1000))
        self.post_place_delay = float(self.motion.get("post_place_delay_seconds", 0.2))
        self.max_stack_levels = int(motion_config["scene"]["max_stack_levels"])
        self.approach_servo2 = int(self.motion.get("approach_servo2", 80))
        self.approach_servo3 = int(self.motion.get("approach_servo3", 50))
        self.approach_servo4 = int(self.motion.get("approach_servo4", 50))
        self.level_pose_offsets = {
            int(level): [int(value) for value in offsets]
            for level, offsets in self.motion.get("level_pose_offsets", {"1": [0, 0, 0, 0, 0]}).items()
        }
        self.level_polar_offsets = parse_level_polar_offsets(self.motion)

    def close(self) -> None:
        self.inner.close()

    def observe(self, open_gripper: bool) -> None:
        self.inner.observe(open_gripper=open_gripper)

    def clamp(self, close: bool) -> None:
        self.inner.clamp(close)

    def move(self, pose: Sequence[float], move_time_ms: Optional[int] = None) -> None:
        self.inner.move(pose, move_time_ms)

    def move_verified(self, pose: Sequence[float]) -> None:
        self.inner.move_verified(pose)

    def validate_pose(self, pose: Sequence[float]) -> None:
        self.inner.validate_pose(pose)

    def level_pose(self, base_pose: Sequence[int], level: int, source: str) -> List[int]:
        if not 1 <= level <= self.max_stack_levels:
            raise ValueError("Unsupported stack level: %d" % level)
        if level not in self.level_pose_offsets:
            raise ValueError("Missing level_pose_offsets entry for level %d" % level)
        offset = self.level_pose_offsets[level]
        if len(offset) != 5:
            raise ValueError("Level pose offset must contain 5 servo deltas.")
        pose = [int(round(base_pose[index])) + offset[index] for index in range(5)]
        self.validate_pose(pose)
        print(
            "Level pose %s level=%d base=%r offset=%r pose=%r"
            % (source, level, list(base_pose), offset, pose)
        )
        return pose

    def level_motion_workspace(
        self, workspace: Tuple[float, float], level: int, source: str, factor: float
    ) -> Tuple[float, float]:
        radial_m, tangent_m = self.level_polar_offsets.get(level, (0.0, 0.0))
        corrected = offset_polar(workspace, radial_m * factor, tangent_m * factor)
        if (radial_m, tangent_m) != (0.0, 0.0) and factor != 0.0:
            print(
                "Level polar workspace %s level=%d raw=(%.5f, %.5f) radial=%.1fmm tangent=%.1fmm factor=%.2f corrected=(%.5f, %.5f)"
                % (
                    source,
                    level,
                    workspace[0],
                    workspace[1],
                    radial_m * 1000.0,
                    tangent_m * 1000.0,
                    factor,
                    corrected[0],
                    corrected[1],
                )
            )
        return corrected

    def inverse_kinematics_level(
        self, workspace: Tuple[float, float], level: int, source: str
    ) -> List[int]:
        offset = self.level_polar_offsets.get(level, (0.0, 0.0))
        factors = [1.0, 0.75, 0.5, 0.25, 0.0] if offset != (0.0, 0.0) else [0.0]
        failures: List[str] = []
        for factor in factors:
            motion_workspace = self.level_motion_workspace(workspace, level, source, factor)
            try:
                base_pose = self.inner.inverse_kinematics_xy(
                    motion_workspace, source="%s level=1-base" % source
                )
                return self.level_pose(base_pose, level, source)
            except (RuntimeError, ValueError) as exc:
                failures.append("factor=%.2f workspace=%r error=%s" % (factor, motion_workspace, exc))
                if factor != factors[-1]:
                    print("Level polar workspace fallback: %s" % failures[-1])
        raise RuntimeError(
            "No reachable level %d pose for %s. Tried: %s"
            % (level, source, "; ".join(failures))
        )

    def approach_pose(self, pose: Sequence[int]) -> List[int]:
        return [
            int(round(pose[0])),
            self.approach_servo2,
            self.approach_servo3,
            self.approach_servo4,
            int(round(pose[4])),
        ]

    def plan_pick_pose(self, workspace: Tuple[float, float], level: int, label: str) -> List[int]:
        return self.inverse_kinematics_level(workspace, level, "%s pick" % label)

    def plan_place_pose(self, workspace: Tuple[float, float], level: int, label: str) -> List[int]:
        return self.inverse_kinematics_level(workspace, level, "%s place" % label)

    def execute_transfer(
        self,
        source_workspace: Tuple[float, float],
        source_level: int,
        dest_workspace: Tuple[float, float],
        dest_level: int,
        label: str,
    ) -> None:
        pick_pose = self.plan_pick_pose(source_workspace, source_level, label)
        place_pose = self.plan_place_pose(dest_workspace, dest_level, label)
        pick_approach = self.approach_pose(pick_pose)
        place_approach = self.approach_pose(place_pose)
        self.validate_pose(pick_approach)
        self.validate_pose(place_approach)

        self.observe(open_gripper=True)
        self.inner.arm.Arm_Buzzer_On(1)
        time.sleep(0.5)

        self.move(self.inner.poses["top"], self.top_move_time)
        self.move(pick_approach, self.approach_move_time)
        self.clamp(False)
        self.move_verified(pick_pose)
        self.clamp(True)
        self.move(pick_approach, self.top_move_time)

        self.move(self.inner.poses["top"], self.top_move_time)
        self.move(place_approach, self.approach_move_time)
        self.move(place_pose, self.place_move_time)
        self.clamp(False)
        time.sleep(self.post_place_delay)
        self.move(place_approach, self.top_move_time)
        self.move(self.inner.poses["observe"], self.observe_move_time)


def build_state_from_scan(mapped_blocks: Sequence[object], colors: Sequence[str]) -> dict:
    counts = Counter(item.detection.color for item in mapped_blocks)
    duplicates = [color for color, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(
            "Resync requires one visible block per color in v1; duplicate colors: %s"
            % ", ".join(sorted(duplicates))
        )
    blocks = []
    for item in sorted(
        mapped_blocks,
        key=lambda entry: (color_order(colors).get(entry.detection.color, 99), entry.workspace[1], entry.workspace[0]),
    ):
        blocks.append(
            {
                "color": item.detection.color,
                "workspace_x": round(float(item.workspace[0]), 5),
                "workspace_y": round(float(item.workspace[1]), 5),
                "level": 1,
                "stack_id": item.detection.color,
                "support_color": None,
                "last_scan_target_id": item.target_id,
                "supporting_views": int(item.supporting_views),
                "coordinate_spread_m": round(float(item.coordinate_spread_m), 5),
            }
        )
    return {
        "version": 1,
        "updated_at": utc_now(),
        "coordinate_frame": "base_90_global_xy_meters",
        "blocks": blocks,
    }


def print_state(state: dict) -> None:
    blocks = sorted(state.get("blocks", []), key=lambda item: (item["stack_id"], item["level"], item["color"]))
    print("Scene state updated_at=%s blocks=%d" % (state.get("updated_at", ""), len(blocks)))
    for item in blocks:
        print(
            "  %-6s stack=%-6s level=%d workspace=(%.5f, %.5f) support=%s"
            % (
                item["color"],
                item["stack_id"],
                item["level"],
                item["workspace_x"],
                item["workspace_y"],
                item["support_color"] or "-",
            )
        )


def plan_destination(state: dict, request: dict, config: dict) -> dict:
    source = ensure_topmost(state, request["source"])
    target = get_block(state, request["target"])
    relation = request["relation"]
    default_distance_cm = float(config["scene"]["default_distance_cm"])
    if relation == "above":
        target_stack = target["stack_id"]
        current_top = top_block(state, target_stack)
        if (
            source["stack_id"] == target["stack_id"]
            and source["level"] == current_top["level"]
            and source["level"] > target["level"]
        ):
            return {
                "source": source,
                "target": target,
                "relation": relation,
                "dest_x": float(source["workspace_x"]),
                "dest_y": float(source["workspace_y"]),
                "dest_level": int(source["level"]),
                "dest_stack_id": source["stack_id"],
                "dest_support_color": source["support_color"],
                "noop": True,
            }
        dest_level = int(current_top["level"]) + 1
        if dest_level > int(config["scene"]["max_stack_levels"]):
            raise RuntimeError("Destination stack would exceed the configured maximum level.")
        return {
            "source": source,
            "target": target,
            "relation": relation,
            "dest_x": float(target["workspace_x"]),
            "dest_y": float(target["workspace_y"]),
            "dest_level": dest_level,
            "dest_stack_id": target_stack,
            "dest_support_color": current_top["color"],
            "noop": False,
        }

    distance_cm = float(request["distance_cm"] or default_distance_cm)
    distance_m = round(distance_cm / 100.0, 5)
    radial_m, tangent_m = relation_to_polar_offset(relation, distance_m)
    dest_x, dest_y = offset_polar(
        (float(target["workspace_x"]), float(target["workspace_y"])),
        radial_m=radial_m,
        tangent_m=tangent_m,
    )
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "dest_x": dest_x,
        "dest_y": dest_y,
        "dest_level": 1,
        "dest_stack_id": source["color"],
        "dest_support_color": None,
        "noop": False,
    }


def apply_move_to_state(state: dict, source_color: str, destination: dict) -> dict:
    updated = copy.deepcopy(state)
    block = get_block(updated, source_color)
    block["workspace_x"] = round(float(destination["dest_x"]), 5)
    block["workspace_y"] = round(float(destination["dest_y"]), 5)
    block["level"] = int(destination["dest_level"])
    block["stack_id"] = str(destination["dest_stack_id"])
    block["support_color"] = destination["dest_support_color"]
    updated["updated_at"] = utc_now()
    return updated


def append_history(history: dict, entry: dict) -> dict:
    updated = copy.deepcopy(history)
    updated.setdefault("entries", []).append(entry)
    updated["updated_at"] = utc_now()
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relative block arrangement and undo based on ai-visual scanning."
    )
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ik-service", default="/get_kinemarics")
    parser.add_argument("--sample-frames", type=int, default=20)
    parser.add_argument("--min-hits", type=int, default=12)
    parser.add_argument("--max-jitter", type=float, default=12.0)
    subparsers = parser.add_subparsers(dest="action", required=True)

    resync = subparsers.add_parser("resync", help="Scan the current scene and rebuild scene_state.")
    resync.add_argument("--colors", default="red,yellow,green,blue")
    resync.add_argument("--dry-run", action="store_true", help="Plan and print, but do not move the arm or write state.")

    move = subparsers.add_parser("move", help="Execute one relative arrangement command.")
    move.add_argument("--command", help='Natural-language command, e.g. "把红色方块放到蓝色方块左边3cm"')
    move.add_argument("--source")
    move.add_argument("--target")
    move.add_argument("--relation")
    move.add_argument("--distance-cm", type=float)
    move.add_argument("--dry-run", action="store_true", help="Plan and print, but do not move the arm or write state.")

    undo = subparsers.add_parser("undo", help="Undo the most recent successful move.")
    undo.add_argument("--dry-run", action="store_true", help="Plan and print, but do not move the arm or write state.")
    subparsers.add_parser("show-state", help="Print the stored scene state.")
    return parser.parse_args()


def resync_scene(args: argparse.Namespace, config: dict, visual_module) -> int:
    ai_visual_dir = Path(config["ai_visual"]["path"])
    ai_config = load_json(ai_visual_dir / config["ai_visual"]["config_file"])
    feedback_path = ai_visual_dir / config["ai_visual"]["feedback_config_file"]
    feedback_rules = visual_module.load_workspace_corrections(feedback_path)
    colors = normalize_colors(args.colors)
    scan_dir = ROOT / config["ai_visual"]["scan_dir"]
    map_output = ROOT / config["ai_visual"]["map_output"]
    detector = visual_module.ColorBlockDetector(ai_config)

    arm = None
    cap = None
    try:
        arm = ArrangementArm(visual_module, ai_config, config, args.ik_service, enable_ik=False)
        cap = visual_module.open_camera(args.camera)
        mapped_blocks = visual_module.scan_workspace(
            cap,
            arm.inner,
            detector,
            ai_config,
            feedback_rules,
            feedback_path,
            colors,
            args.sample_frames,
            args.min_hits,
            args.max_jitter,
            scan_dir,
            map_output,
        )
        state = build_state_from_scan(mapped_blocks, colors)
        print_state(state)
        if not args.dry_run:
            save_json(args.state_file, state)
            save_json(args.history_file, empty_history())
        else:
            print("Dry-run: scene_state.json and undo_history.json were not written.")
        return 0
    finally:
        if cap is not None:
            cap.release()
        if arm is not None:
            arm.close()


def execute_move(args: argparse.Namespace, config: dict, visual_module) -> int:
    state = load_state(args.state_file)
    if not state.get("blocks"):
        raise RuntimeError("Scene state is empty. Run resync first.")
    history = load_history(args.history_file)
    request = resolve_move_request(args, config)
    destination = plan_destination(state, request, config)
    if destination["noop"]:
        print("Command is already satisfied; no motion executed.")
        return 0

    source = destination["source"]
    print(
        "Move %s -> %s relation=%s dest=(%.5f, %.5f) level=%d"
        % (
            request["source"],
            request["target"],
            request["relation"],
            destination["dest_x"],
            destination["dest_y"],
            destination["dest_level"],
        )
    )
    if args.dry_run:
        print("Dry-run: no arm motion and no state change.")
        return 0

    ai_visual_dir = Path(config["ai_visual"]["path"])
    ai_config = load_json(ai_visual_dir / config["ai_visual"]["config_file"])
    arm = None
    try:
        arm = ArrangementArm(visual_module, ai_config, config, args.ik_service, enable_ik=True)
        arm.execute_transfer(
            (float(source["workspace_x"]), float(source["workspace_y"])),
            int(source["level"]),
            (float(destination["dest_x"]), float(destination["dest_y"])),
            int(destination["dest_level"]),
            "%s->%s" % (request["source"], request["target"]),
        )
    finally:
        if arm is not None:
            arm.close()

    state_before = copy.deepcopy(state)
    updated_state = apply_move_to_state(state, request["source"], destination)
    updated_history = append_history(
        history,
        {
            "timestamp": utc_now(),
            "kind": "move",
            "command": request["command_text"]
            or "%s %s %s" % (request["source"], request["relation"], request["target"]),
            "moved_color": request["source"],
            "state_before": state_before,
        },
    )
    save_json(args.state_file, updated_state)
    save_json(args.history_file, updated_history)
    print_state(updated_state)
    return 0


def execute_undo(args: argparse.Namespace, config: dict, visual_module) -> int:
    history = load_history(args.history_file)
    if not history.get("entries"):
        raise RuntimeError("Undo history is empty.")
    current_state = load_state(args.state_file)
    if not current_state.get("blocks"):
        raise RuntimeError("Scene state is empty. Run resync first.")
    entry = history["entries"][-1]
    previous_state = entry["state_before"]
    moved_color = entry["moved_color"]
    current_block = ensure_topmost(current_state, moved_color)
    previous_block = get_block(previous_state, moved_color)
    print(
        "Undo %s: %s current=(%.5f, %.5f, level=%d) -> previous=(%.5f, %.5f, level=%d)"
        % (
            entry.get("command", ""),
            moved_color,
            current_block["workspace_x"],
            current_block["workspace_y"],
            current_block["level"],
            previous_block["workspace_x"],
            previous_block["workspace_y"],
            previous_block["level"],
        )
    )
    if args.dry_run:
        print("Dry-run: no arm motion and no history change.")
        return 0

    ai_visual_dir = Path(config["ai_visual"]["path"])
    ai_config = load_json(ai_visual_dir / config["ai_visual"]["config_file"])
    arm = None
    try:
        arm = ArrangementArm(visual_module, ai_config, config, args.ik_service, enable_ik=True)
        arm.execute_transfer(
            (float(current_block["workspace_x"]), float(current_block["workspace_y"])),
            int(current_block["level"]),
            (float(previous_block["workspace_x"]), float(previous_block["workspace_y"])),
            int(previous_block["level"]),
            "undo:%s" % moved_color,
        )
    finally:
        if arm is not None:
            arm.close()

    remaining_history = copy.deepcopy(history)
    remaining_history["entries"] = remaining_history.get("entries", [])[:-1]
    remaining_history["updated_at"] = utc_now()
    previous_state["updated_at"] = utc_now()
    save_json(args.state_file, previous_state)
    save_json(args.history_file, remaining_history)
    print_state(previous_state)
    return 0


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    ai_visual_path = (args.project_dir / config["ai_visual"]["relative_path"]).resolve()
    config["ai_visual"]["path"] = str(ai_visual_path)

    if args.action == "resync":
        visual_module = load_visual_module(ai_visual_path)
        return resync_scene(args, config, visual_module)
    if args.action == "move":
        visual_module = None if args.dry_run else load_visual_module(ai_visual_path)
        return execute_move(args, config, visual_module)
    if args.action == "undo":
        visual_module = None if args.dry_run else load_visual_module(ai_visual_path)
        return execute_undo(args, config, visual_module)
    if args.action == "show-state":
        print_state(load_state(args.state_file))
        return 0
    raise ValueError("Unsupported action: %s" % args.action)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
