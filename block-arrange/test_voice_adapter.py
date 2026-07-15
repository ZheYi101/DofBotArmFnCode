#!/usr/bin/env python3

import copy
import unittest
from pathlib import Path

import command_runner
from voice_adapter import parse_voice_text


class VoiceAdapterTests(unittest.TestCase):
    def test_user_sequence(self):
        actions = parse_voice_text(
            "你好, 扫描下当前场景, 然后把红色方块放到蓝色方块上面, "
            "然后把蓝色方块放到红色方块左边"
        )
        self.assertEqual([item["kind"] for item in actions], ["resync", "move", "move"])
        self.assertEqual(
            (actions[1]["source"], actions[1]["target"], actions[1]["relation"]),
            ("red", "blue", "above"),
        )
        self.assertEqual(
            (actions[2]["source"], actions[2]["target"], actions[2]["relation"]),
            ("blue", "red", "left"),
        )

    def test_aliases_and_chinese_distance(self):
        actions = parse_voice_text(
            "请扫描桌面，接着将红色积木放在蓝色色块顶部，再把蓝色移到红色左侧十厘米"
        )
        self.assertEqual(len(actions), 3)
        self.assertEqual(actions[2]["distance_cm"], 10.0)

    def test_unknown_fragment_rejects_whole_transcript(self):
        with self.assertRaisesRegex(ValueError, "Unsupported voice transcript fragment"):
            parse_voice_text("扫描当前场景，然后唱歌，然后把红色放到蓝色上面")

    def test_decimal_chinese_distance(self):
        action = parse_voice_text("把蓝色方块挪到红色方块右侧八点五厘米")[0]
        self.assertEqual(action["distance_cm"], 8.5)

    def test_sequence_uses_one_auto_undo_before_final_move(self):
        actions = parse_voice_text(
            "扫描当前场景，然后把红色方块放到蓝色方块上面，"
            "然后把蓝色方块放到红色方块左边"
        )
        config = command_runner.load_json(Path(__file__).with_name("config.json"))
        state = {
            "version": 1,
            "updated_at": "",
            "coordinate_frame": "base_90_global_xy_meters",
            "blocks": [
                {
                    "color": "red",
                    "workspace_x": -0.10,
                    "workspace_y": 0.28,
                    "level": 1,
                    "stack_id": "red",
                    "support_color": None,
                },
                {
                    "color": "blue",
                    "workspace_x": 0.04,
                    "workspace_y": 0.29,
                    "level": 1,
                    "stack_id": "blue",
                    "support_color": None,
                },
            ],
        }
        history = command_runner.empty_history()

        first = dict(actions[1], command_text=actions[1]["original"])
        first_destination = command_runner.plan_destination(state, first, config)
        state_before = copy.deepcopy(state)
        state = command_runner.apply_move_to_state(state, first["source"], first_destination)
        history = command_runner.append_history(
            history,
            {
                "timestamp": "",
                "kind": "move",
                "command": first["command_text"],
                "moved_color": first["source"],
                "state_before": state_before,
            },
        )

        uncovered, remaining, undo_steps = command_runner.plan_auto_undos(
            state, history, actions[2]["source"]
        )
        self.assertEqual(len(undo_steps), 1)
        self.assertEqual(undo_steps[0]["moved_color"], "red")
        self.assertEqual(remaining["entries"], [])

        final = dict(actions[2], command_text=actions[2]["original"])
        final_destination = command_runner.plan_destination(uncovered, final, config)
        self.assertFalse(final_destination["noop"])
        self.assertEqual(final_destination["dest_level"], 1)


if __name__ == "__main__":
    unittest.main()
