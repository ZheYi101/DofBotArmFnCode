#!/usr/bin/env python3
"""Parse Chinese ASR transcripts into ordered block-arrange actions."""

from __future__ import annotations

import re
from typing import List


COLOR_PATTERN = r"(?:红色|红|黄色|黄|绿色|绿|蓝色|蓝|red|yellow|green|blue)"
BLOCK_PATTERN = r"(?:方块|方快|积木块?|色块)?"
RELATION_PATTERN = (
    r"(?:左边|左侧|左面|左|右边|右侧|右面|右|前面|前边|前方|前|"
    r"后面|后边|后方|后|上面|上方|顶部|上)"
)
NUMBER_PATTERN = r"(?:[0-9]+(?:\.[0-9]+)?|[零〇一二两三四五六七八九十百点]+)"

MOVE_RE = re.compile(
    rf"(?:把|将)?\s*(?P<source>{COLOR_PATTERN})\s*{BLOCK_PATTERN}\s*"
    rf"(?:放到|放在|移动到|移到|挪到|摆到)\s*"
    rf"(?P<target>{COLOR_PATTERN})\s*{BLOCK_PATTERN}\s*"
    rf"(?P<relation>{RELATION_PATTERN})\s*"
    rf"(?:(?P<distance>{NUMBER_PATTERN})\s*(?:cm|厘米|公分)?)?",
    re.IGNORECASE,
)

SCAN_RE = re.compile(
    r"(?:重新\s*)?(?:扫描|识别|更新)(?:一下|下)?\s*"
    r"(?:当前|现在)?\s*(?:的)?\s*(?:场景|桌面|物块|方块)?"
)

COLOR_ALIASES = {
    "红": "red",
    "红色": "red",
    "red": "red",
    "黄": "yellow",
    "黄色": "yellow",
    "yellow": "yellow",
    "绿": "green",
    "绿色": "green",
    "green": "green",
    "蓝": "blue",
    "蓝色": "blue",
    "blue": "blue",
}

RELATION_ALIASES = {
    "左": "left",
    "左边": "left",
    "左侧": "left",
    "左面": "left",
    "右": "right",
    "右边": "right",
    "右侧": "right",
    "右面": "right",
    "前": "front",
    "前面": "front",
    "前边": "front",
    "前方": "front",
    "后": "back",
    "后面": "back",
    "后边": "back",
    "后方": "back",
    "上": "above",
    "上面": "above",
    "上方": "above",
    "顶部": "above",
}

CONNECTOR_RE = re.compile(r"(?:然后|接着|随后|接下来|之后|再然后|再)")
PUNCTUATION_RE = re.compile(r"[\s,，。；;、!！?？:：]+")
FILLER_RE = re.compile(
    r"(?:你好|您好|小雅|机械臂|请|麻烦|帮我|帮忙|执行|开始|一下|"
    r"这个|任务|指令|操作|好的|好|谢谢)"
)

CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def parse_chinese_integer(raw: str) -> int:
    if "百" in raw:
        left, right = raw.split("百", 1)
        hundreds = CHINESE_DIGITS.get(left, 1) * 100
        return hundreds + (parse_chinese_integer(right) if right else 0)
    if "十" in raw:
        left, right = raw.split("十", 1)
        tens = CHINESE_DIGITS.get(left, 1) * 10
        return tens + (CHINESE_DIGITS[right] if right else 0)
    digits = [str(CHINESE_DIGITS[item]) for item in raw]
    return int("".join(digits))


def parse_number(raw: str) -> float:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw):
        return float(raw)
    if "点" not in raw:
        return float(parse_chinese_integer(raw))
    whole, fraction = raw.split("点", 1)
    fraction_digits = "".join(str(CHINESE_DIGITS[item]) for item in fraction)
    return float("%d.%s" % (parse_chinese_integer(whole), fraction_digits))


def normalize_color(raw: str) -> str:
    key = raw.strip().lower()
    if key not in COLOR_ALIASES:
        raise ValueError("Unsupported voice color: %s" % raw)
    return COLOR_ALIASES[key]


def normalize_relation(raw: str) -> str:
    key = raw.strip().lower()
    if key not in RELATION_ALIASES:
        raise ValueError("Unsupported voice relation: %s" % raw)
    return RELATION_ALIASES[key]


def parse_voice_text(text: str) -> List[dict]:
    transcript = text.strip()
    if not transcript:
        raise ValueError("Voice transcript is empty.")

    matches = []
    for match in SCAN_RE.finditer(transcript):
        matches.append((match.start(), match.end(), "resync", match))
    for match in MOVE_RE.finditer(transcript):
        matches.append((match.start(), match.end(), "move", match))
    matches.sort(key=lambda item: (item[0], item[1]))
    if not matches:
        raise ValueError("No supported block-arrange instruction was found in the transcript.")

    actions: List[dict] = []
    consumed = [False] * len(transcript)
    last_end = -1
    for start, end, kind, match in matches:
        if start < last_end:
            continue
        for index in range(start, end):
            consumed[index] = True
        original = transcript[start:end].strip()
        if kind == "resync":
            actions.append({"kind": "resync", "original": original})
        else:
            distance = match.group("distance")
            relation = normalize_relation(match.group("relation"))
            actions.append(
                {
                    "kind": "move",
                    "source": normalize_color(match.group("source")),
                    "target": normalize_color(match.group("target")),
                    "relation": relation,
                    "distance_cm": (
                        None if distance is None or relation == "above" else parse_number(distance)
                    ),
                    "original": original,
                }
            )
        last_end = end

    residual = "".join(
        character if not consumed[index] else " "
        for index, character in enumerate(transcript)
    )
    residual = CONNECTOR_RE.sub(" ", residual)
    residual = FILLER_RE.sub(" ", residual)
    residual = PUNCTUATION_RE.sub("", residual)
    if residual:
        raise ValueError("Unsupported voice transcript fragment: %s" % residual)
    return actions


def describe_action(action: dict) -> str:
    if action["kind"] == "resync":
        return "resync current scene"
    distance = ""
    if action["relation"] != "above":
        if action["distance_cm"] is None:
            distance = " using configured safe distance"
        else:
            distance = " at %.1fcm" % float(action["distance_cm"])
    relation_text = {
        "left": "left of",
        "right": "right of",
        "front": "in front of",
        "back": "behind",
        "above": "above",
    }[action["relation"]]
    return "move %s %s %s%s (auto-undo if covered)" % (
        action["source"],
        relation_text,
        action["target"],
        distance,
    )
