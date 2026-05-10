"""
narrative_planner.py — 叙事大纲规划器

在 ReAct 剪辑之前，用一次 LLM 调用生成整体叙事蓝图：
- 整体情绪弧线
- 每个音频段落的叙事意图、视觉方向、目标情绪
- 关键时刻的视觉指令

输出的大纲会：
1. 展示给用户确认（Human-in-the-Loop）
2. 作为 user_feedback 传给 ReAct Planner 指导选片
"""
from __future__ import annotations

import json
import re

from loguru import logger

from models.audio import AudioMap
from models.video import SceneItem
from utils.clients import QWEN_MAX, get_qwen_client
from utils.prompt_loader import load_prompt


async def run_narrative_planner(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    background_info: str | None = None,
) -> tuple[dict, str]:
    """
    生成叙事大纲。

    Returns:
        (narrative_plan_dict, narrative_plan_md)
        - narrative_plan_dict: 结构化大纲，存入 GraphState
        - narrative_plan_md: Markdown 格式，用于展示给用户
    """
    logger.info("[Narrative Planner] 开始生成叙事大纲")

    system_prompt = load_prompt("planning/narrative_planner_system.md")
    user_msg = _build_user_message(audio_map, scene_table, background_info)

    client = get_qwen_client()
    response = await client.chat.completions.create(
        model=QWEN_MAX,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content or ""
    logger.debug(f"[Narrative Planner] 原始输出（前500字）: {raw[:500]}")

    plan_dict = _parse_json(raw)
    if not plan_dict:
        logger.warning("[Narrative Planner] JSON 解析失败，使用兜底大纲")
        plan_dict = _fallback_plan(audio_map)

    plan_md = _plan_to_md(plan_dict)
    logger.info(f"[Narrative Planner] 大纲生成完成，共 {len(plan_dict.get('segments', []))} 个段落")

    return plan_dict, plan_md


def narrative_plan_to_react_prompt(plan_dict: dict) -> str:
    """
    把叙事大纲转换为 ReAct Planner 的 user_feedback 字符串。
    这个字符串会注入到 ReAct 的初始消息里，作为全局指导方针。
    """
    if not plan_dict:
        return ""

    lines = [
        "## 叙事大纲（请严格遵守）",
        "",
        f"**整体主题**：{plan_dict.get('overall_theme', '未指定')}",
        f"**情绪弧线**：{plan_dict.get('arc', '未指定')}",
        "",
        "### 各段落指导",
    ]

    for seg in plan_dict.get("segments", []):
        lines.append(
            f"- **{seg['name']}** ({seg.get('start', 0):.1f}s–{seg.get('end', 0):.1f}s)："
            f" 叙事={seg.get('narrative_role', '')}，"
            f" 视觉={seg.get('visual_direction', '')}，"
            f" 情绪={seg.get('mood_target', '')}，"
            f" 节奏={seg.get('energy_level', '')}"
        )
        if seg.get("transition_hint"):
            lines.append(f"  → 过渡：{seg['transition_hint']}")

    key_moments = plan_dict.get("key_moments", [])
    if key_moments:
        lines.append("")
        lines.append("### 关键时刻（必须特殊处理）")
        for km in key_moments:
            lines.append(
                f"- **{km.get('time', 0):.1f}s** [{km.get('type', '')}]：{km.get('instruction', '')}"
            )

    if plan_dict.get("notes"):
        lines.append("")
        lines.append(f"### 额外注意事项\n{plan_dict['notes']}")

    return "\n".join(lines)


# ── 内部工具 ───────────────────────────────────────────────────────────────────

def _build_user_message(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    background_info: str | None,
) -> str:
    # 音乐结构
    seg_lines = []
    for s in audio_map.segments:
        seg_lines.append(
            f"  {s.name:20s} {s.start:6.1f}s–{s.end:6.1f}s  "
            f"energy={s.energy}  trend={s.energy_trend}  mood={s.mood}"
            + (f"  visual_hint={s.visual_suggestion}" if s.visual_suggestion else "")
        )

    # 关键时刻
    key_moments = getattr(audio_map, "key_moments", [])
    km_lines = []
    for km in sorted(key_moments, key=lambda x: x.get("final_time_sec", 0)):
        km_lines.append(
            f"  {km.get('final_time_sec', 0):6.1f}s  "
            f"{km.get('type', '')}  importance={km.get('importance', '')}  "
            f"emotion={km.get('emotion', '')}"
        )

    # 素材库情绪分布
    mood_counter: dict[str, int] = {}
    intensity_buckets = {"low(0-0.3)": 0, "mid(0.3-0.7)": 0, "high(0.7-1.0)": 0}
    for s in scene_table:
        mood = s.mood or "unknown"
        mood_counter[mood] = mood_counter.get(mood, 0) + 1
        intensity = float((s.visual_profile or {}).get("arousal", 0.5))
        if intensity < 0.3:
            intensity_buckets["low(0-0.3)"] += 1
        elif intensity < 0.7:
            intensity_buckets["mid(0.3-0.7)"] += 1
        else:
            intensity_buckets["high(0.7-1.0)"] += 1

    mood_summary = "  " + "  ".join(f"{k}:{v}个" for k, v in sorted(mood_counter.items(), key=lambda x: -x[1])[:8])
    intensity_summary = "  " + "  ".join(f"{k}:{v}个" for k, v in intensity_buckets.items())

    return (
        f"## 背景信息\n{background_info or '未提供'}\n\n"
        f"## 音乐结构\n"
        f"BPM: {audio_map.bpm:.1f}  总时长: {audio_map.total_duration:.1f}s  段落数: {len(audio_map.segments)}\n\n"
        f"段落列表：\n" + "\n".join(seg_lines) + "\n\n"
        + (f"关键时刻：\n" + "\n".join(km_lines) + "\n\n" if km_lines else "")
        + f"## 素材库概况\n"
        f"总场景数: {len(scene_table)}\n"
        f"情绪分布：\n{mood_summary}\n"
        f"强度分布：\n{intensity_summary}\n\n"
        f"## 任务\n"
        f"请根据以上信息，制定完整的叙事大纲。直接输出 JSON，不要有任何额外文字。"
    )


def _parse_json(raw: str) -> dict | None:
    """从 LLM 输出中提取 JSON。"""
    # 先尝试直接解析
    try:
        return json.loads(raw.strip())
    except Exception:
        pass

    # 提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass

    # 提取第一个 { ... }
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def _fallback_plan(audio_map: AudioMap) -> dict:
    """JSON 解析失败时的兜底大纲，基于音频段落直接生成。"""
    segments = []
    for s in audio_map.segments:
        energy_level = "high" if s.energy >= 7 else ("low" if s.energy <= 3 else "mid")
        segments.append({
            "name": s.name,
            "start": s.start,
            "end": s.end,
            "narrative_role": s.description or "过渡段落",
            "visual_direction": s.visual_suggestion or "根据情绪选择合适场景",
            "mood_target": s.mood,
            "energy_level": energy_level,
            "transition_hint": "",
        })
    return {
        "overall_theme": "情绪驱动的视觉叙事",
        "arc": "起伏变化",
        "segments": segments,
        "key_moments": [],
        "notes": "（兜底大纲，叙事规划器解析失败）",
    }


def _plan_to_md(plan_dict: dict) -> str:
    """把大纲 dict 转为 Markdown，用于展示给用户。"""
    lines = [
        f"**整体主题**：{plan_dict.get('overall_theme', '未指定')}",
        f"**情绪弧线**：{plan_dict.get('arc', '未指定')}",
        "",
        "| 段落 | 时间 | 叙事作用 | 视觉方向 | 情绪 | 节奏 |",
        "|------|------|----------|----------|------|------|",
    ]

    for seg in plan_dict.get("segments", []):
        name = seg.get("name", "")
        time_range = f"{seg.get('start', 0):.1f}s–{seg.get('end', 0):.1f}s"
        role = seg.get("narrative_role", "")
        visual = seg.get("visual_direction", "")
        mood = seg.get("mood_target", "")
        energy = seg.get("energy_level", "")
        # 截断过长内容
        role = (role[:25] + "…") if len(role) > 25 else role
        visual = (visual[:30] + "…") if len(visual) > 30 else visual
        lines.append(f"| {name} | {time_range} | {role} | {visual} | {mood} | {energy} |")

    key_moments = plan_dict.get("key_moments", [])
    if key_moments:
        lines.append("")
        lines.append("**关键时刻**：")
        for km in key_moments:
            lines.append(f"- `{km.get('time', 0):.1f}s` [{km.get('type', '')}] {km.get('instruction', '')}")

    if plan_dict.get("notes"):
        lines.append("")
        lines.append(f"**注意事项**：{plan_dict['notes']}")

    return "\n".join(lines)
