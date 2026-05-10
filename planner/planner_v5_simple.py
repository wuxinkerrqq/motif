"""
planner_v5_simple.py — 全素材直给 GPT-5.5，分两次调用

调用 1: 输入 audio_map + 全 scene_table → 输出 L1 + L2 锚定方案
调用 2: 输入 L1/L2 结果 + 剩余 scene → 输出 L3 填充 → 完整 render_plan

优势：
  - 零检索损失（LLM 直接看全素材）
  - 叙事天然连贯（全局视野）
  - 成本 ~$0.25（vs v4 $2-3）
  - 代码量极少
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from models.audio import AudioMap
from models.video import SceneItem
from utils.clients import GPT_5_5, get_openai_client


@dataclass
class PlannerResult:
    committed: list[dict] = field(default_factory=list)
    global_plan: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    finished: bool = False
    iterations: int = 0


# ═══════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════

CALL1_SYSTEM = """你是专业 AMV 剪辑师。你将看到一首音乐的完整分析和全部视频素材库。

你的任务是【第一步：锚定主结构】，为音乐的关键位置选定最契合的视频片段。

## 工作流

### Step 1 — L1 段落分界点（最高优先级）

音乐的段落分界点是整首歌的骨架。为以下每个分界点选一个 scene：
- 开场（0s 起）：选一个能"开篇定调"的 scene
- 每个段落切换处：选一个能"承上启下"的 scene
- 副歌爆发点：选最有冲击力的 scene（is_climax_material 优先）
- 收尾：选一个能"余韵收束"的 scene

每个 L1 占坑需要指定：
- scene_id
- clip_offset：从 scene 的第几秒开始截（用于对齐动作瞬间到音乐锚点）
- duration：这个占坑持续多长（开场 2-4s，接缝 1-1.5s，收尾填满 end 段）
- transition_type：转场类型（见下方"可用转场"）

## 可用转场类型（transition_type）

只能从以下列表里选：
- `hard_cut`：直接硬切，默认，zero 特效（transition_duration=0）
- `flash_white` / `flash_black`：闪白/闪黑，适合节拍锚点、爆发点（建议 0.05-0.12s）
- `camera_shake_cut`：相机抖动，适合冲击感段落（建议 0.15-0.30s）
- `zoom_punch`：冲击缩放（从 1.3x 快速回到 1x），适合爆发/高潮（建议 0.15-0.25s）
- `zoom_out`：拉远缩放，适合段落过渡/收束（建议 0.2-0.3s）
- `rgb_split`：色偏抖动（故障感），适合紧张/异常氛围（建议 0.15-0.25s）
- `radial_blur`：径向模糊，适合速度感/冲刺（建议 0.15-0.25s）
- `whip_pan`：水平动态模糊，适合段落切换（建议 0.15-0.25s）
- `fade_in`：平滑淡入，适合开场/安静段（建议 0.3-0.5s）
- `dissolve`：溶解渐显，适合温柔过渡（建议 0.2-0.4s）

**锚点位优先用 flash_white / zoom_punch / camera_shake_cut；非锚点绝大多数用 hard_cut。**

### Step 2 — L2 段内关键锚点

看 key_moments 列表中 importance 较高的点（排除距离 L1 < 2s 的）。
为这些锚点选 scene，要求：
- scene 的动作/情绪顶点应与锚点时间对齐
- 不要选已被 L1 使用的 scene_id
- 每段内 L2 数量控制在 1-3 个（不要太密）
- 相邻 L2 间隔 ≥ 3s

## 输出格式（严格 JSON）

```json
{
  "L1": [
    {
      "audio_time": 0.0,
      "role": "opening",
      "scene_id": 12,
      "clip_offset": 0.0,
      "duration": 3.5,
      "speed_factor": 1.0,
      "transition_type": "hard_cut",
      "transition_duration": 0.0,
      "reason": "一句话说明为什么选这个"
    }
  ],
  "L2": [
    {
      "audio_time": 5.97,
      "scene_id": 7,
      "clip_offset": 0.5,
      "duration": 1.0,
      "speed_factor": 1.0,
      "transition_type": "flash_white",
      "transition_duration": 0.08,
      "reason": "..."
    }
  ]
}
```

## 约束

- scene 不能重复使用（一个 scene_id 只能出现一次）
- clip_offset + duration × speed_factor 不能超过 scene 的 duration
- speed_factor 范围 [0.5, 1.5]（<1 慢放，>1 快放）
- 开场必须从 audio_time=0 开始
- 收尾必须覆盖到歌曲结束
- L1 + L2 总数不要超过 20 个（留足空间给后续填充）
"""

CALL2_SYSTEM = """你是专业 AMV 剪辑师。第一步已经锚定了关键位置的 scene，现在你需要【填充剩余空白区间】。

## 已锚定的时间线

{anchored_timeline}

## 待填充的 gap 区间

{gaps_list}

## 可用素材（排除已使用的）

{remaining_scenes}

## 你的任务

为每个 gap 区间选择 scene 填满，要求：
- mood 与所属音乐段落一致
- 相邻 scene 情绪跳跃不突兀（不要上一秒哭泣下一秒燃爆）
- 多样化源视频，避免连续使用同一个 mp4
- clip 之间无缝衔接，不要留 <0.5s 的空隙
- 一个 scene 只能用一次

## 输出格式（严格 JSON 数组）

```json
[
  {
    "audio_start": 3.5,
    "audio_end": 5.47,
    "scene_id": 15,
    "clip_offset": 0.0,
    "speed_factor": 1.0,
    "transition_type": "hard_cut",
    "transition_duration": 0.0
  }
]
```

## 约束

- audio_start/audio_end 必须精确覆盖 gap（不重叠、不遗漏）
- clip_offset + (audio_end - audio_start) × speed_factor ≤ scene.duration
- speed_factor 范围 [0.5, 1.5]
- 每个 gap 可以用 1 个或多个 scene 填充（按需切分）
"""


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _format_audio_map(audio_map: AudioMap) -> str:
    lines = [
        f"BPM: {audio_map.bpm}  总时长: {audio_map.total_duration:.2f}s",
        f"叙事概要: {audio_map.narrative_summary}",
        f"情绪轨迹: {' → '.join(audio_map.mood_arc)}",
        "",
        "## 段落",
    ]
    for s in audio_map.segments:
        vp = getattr(s, "visual_profile", None) or {}
        lines.append(
            f"  [{s.name}]  {s.start:.2f}~{s.end:.2f}s  "
            f"energy={getattr(s, 'energy_level', '?')}  mood={s.mood}  "
            f"pacing={getattr(s, 'pacing_hint', '?')}"
        )
        if s.description:
            lines.append(f"    {s.description}")

    lines.append("\n## Key Moments（按 importance 降序，top 15）")
    kms = sorted(audio_map.key_moments_v2, key=lambda k: k.importance, reverse=True)[:15]
    for k in kms:
        lines.append(
            f"  t={k.time:6.2f}s  imp={k.importance:.3f}  tier={k.tier}  {k.description[:50]}"
        )
    return "\n".join(lines)


def _format_scene_table(scene_table: list[SceneItem]) -> str:
    lines = [f"共 {len(scene_table)} 个 scene\n"]
    for s in scene_table:
        vp = s.visual_profile or {}
        climax = " [CLIMAX]" if s.is_climax_material else ""
        outro = " [OUTRO]" if s.is_outro_material else ""
        lines.append(
            f"scene_id={s.scene_id}  "
            f"source={Path(s.source_file).name}  "
            f"[{s.start:.2f}~{s.end:.2f}s] dur={s.duration:.2f}s  "
            f"mood={s.mood}  motion={vp.get('motion_intensity', 0):.2f}  "
            f"V/A/D={vp.get('valence',0):.2f}/{vp.get('arousal',0):.2f}/{vp.get('dominance',0):.2f}"
            f"{climax}{outro}\n"
            f"  {s.scene_description[:80]}\n"
            f"  characters={s.characters}"
        )
    return "\n".join(lines)


def _parse_json_from_response(raw: str) -> dict | list | None:
    import re
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    text = match.group(1).strip() if match else raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{") if "{" in text else text.find("[")
        if start >= 0:
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                return None
    return None


def _compute_gaps(
    l1_l2_clips: list[dict],
    total_duration: float,
) -> list[tuple[float, float]]:
    """从 L1+L2 占坑列表算出剩余 gap。"""
    occupied = sorted(l1_l2_clips, key=lambda c: c["audio_start"])
    gaps = []
    cursor = 0.0
    for c in occupied:
        if c["audio_start"] > cursor + 0.01:
            gaps.append((round(cursor, 3), round(c["audio_start"], 3)))
        cursor = max(cursor, c["audio_end"])
    if cursor < total_duration - 0.01:
        gaps.append((round(cursor, 3), round(total_duration, 3)))
    return gaps


def _l1l2_to_clips(result: dict) -> list[dict]:
    """把 Call 1 的 L1+L2 输出转成统一的 clip 列表。"""
    clips = []
    for item in result.get("L1", []):
        t = item["audio_time"]
        dur = item["duration"]
        clips.append({
            "audio_start": round(t, 3),
            "audio_end": round(t + dur, 3),
            "scene_id": item["scene_id"],
            "clip_offset": item.get("clip_offset", 0.0),
            "speed_factor": item.get("speed_factor", 1.0),
            "transition_type": item.get("transition_type", "hard_cut"),
            "transition_duration": item.get("transition_duration", 0.0),
            "layer": "L1",
        })
    for item in result.get("L2", []):
        t = item["audio_time"]
        dur = item.get("duration", 1.0)
        clips.append({
            "audio_start": round(t - dur / 2, 3),
            "audio_end": round(t + dur / 2, 3),
            "scene_id": item["scene_id"],
            "clip_offset": item.get("clip_offset", 0.0),
            "speed_factor": item.get("speed_factor", 1.0),
            "transition_type": item.get("transition_type", "hard_cut"),
            "transition_duration": item.get("transition_duration", 0.0),
            "layer": "L2",
        })
    return clips


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

async def run_planner_v5(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    clip_embeddings=None,
    clip_scene_ids=None,
    background_info: str = "",
    log_path: Path | None = None,
) -> PlannerResult:
    started = datetime.now()
    client = get_openai_client()

    audio_text = _format_audio_map(audio_map)
    scene_text = _format_scene_table(scene_table)

    # ── Call 1: L1 + L2 锚定 ─────────────────────────────────
    logger.info("[v5] Call 1: 锚定 L1 + L2...")
    user_msg_1 = (
        f"## 背景\n{background_info}\n\n"
        f"## 音乐分析\n{audio_text}\n\n"
        f"## 全部视频素材\n{scene_text}"
    )

    resp1 = await asyncio.wait_for(
        client.chat.completions.create(
            model=GPT_5_5,
            messages=[
                {"role": "system", "content": CALL1_SYSTEM},
                {"role": "user", "content": user_msg_1},
            ],
        ),
        timeout=600.0,
    )
    raw1 = resp1.choices[0].message.content
    result1 = _parse_json_from_response(raw1)
    if not result1:
        logger.error(f"[v5] Call 1 解析失败:\n{raw1[:500]}")
        return PlannerResult(elapsed_sec=(datetime.now() - started).total_seconds())

    l1_count = len(result1.get("L1", []))
    l2_count = len(result1.get("L2", []))
    logger.info(f"[v5] Call 1 完成: L1={l1_count}, L2={l2_count}")

    # ── 计算 gap ─────────────────────────────────────────────
    anchored_clips = _l1l2_to_clips(result1)
    used_ids = {c["scene_id"] for c in anchored_clips}
    gaps = _compute_gaps(anchored_clips, audio_map.total_duration)
    logger.info(f"[v5] Gap 数量: {len(gaps)}, 总时长: {sum(e-s for s,e in gaps):.2f}s")

    # ── Call 2: L3 填充 ──────────────────────────────────────
    logger.info("[v5] Call 2: 填充 L3 gaps...")

    anchored_text = "\n".join(
        f"  [{c['layer']}] {c['audio_start']:.2f}~{c['audio_end']:.2f}s  scene={c['scene_id']}"
        for c in sorted(anchored_clips, key=lambda x: x["audio_start"])
    )
    gaps_text = "\n".join(
        f"  Gap: [{s:.2f}s ~ {e:.2f}s]  dur={e-s:.2f}s"
        for s, e in gaps
    )
    remaining = [s for s in scene_table if s.scene_id not in used_ids]
    remaining_text = _format_scene_table(remaining)

    call2_system = (
        CALL2_SYSTEM
        .replace("{anchored_timeline}", anchored_text)
        .replace("{gaps_list}", gaps_text)
        .replace("{remaining_scenes}", remaining_text)
    )

    resp2 = await asyncio.wait_for(
        client.chat.completions.create(
            model=GPT_5_5,
            messages=[
                {"role": "system", "content": call2_system},
                {"role": "user", "content": "请填充所有 gap，输出完整 JSON 数组。"},
            ],
        ),
        timeout=600.0,
    )
    raw2 = resp2.choices[0].message.content
    result2 = _parse_json_from_response(raw2)
    if not result2 or not isinstance(result2, list):
        logger.error(f"[v5] Call 2 解析失败:\n{raw2[:500]}")
        result2 = []

    logger.info(f"[v5] Call 2 完成: L3={len(result2)} clips")

    # ── 合并 committed ────────────────────────────────────────
    committed = []
    for c in anchored_clips:
        committed.append({
            "audio_start": c["audio_start"],
            "audio_end": c["audio_end"],
            "scene_id": c["scene_id"],
            "speed_factor": c.get("speed_factor", 1.0),
            "transition_type": c.get("transition_type", "hard_cut"),
            "transition_duration": c.get("transition_duration", 0.0),
        })
    for c in result2:
        committed.append({
            "audio_start": c.get("audio_start", 0),
            "audio_end": c.get("audio_end", 0),
            "scene_id": c.get("scene_id", 0),
            "speed_factor": c.get("speed_factor", 1.0),
            "transition_type": c.get("transition_type", "hard_cut"),
            "transition_duration": c.get("transition_duration", 0.0),
        })

    committed.sort(key=lambda x: x["audio_start"])

    elapsed = (datetime.now() - started).total_seconds()
    covered = sum(c["audio_end"] - c["audio_start"] for c in committed)
    finished = abs(covered - audio_map.total_duration) < 1.0

    logger.info(
        f"[v5] 完成。{len(committed)} 片段 / 2 次调用 / {elapsed:.1f}s / "
        f"覆盖 {covered:.1f}/{audio_map.total_duration:.1f}s = {covered/audio_map.total_duration*100:.1f}%"
    )

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({
            "version": "v5",
            "model": GPT_5_5,
            "elapsed_sec": elapsed,
            "finished": finished,
            "coverage": f"{covered:.1f}/{audio_map.total_duration:.1f}s",
            "call1_result": result1,
            "call2_result": result2,
            "committed": committed,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    return PlannerResult(
        committed=committed,
        global_plan=result1,
        elapsed_sec=elapsed,
        finished=finished,
        iterations=2,
    )
