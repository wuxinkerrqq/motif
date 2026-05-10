"""
planner_v4.py — ReAct + v3 全局规划的混合架构（GPT-5.5）

架构：
  Stage 1（GPT-5.5 一次调用，复用 v3 的）：
    输入完整 audio_map → 输出每段的 retrieval_query + weight_profile
  Stage 2（每段独立 ReAct 小循环，GPT-5.5）：
    每段重置 messages，注入 Stage 1 给的 query + 锚点 + cut_points
    LLM 用 search_scenes / get_scene_detail / commit_clip 工具实时决策
    工具层自动算 speed_factor + 锚点位转场 + scene 时长校验

关键改进 vs v1：
  - 不再一个 80 轮大循环烧 context，每段独立
  - LLM 不用从头摸索全局，Stage 1 已给方向
  - 工具自动校验，LLM 专注审美
关键改进 vs v3：
  - LLM 实时驾驶（agent 灵魂），不再让算法机械选
  - 多样性、连贯性靠 LLM 判断而非硬规则
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from agents.planner_tools import (
    commit_clip,
    get_scene_detail,
    search_scenes,
)
from planner.planner_v3 import (
    compute_cut_points_and_energies,
    stage1_screenwriter,
)
from models.audio import AudioMap, AudioSegment
from models.video import SceneItem
from utils.clients import GPT_5_5, get_openai_client


LOCAL_PROMPT = """你是专业 AMV 剪辑师，正在为一个音乐段落选配 scene。

## 上级已经给定的本段规划

- 段落 [{label}]：音乐 [{seg_start:.2f}s - {seg_end:.2f}s]
- retrieval_query：{query}
- 镜头计划（{n_shots} 个，按顺序逐一完成）：
{shots_text}

## 你的工作

对每个 shot 按顺序：
1. 调 `search_scenes(query=..., audio_start=该 shot 起点)` 找候选
   - query 应该**针对当前 shot 的具体时间位置和情绪**来写，不要每个 shot 都用同样的 query
   - 参考上级的 retrieval_query 作为方向，但每个 shot 要有变化（比如段落前半偏蓄势、后半偏爆发）
   - 好的 query 示例："角色特写+坚定眼神"、"远景+城市全貌+压迫感"、"快速动作+刀光剑影"
2. 看候选列表，选一个最匹配本 shot 意图的 scene_id
3. 调 `commit_clip(audio_start, audio_end, scene_id, transition_type, transition_duration)`
   - audio_start/end 严格按计划，不要改
   - 锚点位（计划里标注的 ★）用 flash_white@0.08，普通切点用 hard_cut
   - 系统会自动算 speed_factor，你不用管

## 锚点优先原则

标注 ★ 的 shot 是音乐的**燃点/爆发点/转折点**——观众最能感知到的高光时刻。
对这些 shot：
- **必须优先处理**：如果你还没到锚点 shot，可以先跳过当前 shot，优先搜索并 commit 锚点 shot，再回头填普通 shot
- **搜索时写更具体的 query**：比如"一刀斩杀瞬间+慢放"、"角色睁眼特写+爆发"、"爆炸冲击波+闪光"
- **选最高 arousal / is_climax_material 的 scene**：锚点位不要放平淡镜头
- 转场用 flash_white@0.08（已标注）

标注 ● 的是普通切点，正常处理即可。

## 多样性要求

素材库可能存在某个源文件 scene 数量特别多（比如"母亲回忆"），CLIP 检索会偏向它。
请**主动选择不同源文件**，避免连续多个 shot 都来自同一个 mp4。系统会在工作记忆里告诉你已用源文件分布。

## 工作流约束

- 全部 {n_shots} 个 shot 都 commit 完后立即结束，不要追加额外 shot
- 不要复用已 commit 的 scene_id（系统会自动过滤）
- 不要调用 get_music_overview 等额外信息工具——上面信息已经够
"""

LOCAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_scenes",
            "description": "搜索匹配镜头意图的视频场景，返回 top-K 候选。系统自动从音乐段落注入 target_mood / target_profile。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言镜头描述"},
                    "audio_start": {"type": "number", "description": "镜头开始时间（秒）"},
                    "k": {"type": "integer", "description": "返回数量，默认 8"},
                },
                "required": ["query", "audio_start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_detail",
            "description": "查看某个候选 scene 的完整信息（来源、时长、描述、visual_profile）。选 scene 前可用来对比。",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "integer"},
                },
                "required": ["scene_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_clip",
            "description": "提交一个镜头到时间线。系统自动校验 scene 时长并计算 speed_factor。",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_start": {"type": "number"},
                    "audio_end": {"type": "number"},
                    "scene_id": {"type": "integer"},
                    "transition_type": {
                        "type": "string",
                        "enum": ["hard_cut", "flash_white", "flash_black", "dissolve"],
                    },
                    "transition_duration": {"type": "number"},
                },
                "required": ["audio_start", "audio_end", "scene_id", "transition_type"],
            },
        },
    },
]


@dataclass
class PlannerResult:
    committed: list[dict] = field(default_factory=list)
    global_plan: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    finished: bool = False
    iterations: int = 0


def _build_shots_text(shot_specs: list[dict], anchors_at: dict[int, str]) -> str:
    lines = []
    for i, s in enumerate(shot_specs):
        mark = anchors_at.get(i, "")
        lines.append(
            f"  Shot {i+1}{mark}: 音乐 [{s['audio_start']:.2f}s - {s['audio_end']:.2f}s] dur={s['target_dur']:.2f}s"
        )
    return "\n".join(lines)


def _build_shot_specs(
    audio_segment: AudioSegment,
    audio_map: AudioMap,
    max_shots: int | None = None,
) -> tuple[list[dict], dict[int, str]]:
    """复用 v3 的 cut_points 计算，输出 shot_specs + 哪些是锚点（标★）。"""
    shot_durations, _ = compute_cut_points_and_energies(audio_segment, audio_map, max_shots=max_shots)
    shot_specs = []
    cursor = audio_segment.start
    for d in shot_durations:
        shot_specs.append({
            "audio_start": round(cursor, 3),
            "audio_end": round(cursor + d, 3),
            "target_dur": d,
        })
        cursor += d

    anchors_at: dict[int, str] = {}
    seg_anchors = [
        k for k in audio_map.key_moments_v2
        if audio_segment.start <= k.time <= audio_segment.end
    ]
    for i, s in enumerate(shot_specs):
        for k in seg_anchors:
            if abs(s["audio_start"] - k.time) < 0.3:
                if k.tier == "narrative_anchor":
                    anchors_at[i] = " ★必踩锚点（强烈推荐 flash_white@0.08）"
                elif k.tier == "section_beat":
                    anchors_at[i] = " ●切点（推荐 hard_cut）"
                break
    return shot_specs, anchors_at


def _build_working_memory(
    seg_committed: list[dict],
    scene_lookup: dict,
    shot_specs: list[dict],
    anchors_at: dict[int, str] | None = None,
) -> str:
    if not seg_committed:
        hint = ""
        if anchors_at and 0 in anchors_at:
            hint = f"\n⚡ Shot 1 是锚点（燃点）！优先选高 arousal / 动作爆发类素材。"
        return f"=== 进度 ===\n本段共 {len(shot_specs)} 个 shot，尚未提交任何。从 Shot 1 开始。{hint}\n==="

    src_counter: Counter = Counter()
    for c in seg_committed:
        scene = scene_lookup.get(c["scene_id"])
        if scene:
            src_counter[Path(scene.source_file).name] += 1

    lines = [f"=== 进度 ===", f"本段已提交 {len(seg_committed)}/{len(shot_specs)} 个 shot"]
    lines.append("已用源文件分布（注意均衡）：")
    for src, cnt in src_counter.most_common():
        warn = " ⚠⚠ 已偏多" if cnt >= 3 else ""
        lines.append(f"  {src}: {cnt}{warn}")
    next_idx = len(seg_committed)
    if next_idx < len(shot_specs):
        s = shot_specs[next_idx]
        lines.append(f"下一个 Shot {next_idx+1}: 音乐 [{s['audio_start']:.2f}s - {s['audio_end']:.2f}s]")
        if anchors_at and next_idx in anchors_at:
            lines.append(f"⚡ 这是锚点 shot！搜索时写具体动作 query，选最燃的素材。")
    lines.append("===")
    return "\n".join(lines)


async def stage2_react_segment(
    seg_guidance: dict,
    audio_segment: AudioSegment,
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    scene_lookup: dict[int, SceneItem],
    clip_embeddings: np.ndarray,
    clip_scene_ids: list[int],
    used_scene_ids: set[int],
    max_shots: int | None = None,
    max_iterations: int | None = None,
) -> tuple[list[dict], int]:
    label = audio_segment.name
    query = seg_guidance.get("retrieval_query", "")

    shot_specs, anchors_at = _build_shot_specs(audio_segment, audio_map, max_shots=max_shots)
    if not shot_specs:
        logger.warning(f"[v4 {label}] 无 shot，跳过")
        return [], 0

    # 动态迭代上限：每 shot 约需 2-3 轮（search + commit），留 10 轮缓冲
    if max_iterations is None:
        max_iterations = len(shot_specs) * 3 + 10

    shots_text = _build_shots_text(shot_specs, anchors_at)
    system_prompt = LOCAL_PROMPT.format(
        label=label,
        seg_start=audio_segment.start,
        seg_end=audio_segment.end,
        query=query,
        n_shots=len(shot_specs),
        shots_text=shots_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "开始处理 Shot 1。先 search_scenes，然后从候选里选最匹配的，调 commit_clip 提交。"},
    ]

    client = get_openai_client()
    seg_committed: list[dict] = []
    search_history: list[tuple[str, tuple[int, ...]]] = []
    iters = 0

    for iteration in range(max_iterations):
        iters = iteration + 1
        if len(seg_committed) >= len(shot_specs):
            logger.info(f"[v4 {label}] 完成 {len(shot_specs)} shots，共 {iters} 轮")
            break

        # 工作记忆刷新到 system prompt
        wm = _build_working_memory(seg_committed, scene_lookup, shot_specs, anchors_at)
        messages[0]["content"] = system_prompt + "\n\n" + wm

        logger.info(f"[v4 {label} iter {iters}] 调用 GPT-5.5（已 commit {len(seg_committed)}/{len(shot_specs)}）...")
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=GPT_5_5,
                    messages=messages,
                    tools=LOCAL_TOOL_SCHEMAS,
                    tool_choice="auto",
                ),
                timeout=300.0,
            )
        except Exception as e:
            logger.error(f"[v4 {label} iter {iters}] LLM error: {e}")
            break

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(msg.model_dump(exclude_unset=True))

        if finish_reason == "stop" or not getattr(msg, "tool_calls", None):
            if len(seg_committed) >= len(shot_specs):
                break
            messages.append({
                "role": "user",
                "content": f"还差 {len(shot_specs) - len(seg_committed)} 个 shot 未提交，请继续。",
            })
            continue

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except Exception:
                tool_args = {}

            result_text = await _execute_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                audio_map=audio_map,
                audio_segment=audio_segment,
                scene_table=scene_table,
                scene_lookup=scene_lookup,
                clip_embeddings=clip_embeddings,
                clip_scene_ids=clip_scene_ids,
                used_scene_ids=used_scene_ids,
                seg_committed=seg_committed,
                search_history=search_history,
            )

            args_short = json.dumps(tool_args, ensure_ascii=False)[:80]
            preview = result_text.replace("\n", " | ")[:160]
            logger.info(f"  [{label} iter {iters}] {tool_name}({args_short}) → {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

        # context 裁剪：保留 system + 用户初始 + 最近 16 条
        if len(messages) > 22:
            tail_start = len(messages) - 16
            while tail_start < len(messages) and messages[tail_start].get("role") == "tool":
                tail_start += 1
            messages = messages[:2] + messages[tail_start:]

    return seg_committed, iters


async def _execute_tool(
    tool_name: str,
    tool_args: dict,
    audio_map: AudioMap,
    audio_segment: AudioSegment,
    scene_table: list[SceneItem],
    scene_lookup: dict[int, SceneItem],
    clip_embeddings: np.ndarray,
    clip_scene_ids: list[int],
    used_scene_ids: set[int],
    seg_committed: list[dict],
    search_history: list,
) -> str:
    if tool_name == "search_scenes":
        target_mood = audio_segment.mood
        target_profile = getattr(audio_segment, "visual_profile", None) or {}
        return search_scenes(
            scene_table=scene_table,
            query=tool_args.get("query", ""),
            target_profile=target_profile,
            target_mood=target_mood,
            exclude_ids=set(used_scene_ids),
            k=int(tool_args.get("k", 8)),
            clip_embeddings=clip_embeddings,
            clip_scene_ids=clip_scene_ids,
            search_history=search_history,
        )

    if tool_name == "get_scene_detail":
        return get_scene_detail(scene_table, int(tool_args.get("scene_id", 0)))

    if tool_name == "commit_clip":
        # 注意：seg_committed 是本段的 buffer；committed 是全局——这里 v4 把 seg_committed 当 committed 传进去
        result = commit_clip(
            committed=seg_committed,
            audio_start=float(tool_args.get("audio_start", 0)),
            audio_end=float(tool_args.get("audio_end", 0)),
            scene_id=int(tool_args.get("scene_id", 0)),
            transition_type=tool_args.get("transition_type", "hard_cut"),
            transition_duration=float(tool_args.get("transition_duration", 0.0)),
            scene_lookup=scene_lookup,
        )
        if result.startswith("✓"):
            used_scene_ids.add(int(tool_args.get("scene_id", 0)))
        return result

    return f"未知工具：{tool_name}"


async def run_planner_v4(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    clip_embeddings: np.ndarray,
    clip_scene_ids: list[int],
    background_info: str = "",
    log_path: Path | None = None,
) -> PlannerResult:
    started = datetime.now()

    # Stage 1：复用 v3
    plan = await stage1_screenwriter(audio_map, scene_table, background_info)

    # Stage 2：每段独立 ReAct
    scene_lookup = {s.scene_id: s for s in scene_table}
    committed: list[dict] = []
    used_scene_ids: set[int] = set()
    total_iters = 0

    name_queue: dict[str, list[AudioSegment]] = {}
    for s in audio_map.segments:
        name_queue.setdefault(s.name, []).append(s)

    total_scenes = len(scene_table)
    total_dur = audio_map.total_duration
    budget = int(total_scenes * 0.85)

    for seg_guide in plan.get("segments", []):
        target_start = seg_guide.get("audio_start", -1)
        audio_seg = next(
            (s for s in audio_map.segments if abs(s.start - target_start) < 0.5),
            None,
        )
        if audio_seg is None:
            queue = name_queue.get(seg_guide["label"], [])
            if queue:
                audio_seg = queue.pop(0)
        if audio_seg is None:
            logger.warning(f"[v4] 无法匹配段落 {seg_guide['label']}")
            continue

        seg_dur = audio_seg.end - audio_seg.start
        seg_max_shots = max(2, int(budget * seg_dur / total_dur))
        logger.info(f"[v4] 开始段 {audio_seg.name}（配额 {seg_max_shots} shots）")

        seg_committed, iters = await stage2_react_segment(
            seg_guidance=seg_guide,
            audio_segment=audio_seg,
            audio_map=audio_map,
            scene_table=scene_table,
            scene_lookup=scene_lookup,
            clip_embeddings=clip_embeddings,
            clip_scene_ids=clip_scene_ids,
            used_scene_ids=used_scene_ids,
            max_shots=seg_max_shots,
        )
        total_iters += iters
        committed.extend(seg_committed)

    elapsed = (datetime.now() - started).total_seconds()
    covered = sum(c["audio_end"] - c["audio_start"] for c in committed)
    finished = abs(covered - audio_map.total_duration) < 1.0

    logger.info(
        f"[v4] 完成。{len(committed)} 片段 / {total_iters} 轮 / {elapsed:.1f}s / "
        f"覆盖 {covered:.1f}/{audio_map.total_duration:.1f}s = {covered/audio_map.total_duration*100:.1f}%"
    )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps({
                "version": "v4",
                "model": GPT_5_5,
                "elapsed_sec": elapsed,
                "iterations": total_iters,
                "finished": finished,
                "coverage": f"{covered:.1f}/{audio_map.total_duration:.1f}s",
                "global_plan": plan,
                "committed": committed,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return PlannerResult(
        committed=committed,
        global_plan=plan,
        elapsed_sec=elapsed,
        finished=finished,
        iterations=total_iters,
    )
