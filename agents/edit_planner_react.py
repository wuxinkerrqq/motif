"""
edit_planner_react.py — 基于真实 ReAct 工具调用的剪辑规划器

与旧版 Screenwriter 的区别：
- 逐步推理，每步调用工具，可回溯
- commit_clip 自动触发连贯性审查
- 情绪匹配优先于 embedding 相似度
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from agents.aesthetic_judge import judge_best_scene as _judge_best_scene
from agents.continuity_reviewer import check_continuity as _check_continuity
from agents.planner_tools import (
    TOOL_SCHEMAS,
    commit_clip,
    get_music_overview,
    get_scene_detail,
    get_segment_detail,
    get_skill,
    get_special_events,
    inspect_timeline,
    remove_clip,
    search_scenes,
)
from models.audio import AudioMap
from models.plan import RenderItem
from models.video import SceneItem
from utils.clients import QWEN_MAX, get_qwen_client
from utils.prompt_loader import load_prompt, load_skill
from utils.snap_to_beats import snap_timeline


async def run_react_planner_v2(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    background_info: str | None = None,
    editing_style: str = "visual_driven",
    user_feedback: str | None = None,
    runtime_config: dict | None = None,
    max_iterations: int = 200,
    clip_embeddings: np.ndarray | None = None,
    clip_scene_ids: list[int] | None = None,
) -> tuple[list[RenderItem], list[dict]]:
    logger.info("[ReAct Planner v2] 开始规划")
    if clip_embeddings is not None:
        logger.info(f"[ReAct Planner v2] 使用 CLIP 视觉 embedding（{len(clip_embeddings)} 个场景）")
    else:
        logger.info("[ReAct Planner v2] 使用文本 embedding（CLIP 未加载）")

    committed: list[dict] = []
    scene_lookup = {s.scene_id: s for s in scene_table}
    recent_queries: list[str] = []
    searches_since_commit: int = 0

    static_system_prompt = load_prompt("planning/react_planner_system.md")

    scene_stats = _build_scene_stats(scene_table)
    # 判断素材是否充足，用于工作记忆提示
    total_scene_duration = sum(s.duration for s in scene_table)
    material_scarce = total_scene_duration < audio_map.total_duration * 1.5

    initial_msg = (
        f"## 任务\n"
        f"为以下音乐制作完整的 AMV 剪辑规划。\n\n"
        f"## 背景信息\n{background_info or '未提供'}\n\n"
        f"## 剪辑风格\n{editing_style}\n\n"
        f"## 用户反馈\n{user_feedback or '无'}\n\n"
        f"## 素材库概况\n{scene_stats}\n\n"
        f"请先调用 get_music_overview 了解音乐结构，然后开始逐段规划。"
    )

    messages: list[dict] = [
        {"role": "system", "content": static_system_prompt},
        {"role": "user", "content": initial_msg},
    ]

    client = get_qwen_client()

    for iteration in range(max_iterations):
        # 每轮更新 system prompt，注入工作记忆摘要
        messages[0]["content"] = static_system_prompt + "\n\n" + _build_working_memory(
            committed, audio_map, material_scarce
        )

        response = await client.chat.completions.create(
            model=QWEN_MAX,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.4,
            max_tokens=2000,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        messages.append(msg.model_dump(exclude_unset=True))

        if finish_reason == "stop" or not getattr(msg, "tool_calls", None):
            gaps = _find_gaps(committed, audio_map.total_duration)
            if not gaps:
                logger.info(f"[ReAct Planner v2] 规划完成，共 {len(committed)} 个片段")
                break
            gap_desc = ", ".join(f"{g[0]:.1f}s–{g[1]:.1f}s" for g in gaps)
            messages.append({
                "role": "user",
                "content": f"还有未覆盖的区间：{gap_desc}，请继续填充。",
            })
            logger.info(f"[ReAct Planner v2] 第{iteration+1}轮：仍有空缺 {gap_desc}")
            continue

        tool_results: list[dict] = []
        seen_calls: set[str] = set()
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except Exception:
                tool_args = {}

            dedup_key = f"{tool_name}:{tc.function.arguments}"
            if dedup_key in seen_calls:
                logger.debug(f"[Tool] 跳过重复调用：{tool_name}")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "（重复调用已跳过，请使用上一次的结果）",
                })
                continue
            seen_calls.add(dedup_key)

            if tool_name == "search_scenes":
                searches_since_commit += 1
            elif tool_name == "commit_clip":
                searches_since_commit = 0

            result_text = await _execute_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                audio_map=audio_map,
                scene_table=scene_table,
                scene_embeddings=scene_embeddings,
                scene_ids=scene_ids,
                committed=committed,
                scene_lookup=scene_lookup,
                recent_queries=recent_queries,
                clip_embeddings=clip_embeddings,
                clip_scene_ids=clip_scene_ids,
            )

            if tool_name == "search_scenes" and searches_since_commit >= 3:
                gaps = _find_gaps(committed, audio_map.total_duration)
                next_gap = f"{gaps[0][0]:.1f}s–{gaps[0][1]:.1f}s" if gaps else "无空缺"
                result_text += (
                    f"\n\n🚨 强制提示：你已连续搜索 {searches_since_commit} 次但没有提交任何镜头！"
                    f"搜索结果中已有合适场景，请立即调用 commit_clip 提交一个场景。"
                    f"当前最优先填充的区间是 {next_gap}。"
                    f"不要再搜索了，直接提交！"
                )

            logger.debug(f"[Tool] {tool_name}({_truncate(str(tool_args), 80)}) → {_truncate(result_text, 120)}")

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

        messages.extend(tool_results)

        # context 裁剪：工作记忆已在 system prompt 中，历史只需保留最近 20 条
        if len(messages) > 24:
            tail_start = len(messages) - 20
            while tail_start < len(messages) and messages[tail_start].get("role") == "tool":
                tail_start += 1
            messages = messages[:2] + messages[tail_start:]

    else:
        logger.warning(f"[ReAct Planner v2] 达到最大迭代次数 {max_iterations}")

    # ── 转换为 RenderItem ──────────────────────────────────────────────────────
    render_plan = _committed_to_render_plan(committed, scene_lookup)

    if not render_plan:
        logger.warning("[ReAct Planner v2] 无有效片段，回退到兜底规划")
        from agents.edit_planner import _fallback_plan
        render_plan = _fallback_plan(audio_map, scene_table)

    # 填补剩余空缺
    from agents.edit_planner import _fill_gaps
    render_plan = _fill_gaps(render_plan, audio_map, scene_table, scene_embeddings, scene_ids)

    # beat snap
    tolerance = (runtime_config or {}).get("BEAT_SNAP_TOLERANCE", 0.15)
    render_dicts = [r.model_dump() for r in render_plan]
    render_dicts = snap_timeline(render_dicts, audio_map.beat_array, tolerance)
    render_plan = [RenderItem(**d) for d in render_dicts]

    total_covered = sum(r.audio_end - r.audio_start for r in render_plan)
    logger.info(
        f"[ReAct Planner v2] 完成：{len(render_plan)} 个片段，"
        f"覆盖 {total_covered:.1f}s / {audio_map.total_duration:.1f}s"
    )
    return render_plan, []


# ──────────────────────────────────────────────────────────────────────────────
# 工具执行器
# ──────────────────────────────────────────────────────────────────────────────

async def _execute_tool(
    tool_name: str,
    tool_args: dict,
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    committed: list[dict],
    scene_lookup: dict[int, SceneItem],
    recent_queries: list[str] | None = None,
    clip_embeddings: np.ndarray | None = None,
    clip_scene_ids: list[int] | None = None,
) -> str:
    if recent_queries is None:
        recent_queries = []

    if tool_name == "get_skill":
        return get_skill(tool_args.get("skill_name", ""))

    if tool_name == "get_music_overview":
        return get_music_overview(audio_map)

    if tool_name == "get_special_events":
        return get_special_events(audio_map)

    if tool_name == "get_segment_detail":
        return get_segment_detail(audio_map, tool_args.get("segment_name", ""))

    if tool_name == "search_scenes":
        query = tool_args.get("query", "")

        # 检测重复 query，注入警告
        warning = ""
        if query and recent_queries and query == recent_queries[-1]:
            warning = (
                "\n⚠️ 警告：你刚才用了完全相同的 query，结果不会有变化！"
                "请换一个完全不同的角度描述画面需求（换情绪/换视觉/换动作维度）。"
            )
        recent_queries.append(query)
        if len(recent_queries) > 10:
            recent_queries.pop(0)

        # 自动排除已 commit 的场景（避免重复选用）
        committed_ids = {c["scene_id"] for c in committed}

        # 从当前音乐段落自动推断 target_mood / target_profile
        # 让 LLM 不必手动复述音乐侧的 visual_profile，工具直接抄即可
        audio_start = float(tool_args.get("audio_start", 0.0))
        audio_seg = _infer_audio_segment(audio_map, audio_start)
        seg_obj = next((s for s in audio_map.segments if s.start <= audio_start < s.end), None)
        auto_mood = tool_args.get("target_mood") or audio_seg.get("mood")
        auto_profile = tool_args.get("target_profile")
        if not auto_profile and seg_obj is not None:
            auto_profile = getattr(seg_obj, "visual_profile", None)

        result = search_scenes(
            scene_table=scene_table,
            query=query,
            target_profile=auto_profile,
            target_mood=auto_mood,
            exclude_ids=committed_ids if committed_ids else None,
            k=int(tool_args.get("k", 8)),
            clip_embeddings=clip_embeddings,
            clip_scene_ids=clip_scene_ids,
        )

        # ── 审美判断：从候选中选出最佳场景 ──────────────────────────────────
        try:
            candidates, candidate_scores = _get_candidate_objects(
                result_text=result,
                scene_lookup=scene_lookup,
            )
            if candidates:
                # 构建当前音乐段落信息（从 tool_args 或 audio_map 推断）
                audio_segment = _infer_audio_segment(
                    audio_map=audio_map,
                    audio_start=float(tool_args.get("audio_start", 0.0)),
                )
                committed_scene_objs = [
                    scene_lookup[c["scene_id"]]
                    for c in committed
                    if c["scene_id"] in scene_lookup
                ]
                best_id, reason = await _judge_best_scene(
                    candidates=candidates,
                    candidate_scores=candidate_scores,
                    audio_segment=audio_segment,
                    committed_scenes=committed_scene_objs,
                )
                result += f"\n\n🎯 审美推荐：scene_id={best_id}（{reason}）\n建议直接 commit 这个场景。"
        except Exception as e:
            logger.debug(f"[AestheticJudge] 跳过：{e}")

        return result + warning

    if tool_name == "get_scene_detail":
        return get_scene_detail(scene_table, int(tool_args.get("scene_id", 0)))

    if tool_name == "inspect_timeline":
        return inspect_timeline(committed, audio_map)

    if tool_name == "commit_clip":
        audio_start = float(tool_args.get("audio_start", 0))
        audio_end = float(tool_args.get("audio_end", 0))
        scene_id = int(tool_args.get("scene_id", 0))
        speed = float(tool_args.get("speed_factor", 1.0))
        t_type = tool_args.get("transition_type", "hard_cut")
        t_dur = float(tool_args.get("transition_duration", 0.0))

        result = commit_clip(committed, audio_start, audio_end, scene_id, speed, t_type, t_dur)

        # 自动连贯性检查
        if result.startswith("✓") and len(committed) >= 2:
            sorted_clips = sorted(committed, key=lambda c: c["audio_start"])
            prev_clip = sorted_clips[-2]
            curr_clip = sorted_clips[-1]
            scene_a = scene_lookup.get(prev_clip["scene_id"])
            scene_b = scene_lookup.get(curr_clip["scene_id"])
            if scene_a and scene_b:
                continuity = await _check_continuity(scene_a, scene_b, t_type)
                if continuity["pass"]:
                    result += f"\n连贯性：✓ {continuity['reason']}"
                else:
                    result += (
                        f"\n连贯性：✗ {continuity['reason']}"
                        f"  建议：{continuity['suggestion']}"
                        f"\n→ 请调用 remove_clip(audio_start={audio_start}) 撤销，换一个场景重试。"
                    )

        return result

    if tool_name == "remove_clip":
        return remove_clip(committed, float(tool_args.get("audio_start", 0)))

    if tool_name == "check_continuity":
        prev_id = int(tool_args.get("prev_scene_id", 0))
        next_id = int(tool_args.get("next_scene_id", 0))
        cut_type = tool_args.get("cut_type", "hard_cut")
        scene_a = scene_lookup.get(prev_id)
        scene_b = scene_lookup.get(next_id)
        if not scene_a or not scene_b:
            return f"错误：找不到 scene_id={prev_id} 或 scene_id={next_id}"
        continuity = await _check_continuity(scene_a, scene_b, cut_type)
        if continuity["pass"]:
            return f"✓ 连贯性通过：{continuity['reason']}"
        return f"✗ 连贯性不通过：{continuity['reason']}。建议：{continuity['suggestion']}"

    return f"未知工具：{tool_name}"


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _build_working_memory(
    committed: list[dict],
    audio_map: AudioMap,
    material_scarce: bool = False,
) -> str:
    """生成实时工作记忆摘要，注入 system prompt，让 LLM 始终知道当前进度。"""
    total = audio_map.total_duration
    gaps = _find_gaps(committed, total)
    covered = total - sum(g[1] - g[0] for g in gaps)

    lines = ["=== 当前剪辑进度（实时更新）==="]

    if not committed:
        lines.append("尚未提交任何片段，请从头开始规划。")
    else:
        lines.append(f"已提交：{len(committed)} 个片段 | 覆盖 {covered:.1f}s / {total:.1f}s")

        # 已用场景统计（含复用次数）
        from collections import Counter
        scene_counts = Counter(c["scene_id"] for c in committed)
        used_str = ", ".join(
            f"#{sid}(×{cnt})" if cnt > 1 else f"#{sid}"
            for sid, cnt in sorted(scene_counts.items())
        )
        lines.append(f"已用场景：{used_str}")

        if material_scarce:
            lines.append("⚠ 素材较少，必要时可复用场景")
        else:
            lines.append("优先避免重复使用同一场景")

        # 最近 3 次提交
        recent = sorted(committed, key=lambda c: c["audio_start"])[-3:]
        lines.append("最近提交：")
        for c in recent:
            seg = _infer_audio_segment(audio_map, c["audio_start"])
            lines.append(
                f"  {c['audio_start']:.1f}s–{c['audio_end']:.1f}s  "
                f"#{c['scene_id']}  mood={seg.get('mood','?')}"
            )

    # 当前空缺
    if gaps:
        next_gap = gaps[0]
        seg = _infer_audio_segment(audio_map, next_gap[0])
        lines.append(
            f"下一个空缺：{next_gap[0]:.1f}s–{next_gap[1]:.1f}s  "
            f"→ 段落情绪={seg.get('mood','?')}  能量={seg.get('energy','?')}"
        )
        if len(gaps) > 1:
            lines.append(f"其余空缺：{', '.join(f'{g[0]:.1f}s–{g[1]:.1f}s' for g in gaps[1:])}")
    else:
        lines.append("✅ 时间线已全部覆盖，可以结束规划。")

    lines.append("===")
    return "\n".join(lines)


def _build_scene_stats(scene_table) -> str:
    from collections import Counter
    total = len(scene_table)
    durations = [s.duration for s in scene_table]
    avg_dur = sum(durations) / total if total else 0
    moods = Counter(s.mood for s in scene_table)
    mood_str = ", ".join(f"{m}:{c}" for m, c in moods.most_common(6))
    sources = Counter(Path(s.source_file).name for s in scene_table)
    src_str = ", ".join(f"{f}({c}个)" for f, c in sources.most_common())
    return (
        f"共 {total} 个场景，平均时长 {avg_dur:.1f}s\n"
        f"情绪分布：{mood_str}\n"
        f"来源文件：{src_str}"
    )


def _find_gaps(committed: list[dict], total_duration: float) -> list[tuple[float, float]]:
    if not committed:
        return [(0.0, total_duration)]
    sorted_clips = sorted(committed, key=lambda c: c["audio_start"])
    gaps = []
    cursor = 0.0
    for c in sorted_clips:
        if c["audio_start"] - cursor > 0.2:
            gaps.append((cursor, c["audio_start"]))
        cursor = max(cursor, c["audio_end"])
    if total_duration - cursor > 0.2:
        gaps.append((cursor, total_duration))
    return gaps


def _committed_to_render_plan(
    committed: list[dict],
    scene_lookup: dict[int, SceneItem],
) -> list[RenderItem]:
    render_plan = []
    for i, c in enumerate(sorted(committed, key=lambda x: x["audio_start"]), start=1):
        scene = scene_lookup.get(c["scene_id"])
        if not scene:
            logger.warning(f"[ReAct] scene_id={c['scene_id']} 不存在，跳过")
            continue
        audio_dur = c["audio_end"] - c["audio_start"]
        speed = c.get("speed_factor", 1.0)
        source_dur = min(audio_dur * speed, scene.duration)
        render_plan.append(RenderItem(
            order=i,
            audio_start=c["audio_start"],
            audio_end=c["audio_end"],
            scene_id=c["scene_id"],
            source_file=scene.source_file,
            clip_start=scene.start,
            clip_end=round(scene.start + source_dur, 3),
            speed_factor=speed,
            beat_snap_offset=0.0,
            cut_type=c.get("transition_type", "hard_cut"),
            transition_duration=c.get("transition_duration", 0.0),
        ))
    return render_plan


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _get_candidate_objects(
    result_text: str,
    scene_lookup: dict,
) -> tuple[list, list[float]]:
    """从 search_scenes 返回的文字中解析出 scene_id 和 score。"""
    import re
    candidates = []
    scores = []
    for match in re.finditer(r"scene_id=\s*(\d+)\s+score=([\d.]+)", result_text):
        sid = int(match.group(1))
        score = float(match.group(2))
        scene = scene_lookup.get(sid)
        if scene:
            candidates.append(scene)
            scores.append(score)
    return candidates, scores


def _infer_audio_segment(audio_map: AudioMap, audio_start: float) -> dict:
    """根据 audio_start 找到对应的音乐段落，返回段落信息字典。"""
    seg = next(
        (s for s in audio_map.segments if s.start <= audio_start < s.end),
        None,
    )
    if seg is None and audio_map.segments:
        seg = audio_map.segments[-1]
    if seg is None:
        return {"mood": "未知", "energy": 5, "energy_trend": "stable", "description": ""}
    return {
        "mood": seg.mood,
        "energy": seg.energy,
        "energy_trend": seg.energy_trend,
        "description": getattr(seg, "description", "") or "",
    }
