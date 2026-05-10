from __future__ import annotations

import json
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from models.audio import AudioMap
from models.plan import RenderItem
from models.video import SceneItem
from utils.clients import QWEN_MAX, QWEN_PLUS, get_qwen_client
from utils.embedder import build_scene_text, embed_texts, top_k_similar
from utils.prompt_loader import load_and_render, load_skill
from utils.snap_to_beats import snap_timeline

MOOD_GROUPS = {
    "high_energy": {
        "tense", "epic", "excited", "determined",
        "triumphant", "relentless", "resolute", "transcendent",
    },
    "sad": {"melancholic", "somber", "lonely", "anxious", "reflective"},
    "calm": {"calm", "peaceful", "intimate"},
}


async def generate_narrative_framework(
    audio_map: AudioMap,
    background_info: str | None,
    material_analysis: dict | None,
) -> dict:
    """
    第一步：根据 background_info + 音乐分析 + 素材类型，
    让 LLM 生成叙事骨架，约束后续意图规划。
    """
    client = get_qwen_client()

    narrative_stages = []
    material_type = "通用影视素材"

    if material_analysis:
        narrative_stages = material_analysis.get("narrative_stages", [])
        material_type = material_analysis.get("material_type", "通用影视素材")

    segments_brief = [
        f"{s.name}({s.start:.1f}-{s.end:.1f}s, energy={s.energy}, mood={s.mood})"
        for s in audio_map.segments
    ]

    prompt = f"""
你是一个专业的混剪导演。

## 作品背景
{background_info or "未提供"}

## 素材类型
{material_type}

## 素材的叙事阶段（AI 自动分析）
{", ".join(narrative_stages) if narrative_stages else "未知"}

## 音乐段落结构
{chr(10).join(segments_brief)}

## 任务
请根据以上信息，为这次混剪制定一个叙事骨架：
1. 把音乐段落和叙事阶段一一对应
2. 指定哪些段落是关键锚点（高潮、结尾、重要转折）
3. 给出每个段落的叙事方向约束

严格输出 JSON，不要有多余文字：

{{
  "narrative_summary": "整体叙事方向（一句话）",
  "segment_mapping": [
    {{
      "segment_name": "音乐段落名",
      "narrative_stage": "对应的叙事阶段",
      "anchor": true或false,
      "prefer_outro": true或false,
      "prefer_climax": true或false,
      "direction": "这个段落的画面方向约束（20-40字，具体说明应该选什么类型的画面）"
    }}
  ]
}}
"""

    try:
        response = await client.chat.completions.create(
            model=QWEN_MAX,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        framework = json.loads(clean)
        logger.info(f"[Edit Planner] 叙事骨架: {framework.get('narrative_summary')}")
        return framework
    except Exception as e:
        logger.warning(f"[Edit Planner] 叙事骨架生成失败: {e}，跳过")
        return {}


async def run_edit_planner(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    editing_style: str = "visual_driven",
    background_info: str | None = None,
    user_feedback: str | None = None,
    runtime_config: dict | None = None,
    material_analysis: dict | None = None,
) -> tuple[list[RenderItem], list[dict]]:
    logger.info("[Edit Planner] 开始 ReAct 叙事规划")
    return await _run_react_planner(
        audio_map=audio_map,
        scene_table=scene_table,
        scene_embeddings=scene_embeddings,
        scene_ids=scene_ids,
        editing_style=editing_style,
        background_info=background_info,
        user_feedback=user_feedback,
        runtime_config=runtime_config,
    )


def _energy_to_intensity(energy: int) -> tuple[float, float]:
    """Map audio segment energy (1-10) to scene intensity filter range."""
    if energy >= 8:
        return 0.65, 1.0
    elif energy >= 6:
        return 0.45, 0.75
    elif energy >= 4:
        return 0.25, 0.60
    else:
        return 0.0, 0.40


async def _run_screenwriter(
    audio_map: AudioMap,
    editing_style: str,
    background_info: str | None,
    user_feedback: str | None = None,
    avg_scene_dur: float = 3.0,
    max_scene_dur: float = 8.0,
) -> list[dict]:
    """
    Screenwriter phase: one Qwen Max call that pre-plans the entire shot list.
    Returns list of slots covering the full audio timeline.
    Each slot: {audio_start, audio_end, query, min_intensity, max_intensity,
                speed_factor, transition_type, transition_duration, anchor}
    """
    client = get_qwen_client()

    segments_lines = []
    for s in audio_map.segments:
        line = (
            f"  {s.name}: {s.start:.1f}s–{s.end:.1f}s  "
            f"energy={s.energy}  mood={s.mood}  trend={s.energy_trend}"
        )
        if s.description:
            line += f"  描述: {s.description[:60]}"
        if s.visual_suggestion:
            line += f"  视觉建议: {s.visual_suggestion[:60]}"
        segments_lines.append(line)
    segments_text = "\n".join(segments_lines)

    if audio_map.special_events:
        events_lines = []
        for e in sorted(audio_map.special_events, key=lambda x: x.time):
            ev = f"  {e.time:.1f}s  {e.type}"
            if e.intensity is not None:
                ev += f"  intensity={e.intensity:.2f}"
            events_lines.append(ev)
        special_events_text = "\n".join(events_lines)
    else:
        special_events_text = "  （无特殊事件）"

    r1_section = ""
    if audio_map.r1_understanding:
        r1_section = f"\n## 音乐感性理解（AI 导演视角，请充分参考）\n{audio_map.r1_understanding}\n"

    prompt = f"""你是一位专业的混剪导演，正在为一首音乐制定完整的镜头脚本。

## 音乐信息
BPM: {audio_map.bpm:.1f}  总时长: {audio_map.total_duration:.1f}s

## 音乐段落（代码分析）
{segments_text}

## 特殊事件（高潮锚点）
{special_events_text}
{r1_section}
## 剪辑风格
{editing_style}

## 背景信息
{background_info or "未提供"}

## 用户反馈
{user_feedback or "无"}

## 任务
请为整首音乐制定完整的镜头脚本，输出每个镜头的时间范围和搜索描述。

## 素材时长参考（重要）
素材库中场景的平均时长约为 {avg_scene_dur:.1f}s，最长约 {max_scene_dur:.1f}s。
请将每个镜头槽位的时长设置为 {avg_scene_dur:.1f}s 左右（不超过 {max_scene_dur:.1f}s）。
这样每个槽位恰好对应一个场景，每个镜头都有独立的画面描述，避免重复堆砌。

## 强度映射规则（仅供参考，系统会自动校正）
- energy 8-10 → min_intensity=0.65, max_intensity=1.0
- energy 6-7  → min_intensity=0.45, max_intensity=0.75
- energy 4-5  → min_intensity=0.25, max_intensity=0.60
- energy 1-3  → min_intensity=0.0,  max_intensity=0.40

## 输出格式
严格输出 JSON 数组，不要有多余文字：

[
  {{
    "audio_start": 0.0,
    "audio_end": 5.0,
    "query": "角色站在高处俯瞰城市，表情坚定，光线柔和",
    "min_intensity": 0.0,
    "max_intensity": 0.4,
    "speed_factor": 1.0,
    "transition_type": "hard_cut",
    "transition_duration": 0.0,
    "anchor": false
  }}
]

## 重要约束
1. 所有镜头必须连续覆盖整个音频（0.0 到 {audio_map.total_duration:.1f}s），不能有空缺
2. query 必须具体生动，描述画面内容、动作、情绪（不要写"高能场景"这种泛泛的描述）
3. 特殊事件时间点（如 explosion_after_silence、drop）必须是槽位边界，并设 anchor=true、使用 flash_white 或 flash_black 转场
4. 每个镜头都要有独特的 query，不要在相邻镜头里重复相同的描述词
5. speed_factor 变速规则：
   - energy 1-3 的柔和段落、情感特写、角色回忆：0.4-0.7（慢动作）
   - energy 4-7 的普通叙事段落：1.0（原速）
   - energy 8-10 的高潮爆发、战斗、追逐：1.2-1.8（加速）
   - drop/explosion_after_silence 后的第一个镜头：强制 1.5-2.0（极速卡点冲击感）
   - dissolve 转场镜头建议原速（1.0），加速镜头建议 hard_cut"""

    logger.info("[Screenwriter] 调用 Qwen Max 生成镜头脚本...")
    try:
        response = await client.chat.completions.create(
            model=QWEN_MAX,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=6000,
        )
        raw = response.choices[0].message.content
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        shot_plan = json.loads(clean)
        if not isinstance(shot_plan, list):
            raise ValueError("输出不是 JSON 数组")
        logger.info(f"[Screenwriter] 生成 {len(shot_plan)} 个镜头槽位")
        return shot_plan
    except Exception as e:
        logger.error(f"[Screenwriter] 失败: {e}")
        return []


def _search_best_scene(
    scene_table: list["SceneItem"],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    query: str,
    min_intensity: float,
    max_intensity: float,
    exclude_ids: set[int],
) -> int | None:
    """Hard-filter by intensity, then embedding search. Returns best scene_id or None."""
    from utils.embedder import embed_texts, top_k_similar

    filtered = [
        s for s in scene_table
        if min_intensity <= float((s.visual_profile or {}).get("arousal", 0.5)) <= max_intensity
        and s.scene_id not in exclude_ids
    ]
    if len(filtered) < 3:
        filtered = [s for s in scene_table if s.scene_id not in exclude_ids]
    if not filtered:
        return None

    query_vec = embed_texts([query])[0]
    id_to_idx = {sid: i for i, sid in enumerate(scene_ids)}
    valid_pairs = [(s.scene_id, id_to_idx[s.scene_id]) for s in filtered if s.scene_id in id_to_idx]
    if not valid_pairs:
        return None

    valid_ids, valid_indices = zip(*valid_pairs)
    pool_embeddings = scene_embeddings[list(valid_indices)]
    top = top_k_similar(
        query_embedding=query_vec,
        corpus_embeddings=pool_embeddings,
        k=1,
        id_list=list(valid_ids),
    )
    return top[0][0] if top else None


async def _run_react_planner(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    editing_style: str = "visual_driven",
    background_info: str | None = None,
    user_feedback: str | None = None,
    runtime_config: dict | None = None,
) -> tuple[list[RenderItem], list[dict]]:
    """Screenwriter → direct slot filling pipeline."""
    from utils.snap_to_beats import snap_to_beat

    # ── 计算素材时长统计 ──────────────────────────────────────────────────────
    durations = sorted(s.duration for s in scene_table)
    avg_scene_dur = sum(durations) / len(durations) if durations else 3.0
    max_scene_dur = durations[-1] if durations else 8.0
    # 用 p75 作为建议时长上限，避免极端长场景拉偏均值
    p75_dur = durations[int(len(durations) * 0.75)] if durations else 5.0

    # ── Phase 1: Screenwriter ─────────────────────────────────────────────────
    shot_plan = await _run_screenwriter(
        audio_map, editing_style, background_info, user_feedback,
        avg_scene_dur=round(avg_scene_dur, 1),
        max_scene_dur=round(p75_dur, 1),
    )

    if not shot_plan:
        logger.warning("[Planner] Screenwriter 失败，使用兜底规划")
        render_plan = _fallback_plan(audio_map, scene_table)
        return _finalize_plan(render_plan, audio_map, scene_table, scene_embeddings, scene_ids, runtime_config)

    logger.info(f"[Planner] Screenwriter 生成 {len(shot_plan)} 个槽位，avg_scene_dur={avg_scene_dur:.1f}s")

    # ── 后处理1：用音频 energy 覆盖 Screenwriter 的 intensity 范围 ────────────
    for slot in shot_plan:
        try:
            mid = (float(slot["audio_start"]) + float(slot["audio_end"])) / 2
        except (KeyError, TypeError):
            continue
        seg = next((s for s in audio_map.segments if s.start <= mid < s.end), None)
        energy = seg.energy if seg else 5
        slot["min_intensity"], slot["max_intensity"] = _energy_to_intensity(energy)

    # ── 后处理2：将槽位边界吸附到最近的 beat ─────────────────────────────────
    beat_arr = audio_map.beat_array
    if beat_arr:
        for slot in shot_plan:
            try:
                snapped_start, _ = snap_to_beat(float(slot["audio_start"]), beat_arr, tolerance=0.3)
                snapped_end, _ = snap_to_beat(float(slot["audio_end"]), beat_arr, tolerance=0.3)
                slot["audio_start"] = snapped_start
                slot["audio_end"] = snapped_end
            except (KeyError, TypeError):
                continue
        # 修复相邻槽位因吸附产生的微小重叠/空隙
        for i in range(1, len(shot_plan)):
            prev_end = shot_plan[i - 1].get("audio_end", 0)
            curr_start = shot_plan[i].get("audio_start", 0)
            if abs(float(curr_start) - float(prev_end)) < 0.15:
                shot_plan[i]["audio_start"] = prev_end

    # ── Phase 2: Direct slot filling ──────────────────────────────────────────
    scene_lookup = {s.scene_id: s for s in scene_table}
    committed: list[dict] = []
    used_ids: set[int] = set()

    for i, slot in enumerate(shot_plan):
        try:
            a_start = float(slot["audio_start"])
            a_end = float(slot["audio_end"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"[Slot {i+1}] 无效时间范围，跳过: {slot}")
            continue

        query = slot.get("query", "")
        min_int = float(slot.get("min_intensity", 0.0))
        max_int = float(slot.get("max_intensity", 1.0))
        speed = round(float(slot.get("speed_factor", 1.0)), 2)
        t_type = slot.get("transition_type", "hard_cut")
        t_dur = float(slot.get("transition_duration", 0.0))

        current_time = a_start
        clips_in_slot = 0

        # 内部循环：用同一个 query 填满整个槽位（场景短时多次搜索）
        while current_time < a_end - 0.05:
            scene_id = _search_best_scene(
                scene_table=scene_table,
                scene_embeddings=scene_embeddings,
                scene_ids=scene_ids,
                query=query,
                min_intensity=min_int,
                max_intensity=max_int,
                exclude_ids=used_ids,
            )

            if scene_id is None:
                logger.warning(f"[Slot {i+1}/{len(shot_plan)}] 素材耗尽，已填 {current_time - a_start:.1f}s")
                break

            scene = scene_lookup[scene_id]
            audio_remaining = a_end - current_time
            source_needed = audio_remaining * speed
            source_to_use = min(scene.duration, source_needed)
            actual_audio = source_to_use / speed

            committed.append({
                "audio_start": round(current_time, 3),
                "audio_end": round(current_time + actual_audio, 3),
                "scene_id": scene_id,
                "speed_factor": speed,
                # 只有槽位第一个片段应用转场
                "transition_type": t_type if clips_in_slot == 0 else "hard_cut",
                "transition_duration": t_dur if clips_in_slot == 0 else 0.0,
            })
            used_ids.add(scene_id)
            logger.info(
                f"[Slot {i+1}/{len(shot_plan)}] scene={scene_id} "
                f"[{current_time:.1f}s-{current_time + actual_audio:.1f}s] "
                f"src={source_to_use:.1f}s  arousal={(scene.visual_profile or {}).get('arousal', 0):.2f}"
            )
            current_time += actual_audio
            clips_in_slot += 1

    # ── Phase 3: Convert committed → RenderItem ───────────────────────────────
    render_plan: list[RenderItem] = []
    for i, c in enumerate(sorted(committed, key=lambda x: x["audio_start"]), start=1):
        scene = scene_lookup.get(c["scene_id"])
        if not scene:
            continue
        audio_dur = c["audio_end"] - c["audio_start"]
        speed = c.get("speed_factor", 1.0)
        source_dur = audio_dur * speed  # committed 时已经按实际 scene.duration 截断过
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

    if not render_plan:
        logger.warning("[Planner] 无有效片段，使用兜底规划")
        render_plan = _fallback_plan(audio_map, scene_table)

    return _finalize_plan(render_plan, audio_map, scene_table, scene_embeddings, scene_ids, runtime_config)


def _finalize_plan(
    render_plan: list[RenderItem],
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    runtime_config: dict | None,
) -> tuple[list[RenderItem], list[dict]]:
    """Coverage check → fill gaps → beat snap. Shared by all planner paths."""
    total_covered = sum(r.audio_end - r.audio_start for r in render_plan)
    coverage = total_covered / audio_map.total_duration
    logger.info(f"[Planner] 覆盖率: {total_covered:.1f}s / {audio_map.total_duration:.1f}s = {coverage:.1%}")

    if coverage < 0.7:
        logger.warning("[Planner] 覆盖率不足 70%，使用兜底规划")
        render_plan = _fallback_plan(audio_map, scene_table)

    if coverage < 0.99:
        render_plan = _fill_gaps(render_plan, audio_map, scene_table, scene_embeddings, scene_ids)
        total_covered = sum(r.audio_end - r.audio_start for r in render_plan)
        coverage = total_covered / audio_map.total_duration
        logger.info(f"[Planner] 补缺口后覆盖率: {total_covered:.1f}s / {audio_map.total_duration:.1f}s = {coverage:.1%}")

    tolerance = (runtime_config or {}).get("BEAT_SNAP_TOLERANCE", 0.15)
    render_dicts = [r.model_dump() for r in render_plan]
    render_dicts = snap_timeline(render_dicts, audio_map.beat_array, tolerance)
    render_plan = [RenderItem(**d) for d in render_dicts]

    logger.info(f"[Edit Planner] 完成，共 {len(render_plan)} 个片段")
    return render_plan, []


async def _run_edit_planner_legacy(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    editing_style: str = "visual_driven",
    background_info: str | None = None,
    user_feedback: str | None = None,
    runtime_config: dict | None = None,
    material_analysis: dict | None = None,
) -> tuple[list[RenderItem], list[dict]]:
    """旧版单次 LLM 规划（保留备用，不再调用）。"""
    logger.info("[Edit Planner Legacy] 开始叙事规划")

    # 叙事骨架生成已禁用（对简单素材会过度约束）
    narrative_framework = {}

    # ── 构建 prompt ───────────────────────────────────────────────────────────
    segments_json = json.dumps(
        [
            {
                "name": s.name,
                "start": s.start,
                "end": s.end,
                "energy": s.energy,
                "energy_trend": s.energy_trend,
                "energy_peak": s.energy_peak,
                "mood": s.mood,
                "description": s.description[:60] if s.description else "",
            }
            for s in audio_map.segments
        ],
        ensure_ascii=False, indent=2
    )

    special_events_json = (
        json.dumps(
            [e.model_dump(exclude_none=True) for e in audio_map.special_events],
            ensure_ascii=False, indent=2
        )
        if audio_map.special_events else "[]"
    )

    moods = Counter(s.mood for s in scene_table)
    durations = [s.duration for s in scene_table]

    mood_distribution = ", ".join(
        f"{mood}:{count}" for mood, count in sorted(moods.items(), key=lambda x: -x[1])
    )
    density_distribution = ""  # 已废弃，保留变量避免下游 KeyError
    duration_stats = (
        f"avg={sum(durations)/len(durations):.1f}s "
        f"min={min(durations):.1f}s "
        f"max={max(durations):.1f}s"
    )

    file_scene_groups: dict[str, list] = defaultdict(list)
    for s in scene_table:
        file_scene_groups[Path(s.source_file).name].append(s)
    source_lines = []
    for fname in sorted(file_scene_groups.keys()):
        fscenes = file_scene_groups[fname]
        fmoods = Counter(s.mood for s in fscenes)
        mood_str = ", ".join(f"{m}:{c}" for m, c in sorted(fmoods.items(), key=lambda x: -x[1]))
        source_lines.append(f"- {fname}（{len(fscenes)} 个场景，情绪: {mood_str}）")
    source_files_info = "\n".join(source_lines) if source_lines else "（无素材）"

    # 叙事骨架注入 prompt
    framework_section = ""
    if narrative_framework:
        framework_section = f"""
## 叙事骨架约束（必须严格遵守）

整体叙事：{narrative_framework.get('narrative_summary', '')}

各段落的叙事方向约束：
{json.dumps(narrative_framework.get('segment_mapping', []), ensure_ascii=False, indent=2)}

**重要：你的意图 JSON 必须和以上叙事骨架严格对应，
不能把结局画面放在开头，不能违反叙事阶段的顺序。**
"""

    narrative_skill = load_skill("narrative_skill")
    music_narrative_skill = load_skill("music_narrative_skill")

    system_prompt = load_and_render(
        "planning/edit_planner_system.md",
        narrative_skill=narrative_skill,
        music_narrative_skill=music_narrative_skill,
    )

    # 叙事骨架注入已禁用
    pass

    user_prompt = load_and_render(
        "planning/edit_planner_user.md",
        editing_style=editing_style,
        background_info=background_info or "未提供",
        user_feedback=user_feedback or "无",
        bpm=f"{audio_map.bpm:.1f}",
        total_duration=f"{audio_map.total_duration:.1f}",
        segments_json=segments_json,
        special_events_json=special_events_json,
        source_files_info=source_files_info,
        scene_count=len(scene_table),
        mood_distribution=mood_distribution,
        density_distribution=density_distribution,
        duration_stats=duration_stats,
    )

    logger.info("[Edit Planner] 调用 qwen-max 生成叙事意图")
    client = get_qwen_client()

    response = await client.chat.completions.create(
        model=QWEN_MAX,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=4000,
        extra_body={"enable_thinking": True},
    )

    msg = response.choices[0].message
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        logger.info("[Edit Planner] === 思考过程 ===")
        logger.info(msg.reasoning_content)

    intent_output = msg.content
    logger.info("[Edit Planner] === 意图 JSON 输出 ===")
    logger.info(intent_output)

    intents = _parse_intents(intent_output, audio_map)

    if not intents:
        logger.warning("[Edit Planner] 意图解析失败，使用默认意图")
        intents = _default_intents(audio_map)

    render_plan = _fill_timeline_by_embedding(
        intents=intents,
        scene_table=scene_table,
        scene_embeddings=scene_embeddings,
        scene_ids=scene_ids,
        total_duration=audio_map.total_duration,
    )

    if not render_plan:
        logger.warning("[Edit Planner] 向量检索失败，使用兜底规划")
        render_plan = _fallback_plan(audio_map, scene_table)
        intents = _default_intents(audio_map)

    total_audio_covered = sum(r.audio_end - r.audio_start for r in render_plan)
    coverage = total_audio_covered / audio_map.total_duration
    logger.info(f"[Edit Planner] 音频覆盖率: {total_audio_covered:.1f}s / {audio_map.total_duration:.1f}s = {coverage:.1%}")

    total_scene_duration = sum(s.duration for s in scene_table)
    if total_scene_duration < audio_map.total_duration:
        logger.warning(
            f"[Edit Planner] ⚠️ 素材总时长 {total_scene_duration:.1f}s < 音乐时长 {audio_map.total_duration:.1f}s，"
            f"缺口 {audio_map.total_duration - total_scene_duration:.1f}s，将自动复用素材填充"
        )

    if coverage < 0.7:
        logger.warning("[Edit Planner] 覆盖率不足 70%，使用兜底规划")
        render_plan = _fallback_plan(audio_map, scene_table)

    if coverage < 0.99:
        render_plan = _fill_gaps(render_plan, audio_map, scene_table, scene_embeddings, scene_ids)
        total_audio_covered = sum(r.audio_end - r.audio_start for r in render_plan)
        coverage = total_audio_covered / audio_map.total_duration
        logger.info(f"[Edit Planner] 补缺口后覆盖率: {total_audio_covered:.1f}s / {audio_map.total_duration:.1f}s = {coverage:.1%}")

    tolerance = (runtime_config or {}).get("BEAT_SNAP_TOLERANCE", 0.15)
    render_dicts = [r.model_dump() for r in render_plan]
    render_dicts = snap_timeline(render_dicts, audio_map.beat_array, tolerance)

    render_plan = [RenderItem(**d) for d in render_dicts]
    logger.info(f"[Edit Planner Legacy] 完成，共 {len(render_plan)} 个片段")

    return render_plan, intents


def _fill_timeline_by_embedding(
    intents: list[dict],
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    total_duration: float,
) -> list[RenderItem]:
    used_ids: set[int] = set()
    # 每个源视频已使用到的最大 end 时间戳，保证同一视频内场景不倒流
    last_end_by_source: dict[str, float] = {}
    scene_lookup = {s.scene_id: s for s in scene_table}
    plan: list[RenderItem] = []
    order = 1

    narrative_positions = _compute_narrative_positions(scene_table)
    logger.info(f"[Edit Planner] 叙事位置计算完成，共 {len(scene_table)} 个场景")

    anchor_intents = [i for i in intents if i.get("anchor")]
    normal_intents = [i for i in intents if not i.get("anchor")]

    def _run_intent(intent):
        nonlocal order
        result = _fill_segment_by_embedding(
            intent=intent,
            scene_table=scene_table,
            scene_embeddings=scene_embeddings,
            scene_ids=scene_ids,
            scene_lookup=scene_lookup,
            used_ids=used_ids,
            start_order=order,
            narrative_positions=narrative_positions,
            total_duration=total_duration,
            last_end_by_source=last_end_by_source,
        )
        plan.extend(result)
        order += len(result)
        # used_ids and last_end_by_source already updated in-place by _fill_segment_by_embedding

    for intent in anchor_intents:
        _run_intent(intent)
    for intent in normal_intents:
        _run_intent(intent)

    plan.sort(key=lambda r: r.audio_start)
    for i, r in enumerate(plan):
        r.order = i + 1

    return plan


def _compute_narrative_positions(scene_table: list[SceneItem]) -> dict[int, float]:
    """
    计算每个场景的全局叙事位置（0-1）。
    按 source_file 文件名排序，建立全局时间轴。
    假设用户按剧情顺序截取素材，文件名排序反映叙事顺序。
    """
    file_scenes: dict[str, list] = defaultdict(list)
    for s in scene_table:
        fname = Path(s.source_file).name
        file_scenes[fname].append(s)

    # 按文件名排序（反映截取顺序）
    sorted_files = sorted(file_scenes.keys())

    # 计算每个文件的全局偏移
    file_offsets = {}
    offset = 0.0
    for fname in sorted_files:
        file_offsets[fname] = offset
        scenes_in_file = file_scenes[fname]
        file_duration = max(s.end for s in scenes_in_file)
        offset += file_duration

    total_duration = offset if offset > 0 else 1.0

    # 计算每个场景的全局叙事位置
    positions = {}
    for s in scene_table:
        fname = Path(s.source_file).name
        global_time = file_offsets.get(fname, 0.0) + s.start
        positions[s.scene_id] = global_time / total_duration

    return positions


def _fill_segment_by_embedding(
    intent: dict,
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
    scene_lookup: dict,
    used_ids: set[int],
    start_order: int,
    narrative_positions: dict[int, float] | None = None,
    total_duration: float = 1.0,
    last_end_by_source: dict[str, float] | None = None,
) -> list[RenderItem]:
    audio_start = intent["audio_start"]
    audio_end = intent["audio_end"]
    intent_text = intent["intent"]

    prefer_outro = intent.get("prefer_outro", False)
    prefer_climax = intent.get("prefer_climax", False)
    temporal_mode = intent.get("temporal_mode", "free")
    prefer_sources = intent.get("prefer_sources", [])
    speed_factor = max(0.3, min(3.0, float(intent.get("speed_factor", 1.0))))
    transition_type = intent.get("transition_type", "hard_cut")
    transition_duration = float(intent.get("transition_duration", 0.0))

    query_vec = embed_texts([intent_text])[0]

    base_pool = scene_table
    if prefer_outro:
        outro_pool = [s for s in scene_table if s.is_outro_material]
        if outro_pool:
            base_pool = outro_pool
    elif prefer_climax:
        climax_pool = [s for s in scene_table if s.is_climax_material]
        if climax_pool:
            base_pool = climax_pool

    # Split into preferred pool and global fallback pool
    if prefer_sources:
        prefer_set = set(prefer_sources)
        primary_pool = [s for s in base_pool if Path(s.source_file).name in prefer_set]
        fallback_pool = [s for s in base_pool if Path(s.source_file).name not in prefer_set]
        if not primary_pool:
            logger.warning(f"  [{intent.get('name','?')}] prefer_sources={prefer_sources} 无匹配，使用全量素材池")
            primary_pool = base_pool
            fallback_pool = []
    else:
        primary_pool = base_pool
        fallback_pool = []

    def _pool_from(src: list[SceneItem], strict: bool = True) -> list[SceneItem]:
        out = []
        for s in src:
            if s.scene_id in used_ids:
                continue
            if strict and temporal_mode == "forward" and last_end_by_source:
                min_start = last_end_by_source.get(s.source_file, 0.0) - 0.5
                if s.start < min_start:
                    continue
            out.append(s)
        return out

    result = []
    current_time = audio_start
    order = start_order

    while current_time < audio_end - 0.1:
        pool = _pool_from(primary_pool, strict=True)
        if not pool and temporal_mode == "forward":
            pool = _pool_from(primary_pool, strict=False)
            if pool:
                logger.warning(f"  [{intent.get('name','?')}] forward 时序约束下无候选，放宽限制")
        if not pool and fallback_pool:
            pool = _pool_from(fallback_pool, strict=False)
            if pool:
                logger.info(f"  [{intent.get('name','?')}] prefer_sources 已耗尽，降级到全量素材库")

        if not pool:
            logger.warning(f"  [fill] 段落 {intent.get('name', '?')} 素材耗尽")
            break

        pool_ids = [s.scene_id for s in pool]
        pool_indices = [scene_ids.index(sid) for sid in pool_ids if sid in scene_ids]
        pool_embeddings = scene_embeddings[pool_indices]
        pool_scene_ids = [pool_ids[i] for i in range(len(pool_ids)) if pool_ids[i] in scene_ids]

        top = top_k_similar(
            query_embedding=query_vec,
            corpus_embeddings=pool_embeddings,
            k=30,
            exclude_ids=used_ids,
            id_list=pool_scene_ids,
        )

        if not top:
            logger.warning(f"  [fill] 段落 {intent.get('name', '?')} 向量检索无结果")
            break

        scene_id, sim_score = top[0]

        scene = scene_lookup.get(scene_id)
        if not scene:
            break

        used_ids.add(scene_id)
        if last_end_by_source is not None:
            src = scene.source_file
            last_end_by_source[src] = max(last_end_by_source.get(src, 0.0), scene.end)

        # 变速：慢镜需要取更多源素材（source_duration = audio_duration * speed_factor）
        audio_remaining = audio_end - current_time
        source_needed = audio_remaining * speed_factor
        source_to_use = min(scene.duration, source_needed)
        audio_covered = source_to_use / speed_factor

        # 只有段落第一个镜头切入时应用转场特效
        is_first = len(result) == 0
        item_cut_type = transition_type if is_first else "hard_cut"
        item_transition_dur = transition_duration if is_first else 0.0

        result.append(RenderItem(
            order=order,
            audio_start=round(current_time, 3),
            audio_end=round(current_time + audio_covered, 3),
            scene_id=scene_id,
            source_file=scene.source_file,
            clip_start=scene.start,
            clip_end=round(scene.start + source_to_use, 3),
            speed_factor=speed_factor,
            beat_snap_offset=0.0,
            cut_type=item_cut_type,
            transition_duration=item_transition_dur,
        ))

        logger.debug(
            f"  [{intent.get('name','?')}] scene {scene_id:03d} "
            f"sim={sim_score:.3f} "
            f"src={source_to_use:.1f}s audio={audio_covered:.1f}s"
        )

        current_time += audio_covered
        order += 1

    return result


def _parse_intents(raw: str, audio_map: AudioMap) -> list[dict]:
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        data = json.loads(clean)
        intents = data if isinstance(data, list) else data.get("segments", [])
        if intents:
            logger.info(f"[Edit Planner] 解析到 {len(intents)} 个意图段落")
            return intents
    except Exception as e:
        logger.warning(f"[Edit Planner] 意图解析失败: {e}")
    return _default_intents(audio_map)


def _default_intents(audio_map: AudioMap) -> list[dict]:
    intents = []
    for seg in audio_map.segments:
        is_anchor = seg.name in ("drop", "outro", "chorus") or seg.energy_peak >= 9
        intent_text = f"情绪：{seg.mood}，能量：{seg.energy}，段落：{seg.name}"
        if seg.description:
            intent_text += f"，{seg.description[:50]}"
        intents.append({
            "name": seg.name,
            "audio_start": seg.start,
            "audio_end": seg.end,
            "intent": intent_text,
            "anchor": is_anchor,
            "prefer_outro": seg.name == "outro",
            "prefer_climax": seg.name in ("drop", "chorus") and seg.energy_peak >= 8,
            "temporal_mode": "free",
            "prefer_sources": [],
            "speed_factor": 1.0,
            "transition_type": "hard_cut",
            "transition_duration": 0.0,
        })
    return intents


def _fill_gaps(
    render_plan: list[RenderItem],
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    scene_embeddings: np.ndarray,
    scene_ids: list[int],
) -> list[RenderItem]:
    """填充未覆盖的音频空隙，允许复用已有场景。"""
    sorted_plan = sorted(render_plan, key=lambda r: r.audio_start)
    scene_lookup = {s.scene_id: s for s in scene_table}
    order_base = max((r.order for r in sorted_plan), default=0) + 1

    # 找出所有空隙
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for r in sorted_plan:
        if r.audio_start - cursor > 0.1:
            gaps.append((cursor, r.audio_start))
        cursor = max(cursor, r.audio_end)
    if audio_map.total_duration - cursor > 0.1:
        gaps.append((cursor, audio_map.total_duration))

    if not gaps:
        return render_plan

    new_items: list[RenderItem] = []

    for gap_start, gap_end in gaps:
        logger.info(f"[Gap Fill] 填充空隙 {gap_start:.1f}s - {gap_end:.1f}s ({gap_end - gap_start:.1f}s)")

        from utils.clip_embedder import embed_texts_clip
        query_vec = embed_texts_clip([f"空隙填充，时间位置 {gap_start:.1f}s"])[0]
        current_time = gap_start
        reused: set[int] = set()

        while current_time < gap_end - 0.1:
            pool = [s for s in scene_table if s.scene_id not in reused]
            if not pool:
                reused.clear()
                pool = list(scene_table)
            if not pool:
                break

            pool_ids = [s.scene_id for s in pool]
            pool_indices = [scene_ids.index(sid) for sid in pool_ids if sid in scene_ids]
            pool_embeddings = scene_embeddings[pool_indices]
            pool_scene_ids = [pid for pid in pool_ids if pid in scene_ids]

            top = top_k_similar(
                query_embedding=query_vec,
                corpus_embeddings=pool_embeddings,
                k=1,
                exclude_ids=set(),
                id_list=pool_scene_ids,
            )
            if not top:
                break

            scene_id, _ = top[0]
            scene = scene_lookup.get(scene_id)
            if not scene:
                break

            reused.add(scene_id)
            clip_duration = min(scene.duration, gap_end - current_time)

            new_items.append(RenderItem(
                order=order_base,
                audio_start=round(current_time, 3),
                audio_end=round(current_time + clip_duration, 3),
                scene_id=scene_id,
                source_file=scene.source_file,
                clip_start=scene.start,
                clip_end=round(scene.start + clip_duration, 3),
                speed_factor=1.0,
                beat_snap_offset=0.0,
            ))
            logger.debug(f"  [gap] scene {scene_id:03d} → {current_time:.1f}s-{current_time + clip_duration:.1f}s (复用)")
            current_time += clip_duration
            order_base += 1

    render_plan = render_plan + new_items
    render_plan.sort(key=lambda r: r.audio_start)
    for i, r in enumerate(render_plan):
        r.order = i + 1
    return render_plan


def _fallback_plan(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
) -> list[RenderItem]:
    logger.warning("[Edit Planner] 使用兜底规划（顺序平铺）")
    plan = []
    audio_cursor = 0.0
    scene_idx = 0
    order = 1

    while audio_cursor < audio_map.total_duration and scene_idx < len(scene_table):
        scene = scene_table[scene_idx]
        audio_end = min(audio_cursor + scene.duration, audio_map.total_duration)
        plan.append(RenderItem(
            order=order,
            audio_start=round(audio_cursor, 3),
            audio_end=round(audio_end, 3),
            scene_id=scene.scene_id,
            source_file=scene.source_file,
            clip_start=scene.start,
            clip_end=scene.end,
            speed_factor=1.0,
            beat_snap_offset=0.0,
        ))
        audio_cursor = audio_end
        scene_idx += 1
        order += 1

    return plan

def patch_render_plan(
    render_plan: list[RenderItem],
    intents: list[dict],
    segment_issues: list,
    scene_table: list[SceneItem],
    scene_embeddings,
    scene_ids: list[int],
    audio_map: AudioMap,
    banned_scene_ids: set[int] | None = None,
) -> tuple[list[RenderItem], list[dict]]:
    """
    填空题模式：只修改有问题的段落，保留其他段落不动。
    """
    problem_segments = {si.segment_name for si in segment_issues}
    logger.info(f"[Patch] 需要修改的段落: {problem_segments}")

    intent_lookup = {i["name"]: i for i in intents}

    kept_items = []
    used_ids: set[int] = set(banned_scene_ids or [])  # 黑名单场景直接加入已用集合

    for r in render_plan:
        seg_name = _find_segment_name(r.audio_start, intents)
        if seg_name not in problem_segments:
            kept_items.append(r)
            used_ids.add(r.scene_id)

    logger.info(f"[Patch] 保留 {len(kept_items)} 条，重新处理 {len(render_plan) - len(kept_items)} 条")

    scene_lookup = {s.scene_id: s for s in scene_table}
    narrative_positions = _compute_narrative_positions(scene_table)
    order = max((r.order for r in kept_items), default=0) + 1

    for si in segment_issues:
        intent = intent_lookup.get(si.segment_name)
        if not intent:
            logger.warning(f"[Patch] 找不到段落 {si.segment_name} 的意图，跳过")
            continue

        if si.suggested_scene_ids:
            logger.info(f"[Patch] {si.segment_name}: 直接使用 scene_id {si.suggested_scene_ids}")
            new_items = _fill_with_forced_scenes(
                intent=intent,
                forced_ids=si.suggested_scene_ids,
                scene_lookup=scene_lookup,
                used_ids=used_ids,
                start_order=order,
                scene_table=scene_table,
                scene_embeddings=scene_embeddings,
                scene_ids=scene_ids,
                narrative_positions=narrative_positions,
                total_duration=audio_map.total_duration,
            )
        else:
            logger.info(f"[Patch] {si.segment_name}: 重新向量检索")
            new_items = _fill_segment_by_embedding(
                intent=intent,
                scene_table=scene_table,
                scene_embeddings=scene_embeddings,
                scene_ids=scene_ids,
                scene_lookup=scene_lookup,
                used_ids=used_ids,
                start_order=order,
                narrative_positions=narrative_positions,
                total_duration=audio_map.total_duration,
            )

        kept_items.extend(new_items)
        used_ids.update(r.scene_id for r in new_items)
        order += len(new_items)

    kept_items.sort(key=lambda r: r.audio_start)
    for i, r in enumerate(kept_items):
        r.order = i + 1

    logger.info(f"[Patch] 修复完成，共 {len(kept_items)} 条")
    return kept_items, intents


def _fill_with_forced_scenes(
    intent: dict,
    forced_ids: list[int],
    scene_lookup: dict,
    used_ids: set[int],
    start_order: int,
    scene_table=None,
    scene_embeddings=None,
    scene_ids=None,
    narrative_positions=None,
    total_duration: float = 1.0,
) -> list[RenderItem]:
    """先放 forced 场景，剩余时长继续向量检索填满。"""
    audio_start = intent["audio_start"]
    audio_end = intent["audio_end"]
    current_time = audio_start
    result = []
    order = start_order

    # 第一步：放置 forced 场景
    for scene_id in forced_ids:
        if scene_id in used_ids:
            continue
        scene = scene_lookup.get(scene_id)
        if not scene:
            continue
        if current_time >= audio_end - 0.1:
            break

        used_ids.add(scene_id)
        clip_duration = min(scene.duration, audio_end - current_time)

        result.append(RenderItem(
            order=order,
            audio_start=round(current_time, 3),
            audio_end=round(current_time + clip_duration, 3),
            scene_id=scene_id,
            source_file=scene.source_file,
            clip_start=scene.start,
            clip_end=round(scene.start + clip_duration, 3),
            speed_factor=1.0,
            beat_snap_offset=0.0,
        ))
        logger.info(f"  [forced] 锁定 scene {scene_id} → {current_time:.1f}s-{current_time+clip_duration:.1f}s")
        current_time += clip_duration
        order += 1

    # 第二步：剩余时长继续向量检索填满
    if current_time < audio_end - 0.1 and scene_table and scene_embeddings is not None and scene_ids:
        remaining = audio_end - current_time
        logger.info(f"  [forced] 剩余 {remaining:.1f}s 继续向量检索填满")
        remaining_intent = dict(intent)
        remaining_intent["audio_start"] = current_time
        extra = _fill_segment_by_embedding(
            intent=remaining_intent,
            scene_table=scene_table,
            scene_embeddings=scene_embeddings,
            scene_ids=scene_ids,
            scene_lookup=scene_lookup,
            used_ids=used_ids,
            start_order=order,
            narrative_positions=narrative_positions,
            total_duration=total_duration,
        )
        result.extend(extra)

    return result


def _find_segment_name(audio_start: float, intents: list[dict]) -> str:
    for intent in intents:
        if intent["audio_start"] <= audio_start < intent["audio_end"]:
            return intent["name"]
    return ""
