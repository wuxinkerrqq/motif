"""
planner_tools.py — ReAct edit planner 的工具函数集

每个函数接收运行时数据（audio_map / scene_table 等）和 LLM 传入的参数，
返回纯文本字符串供 LLM 阅读。不依赖 LangChain @tool，直接由 ReAct 循环调用。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.audio import AudioMap
    from models.video import SceneItem


# ── 音乐探索工具 ───────────────────────────────────────────────────────────────

def get_music_overview(audio_map: "AudioMap") -> str:
    """返回音乐整体结构概览（紧凑文本）。"""
    lines = [f"BPM: {audio_map.bpm:.1f}  总时长: {audio_map.total_duration:.1f}s"]

    narrative = getattr(audio_map, "narrative_summary", "") or ""
    if narrative:
        lines.append(f"叙事弧线: {narrative}")

    mood_arc = getattr(audio_map, "mood_arc", []) or []
    if mood_arc:
        lines.append(f"情绪轨迹: {' → '.join(mood_arc)}")

    km_v2_count = len(getattr(audio_map, "key_moments_v2", []) or [])
    km_v1_count = len(getattr(audio_map, "key_moments", []) or [])
    total_moments = km_v2_count or km_v1_count
    lines.append(f"段落数: {len(audio_map.segments)}  关键时刻数: {total_moments}")
    lines.append("")
    lines.append("段落列表（name | 时间 | energy | mood | V/A/D | pacing）：")

    for s in audio_map.segments:
        line = (
            f"  {s.name:20s} {s.start:6.1f}s–{s.end:6.1f}s  "
            f"energy={s.energy}  trend={s.energy_trend:15s}  mood={s.mood}"
        )
        vp = getattr(s, "visual_profile", None)
        if vp:
            line += f"  V/A/D={vp.get('valence',0):.2f}/{vp.get('arousal',0):.2f}/{vp.get('dominance',0):.2f}"
        pacing = getattr(s, "pacing_hint", None)
        if pacing and pacing.get("beats_per_shot_range"):
            r = pacing["beats_per_shot_range"]
            line += f"  pacing={r[0]}-{r[1]}拍/镜"
        if getattr(s, "energy_level", None):
            line += f"  level={s.energy_level}"
        lines.append(line)

    return "\n".join(lines)


def get_special_events(audio_map: "AudioMap") -> str:
    """返回关键剪辑锚点列表。优先 key_moments_v2。"""
    km_v2 = getattr(audio_map, "key_moments_v2", []) or []
    if km_v2:
        lines = ["关键锚点（★必踩 / ●尽量踩 / ·可选）："]
        for k in sorted(km_v2, key=lambda x: x.time):
            tier_mark = {
                "narrative_anchor": "★",
                "section_beat":     "●",
                "rhythmic_hit":     "·",
            }.get(k.tier, "?")
            vp = k.visual_profile or {}
            vad = ""
            if vp:
                vad = f" V/A/D={vp.get('valence',0):.2f}/{vp.get('arousal',0):.2f}/{vp.get('dominance',0):.2f}"
            desc = k.description or ""
            lines.append(
                f"  {tier_mark} {k.time:6.2f}s  {k.tier:18s}  "
                f"{k.anchor_type:30s}  imp={k.importance:.2f}  "
                f"trans={k.transition_recommendation}{vad}  | {desc}"
            )
        return "\n".join(lines)

    moments = getattr(audio_map, "key_moments", []) or []
    if not moments:
        return "无关键时刻数据"
    lines = ["关键时刻列表（time | type | emotion）："]
    for km in sorted(moments, key=lambda x: x.get("final_time_sec", 0)):
        t = km.get("final_time_sec", 0)
        ktype = km.get("type", "unknown")
        emotion = km.get("emotion", "")
        importance = km.get("importance", "")
        lines.append(f"  {t:6.1f}s  {ktype}  importance={importance}  emotion={emotion}")
    return "\n".join(lines)


def get_segment_detail(audio_map: "AudioMap", segment_name: str) -> str:
    """返回某个音乐段落的完整详情。"""
    seg = next((s for s in audio_map.segments if s.name == segment_name), None)
    if seg is None:
        available = [s.name for s in audio_map.segments]
        return f"未找到段落 '{segment_name}'。可用段落：{available}"
    lines = [
        f"段落: {seg.name}",
        f"时间: {seg.start:.1f}s – {seg.end:.1f}s  (时长 {seg.end - seg.start:.1f}s)",
        f"energy: {seg.energy}  energy_peak: {seg.energy_peak}  energy_trend: {seg.energy_trend}",
        f"mood: {seg.mood}",
        f"描述: {seg.description or '（无）'}",
    ]

    energy_level = getattr(seg, "energy_level", None)
    density_level = getattr(seg, "density_level", None)
    if energy_level or density_level:
        lines.append(f"分级: energy={energy_level or '?'}  density={density_level or '?'}")

    pacing = getattr(seg, "pacing_hint", None)
    if pacing:
        r = pacing.get("beats_per_shot_range", ["?", "?"])
        why = pacing.get("rationale", "")
        lines.append(f"剪辑节奏: {r[0]}-{r[1]} 拍/镜  理由: {why}")

    vp = getattr(seg, "visual_profile", None)
    if vp:
        lines.append(
            f"视觉特征 V/A/D: {vp.get('valence', 0):.2f}/"
            f"{vp.get('arousal', 0):.2f}/{vp.get('dominance', 0):.2f}  "
            f"motion={vp.get('motion_intensity', 0):.2f}  "
            f"grain={vp.get('grain', '?')}  pattern={vp.get('temporal_pattern', '?')}"
        )

    if seg.visual_suggestion:
        lines.append(f"视觉建议: {seg.visual_suggestion}")

    return "\n".join(lines)


# ── 场景搜索工具 ───────────────────────────────────────────────────────────────

# mood 兼容性：粗筛过滤明显跨类的情绪。同组内可互相替补。
MOOD_GROUPS = [
    {"epic", "triumphant", "determined", "excited", "joyful"},
    {"somber", "lonely", "melancholic", "nostalgic"},
    {"tense", "anxious", "determined"},
    {"peaceful", "calm", "intimate", "nostalgic"},
]

GRAIN_ORDER = {"detail": 0, "mid": 1, "broad": 2}
PATTERN_OPPOSITES = {("accelerating", "decelerating"), ("decelerating", "accelerating")}


def _mood_compatible(target_mood: str | None, scene_mood: str | None) -> bool:
    if not target_mood:
        return True
    t = target_mood.lower()
    s = (scene_mood or "").lower()
    if t == s:
        return True
    for group in MOOD_GROUPS:
        if t in group and s in group:
            return True
    return False


def _profile_distance(target: dict, scene_vp: dict) -> float:
    """target 与 scene visual_profile 的距离 [0,1]，越大越远。"""
    keys = ["valence", "arousal", "dominance", "motion_intensity"]
    diff_sq = sum((float(target.get(k, 0.5)) - float(scene_vp.get(k, 0.5))) ** 2 for k in keys)
    vad_dist = (diff_sq ** 0.5) / 2.0

    tg, sg = target.get("grain"), scene_vp.get("grain")
    if tg in GRAIN_ORDER and sg in GRAIN_ORDER:
        gd = abs(GRAIN_ORDER[tg] - GRAIN_ORDER[sg]) / 2.0
    else:
        gd = 0.5

    tp, sp = target.get("temporal_pattern"), scene_vp.get("temporal_pattern")
    if tp and sp:
        if tp == sp:
            pd = 0.0
        elif (tp, sp) in PATTERN_OPPOSITES:
            pd = 1.0
        else:
            pd = 0.5
    else:
        pd = 0.5

    return min(1.0, 0.75 * vad_dist + 0.15 * gd + 0.10 * pd)


def search_scenes(
    scene_table: list["SceneItem"],
    query: str,
    target_profile: dict | None = None,
    target_mood: str | None = None,
    exclude_ids: set[int] | None = None,
    k: int = 8,
    clip_embeddings: np.ndarray | None = None,
    clip_scene_ids: list[int] | None = None,
    search_history: list | None = None,
    # 兼容旧调用签名，已废弃，忽略
    scene_embeddings: np.ndarray | None = None,
    scene_ids: list[int] | None = None,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
    mood_filter: str | None = None,
    target_intensity: float | None = None,
    heuristic: str | None = None,
) -> str:
    """
    场景检索。final_score = 0.5 * CLIP_emb + 0.5 * (1 - profile_dist)
    target_mood 用于粗筛；target_profile 缺省则只走 CLIP。
    """
    from utils.embedder import top_k_similar
    from utils.clip_embedder import embed_texts_clip

    exclude_ids = exclude_ids or set()

    pool = [
        s for s in scene_table
        if s.scene_id not in exclude_ids and _mood_compatible(target_mood, s.mood)
    ]
    if len(pool) < 5:
        pool = [s for s in scene_table if s.scene_id not in exclude_ids]
    if not pool:
        return "素材库已耗尽，没有可用场景。"

    pool_ids = [s.scene_id for s in pool]

    if clip_embeddings is None or clip_scene_ids is None or len(clip_embeddings) == 0:
        return "无 CLIP embedding 索引，无法搜索。"

    query_vec = embed_texts_clip([query])[0]
    id_to_idx = {sid: i for i, sid in enumerate(clip_scene_ids)}
    valid_pairs = [(sid, id_to_idx[sid]) for sid in pool_ids if sid in id_to_idx]
    if not valid_pairs:
        return "无法找到有效的 embedding 索引。"
    valid_ids, valid_indices = zip(*valid_pairs)
    pool_emb = clip_embeddings[list(valid_indices)]

    initial_top = top_k_similar(
        query_embedding=query_vec,
        corpus_embeddings=pool_emb,
        k=min(k * 3, len(valid_ids)),
        id_list=list(valid_ids),
    )
    if not initial_top:
        return "语义搜索无结果。"

    scene_lookup = {s.scene_id: s for s in scene_table}
    use_profile = isinstance(target_profile, dict) and len(target_profile) > 0
    reranked: list[tuple[int, float, float, float]] = []
    for sid, emb_score in initial_top:
        s = scene_lookup.get(sid)
        if not s:
            continue
        if use_profile:
            dist = _profile_distance(target_profile, s.visual_profile or {})
            prof_match = 1.0 - dist
            final = 0.5 * emb_score + 0.5 * prof_match
        else:
            prof_match = 0.0
            final = emb_score
        reranked.append((sid, final, emb_score, prof_match))
    reranked.sort(key=lambda x: x[1], reverse=True)
    reranked = reranked[:k]

    # 幂等检测：如果本次 top-3 与历史某次 top-3 重合 ≥ 2，提示直接复用
    top3_ids = tuple(sid for sid, *_ in reranked[:3])
    if search_history is not None:
        for prev_query, prev_top3 in search_history:
            if len(set(top3_ids) & set(prev_top3)) >= 2:
                return (
                    f"⚠ 本次搜索 top-3={list(top3_ids)} 与之前 query='{prev_query[:30]}' "
                    f"的 top-3={list(prev_top3)} 高度重合。\n"
                    f"请直接从已有候选中 commit_clip，不要重复搜索。"
                )
        search_history.append((query, top3_ids))

    header = f"搜索结果（query='{query[:40]}', mood={target_mood or 'any'}, profile={'on' if use_profile else 'off'}）："
    lines = [header]
    for sid, final, emb, prof in reranked:
        s = scene_lookup[sid]
        vp = s.visual_profile or {}
        v = vp.get("valence", 0.5); a = vp.get("arousal", 0.5); d = vp.get("dominance", 0.5)
        m = vp.get("motion_intensity", 0.0)
        grain = vp.get("grain", "?"); pattern = vp.get("temporal_pattern", "?")
        desc_short = (s.scene_description[:60] + "…") if len(s.scene_description) > 60 else s.scene_description
        lines.append(
            f"  scene_id={sid:4d}  score={final:.3f}  (emb={emb:.2f} prof={prof:.2f})  "
            f"dur={s.duration:.1f}s  mood={s.mood}\n"
            f"           VAD={v:.2f}/{a:.2f}/{d:.2f}  motion={m:.2f}  grain={grain}  pat={pattern}\n"
            f"           来源: {Path(s.source_file).name}  [{s.start:.1f}s–{s.end:.1f}s]\n"
            f"           描述: {desc_short}"
        )
    return "\n".join(lines)


def get_scene_detail(scene_table: list["SceneItem"], scene_id: int) -> str:
    scene_lookup = {s.scene_id: s for s in scene_table}
    s = scene_lookup.get(scene_id)
    if s is None:
        return f"未找到 scene_id={scene_id}"
    vp = s.visual_profile or {}
    lines = [
        f"scene_id: {s.scene_id}",
        f"来源: {Path(s.source_file).name}  [{s.start:.1f}s–{s.end:.1f}s]  时长={s.duration:.1f}s",
        f"mood: {s.mood}    is_climax: {s.is_climax_material}  is_outro: {s.is_outro_material}",
        f"visual_profile: V/A/D={vp.get('valence', 0):.2f}/{vp.get('arousal', 0):.2f}/{vp.get('dominance', 0):.2f}  "
        f"motion={vp.get('motion_intensity', 0):.2f}  grain={vp.get('grain', '?')}  pattern={vp.get('temporal_pattern', '?')}",
        f"人物: {', '.join(s.characters) if s.characters else '（无）'}",
        f"描述: {s.scene_description}",
    ]
    return "\n".join(lines)


# ── 时间线工具 ─────────────────────────────────────────────────────────────────

def inspect_timeline(committed: list[dict], audio_map: "AudioMap") -> str:
    """返回当前时间线的覆盖情况。"""
    total = audio_map.total_duration
    if not committed:
        return f"时间线为空。总时长 {total:.1f}s，覆盖率 0%。"

    sorted_clips = sorted(committed, key=lambda c: c["audio_start"])
    covered = sum(c["audio_end"] - c["audio_start"] for c in sorted_clips)
    coverage_pct = covered / total * 100

    lines = [
        f"时间线状态：{len(sorted_clips)} 个片段，覆盖 {covered:.1f}s / {total:.1f}s = {coverage_pct:.1f}%",
        "",
        "已填充区间：",
    ]
    for c in sorted_clips:
        dur = c["audio_end"] - c["audio_start"]
        lines.append(
            f"  [{c['audio_start']:6.1f}s–{c['audio_end']:6.1f}s]  "
            f"scene={c['scene_id']}  dur={dur:.1f}s"
        )

    gaps = []
    prev_end = 0.0
    for c in sorted_clips:
        if c["audio_start"] - prev_end > 0.2:
            gaps.append((prev_end, c["audio_start"]))
        prev_end = max(prev_end, c["audio_end"])
    if total - prev_end > 0.2:
        gaps.append((prev_end, total))

    if gaps:
        lines.append("")
        lines.append("空缺区间：")
        for g_start, g_end in gaps:
            seg_names = []
            for seg in audio_map.segments:
                if seg.start < g_end and seg.end > g_start:
                    seg_names.append(seg.name)
            seg_hint = f"  (段落: {', '.join(seg_names)})" if seg_names else ""
            lines.append(f"  [{g_start:6.1f}s–{g_end:6.1f}s]  时长={g_end - g_start:.1f}s{seg_hint}")
    else:
        lines.append("")
        lines.append("✓ 无空缺，时间线已完整覆盖。")

    return "\n".join(lines)


def commit_clip(
    committed: list[dict],
    audio_start: float,
    audio_end: float,
    scene_id: int,
    speed_factor: float = 1.0,
    transition_type: str = "hard_cut",
    transition_duration: float = 0.0,
    scene_lookup: dict | None = None,
    speed_min: float = 0.5,
    speed_max: float = 1.5,
) -> str:
    """将一个镜头加入时间线。

    新增：scene_lookup 不为空时，自动基于 scene 实际时长校验 + 计算 speed_factor。
      - scene_dur >= target_dur            → speed=1.0，截前 target_dur 秒
      - target_dur*0.5 <= scene_dur < target_dur → speed=scene_dur/target_dur（慢放）
      - scene_dur < target_dur*0.5         → 拒绝（提示重选）
      - scene_dur > target_dur*2.0         → 截前段，speed=1.0（不加速避免观感差）
    """
    if audio_end <= audio_start:
        return f"错误：audio_end ({audio_end}) 必须大于 audio_start ({audio_start})"

    for c in committed:
        if audio_start < c["audio_end"] and audio_end > c["audio_start"]:
            return (
                f"警告：时间段 [{audio_start:.1f}s–{audio_end:.1f}s] 与已有片段 "
                f"[{c['audio_start']:.1f}s–{c['audio_end']:.1f}s] (scene={c['scene_id']}) 重叠，已跳过。"
            )

    target_dur = audio_end - audio_start
    scene_dur = None
    auto_msg = ""

    if scene_lookup is not None:
        scene = scene_lookup.get(scene_id)
        if scene is None:
            return f"错误：scene_id={scene_id} 不存在"
        scene_dur = float(scene.duration)

        ratio = scene_dur / target_dur if target_dur > 0 else 1.0

        if ratio < 0.5:
            return (
                f"❌ 拒绝：scene_id={scene_id} 时长仅 {scene_dur:.2f}s，"
                f"远低于目标音乐时长 {target_dur:.2f}s（比例 {ratio:.2f} < 0.5）。"
                f"必须慢放 >2× 才能填满，画面会失真。请换一个更长的 scene 或拆成多个镜头。"
            )

        if ratio < 1.0:
            speed_factor = max(speed_min, scene_dur / target_dur)
            auto_msg = f"  自动慢放 speed={speed_factor:.2f}（scene {scene_dur:.2f}s → 音乐 {target_dur:.2f}s）"
        else:
            speed_factor = 1.0
            auto_msg = f"  scene {scene_dur:.2f}s 充足，截前 {target_dur:.2f}s"

    committed.append({
        "audio_start": round(audio_start, 3),
        "audio_end": round(audio_end, 3),
        "scene_id": scene_id,
        "speed_factor": round(speed_factor, 3),
        "transition_type": transition_type,
        "transition_duration": round(transition_duration, 3),
        "scene_duration": round(scene_dur, 3) if scene_dur is not None else None,
    })

    audio_dur = audio_end - audio_start
    return (
        f"✓ 已添加：scene={scene_id}  音频 [{audio_start:.1f}s–{audio_end:.1f}s]  "
        f"时长={audio_dur:.1f}s  transition={transition_type}\n{auto_msg}"
    )


def remove_clip(committed: list[dict], audio_start: float) -> str:
    """从时间线中移除一个镜头（按 audio_start 定位）。"""
    for i, c in enumerate(committed):
        if abs(c["audio_start"] - audio_start) < 0.15:
            removed = committed.pop(i)
            return (
                f"✓ 已移除：scene={removed['scene_id']}  "
                f"[{removed['audio_start']:.1f}s–{removed['audio_end']:.1f}s]"
            )
    return f"未找到 audio_start≈{audio_start:.1f}s 的片段，无法移除"


def get_skill(skill_name: str) -> str:
    """按需加载专业剪辑知识。"""
    from utils.prompt_loader import load_skill as _load
    valid = {
        "narrative_skill":       "叙事结构、五段式时间线、特殊音乐事件处理规范",
        "music_narrative_skill": "音乐叙事技法（Leitmotif、对位法、音乐心理学等）",
        "editing_skill":         "镜头选择优先级、切点设计、动作衔接、变速处理",
    }
    if skill_name not in valid:
        return f"未知 skill：{skill_name}。可用：{list(valid.keys())}"
    return _load(skill_name)


# ── OpenAI function-calling schema ────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_skill",
            "description": "查阅专业剪辑知识。遇到复杂叙事结构、特殊音乐事件、或不确定如何处理某段时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "enum": ["narrative_skill", "music_narrative_skill", "editing_skill"],
                        "description": "查阅哪类剪辑知识",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_music_overview",
            "description": "获取音乐整体结构：BPM、总时长、所有段落的名称/时间/能量/情绪/V-A-D/pacing。第一步必须调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_special_events",
            "description": "获取关键剪辑锚点（★narrative_anchor 必踩 / ●section_beat 尽量踩 / ·rhythmic_hit 可选）及推荐转场。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_segment_detail",
            "description": "获取某个音乐段落的完整详情（含 visual_profile / pacing_hint）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_name": {"type": "string", "description": "段落名称，如 'drop' / 'chorus_1'"},
                },
                "required": ["segment_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_scenes",
            "description": (
                "搜索匹配的视频场景。评分 = 0.5 × CLIP语义 + 0.5 × visual_profile 距离匹配。"
                "target_mood 同组兼容；target_profile 直接抄当前音乐段落或锚点的 visual_profile。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言描述画面，如 '爆发瞬间，角色觉醒释放能量'"},
                    "target_mood": {
                        "type": "string",
                        "description": "当前音乐段落的 mood（如 epic/somber/tense/peaceful），用于粗筛，可选",
                    },
                    "target_profile": {
                        "type": "object",
                        "description": "目标 visual_profile，建议直接抄音乐段落/锚点的 visual_profile",
                        "properties": {
                            "valence": {"type": "number"},
                            "arousal": {"type": "number"},
                            "dominance": {"type": "number"},
                            "motion_intensity": {"type": "number"},
                            "grain": {"type": "string", "enum": ["detail", "mid", "broad"]},
                            "temporal_pattern": {
                                "type": "string",
                                "enum": ["accelerating", "decelerating", "stable", "pulsing"],
                            },
                        },
                    },
                    "k": {"type": "integer", "description": "返回数量，默认 8"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_detail",
            "description": "获取某个场景的完整信息（含 visual_profile 与描述）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "integer", "description": "场景 ID"},
                },
                "required": ["scene_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_timeline",
            "description": "查看当前时间线的覆盖情况：已填充区间、空缺位置、覆盖率。定期调用以了解进度。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_clip",
            "description": "将一个镜头加入时间线规划。确认 scene_id 合适后调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_start": {"type": "number", "description": "音频开始时间（秒）"},
                    "audio_end": {"type": "number", "description": "音频结束时间（秒）"},
                    "scene_id": {"type": "integer", "description": "使用的场景 ID"},
                    "transition_type": {
                        "type": "string",
                        "enum": ["hard_cut", "flash_white", "flash_black", "dissolve"],
                        "description": "切入特效；锚点处优先用音频侧的 transition_recommendation",
                    },
                    "transition_duration": {
                        "type": "number",
                        "description": "转场时长（秒）：flash 0.05-0.12，dissolve 0.2-0.4，hard_cut 填 0.0",
                    },
                },
                "required": ["audio_start", "audio_end", "scene_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_clip",
            "description": "从时间线中移除一个镜头。连贯性检查不通过时调用，然后换一个场景重试。",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_start": {"type": "number", "description": "要移除的镜头的 audio_start（秒）"},
                },
                "required": ["audio_start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_continuity",
            "description": "检查两个相邻场景的切换是否合理。commit_clip 已自动调用，通常不需要手动调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prev_scene_id": {"type": "integer"},
                    "next_scene_id": {"type": "integer"},
                    "cut_type": {
                        "type": "string",
                        "enum": ["hard_cut", "flash_white", "flash_black", "dissolve"],
                    },
                },
                "required": ["prev_scene_id", "next_scene_id"],
            },
        },
    },
]
