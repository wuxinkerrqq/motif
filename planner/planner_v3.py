"""
planner_v3.py — DIRECT 风格的算法驱动规划

架构：
  Stage 1（GPT-5.5 一次调用）：
    输入完整 audio_map → 输出每段的 retrieval_query + weight_profile
  Stage 2（纯算法，beam search）：
    对每段：
      1. 用音乐侧锚点+均分计算 cut_points / shot_durations
      2. 用每段的 visual_profile.arousal 作 target_energy
      3. 候选池 prompt 排序 top-60
      4. Beam search 选最优 scene 序列
  - 不让 LLM 选 scene
  - 不让 LLM 算时间
  - 只让 LLM 写 query 和选 weight profile
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from planner.beam_search import beam_search_segment, build_candidate_pool
from planner.scoring import get_weight_profile
from models.audio import AudioMap, AudioSegment, KeyMoment
from models.video import SceneItem
from utils.clients import GPT_5_5, get_openai_client
from utils.clip_embedder import embed_texts_clip

SCREENWRITER_PROMPT = Path(__file__).parent / "prompts" / "v3_screenwriter.md"


@dataclass
class PlannerResult:
    committed: list[dict] = field(default_factory=list)
    global_plan: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    finished: bool = False


# ── Stage 1：GPT-5.5 段级指引 ────────────────────────────────────────────────

async def stage1_screenwriter(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    background_info: str,
) -> dict:
    system_prompt = SCREENWRITER_PROMPT.read_text(encoding="utf-8")
    music_desc = _format_music(audio_map)
    scene_desc = _format_scenes_summary(scene_table)

    user_msg = (
        f"## 背景\n{background_info}\n\n"
        f"## 音乐结构\n{music_desc}\n\n"
        f"## 素材库摘要\n{scene_desc}\n\n"
        f"请为每个段落输出 retrieval_query + weight_profile。"
    )

    client = get_openai_client()
    logger.info("[v3 stage1] 调用 GPT-5.5...")
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=GPT_5_5,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        ),
        timeout=600.0,
    )
    raw = response.choices[0].message.content
    plan = _parse_json(raw)
    if not plan:
        raise RuntimeError(f"Stage1 解析失败:\n{raw[:500]}")

    logger.info(f"[v3 stage1] 完成，{len(plan.get('segments', []))} 段指引")
    for seg in plan.get("segments", []):
        logger.info(
            f"  [{seg['label']}] profile={seg.get('weight_profile', '?')}  "
            f"query={seg.get('retrieval_query', '')[:60]}..."
        )
    return plan


def _format_music(audio_map: AudioMap) -> str:
    lines = [
        f"BPM: {audio_map.bpm:.1f}  总时长: {audio_map.total_duration:.1f}s",
        f"叙事弧线: {audio_map.narrative_summary}",
        f"情绪轨迹: {' → '.join(audio_map.mood_arc)}",
        "",
        "段落列表：",
    ]
    for s in audio_map.segments:
        vp = getattr(s, "visual_profile", None) or {}
        line = (
            f"  - {s.name}  [{s.start:.1f}s–{s.end:.1f}s]  "
            f"energy={getattr(s, 'energy_level', '?')}  mood={s.mood}  "
            f"V/A/D={vp.get('valence',0):.2f}/{vp.get('arousal',0):.2f}/{vp.get('dominance',0):.2f}"
        )
        lines.append(line)
        if s.description:
            lines.append(f"    {s.description}")
    return "\n".join(lines)


def _format_scenes_summary(scene_table: list[SceneItem]) -> str:
    from collections import Counter
    total = len(scene_table)
    moods = Counter(s.mood for s in scene_table)
    mood_str = ", ".join(f"{m}:{c}" for m, c in moods.most_common(8))
    return f"共 {total} 个 scene，情绪分布：{mood_str}"


def _parse_json(raw: str) -> dict | None:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ── 切点 / 能量曲线生成（音乐侧客观计算）─────────────────────────────────────

def compute_cut_points_and_energies(
    audio_segment: AudioSegment,
    audio_map: AudioMap,
    max_shots: int | None = None,
) -> tuple[list[float], list[float]]:
    """
    给一个音乐段落算 shot 切点 + 每个 shot 的目标能量。
    切点策略：
      1. 段内的 ★narrative_anchor / ●section_beat 作为强制切点
      2. 不够的位置按 pacing_hint 的 beats_per_shot 均分补足
      3. shot 总数受 max_shots 上限约束（避免素材吃光）
    """
    seg_start, seg_end = audio_segment.start, audio_segment.end
    seg_dur = seg_end - seg_start

    # 段内的 ●/★ 锚点
    anchors = sorted([
        k for k in audio_map.key_moments_v2
        if seg_start < k.time < seg_end
        and k.tier in ("narrative_anchor", "section_beat")
    ], key=lambda k: k.time)
    anchor_times = [k.time for k in anchors]

    # pacing 推算每镜目标时长
    pacing = getattr(audio_segment, "pacing_hint", None) or {}
    bps_range = pacing.get("beats_per_shot_range", [4, 8])
    bpm = audio_map.bpm or 120.0
    # 高能段强制最低 1.5s/镜，避免画面过碎
    target_shot_dur = (bps_range[0] + bps_range[1]) / 2 * 60.0 / bpm
    target_shot_dur = max(1.5, target_shot_dur)

    # 期望镜头数
    n_target = max(len(anchor_times) + 1, round(seg_dur / target_shot_dur))
    if max_shots is not None:
        n_target = min(n_target, max_shots)
        n_target = max(n_target, len(anchor_times) + 1)  # 保锚点

    # 按锚点分块，块内均分
    boundaries = [seg_start] + anchor_times + [seg_end]
    cut_points = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        block_dur = b - a
        block_n = max(1, round(block_dur / seg_dur * n_target))
        for j in range(1, block_n + 1):
            cut_points.append(a + block_dur * j / block_n)

    # 去重 + 排序
    cut_points = sorted(set(round(c, 3) for c in cut_points))
    # 保证最后一个切点是 seg_end
    if abs(cut_points[-1] - seg_end) > 0.05:
        cut_points.append(seg_end)
        cut_points = sorted(set(round(c, 3) for c in cut_points))

    # shot_durations
    shot_durations = []
    cursor = seg_start
    for cp in cut_points:
        shot_durations.append(round(cp - cursor, 3))
        cursor = cp

    # 合并过短的 shot（< 0.3s）：把它的时长加到前一个 shot 上
    MIN_SHOT_DUR = 0.3
    merged_durs = []
    merged_cuts = []
    for d, cp in zip(shot_durations, cut_points):
        if d < MIN_SHOT_DUR and merged_durs:
            merged_durs[-1] = round(merged_durs[-1] + d, 3)
            merged_cuts[-1] = cp
        else:
            merged_durs.append(d)
            merged_cuts.append(cp)
    shot_durations = merged_durs
    cut_points = merged_cuts

    # 每个 shot 的 target_energy = 段落 visual_profile.arousal
    # 锚点位置 boost：附近的 shot 用锚点自己的 arousal
    seg_vp = getattr(audio_segment, "visual_profile", None) or {}
    base_energy = float(seg_vp.get("arousal", 0.5))
    shot_energies = [base_energy] * len(shot_durations)
    for i, cp in enumerate(cut_points):
        for k in anchors:
            if abs(cp - k.time) < 0.5:
                kvp = k.visual_profile or {}
                if kvp:
                    shot_energies[i] = float(kvp.get("arousal", base_energy))
                break

    return shot_durations, shot_energies


# ── Stage 2：算法填充 ────────────────────────────────────────────────────────

def stage2_fill_segment(
    seg_guidance: dict,
    audio_segment: AudioSegment,
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    clip_embeddings: np.ndarray,
    clip_scene_ids: list[int],
    used_scene_ids: set[int],
    last_scene_clip: np.ndarray | None,
    last_scene_motion: float | None,
    max_shots: int | None = None,
) -> tuple[list[dict], np.ndarray | None, float | None]:
    """返回 (committed, last_clip, last_motion) 用于段间衔接。"""
    label = seg_guidance["label"]
    query = seg_guidance["retrieval_query"]
    profile_name = seg_guidance.get("weight_profile", "Default_Priority")

    # 1) 切点 + 能量
    shot_durations, shot_energies = compute_cut_points_and_energies(
        audio_segment, audio_map, max_shots=max_shots
    )
    logger.info(
        f"[v3 stage2 {label}] {len(shot_durations)} shots, "
        f"durs={[round(d,2) for d in shot_durations]}"
    )

    # 2) prompt embed
    prompt_embed = embed_texts_clip([query])[0]

    # 3) 候选池：高能 profile 时启用 climax_material boost
    is_high_energy = profile_name in ("Energy_Priority", "Visual_Complexity_Priority")
    pool = build_candidate_pool(
        scene_table=scene_table,
        clip_embeddings=clip_embeddings,
        clip_scene_ids=clip_scene_ids,
        prompt_embed=prompt_embed,
        used_scene_ids=used_scene_ids,
        pool_size=60,
        is_high_energy=is_high_energy,
    )
    logger.info(f"[v3 stage2 {label}] 候选池 {len(pool)} 个")

    # 4) score config
    config = get_weight_profile(profile_name)
    config.prompt_embed = prompt_embed

    # 5) beam search
    result = beam_search_segment(
        pool=pool,
        clip_embeddings=clip_embeddings,
        clip_scene_ids=clip_scene_ids,
        shot_durations=shot_durations,
        shot_energies=shot_energies,
        score_config=config,
        last_scene_clip=last_scene_clip,
        last_scene_motion=last_scene_motion,
    )
    if result is None:
        logger.error(f"[v3 stage2 {label}] beam search 失败")
        return [], last_scene_clip, last_scene_motion

    # 6) 转 committed
    sid_to_idx = {sid: i for i, sid in enumerate(clip_scene_ids)}
    scene_lookup = {s.scene_id: s for s in scene_table}
    committed: list[dict] = []
    cursor = audio_segment.start
    last_clip_out = last_scene_clip
    last_motion_out = last_scene_motion

    # 取本段的锚点列表，用于切点位特殊转场
    seg_anchors = sorted([
        k for k in audio_map.key_moments_v2
        if audio_segment.start <= k.time <= audio_segment.end
        and k.tier in ("narrative_anchor", "section_beat")
    ], key=lambda k: k.time)

    def _find_anchor_at(t: float):
        for k in seg_anchors:
            if abs(k.time - t) < 0.3:
                return k
        return None

    for pick in result.picks:
        scene = scene_lookup.get(pick.scene_id)
        if scene is None:
            continue
        target_dur = pick.target_dur
        scene_dur = scene.duration

        # 自动算 speed_factor（同 commit_clip 的逻辑）
        if scene_dur >= target_dur:
            speed = 1.0
        else:
            speed = max(0.5, scene_dur / target_dur)

        # 锚点位特殊转场：★/● 锚点处用 transition_recommendation
        # 高能锚点（importance>0.85 + section_change）即使推荐 hard_cut 也强制升级 flash_white
        anchor = _find_anchor_at(cursor)
        if anchor:
            trans_type = anchor.transition_recommendation.split("@")[0]
            trans_dur = 0.08
            if "@" in anchor.transition_recommendation:
                try:
                    trans_dur = float(anchor.transition_recommendation.split("@")[1].rstrip("s"))
                except Exception:
                    pass
            # 高能 section_change 强制 flash_white
            if (anchor.importance >= 0.85
                and anchor.tier == "narrative_anchor"
                and "section_change" in (anchor.anchor_type or "")):
                trans_type = "flash_white"
                trans_dur = 0.08
        else:
            trans_type = "hard_cut"
            trans_dur = 0.0

        committed.append({
            "audio_start": round(cursor, 3),
            "audio_end": round(cursor + target_dur, 3),
            "scene_id": pick.scene_id,
            "speed_factor": round(speed, 3),
            "transition_type": trans_type,
            "transition_duration": round(trans_dur, 3),
            "score": {
                "prompt": round(pick.score.prompt, 3),
                "semantic": round(pick.score.semantic, 3),
                "motion": round(pick.score.motion, 3),
                "energy": round(pick.score.energy, 3),
                "combined": round(pick.score.combined, 3),
            },
            "scene_duration": round(scene_dur, 3),
        })
        cursor += target_dur
        used_scene_ids.add(pick.scene_id)

        idx = sid_to_idx.get(pick.scene_id)
        if idx is not None:
            last_clip_out = clip_embeddings[idx]
        last_motion_out = float((scene.visual_profile or {}).get("motion_intensity", 0.0))

    return committed, last_clip_out, last_motion_out


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def run_planner_v3(
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    clip_embeddings: np.ndarray,
    clip_scene_ids: list[int],
    background_info: str = "",
    log_path: Path | None = None,
) -> PlannerResult:
    started = datetime.now()

    # Stage 1
    plan = await stage1_screenwriter(audio_map, scene_table, background_info)

    # Stage 2: 逐段 beam search
    committed: list[dict] = []
    used_scene_ids: set[int] = set()
    last_clip = None
    last_motion = None

    seg_lookup = {s.name: s for s in audio_map.segments}

    # 同名段（如两个 chorus）按出现顺序消费
    name_to_queue: dict[str, list[AudioSegment]] = {}
    for s in audio_map.segments:
        name_to_queue.setdefault(s.name, []).append(s)

    # 按段落时长占比分配 shot 配额，避免单段吃光素材
    # 留 20% 余量给候选池剪裁
    total_scenes = len(scene_table)
    total_dur = audio_map.total_duration
    budget = int(total_scenes * 0.85)

    for seg_guide in plan.get("segments", []):
        label = seg_guide["label"]
        target_start = seg_guide.get("audio_start", -1)

        # 优先按 audio_start 精确匹配（处理同名段）
        audio_seg = next(
            (s for s in audio_map.segments if abs(s.start - target_start) < 0.5),
            None,
        )
        # 兜底：按 name 队列消费
        if audio_seg is None:
            queue = name_to_queue.get(label, [])
            if queue:
                audio_seg = queue.pop(0)
        if audio_seg is None:
            logger.warning(f"[v3] 无法匹配段落 {label} (start={target_start})")
            continue

        seg_dur = audio_seg.end - audio_seg.start
        seg_max_shots = max(2, int(budget * seg_dur / total_dur))
        logger.info(f"[v3] 段 {label} 配额 {seg_max_shots} shots")

        seg_committed, last_clip, last_motion = stage2_fill_segment(
            seg_guidance=seg_guide,
            audio_segment=audio_seg,
            audio_map=audio_map,
            scene_table=scene_table,
            clip_embeddings=clip_embeddings,
            clip_scene_ids=clip_scene_ids,
            used_scene_ids=used_scene_ids,
            last_scene_clip=last_clip,
            last_scene_motion=last_motion,
            max_shots=seg_max_shots,
        )
        committed.extend(seg_committed)

    elapsed = (datetime.now() - started).total_seconds()
    total_dur = audio_map.total_duration
    covered = sum(c["audio_end"] - c["audio_start"] for c in committed)
    finished = abs(covered - total_dur) < 1.0

    logger.info(
        f"[v3] 完成。{len(committed)} 片段 / {elapsed:.1f}s / "
        f"覆盖 {covered:.1f}/{total_dur:.1f}s = {covered/total_dur*100:.1f}%"
    )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps({
                "version": "v3",
                "model": GPT_5_5,
                "elapsed_sec": elapsed,
                "finished": finished,
                "coverage": f"{covered:.1f}/{total_dur:.1f}s = {covered/total_dur*100:.1f}%",
                "global_plan": plan,
                "committed": committed,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[v3] log 已保存：{log_path}")

    return PlannerResult(
        committed=committed,
        global_plan=plan,
        elapsed_sec=elapsed,
        finished=finished,
    )
