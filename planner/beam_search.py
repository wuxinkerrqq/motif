"""
beam_search.py — 段内 beam search 选最优 scene 序列

参考 DIRECT-Claw editor.py 的 generate_segment_video。
输入：候选池 + 该段每个 shot 的目标长度 + 评分配置
输出：top-N 条最优 scene 序列（EditResult 列表）
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from planner.scoring import Score, ScoreConfig
from models.video import SceneItem


@dataclass
class ShotPick:
    scene_id: int
    target_dur: float           # 这个 shot 应该播放多少秒（音乐时长）
    score: Score


@dataclass
class EditResult:
    picks: list[ShotPick] = field(default_factory=list)
    total_score: float = 0.0

    def append(self, pick: ShotPick) -> None:
        self.picks.append(pick)
        self.total_score += pick.score.combined


def build_candidate_pool(
    scene_table: list[SceneItem],
    clip_embeddings: np.ndarray,
    clip_scene_ids: list[int],
    prompt_embed: np.ndarray,
    used_scene_ids: set[int],
    pool_size: int = 60,
    is_high_energy: bool = False,
    max_per_source: int = 12,
) -> list[SceneItem]:
    """按 prompt 相似度排序，返回 top-pool_size 个未使用的 scene。
    is_high_energy=True 时给 is_climax_material=True 的 scene 加分（前置）。
    max_per_source: 单个源 mp4 在候选池里最多保留多少个 scene（强制多样性）。
    """
    from pathlib import Path as _Path
    sid_to_idx = {sid: i for i, sid in enumerate(clip_scene_ids)}
    pe = prompt_embed.flatten()
    pe_norm = np.linalg.norm(pe)
    if pe_norm == 0:
        pe_norm = 1.0

    scored: list[tuple[SceneItem, float]] = []
    for s in scene_table:
        if s.scene_id in used_scene_ids:
            continue
        idx = sid_to_idx.get(s.scene_id)
        if idx is None:
            continue
        ce = clip_embeddings[idx].flatten()
        ce_norm = np.linalg.norm(ce)
        if ce_norm == 0:
            continue
        cos = float(np.dot(pe, ce) / (pe_norm * ce_norm))

        if is_high_energy:
            if s.is_climax_material:
                cos += 0.15
            if s.is_outro_material:
                cos -= 0.10

        scored.append((s, cos))

    scored.sort(key=lambda x: x[1], reverse=True)

    # 单源文件限额：从高分往下取，每个源 mp4 最多 max_per_source 个
    source_count: dict[str, int] = {}
    pool: list[SceneItem] = []
    for s, _ in scored:
        src = _Path(s.source_file).name
        if source_count.get(src, 0) >= max_per_source:
            continue
        pool.append(s)
        source_count[src] = source_count.get(src, 0) + 1
        if len(pool) >= pool_size:
            break

    return pool


def beam_search_segment(
    pool: list[SceneItem],
    clip_embeddings: np.ndarray,
    clip_scene_ids: list[int],
    shot_durations: list[float],
    shot_energies: list[float],
    score_config: ScoreConfig,
    last_scene_clip: np.ndarray | None = None,
    last_scene_motion: float | None = None,
    beam_size: int = 6,
    exploration: int = 5,
) -> EditResult | None:
    """
    对一个段落跑 beam search。
      shot_durations[i]: 第 i 个镜头需要的音乐时长
      shot_energies[i]:  第 i 个镜头对应的音乐能量（target_motion）
    返回 top-1 EditResult（找不到返回 None）。
    """
    sid_to_idx = {sid: i for i, sid in enumerate(clip_scene_ids)}

    beams: list[tuple[EditResult, int | None, float | None]] = [
        (EditResult(), None, last_scene_motion),  # (result, last_scene_id, last_motion)
    ]
    # 每个 beam 还需带上 used_scene_ids 集合
    beams_used: list[set[int]] = [set()]

    for shot_i, (target_dur, target_energy) in enumerate(zip(shot_durations, shot_energies)):
        score_config.energy_value = target_energy

        new_beams = []
        new_used = []
        for (state, last_sid, last_motion), used in zip(beams, beams_used):
            # 该 beam 的 prev_clip
            if last_sid is None and last_scene_clip is not None:
                prev_clip = last_scene_clip
                prev_motion = last_scene_motion
            elif last_sid is not None:
                idx = sid_to_idx.get(last_sid)
                prev_clip = clip_embeddings[idx] if idx is not None else None
                prev_motion = last_motion
            else:
                prev_clip = None
                prev_motion = None

            # 候选池里去掉已用
            candidates = [s for s in pool if s.scene_id not in used]
            if not candidates:
                continue

            # 计算每个候选的 score
            scored = []
            for s in candidates:
                idx = sid_to_idx.get(s.scene_id)
                if idx is None:
                    continue
                new_clip = clip_embeddings[idx]
                new_motion = float((s.visual_profile or {}).get("motion_intensity", 0.0))
                score = score_config.get_score(
                    last_scene_clip=prev_clip,
                    last_scene_motion=prev_motion,
                    new_scene_clip=new_clip,
                    new_scene_motion=new_motion,
                )
                # 时长惩罚：scene 太短或太长扣分
                ratio = s.duration / target_dur if target_dur > 0 else 1.0
                if ratio < 0.5 or ratio > 3.0:
                    score.combined *= 0.3   # 严重不匹配重罚
                elif ratio < 0.7 or ratio > 2.0:
                    score.combined *= 0.7

                scored.append((s, score, new_motion))

            scored.sort(key=lambda x: x[1].combined, reverse=True)
            top_explore = scored[:exploration]

            for s, sc, motion in top_explore:
                new_state = deepcopy(state)
                new_state.append(ShotPick(scene_id=s.scene_id, target_dur=target_dur, score=sc))
                new_beams.append((new_state, s.scene_id, motion))
                new_u = used.copy()
                new_u.add(s.scene_id)
                new_used.append(new_u)

        # 排序后保留 top-beam_size
        order = sorted(range(len(new_beams)), key=lambda i: new_beams[i][0].total_score, reverse=True)[:beam_size]
        beams = [new_beams[i] for i in order]
        beams_used = [new_used[i] for i in order]

        if not beams:
            logger.warning(f"  [beam] shot {shot_i+1} 无可用候选，提前终止")
            return None

    return beams[0][0]
