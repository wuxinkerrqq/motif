"""
scoring.py — DIRECT 风格的多维度评分（4 维：prompt + semantic + motion + energy）

参考：DIRECT-Claw editing_utils.py + interaction_utils.py 的 ScoreConfig
简化：去掉 saliency 维度（我们没有 U-2-Net）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Score:
    prompt: float
    semantic: float
    motion: float
    energy: float
    combined: float


@dataclass
class ScoreConfig:
    """评分权重配置（参考 DIRECT 的 5 个 profile）"""
    prompt_embed: np.ndarray | None = None
    prompt_weight: float = 16.0
    semantic_weight: float = 1.0
    motion_weight: float = 3.0
    energy_weight: float = 4.0
    energy_value: float = 0.5  # 当前镜头的目标能量值（音乐推断）

    def get_score(
        self,
        last_scene_clip: np.ndarray | None,
        last_scene_motion: float | None,
        new_scene_clip: np.ndarray,
        new_scene_motion: float,
    ) -> Score:
        assert self.prompt_embed is not None, "prompt_embed 未设置"

        prompt_score = _cos(self.prompt_embed, new_scene_clip)
        energy_score = _energy_score(self.energy_value, new_scene_motion)

        if last_scene_clip is None:
            semantic_score = 0.0
            motion_score = 0.0
        else:
            semantic_score = _cos(last_scene_clip, new_scene_clip)
            motion_score = _motion_continuity(last_scene_motion or 0.0, new_scene_motion)

        combined = float(
            self.prompt_weight * prompt_score
            + self.semantic_weight * semantic_score
            + self.motion_weight * motion_score
            + self.energy_weight * energy_score
        )
        return Score(
            prompt=float(prompt_score),
            semantic=float(semantic_score),
            motion=float(motion_score),
            energy=float(energy_score),
            combined=combined,
        )


# ── 内部评分函数 ─────────────────────────────────────────────────────────────

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _motion_continuity(prev_motion: float, curr_motion: float) -> float:
    """运动量差异越小，连贯性分数越高（[0, 1]）。"""
    diff = abs(prev_motion - curr_motion)
    return float(np.exp(-diff * 4.0))  # diff=0 → 1.0；diff=0.5 → 0.13


def _energy_score(target_energy: float, scene_motion: float) -> float:
    """DIRECT 的 energy 评分公式（log-space + 指数衰减），越接近越高。"""
    log_v = np.log(target_energy + 1e-3)
    log_v0 = np.log(scene_motion + 1e-3)
    diff = min(log_v - log_v0, (target_energy - scene_motion) / 40)
    return float(np.exp(-diff ** 2 / 2))


# ── 预设权重 profile（抄 DIRECT 5 选 1，去掉 saliency 维度）──────────────────

WEIGHT_PROFILES = {
    "Motion_Continuity_Priority": ScoreConfig(
        prompt_weight=16, semantic_weight=1, motion_weight=10, energy_weight=4,
    ),
    "Semantic_Priority": ScoreConfig(
        prompt_weight=48, semantic_weight=2, motion_weight=2, energy_weight=4,
    ),
    "Visual_Complexity_Priority": ScoreConfig(
        prompt_weight=16, semantic_weight=1, motion_weight=8, energy_weight=4,
    ),
    "Default_Priority": ScoreConfig(
        prompt_weight=16, semantic_weight=1, motion_weight=3, energy_weight=4,
    ),
    "Energy_Priority": ScoreConfig(
        prompt_weight=16, semantic_weight=1, motion_weight=2, energy_weight=10,
    ),
}


def get_weight_profile(name: str) -> ScoreConfig:
    """返回新的 ScoreConfig 拷贝（不能直接共享，prompt_embed 会被改写）。"""
    base = WEIGHT_PROFILES.get(name) or WEIGHT_PROFILES["Default_Priority"]
    return ScoreConfig(
        prompt_embed=None,
        prompt_weight=base.prompt_weight,
        semantic_weight=base.semantic_weight,
        motion_weight=base.motion_weight,
        energy_weight=base.energy_weight,
        energy_value=0.5,
    )
