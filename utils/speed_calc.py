from __future__ import annotations

from config import DefaultConfig


def get_speed_factor(
    clip_duration: float,
    target_duration: float,
    min_speed: float = DefaultConfig.SPEED_FACTOR_MIN,
    max_speed: float = DefaultConfig.SPEED_FACTOR_MAX,
) -> float:
    """
    计算变速倍率。
    clip_duration / target_duration > 1 → 加速
    clip_duration / target_duration < 1 → 慢动作
    限制在 [min_speed, max_speed] 范围内。
    """
    if target_duration <= 0:
        return 1.0

    factor = clip_duration / target_duration
    return round(max(min_speed, min(max_speed, factor)), 3)


def needs_clip_replacement(
    clip_duration: float,
    target_duration: float,
    extreme_ratio: float = DefaultConfig.SPEED_EXTREME_RATIO,
) -> bool:
    """
    判断时长差距是否过大，需要换素材而非变速。
    """
    if target_duration <= 0:
        return True
    ratio = clip_duration / target_duration
    return ratio > extreme_ratio or ratio < (1.0 / extreme_ratio)