from __future__ import annotations

from typing import Literal

from models.state import GraphState


def coverage_router(
    state: GraphState,
) -> Literal["pass", "fallback"]:
    """
    检查 render_plan 的覆盖率。
    MVP 阶段：覆盖率不足也直接 pass（兜底规划已在 edit_planner 内部处理）。
    """
    render_plan = state.get("render_plan") or []
    audio_map = state.get("audio_map")

    if not render_plan or not audio_map:
        return "fallback"

    total_video = sum(r.clip_end - r.clip_start for r in render_plan)
    coverage = total_video / audio_map.total_duration if audio_map.total_duration > 0 else 0

    if coverage < 0.5:
        return "fallback"

    return "pass"