from __future__ import annotations

from typing import Literal, TypedDict

from models.audio import AudioMap
from models.plan import RenderItem
from models.video import SceneItem


class GraphState(TypedDict, total=False):
    """
    LangGraph 全局状态。
    total=False 表示所有字段都是可选的，节点只需返回自己更新的字段。
    """

    # ── 用户输入 ──────────────────────────────────────────────────────────────
    music_path: str
    video_paths: list[str]
    background_info: str | None
    video_metadata: dict | None   # {filename: {"source", "episode", "context"}}，由意图解析生成
    editing_style: Literal["visual_driven", "story_driven", "emotion_driven"]
    output_path: str
    project_name: str              # projects/<project_name>/ 下集中所有产物

    # ── 预处理产物 ─────────────────────────────────────────────────────────────
    stripped_video_paths: list[str]

    # ── 音频分析产物 ───────────────────────────────────────────────────────────
    audio_map: AudioMap | None

    # ── 视频分析产物 ───────────────────────────────────────────────────────────
    scene_table: list[SceneItem] | None
    scene_embeddings: object | None
    scene_ids: list[int] | None
    clip_embeddings: object | None   # CLIP 视觉 embedding，可选
    clip_scene_ids: list[int] | None

    # ── 缓存路径 ───────────────────────────────────────────────────────────────
    audio_json_path: str | None
    scene_table_path: str | None

    # ── 规划产物 ───────────────────────────────────────────────────────────────
    render_plan: list[RenderItem] | None

    # ── 动态配置 ───────────────────────────────────────────────────────────────
    runtime_config: dict

    # ── 流程控制 ───────────────────────────────────────────────────────────────
    planner_retry_count: int
    current_errors: list[str]

    # ── 素材分析 ───────────────────────────────────────────────────────────────
    material_analysis: dict | None

    # ── Reviewer 相关 ─────────────────────────────────────────────────────────
    last_intents: list[dict] | None
    last_segment_issues: list | None   # Reviewer 最后输出的段落问题，用于 patch 路由判断
    banned_scene_ids: list | None      # 被 Reviewer 否定过的场景，永久不再使用
    reviewer_score: int | None
    reviewer_pass: bool | None
    material_shortage: bool | None
    material_shortage_reason: str | None

    # ── 最终产物 ───────────────────────────────────────────────────────────────
    output_video_path: str | None