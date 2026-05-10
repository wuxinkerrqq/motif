from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from langgraph.graph import StateGraph, END

from models.state import GraphState


async def precheck_node(state: GraphState) -> dict:
    from agents.manager import precheck_node as _precheck
    return await _precheck(state)


async def preprocess_node(state: GraphState) -> dict:
    from video.preprocessor import strip_audio_batch
    video_paths = state["video_paths"]
    stripped = strip_audio_batch(video_paths, "output/stripped")
    logger.info(f"[Graph] 预处理完成: {len(stripped)} 个视频去音轨")
    return {"stripped_video_paths": stripped}


async def audio_analyzer_node(state: GraphState) -> dict:
    from utils.audio_cache import load_audio_map

    music_path = state["music_path"]
    audio_json_path = state.get("audio_json_path")

    if not audio_json_path:
        candidate = Path(music_path).with_suffix(".json")
        if candidate.exists():
            audio_json_path = str(candidate)
            logger.info(f"[Graph] 自动找到 audio_json: {audio_json_path}")

    if audio_json_path and Path(audio_json_path).exists():
        logger.info(f"[Graph] 加载已有 audio_map: {audio_json_path}")
        audio_map = load_audio_map(audio_json_path, music_path=music_path)
        return {"audio_map": audio_map}
    else:
        logger.info("[Graph] 重新运行 Audio Analyzer")
        from audio.analyzer_agent import run_audio_analyzer
        audio_map, config_patch = await run_audio_analyzer(
            music_path=music_path,
            background_info=state.get("background_info"),
            editing_style=state.get("editing_style", "visual_driven"),
        )
        runtime_config = {**state.get("runtime_config", {}), **config_patch}
        return {"audio_map": audio_map, "runtime_config": runtime_config}


async def video_tagger_node(state: GraphState) -> dict:
    from models.video import SceneItem

    video_paths = state["video_paths"]
    stripped_video_paths = state.get("stripped_video_paths") or video_paths
    scene_table_path = state.get("scene_table_path")

    if not scene_table_path:
        dirs = list({str(Path(p).parent) for p in video_paths})
        if len(dirs) == 1:
            video_dir = Path(dirs[0])
            candidate = video_dir.parent / f"{video_dir.name}_scene_table.json"
            if candidate.exists():
                scene_table_path = str(candidate)
                logger.info(f"[Graph] 自动找到 scene_table: {scene_table_path}")
        if not scene_table_path and len(video_paths) == 1:
            candidate2 = Path(video_paths[0]).parent / f"{Path(video_paths[0]).stem}_scene_table.json"
            if candidate2.exists():
                scene_table_path = str(candidate2)

    save_path = scene_table_path

    if scene_table_path and Path(scene_table_path).exists():
        logger.info(f"[Graph] 加载已有 scene_table: {scene_table_path}")
        raw = json.load(open(scene_table_path, encoding="utf-8"))
        scene_table = [SceneItem(**s) for s in raw]
    else:
        logger.info("[Graph] 重新运行 Video Tagger")
        from video.tagger_agent import run_video_tagger
        music_stem = Path(state["music_path"]).stem
        save_path = f"output/{music_stem}_scene_table.json"
        scene_table, _, _ = await run_video_tagger(
            video_paths=video_paths,
            stripped_video_paths=stripped_video_paths,
            frames_dir="output/frames",
            background_info=state.get("background_info"),
            video_metadata=state.get("video_metadata"),
            scene_table_save_path=save_path,
        )

    # 加载 material_analysis（如果存在）
    material_analysis = {}
    if save_path:
        analysis_path = save_path.replace("_scene_table.json", "_material_analysis.json")
        if Path(analysis_path).exists():
            material_analysis = json.load(open(analysis_path, encoding="utf-8"))

    # 构建 CLIP 视觉 embedding（唯一的向量索引）
    clip_embeddings, clip_scene_ids = None, None
    if save_path and Path(save_path).exists():
        from video.scene_embedder import build_clip_index
        clip_embeddings, clip_scene_ids = build_clip_index(save_path)
        logger.info(f"[Graph] CLIP embedding 就绪，shape={clip_embeddings.shape}")

    # scene_embeddings 直接复用 CLIP（同空间），废弃独立的文本 embedding 通道
    return {
        "scene_table": scene_table,
        "scene_embeddings": clip_embeddings,
        "scene_ids": clip_scene_ids,
        "clip_embeddings": clip_embeddings,
        "clip_scene_ids": clip_scene_ids,
        "material_analysis": material_analysis,
    }


async def edit_planner_node(state: GraphState) -> dict:
    """v5 全素材直给 GPT-5.5（Call 1 锚 L1/L2 + Call 2 填 L3）。"""
    from planner.planner_v5_simple import run_planner_v5
    from models.plan import RenderItem

    audio_map = state["audio_map"]
    scene_table = state["scene_table"]
    clip_embeddings = state.get("clip_embeddings")
    clip_scene_ids = state.get("clip_scene_ids")

    music_stem = Path(state["music_path"]).stem
    trace_path = Path(f"output/v5_trace_{music_stem}.json")

    result = await run_planner_v5(
        audio_map=audio_map,
        scene_table=scene_table,
        clip_embeddings=clip_embeddings,
        clip_scene_ids=clip_scene_ids,
        background_info=state.get("background_info") or "",
        log_path=trace_path,
    )

    # committed dict → RenderItem
    scene_lookup = {s.scene_id: s for s in scene_table}
    render_plan: list[RenderItem] = []
    for i, c in enumerate(sorted(result.committed, key=lambda x: x["audio_start"]), start=1):
        scene = scene_lookup.get(c["scene_id"])
        if not scene:
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

    plan_path = f"output/render_plan_{music_stem}.json"
    json.dump(
        [r.model_dump() for r in render_plan],
        open(plan_path, "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )
    logger.info(
        f"[Graph] v5 render_plan 已保存: {plan_path}，共 {len(render_plan)} 条，"
        f"{result.elapsed_sec:.1f}s，覆盖 {'完整' if result.finished else '有 gap'}"
    )

    return {
        "render_plan": render_plan,
        "last_intents": [],
        "last_segment_issues": [],
        "current_errors": [],
    }


async def reviewer_node(state: GraphState) -> dict:
    """资深观众 Reviewer 节点。"""
    from agents.reviewer import run_reviewer, build_reflection_prompt
    from agents.edit_planner import patch_render_plan

    render_plan = state["render_plan"]
    intents = state.get("last_intents", [])
    audio_map = state["audio_map"]
    scene_table = state["scene_table"]
    scene_embeddings = state.get("scene_embeddings")
    scene_ids = state.get("scene_ids")
    background_info = state.get("background_info")
    retry_count = state.get("planner_retry_count", 0)

    result = await run_reviewer(
        render_plan=render_plan,
        intents=intents,
        audio_map=audio_map,
        scene_table=scene_table,
        background_info=background_info,
        retry_count=retry_count,
    )

    if result.material_shortage:
        logger.warning(
            "[Reviewer] ⚠️ 素材严重不足！\n"
            f"  原因：{result.material_shortage_reason}\n"
            "  建议：请添加更多视频素材后重新运行"
        )

    # 打回时：填空题模式，只修改有问题的段落
    patched_plan = render_plan
    if not result.pass_ and result.segment_issues:
        logger.info(f"[Reviewer] 填空题模式：只修改 {len(result.segment_issues)} 个问题段落")

        # 更新黑名单：只封 Reviewer 明确指出的 scene_id
        # 不封整个段落，避免可用素材过度萎缩
        banned = set(state.get("banned_scene_ids") or [])
        for si in result.segment_issues:
            for sid in si.suggested_scene_ids:
                pass  # suggested 是要用的，不封
            # 只封当前段落里 Reviewer 没有推荐的场景
            from agents.edit_planner import _find_segment_name
            problem_items = [r for r in render_plan
                           if _find_segment_name(r.audio_start, intents) == si.segment_name]
            suggested = set(si.suggested_scene_ids)
            for r in problem_items:
                if r.scene_id not in suggested:
                    banned.add(r.scene_id)
        logger.info(f"[Reviewer] 黑名单更新，共 {len(banned)} 个场景")

        patched_plan, intents = patch_render_plan(
            render_plan=render_plan,
            intents=intents,
            segment_issues=result.segment_issues,
            scene_table=scene_table,
            scene_embeddings=scene_embeddings,
            scene_ids=scene_ids,
            audio_map=audio_map,
            banned_scene_ids=banned,
        )
    else:
        banned = set(state.get("banned_scene_ids") or [])

    reflection = build_reflection_prompt(result) if not result.pass_ else ""

    return {
        "render_plan": patched_plan,
        "last_intents": intents,
        "last_segment_issues": result.segment_issues if result.segment_issues else [],
        "banned_scene_ids": list(banned),
        "reviewer_score": result.score,
        "reviewer_pass": result.pass_,
        "material_shortage": result.material_shortage,
        "material_shortage_reason": result.material_shortage_reason,
        "current_errors": [reflection] if reflection else [],
        "planner_retry_count": retry_count + (0 if result.pass_ else 1),
    }


async def renderer_node(state: GraphState) -> dict:
    from pipeline.renderer import render_video

    render_plan = state["render_plan"]
    video_paths = state["video_paths"]
    output_path = state.get("output_path", "output/final/result.mp4")

    name_to_path = {Path(p).name: p for p in video_paths}
    for r in render_plan:
        fname = Path(r.source_file).name
        if fname in name_to_path and not Path(r.source_file).exists():
            r.source_file = name_to_path[fname]

    output = render_video(
        render_plan=render_plan,
        music_path=state["music_path"],
        output_path=output_path,
        temp_dir="output/clips",
    )
    return {"output_video_path": output}


async def deliver_node(state: GraphState) -> dict:
    from agents.manager import deliver_node as _deliver

    if state.get("material_shortage"):
        logger.warning(
            "[Manager] ⚠️  最终提示：视频素材严重不足，影响了混剪质量。\n"
            f"  原因：{state.get('material_shortage_reason', '素材总时长远小于音乐时长')}\n"
            "  建议：补充更多视频素材后重新运行，效果会明显提升。"
        )

    reviewer_score = state.get("reviewer_score", "N/A")
    logger.info(f"[Manager] 最终评审得分：{reviewer_score}/100")

    return await _deliver(state)


# ──────────────────────────────────────────────────────────────────────────────
# 条件边路由
# ──────────────────────────────────────────────────────────────────────────────

def reviewer_router(state: GraphState) -> str:
    """
    Reviewer 评审后的路由：
    - pass → renderer
    - 有段落问题（patch模式）→ reviewer（patch已完成，直接再审）
    - 无段落问题 → edit_planner（整体重新规划）
    """
    if state.get("reviewer_pass", True):
        return "renderer"

    retry_count = state.get("planner_retry_count", 0)
    logger.info(
        f"[Router] Reviewer 打回（得分 {state.get('reviewer_score')}/100），"
        f"第 {retry_count} 次重试"
    )

    # 有段落问题 → patch 已在 reviewer_node 内完成，直接再审
    last_segment_issues = state.get("last_segment_issues", [])
    if last_segment_issues:
        logger.info("[Router] patch 模式：直接再审，不重新规划")
        return "reviewer"

    # 无段落问题 → 整体重新规划
    return "edit_planner"


# ──────────────────────────────────────────────────────────────────────────────
# 图构建
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(
    use_reviewer: bool = True,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
):
    """构建 Motif LangGraph 图（含 Reviewer 填空题模式）。

    Args:
        use_reviewer: 是否启用 Reviewer 节点。
        checkpointer: 保留参数，暂未使用。
        interrupt_before: 保留参数，暂未使用。
    """
    g = StateGraph(GraphState)

    g.add_node("precheck",            precheck_node)
    g.add_node("preprocess",          preprocess_node)
    g.add_node("audio_analyzer",      audio_analyzer_node)
    g.add_node("video_tagger",        video_tagger_node)
    g.add_node("edit_planner",        edit_planner_node)
    g.add_node("renderer",            renderer_node)
    g.add_node("deliver",             deliver_node)

    g.set_entry_point("precheck")
    g.add_edge("precheck",            "preprocess")
    g.add_edge("preprocess",          "audio_analyzer")
    g.add_edge("preprocess",          "video_tagger")
    g.add_edge("audio_analyzer",      "edit_planner")
    g.add_edge("video_tagger",        "edit_planner")
    if use_reviewer:
        g.add_node("reviewer", reviewer_node)
        g.add_edge("edit_planner", "reviewer")
        g.add_conditional_edges(
            "reviewer",
            reviewer_router,
            {
                "renderer":     "renderer",
                "reviewer":     "reviewer",
                "edit_planner": "edit_planner",
            }
        )
    else:
        g.add_edge("edit_planner", "renderer")  # 跳过 Reviewer

    g.add_edge("renderer",       "deliver")
    g.add_edge("deliver",        END)

    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    return g.compile(**compile_kwargs)