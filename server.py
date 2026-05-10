"""
Motif AMV 混剪系统 - Web 服务入口

流程：上传音乐 + 视频 → 可选填背景描述 → 点击开始 → 全自动运行 → 输出视频
"""

from __future__ import annotations

# 强制 HuggingFace 离线模式：CLIP 模型权重已在本地缓存，避免每次启动联网
# 查 metadata 导致的超时重试和后台线程异常
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import shutil
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message="Deserializing unregistered type")

import gradio as gr
from loguru import logger

from utils.project_paths import ProjectPaths

active_tasks: dict[str, dict[str, Any]] = {}

_MOTIF_ROOT = Path(__file__).parent


def _scan_debug_options() -> tuple[list[str], list[str], list[str], list[str]]:
    music_dir  = _MOTIF_ROOT / "test" / "music"
    video_base = _MOTIF_ROOT / "test" / "videos"
    output_dir = _MOTIF_ROOT / "output"

    music_files: list[str] = []
    if music_dir.is_dir():
        for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a"):
            music_files.extend(str(p) for p in sorted(music_dir.glob(ext)))

    video_dirs: list[str] = []
    if video_base.is_dir():
        video_dirs.extend(str(p) for p in sorted(video_base.iterdir()) if p.is_dir())
        if any(video_base.glob("*.mp4")) or any(video_base.glob("*.mov")):
            video_dirs.insert(0, str(video_base))

    audio_jsons: list[str] = []
    if music_dir.is_dir():
        audio_jsons.extend(str(p) for p in sorted(music_dir.glob("*.json")))

    scene_tables: list[str] = []
    if output_dir.is_dir():
        scene_tables.extend(str(p) for p in sorted(output_dir.glob("*_scene_table.json")))

    return music_files, video_dirs, audio_jsons, scene_tables


NODE_LABEL = {
    "precheck":       "Precheck",
    "preprocess":     "Video Preprocessing",
    "audio_analyzer": "Audio Analysis",
    "video_tagger":   "Video Tagging",
    "edit_planner":   "Edit Planning",
    "reviewer":       "Quality Review",
    "renderer":       "Rendering",
    "deliver":        "Deliver",
}

NODE_ORDER = ["precheck", "preprocess", "audio_analyzer", "video_tagger", "edit_planner", "renderer", "deliver"]


def _upload_status(music, videos) -> tuple[str, bool]:
    def _name(f) -> str:
        if f is None: return ""
        if hasattr(f, "orig_name") and f.orig_name: return f.orig_name
        if hasattr(f, "name"): return Path(f.name).name
        if isinstance(f, str): return Path(f).name
        return str(f)

    def _ready(f) -> bool:
        if f is None: return False
        if hasattr(f, "path"): return bool(f.path)
        return bool(f)

    parts, ready = [], True
    if _ready(music):
        parts.append(f"✅ Music: {_name(music)}")
    else:
        parts.append("⏳ Music: not uploaded")
        ready = False
    if videos:
        parts.append(f"✅ Video: {len(videos)} files ready")
    else:
        parts.append("⏳ Video: not uploaded")
        ready = False
    return "　|　".join(parts), ready


# ── 主流程（无中断点） ─────────────────────────────────────────────────────────

async def _run(initial_state: dict, background_str: str | None,
               video_paths: list[str], output_path: str,
               no_reviewer: bool, task_id: str):
    """核心生成器，供两个入口共用。yield (log_str, start_btn_update, video_update, task_id)"""
    import asyncio
    from collections import deque

    log_lines: deque = deque(maxlen=200)
    log_sink_id = None

    def _sink(message):
        record = message.record
        text = f"[{record['time'].strftime('%H:%M:%S')}] {record['message']}"
        log_lines.append(text)

    # 注册 loguru sink 捕获所有后台日志
    log_sink_id = logger.add(_sink, format="{message}", level="INFO")

    def cur() -> str:
        return "\n".join(log_lines)

    yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id

    try:
        from pipeline.graph import build_graph

        graph = build_graph(use_reviewer=no_reviewer)
        config = {"configurable": {"thread_id": task_id}}
        active_tasks[task_id] = {"output_path": output_path}

        if background_str:
            try:
                from agents.manager import parse_video_intent
                filenames = [Path(p).name for p in video_paths]
                log_lines.append("🔍 Parsing description...")
                yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id
                video_metadata = await parse_video_intent(filenames, background_str)
                if video_metadata:
                    initial_state["video_metadata"] = video_metadata
                    log_lines.append(f"✓ Description parsed, matched {len(video_metadata)}/{len(filenames)} files")
                else:
                    log_lines.append("⚠ Description did not match specific files, using as global context")
                yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id
            except Exception as e:
                logger.warning(f"[Server] Intent parsing failed: {e}")
                log_lines.append("⚠ Description parsing failed, skipping")
                yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id

        async for chunk in graph.astream(initial_state, config):
            for node_name in chunk:
                label = NODE_LABEL.get(node_name, node_name)
                log_lines.append(f"━━━ ✓ {label} done ━━━")
            yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id

    except Exception as e:
        logger.exception(f"[Server] Pipeline error: {e}")
        log_lines.append(f"❌ Error: {e}")
        yield cur(), gr.update(interactive=True), gr.update(visible=False), task_id
        if log_sink_id is not None:
            logger.remove(log_sink_id)
        return

    if log_sink_id is not None:
        logger.remove(log_sink_id)

    if Path(output_path).exists():
        log_lines.append("🎉 Done!")
        yield cur(), gr.update(interactive=True), gr.update(value=output_path, visible=True), task_id
    else:
        log_lines.append("❌ Rendering finished but output file not found (check terminal logs)")
        yield cur(), gr.update(interactive=True), gr.update(visible=False), task_id


async def run_pipeline(music_file, video_files, background_text, no_reviewer, _log, task_id):
    def _fp(f) -> Path:
        if hasattr(f, "path") and f.path: return Path(f.path)
        if hasattr(f, "name"): return Path(f.name)
        return Path(str(f))

    if not music_file:
        yield "❌ Please upload a music file", gr.update(interactive=True), gr.update(visible=False), task_id
        return
    if not video_files:
        yield "❌ Please upload at least one video", gr.update(interactive=True), gr.update(visible=False), task_id
        return

    # 项目名 = project_YYYYMMDD_HHMMSS，所有产物进 projects/<name>/
    project_name = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pp = ProjectPaths(project_name)
    pp.create_all()
    task_id = project_name

    # 复制原始音乐到 music/
    music_src = _fp(music_file)
    music_dst = pp.music_dir / music_src.name
    shutil.copy(str(music_src), str(music_dst))
    logger.info(f"[Server] 音乐已保存: {music_dst}")

    # 复制原始视频到 videos/raw/
    video_paths: list[str] = []
    for vf in (video_files if isinstance(video_files, list) else [video_files]):
        src = _fp(vf)
        dst = pp.raw_videos_dir / src.name
        shutil.copy(str(src), str(dst))
        video_paths.append(str(dst))
    logger.info(f"[Server] 视频已保存: {len(video_paths)} 个 → {pp.raw_videos_dir}")

    output_path = str(pp.final_path)

    background_str = (background_text or "").strip() or None
    initial_state: dict[str, Any] = {
        "music_path":          str(music_dst),
        "video_paths":         video_paths,
        "background_info":     background_str,
        "editing_style":       "visual_driven",
        "output_path":         output_path,
        "project_name":        project_name,
        "runtime_config":      {},
        "planner_retry_count": 0,
        "current_errors":      [],
    }

    first = True
    async for chunk in _run(initial_state, background_str, video_paths, output_path, no_reviewer, task_id):
        if first:
            log_str, btn, vid, tid = chunk
            yield f"[{task_id}] 启动 — 项目目录: {pp.root}\n  音乐: {music_dst.name}\n  视频: {len(video_paths)} 个\n" + log_str, btn, vid, tid
            first = False
        else:
            yield chunk


async def run_pipeline_from_path(
    music_path_str, video_dir_str, audio_json_str, scene_table_str,
    background_text, no_reviewer, _log, task_id,
):
    music_path_str  = (music_path_str  or "").strip()
    video_dir_str   = (video_dir_str   or "").strip()
    audio_json_str  = (audio_json_str  or "").strip()
    scene_table_str = (scene_table_str or "").strip()

    _NO_CACHE = "（不使用缓存）"
    if audio_json_str  == _NO_CACHE: audio_json_str  = ""
    if scene_table_str == _NO_CACHE: scene_table_str = ""

    def _err(msg):
        return msg, gr.update(interactive=True), gr.update(visible=False), task_id

    if not music_path_str:
        yield _err("❌ 请填写音乐文件路径"); return
    if not video_dir_str:
        yield _err("❌ 请填写视频素材目录"); return

    music_path = Path(music_path_str)
    video_dir  = Path(video_dir_str)

    if not music_path.exists():
        yield _err(f"❌ 音乐文件不存在：{music_path_str}"); return
    if not video_dir.is_dir():
        yield _err(f"❌ 视频目录不存在：{video_dir_str}"); return

    video_paths: list[str] = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.MP4", "*.MOV", "*.MKV"):
        video_paths.extend(str(p) for p in sorted(video_dir.glob(ext)))

    if not video_paths:
        yield _err(f"❌ 目录中未找到视频文件：{video_dir_str}"); return

    task_id = uuid.uuid4().hex[:8]
    output_path = str(Path("output/final") / f"{task_id}_result.mp4")
    Path("output/final").mkdir(parents=True, exist_ok=True)

    background_str = (background_text or "").strip() or None
    initial_state: dict[str, Any] = {
        "music_path":          str(music_path),
        "video_paths":         video_paths,
        "background_info":     background_str,
        "editing_style":       "visual_driven",
        "output_path":         output_path,
        "runtime_config":      {},
        "planner_retry_count": 0,
        "current_errors":      [],
    }

    if audio_json_str and Path(audio_json_str).exists():
        initial_state["audio_json_path"] = audio_json_str
    if scene_table_str and Path(scene_table_str).exists():
        initial_state["scene_table_path"] = scene_table_str

    first = True
    async for chunk in _run(initial_state, background_str, video_paths, output_path, no_reviewer, task_id):
        if first:
            log_str, btn, vid, tid = chunk
            prefix = f"[{task_id}] Debug — {len(video_paths)} 个视频\n"
            if audio_json_str:  prefix += f"✓ 使用音频缓存：{Path(audio_json_str).name}\n"
            if scene_table_str: prefix += f"✓ 使用SceneTable缓存：{Path(scene_table_str).name}\n"
            yield prefix + log_str, btn, vid, tid
            first = False
        else:
            yield chunk


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Motif") as demo:

    task_id_state = gr.State("")

    gr.Markdown("# Motif — Music-Driven Video Editor")

    with gr.Row():
        # 左列：窄，上传 + 配置
        with gr.Column(scale=1, min_width=260):
            music_input = gr.File(
                label="Music",
                file_types=[".mp3", ".wav", ".flac", ".m4a"],
            )
            video_input = gr.File(
                label="Video Footage (multiple)",
                file_count="multiple",
                file_types=[".mp4", ".mov", ".mkv", ".avi"],
            )
            background_input = gr.Textbox(
                label="Description (optional)",
                placeholder="Describe the source material, desired style, or editing intent...",
                lines=3,
            )
            no_reviewer_input = gr.Checkbox(label="Enable Reviewer", value=False, visible=False)
            upload_status_md = gr.Markdown("⏳ Music: not uploaded　|　⏳ Video: not uploaded")
            start_btn = gr.Button(
                "⏳ Waiting for upload...",
                variant="primary",
                size="lg",
                interactive=False,
            )

        # 右列：宽，日志 + 视频
        with gr.Column(scale=3):
            log_output = gr.Textbox(
                label="Log",
                lines=22,
                max_lines=50,
                interactive=False,
                placeholder="Progress will appear here after clicking Start...",
            )
            video_output = gr.Video(
                label="Output Video",
                visible=False,
            )

    # ── 事件绑定 ──────────────────────────────────────────────────────────────

    def _on_file_change(music, videos):
        status_text, ready = _upload_status(music, videos)
        return gr.update(value=status_text), gr.update(value="🚀 Start" if ready else "⏳ Waiting for upload...", interactive=ready)

    for evt in (music_input.upload, music_input.clear, video_input.upload, video_input.clear):
        evt(fn=_on_file_change, inputs=[music_input, video_input], outputs=[upload_status_md, start_btn])

    _run_outputs = [log_output, start_btn, video_output, task_id_state]

    start_btn.click(
        fn=run_pipeline,
        inputs=[music_input, video_input, background_input, no_reviewer_input, log_output, task_id_state],
        outputs=_run_outputs,
    )


demo.queue()
app = demo.app

if __name__ == "__main__":
    Path("projects").mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        max_file_size="4gb",
        theme=gr.themes.Soft(),
    )
