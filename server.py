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
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message="Deserializing unregistered type")

import gradio as gr
from loguru import logger

UPLOAD_DIR = Path("output/uploads")
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
    "precheck":       "预检",
    "preprocess":     "视频预处理",
    "audio_analyzer": "音频分析",
    "video_tagger":   "视频打标签",
    "edit_planner":   "剪辑规划（ReAct）",
    "reviewer":       "质量评审",
    "renderer":       "视频渲染",
    "deliver":        "交付",
}


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
        parts.append(f"✅ 音乐：{_name(music)}")
    else:
        parts.append("⏳ 音乐：未上传")
        ready = False
    if videos:
        parts.append(f"✅ 视频：{len(videos)} 个文件已就绪")
    else:
        parts.append("⏳ 视频：未上传")
        ready = False
    return "　|　".join(parts), ready


# ── 主流程（无中断点） ─────────────────────────────────────────────────────────

async def _run(initial_state: dict, background_str: str | None,
               video_paths: list[str], output_path: str,
               no_reviewer: bool, task_id: str):
    """核心生成器，供两个入口共用。yield (log_str, start_btn_update, video_update, task_id)"""
    log_lines: list[str] = []

    def add(line: str) -> str:
        log_lines.append(line)
        return "\n".join(log_lines)

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
                add("🔍 解析素材描述...")
                yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id
                video_metadata = await parse_video_intent(filenames, background_str)
                if video_metadata:
                    initial_state["video_metadata"] = video_metadata
                    add(f"✓ 素材描述解析完成，命中 {len(video_metadata)}/{len(filenames)} 个文件")
                else:
                    add("⚠ 素材描述未命中具体文件，作为全局背景使用")
                yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id
            except Exception as e:
                logger.warning(f"[Server] 意图解析失败: {e}")
                add("⚠ 素材描述解析失败，跳过")
                yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id

        async for chunk in graph.astream(initial_state, config):
            for node_name in chunk:
                add(f"✓ {NODE_LABEL.get(node_name, node_name)} 完成")
            yield cur(), gr.update(interactive=False), gr.update(visible=False), task_id

    except Exception as e:
        logger.exception(f"[Server] 流程异常: {e}")
        add(f"❌ 出错：{e}")
        yield cur(), gr.update(interactive=True), gr.update(visible=False), task_id
        return

    if Path(output_path).exists():
        add("🎉 完成！")
        yield cur(), gr.update(interactive=True), gr.update(value=output_path, visible=True), task_id
    else:
        add("❌ 渲染结束但输出文件未找到（请查看终端日志）")
        yield cur(), gr.update(interactive=True), gr.update(visible=False), task_id


async def run_pipeline(music_file, video_files, background_text, no_reviewer, _log, task_id):
    def _fp(f) -> Path:
        if hasattr(f, "path") and f.path: return Path(f.path)
        if hasattr(f, "name"): return Path(f.name)
        return Path(str(f))

    if not music_file:
        yield "❌ 请上传音乐文件", gr.update(interactive=True), gr.update(visible=False), task_id
        return
    if not video_files:
        yield "❌ 请上传视频素材（至少一个）", gr.update(interactive=True), gr.update(visible=False), task_id
        return

    task_id = uuid.uuid4().hex[:8]
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    music_src = _fp(music_file)
    music_dst = task_dir / music_src.name
    shutil.copy(str(music_src), str(music_dst))

    video_paths: list[str] = []
    for vf in (video_files if isinstance(video_files, list) else [video_files]):
        src = _fp(vf)
        dst = task_dir / src.name
        shutil.copy(str(src), str(dst))
        video_paths.append(str(dst))

    output_path = str(Path("output/final") / f"{task_id}_result.mp4")
    Path("output/final").mkdir(parents=True, exist_ok=True)

    background_str = (background_text or "").strip() or None
    initial_state: dict[str, Any] = {
        "music_path":          str(music_dst),
        "video_paths":         video_paths,
        "background_info":     background_str,
        "editing_style":       "visual_driven",
        "output_path":         output_path,
        "runtime_config":      {},
        "planner_retry_count": 0,
        "current_errors":      [],
    }

    first = True
    async for chunk in _run(initial_state, background_str, video_paths, output_path, no_reviewer, task_id):
        if first:
            log_str, btn, vid, tid = chunk
            yield f"[{task_id}] 启动 — 音乐: {music_dst.name}，视频: {len(video_paths)} 个\n" + log_str, btn, vid, tid
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

with gr.Blocks(title="Motif AMV 混剪系统") as demo:

    task_id_state = gr.State("")

    gr.Markdown("# Motif — AI AMV 混剪系统")

    with gr.Row():
        # 左列：窄，上传 + 配置
        with gr.Column(scale=1, min_width=260):
            music_input = gr.File(
                label="音乐文件",
                file_types=[".mp3", ".wav", ".flac", ".m4a"],
            )
            video_input = gr.File(
                label="视频素材（可多选）",
                file_count="multiple",
                file_types=[".mp4", ".mov", ".mkv", ".avi"],
            )
            background_input = gr.Textbox(
                label="背景描述（可选）",
                placeholder="可以简要说明一下音乐来自哪里，视频素材来自哪里，你想剪什么风格等等",
                lines=3,
            )
            no_reviewer_input = gr.Checkbox(label="启用 Reviewer（更慢，质量更高）", value=False, visible=False)
            upload_status_md = gr.Markdown("⏳ 音乐：未上传　|　⏳ 视频：未上传")
            start_btn = gr.Button(
                "⏳ 等待文件上传...",
                variant="primary",
                size="lg",
                interactive=False,
            )

            with gr.Accordion("调试模式（服务器本地路径）", open=False):
                _m, _v, _aj, _st = _scan_debug_options()
                with gr.Row():
                    debug_refresh_btn = gr.Button("刷新列表", size="sm", scale=0)
                music_path_input  = gr.Dropdown(choices=_m,  value=_m[0] if _m else None,   label="音乐文件",          allow_custom_value=True)
                video_dir_input   = gr.Dropdown(choices=_v,  value=_v[0] if _v else None,   label="视频目录",          allow_custom_value=True)
                audio_json_input  = gr.Dropdown(choices=["（不使用缓存）"] + _aj, value="（不使用缓存）", label="音频JSON缓存（可选）",    allow_custom_value=True)
                scene_table_input = gr.Dropdown(choices=["（不使用缓存）"] + _st, value="（不使用缓存）", label="SceneTable缓存（可选）", allow_custom_value=True)
                debug_start_btn   = gr.Button("从服务器路径启动", variant="secondary", size="lg")

        # 右列：宽，日志 + 视频
        with gr.Column(scale=3):
            log_output = gr.Textbox(
                label="运行日志",
                lines=22,
                max_lines=50,
                interactive=False,
                placeholder="点击「开始」后，这里会实时显示进度...",
            )
            video_output = gr.Video(
                label="输出视频",
                visible=False,
            )

    # ── 事件绑定 ──────────────────────────────────────────────────────────────

    def _on_file_change(music, videos):
        status_text, ready = _upload_status(music, videos)
        return gr.update(value=status_text), gr.update(value="🚀 开始" if ready else "⏳ 等待文件上传...", interactive=ready)

    for evt in (music_input.upload, music_input.clear, video_input.upload, video_input.clear):
        evt(fn=_on_file_change, inputs=[music_input, video_input], outputs=[upload_status_md, start_btn])

    _run_outputs = [log_output, start_btn, video_output, task_id_state]

    start_btn.click(
        fn=run_pipeline,
        inputs=[music_input, video_input, background_input, no_reviewer_input, log_output, task_id_state],
        outputs=_run_outputs,
    )
    debug_start_btn.click(
        fn=run_pipeline_from_path,
        inputs=[music_path_input, video_dir_input, audio_json_input, scene_table_input,
                background_input, no_reviewer_input, log_output, task_id_state],
        outputs=_run_outputs,
    )

    def _refresh_debug():
        m, v, aj, st = _scan_debug_options()
        return (
            gr.update(choices=m,  value=m[0] if m else None),
            gr.update(choices=v,  value=v[0] if v else None),
            gr.update(choices=["（不使用缓存）"] + aj, value="（不使用缓存）"),
            gr.update(choices=["（不使用缓存）"] + st, value="（不使用缓存）"),
        )

    debug_refresh_btn.click(fn=_refresh_debug, inputs=[],
                            outputs=[music_path_input, video_dir_input, audio_json_input, scene_table_input])


demo.queue()
app = demo.app

if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        max_file_size="4gb",
    )
