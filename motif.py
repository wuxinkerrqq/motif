#!/usr/bin/env python
"""
motif — 统一入口

用法：
    python motif.py ui                           启动 Web UI
    python motif.py run --music MUSIC --videos VIDEOS_DIR [--output OUT]
                                                  命令行一键出片
    python motif.py render-only --music MUSIC [--stem STEM]
                                                  跳过规划，用已有 render_plan 渲染
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_ui(_):
    """启动 Gradio Web UI"""
    import subprocess
    print("[motif] 启动 Web UI: http://localhost:7860")
    subprocess.run([sys.executable, "server.py"])


def cmd_run(args):
    """命令行完整流程：音频分析 → 视频分析 → 规划 → 渲染"""
    from pipeline.graph import build_graph
    from models.state import GraphState

    music_path = Path(args.music).resolve()
    videos_dir = Path(args.videos).resolve()

    if not music_path.exists():
        print(f"[错误] 音乐文件不存在：{music_path}")
        sys.exit(1)
    if not videos_dir.exists():
        print(f"[错误] 视频目录不存在：{videos_dir}")
        sys.exit(1)

    video_paths = sorted(
        [str(p) for p in videos_dir.rglob("*.mp4")]
        + [str(p) for p in videos_dir.rglob("*.mov")]
        + [str(p) for p in videos_dir.rglob("*.mkv")]
    )
    if not video_paths:
        print(f"[错误] {videos_dir} 下没找到视频文件")
        sys.exit(1)

    music_stem = music_path.stem
    output_path = Path(args.output or f"output/output_{music_stem}.mp4").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[motif] 音乐: {music_path.name}")
    print(f"[motif] 视频: {len(video_paths)} 个")
    print(f"[motif] 背景: {args.background or '(未提供)'}")
    print(f"[motif] 输出: {output_path}")
    print()

    initial_state: GraphState = {
        "music_path": str(music_path),
        "video_paths": video_paths,
        "background_info": args.background or "",
        "output_video_path": str(output_path),
        "editing_style": args.style,
        "video_metadata": {},
    }

    graph = build_graph(enable_reviewer=False)
    config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}

    async def _run():
        async for chunk in graph.astream(initial_state, config):
            for node_name in chunk:
                print(f"[motif] ✓ {node_name} 完成")

    asyncio.run(_run())
    print(f"\n[motif] 完成：{output_path}")


def cmd_render_only(args):
    """跳过规划，直接读 render_plan JSON 渲染"""
    from models.plan import RenderItem
    from pipeline.renderer import render_video

    music_path = Path(args.music).resolve()
    music_stem = args.stem or music_path.stem

    plan_path = Path(args.plan or f"output/render_plan_{music_stem}.json")
    if not plan_path.exists():
        print(f"[错误] 找不到 render_plan: {plan_path}")
        sys.exit(1)

    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = [RenderItem(**r) for r in plan_data]
    print(f"[motif --render-only] 读取 {len(plan)} 条 clip")

    output_path = Path(args.output or f"output/output_{music_stem}.mp4")
    temp_dir = Path(f"output/clips_{music_stem}")
    render_video(
        render_plan=plan,
        music_path=str(music_path),
        output_path=str(output_path),
        temp_dir=str(temp_dir),
    )
    print(f"[motif] 完成：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="motif",
        description="Music-driven auto video editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python motif.py ui
  python motif.py run --music song.mp3 --videos clips/ \\
                      --background "Fate AMV，前半段忧郁后半段燃爆"
  python motif.py render-only --music song.mp3
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ui
    sub.add_parser("ui", help="启动 Web UI（Gradio）")

    # run
    p_run = sub.add_parser("run", help="命令行一键出片（全流程）")
    p_run.add_argument("--music", required=True, help="音乐文件路径")
    p_run.add_argument("--videos", required=True, help="视频素材目录")
    p_run.add_argument("--background", default="",
                       help="自然语言描述：作品背景/剪辑意图/风格偏好均可（可选）")
    p_run.add_argument("--output", default=None, help="输出路径，默认 output/output_<音乐名>.mp4")
    p_run.add_argument("--style", default="visual_driven",
                       choices=["visual_driven", "story_driven", "emotion_driven"],
                       help="剪辑风格")

    # render-only
    p_ro = sub.add_parser("render-only", help="跳过规划，用已有 render_plan 渲染")
    p_ro.add_argument("--music", required=True, help="音乐文件路径")
    p_ro.add_argument("--stem", default=None, help="音乐文件名（无后缀），用于定位缓存")
    p_ro.add_argument("--plan", default=None, help="指定 render_plan JSON 路径")
    p_ro.add_argument("--output", default=None, help="输出路径")

    args = parser.parse_args()

    if args.command == "ui":
        cmd_ui(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "render-only":
        cmd_render_only(args)


if __name__ == "__main__":
    main()
