#!/usr/bin/env python
"""
motif — 统一入口

用法：
    python motif.py ui                           启动 Web UI
    python motif.py run --music MUSIC --videos VIDEOS_DIR [--output OUT]
                                                  命令行一键出片（同时导出 FCP XML）
    python motif.py render-only --music MUSIC [--stem STEM]
                                                  跳过规划，用已有 render_plan 渲染
    python motif.py export-xml --project PROJ    只导出 FCP XML 供 PR / DaVinci 微调
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
    print("[motif] 启动 Web UI...")
    subprocess.run([sys.executable, "server.py"])


def cmd_run(args):
    """命令行完整流程：音频分析 → 视频分析 → 规划 → 渲染"""
    from pipeline.graph import build_graph
    from pipeline.xml_exporter import export_fcpxml
    from models.plan import RenderItem
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

    final_state = {}
    async def _run():
        async for chunk in graph.astream(initial_state, config):
            for node_name, node_output in chunk.items():
                print(f"[motif] ✓ {node_name} 完成")
                final_state.update(node_output or {})

    asyncio.run(_run())

    # 主渲染完成后，额外导出一份 FCP XML 供 PR / DaVinci 微调
    if not args.no_xml and final_state.get("render_plan"):
        try:
            plan_items = final_state["render_plan"]
            plan_dicts = [r.model_dump() if hasattr(r, "model_dump") else r for r in plan_items]
            xml_path = output_path.with_suffix(".xml")
            stripped_dir = Path("output/stripped") if Path("output/stripped").exists() else None
            export_fcpxml(
                plan=plan_dicts,
                music_path=music_path,
                output_xml=xml_path,
                stripped_dir=stripped_dir,
            )
            print(f"[motif] FCP XML: {xml_path}  (可 import 到 Premiere Pro / DaVinci 微调)")
        except Exception as e:
            print(f"[motif] FCP XML 导出失败（不影响成片）: {e}")

    print(f"\n[motif] 完成：{output_path}")


def cmd_render_only(args):
    """跳过规划，直接读 render_plan JSON 渲染。

    两种用法：
      --project PROJ     从 projects/PROJ/ 读取所有文件
      --music M --plan P 显式指定音乐和 plan 路径（兼容老数据）
    """
    from models.plan import RenderItem
    from pipeline.renderer import render_video
    from pipeline.xml_exporter import export_fcpxml
    from utils.project_paths import ProjectPaths

    if args.project:
        pp = ProjectPaths(args.project)
        if not pp.exists():
            print(f"[错误] 项目不存在: {pp.root}")
            sys.exit(1)
        # 自动找音乐和 plan
        music_files = list(pp.music_dir.glob("*"))
        if not music_files:
            print(f"[错误] {pp.music_dir} 下没有音乐文件")
            sys.exit(1)
        music_path = music_files[0]
        plan_path = pp.render_plan_path
        if not plan_path.exists():
            print(f"[错误] 找不到 render_plan: {plan_path}")
            sys.exit(1)
        output_path = pp.final_path
        temp_dir = pp.clips_dir
        xml_path = pp.root / "motif_timeline.xml"
        stripped_dir = pp.stripped_dir
    else:
        if not args.music:
            print("[错误] 必须提供 --project 或 --music")
            sys.exit(1)
        music_path = Path(args.music).resolve()
        music_stem = args.stem or music_path.stem
        plan_path = Path(args.plan or f"output/render_plan_{music_stem}.json")
        if not plan_path.exists():
            print(f"[错误] 找不到 render_plan: {plan_path}")
            sys.exit(1)
        output_path = Path(args.output or f"output/output_{music_stem}.mp4")
        temp_dir = Path(f"output/clips_{music_stem}")
        xml_path = output_path.with_suffix(".xml")
        stripped_dir = Path("output/stripped") if Path("output/stripped").exists() else None

    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = [RenderItem(**r) for r in plan_data]
    print(f"[motif --render-only] 读取 {len(plan)} 条 clip")
    print(f"  music: {music_path}")
    print(f"  plan:  {plan_path}")
    print(f"  output: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # 导出 FCP XML（让用户可在 Premiere Pro / DaVinci Resolve 里做帧精度微调）
    if not args.no_xml:
        export_fcpxml(
            plan=plan_data,
            music_path=music_path,
            output_xml=xml_path,
            stripped_dir=stripped_dir,
        )
        print(f"  xml:    {xml_path}  (可直接 import 到 Premiere Pro / DaVinci)")

    render_video(
        render_plan=plan,
        music_path=str(music_path),
        output_path=str(output_path),
        temp_dir=str(temp_dir),
    )
    print(f"[motif] 完成：{output_path}")


def cmd_export_xml(args):
    """只导出 FCP XML，不做视频渲染。"""
    from pipeline.xml_exporter import export_fcpxml
    from utils.project_paths import ProjectPaths

    if args.project:
        pp = ProjectPaths(args.project)
        if not pp.exists():
            print(f"[错误] 项目不存在: {pp.root}")
            sys.exit(1)
        music_files = list(pp.music_dir.glob("*"))
        if not music_files:
            print(f"[错误] {pp.music_dir} 下没有音乐文件")
            sys.exit(1)
        music_path = music_files[0]
        plan_path = pp.render_plan_path
        xml_path = Path(args.output) if args.output else (pp.root / "motif_timeline.xml")
        stripped_dir = pp.stripped_dir
    else:
        if not (args.music and args.plan):
            print("[错误] 必须提供 --project 或 (--music + --plan)")
            sys.exit(1)
        music_path = Path(args.music).resolve()
        plan_path = Path(args.plan)
        xml_path = Path(args.output) if args.output else plan_path.with_suffix(".xml")
        stripped_dir = Path("output/stripped") if Path("output/stripped").exists() else None

    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    out = export_fcpxml(
        plan=plan_data,
        music_path=music_path,
        output_xml=xml_path,
        stripped_dir=stripped_dir,
    )
    print(f"[motif export-xml] 已生成: {out}")
    print(f"  Premiere Pro: File → Import → 选择这个 xml")
    print(f"  DaVinci Resolve: File → Import → Timeline")


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
    p_run.add_argument("--no-xml", action="store_true", help="跳过 FCP XML 导出")

    # render-only
    p_ro = sub.add_parser("render-only", help="跳过规划，用已有 render_plan 渲染")
    p_ro.add_argument("--project", default=None, help="项目名（projects/<name>/，优先使用）")
    p_ro.add_argument("--music", default=None, help="音乐文件路径（兼容老数据）")
    p_ro.add_argument("--stem", default=None, help="音乐文件名（无后缀），用于定位缓存")
    p_ro.add_argument("--plan", default=None, help="指定 render_plan JSON 路径")
    p_ro.add_argument("--output", default=None, help="输出路径")
    p_ro.add_argument("--no-xml", action="store_true", help="跳过 FCP XML 导出")

    # export-xml（只导 XML 不渲染）
    p_xml = sub.add_parser("export-xml", help="只导出 FCP XML（不渲染视频）")
    p_xml.add_argument("--project", default=None, help="项目名（projects/<name>/）")
    p_xml.add_argument("--music", default=None, help="音乐文件路径")
    p_xml.add_argument("--plan", default=None, help="render_plan JSON 路径")
    p_xml.add_argument("--output", default=None, help="输出 xml 路径")

    args = parser.parse_args()

    if args.command == "ui":
        cmd_ui(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "render-only":
        cmd_render_only(args)
    elif args.command == "export-xml":
        cmd_export_xml(args)


if __name__ == "__main__":
    main()
