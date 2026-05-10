"""统一的项目路径管理。所有中间/最终产物都写到 projects/<name>/ 下。

布局（一音乐一项目）：

    projects/<project_name>/
        music/              原始音乐
        videos/
            raw/            用户上传的原始视频
            stripped/       去音轨版本（分析/渲染都用它）
        analysis/
            audio_map.json
            scene_table.json
            frames/         场景关键帧
        plan/
            render_plan.json
            render_plan_snapped.json
            trace.json      planner 调试日志
        clips/              渲染中间片段
        final.mp4           最终成片
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


PROJECTS_ROOT = Path("projects")


def sanitize(name: str) -> str:
    """把任意字符串转成安全目录名（保留中文、字母、数字、- _ .）。"""
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    return cleaned or "untitled"


class ProjectPaths:
    def __init__(self, project_name: str, root: Path = PROJECTS_ROOT):
        self.name = sanitize(project_name)
        self.root = Path(root) / self.name

    # ── 子目录 ─────────────────────────────────────────────────────────────
    @property
    def music_dir(self) -> Path:      return self.root / "music"
    @property
    def raw_videos_dir(self) -> Path: return self.root / "videos" / "raw"
    @property
    def stripped_dir(self) -> Path:   return self.root / "videos" / "stripped"
    @property
    def analysis_dir(self) -> Path:   return self.root / "analysis"
    @property
    def frames_dir(self) -> Path:     return self.analysis_dir / "frames"
    @property
    def plan_dir(self) -> Path:       return self.root / "plan"
    @property
    def clips_dir(self) -> Path:      return self.root / "clips"

    # ── 具名文件 ────────────────────────────────────────────────────────────
    @property
    def audio_map_path(self) -> Path:    return self.analysis_dir / "audio_map.json"
    @property
    def scene_table_path(self) -> Path:  return self.analysis_dir / "scene_table.json"
    @property
    def material_analysis_path(self) -> Path:
        return self.analysis_dir / "material_analysis.json"
    @property
    def render_plan_path(self) -> Path:  return self.plan_dir / "render_plan.json"
    @property
    def render_plan_snapped_path(self) -> Path:
        return self.plan_dir / "render_plan_snapped.json"
    @property
    def planner_trace_path(self) -> Path: return self.plan_dir / "trace.json"
    @property
    def final_path(self) -> Path:        return self.root / "final.mp4"

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def create_all(self) -> None:
        """创建所有子目录。"""
        for d in (self.music_dir, self.raw_videos_dir, self.stripped_dir,
                  self.frames_dir, self.plan_dir, self.clips_dir):
            d.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.root.exists()

    def clean(self) -> None:
        """删除整个项目目录（用于重跑）。"""
        if self.root.exists():
            shutil.rmtree(self.root)

    def __repr__(self) -> str:
        return f"ProjectPaths(name={self.name!r}, root={self.root})"
