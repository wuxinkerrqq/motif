"""FCP7 XMEML 时间线导出：把 render_plan 转成 Premiere Pro / DaVinci Resolve
可直接 import 的 .xml 文件，让用户在专业 NLE 里做帧精度微调。

用法（作为库）:
    from pipeline.xml_exporter import export_fcpxml
    export_fcpxml(render_plan, music_path, output_xml_path)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

FPS = 24


def _probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, timeout=10,
    )
    return float(r.stdout.strip()) if r.stdout.strip() else 0.0


def _file_url(p: Path) -> str:
    abs_str = str(p.resolve()).replace("\\", "/")
    if abs_str[1:3] == ":/":
        return f"file://localhost/{abs_str}"
    return f"file://{abs_str}"


def _build_xml(
    plan: list[dict],
    music_path: Path,
    sequence_name: str,
    stripped_dir: Path | None = None,
) -> str:
    music_dur = _probe_duration(music_path)
    total_frames = round(music_dur * FPS)

    # 收集所有 source files（去重），优先用 stripped_dir 下的同名文件
    def resolve_source(raw: str) -> Path:
        p = Path(raw)
        if p.exists():
            return p
        if stripped_dir:
            # 尝试 xxx_stripped.mp4
            stem = p.stem
            cand = stripped_dir / f"{stem}_stripped{p.suffix}"
            if cand.exists():
                return cand
            # 尝试原名
            cand2 = stripped_dir / p.name
            if cand2.exists():
                return cand2
        return p

    sources: dict[str, Path] = {}
    for c in plan:
        sf = resolve_source(c["source_file"])
        sources.setdefault(str(sf.resolve()), sf)

    file_ids: dict[str, str] = {p: f"file-{i+1}" for i, p in enumerate(sources)}
    file_durations: dict[str, int] = {
        p_str: max(1, round(_probe_duration(p) * FPS))
        for p_str, p in sources.items()
    }

    seen_fid: set[str] = set()
    video_items = []
    for c in plan:
        order = c["order"]
        sf = resolve_source(c["source_file"])
        fid = file_ids[str(sf.resolve())]

        audio_start_f = round(c["audio_start"] * FPS)
        audio_end_f   = round(c["audio_end"]   * FPS)
        clip_start_f  = round(c["clip_start"]  * FPS)
        clip_end_f    = round(c["clip_end"]    * FPS)
        speed         = c.get("speed_factor", 1.0) or 1.0

        if fid not in seen_fid:
            seen_fid.add(fid)
            file_xml = (
                f'<file id="{fid}">'
                f'<name>{escape(sf.name)}</name>'
                f'<pathurl>{escape(_file_url(sf))}</pathurl>'
                f'<rate><timebase>{FPS}</timebase><ntsc>FALSE</ntsc></rate>'
                f'<duration>{file_durations[str(sf.resolve())]}</duration>'
                f'<media><video><samplecharacteristics>'
                f'<width>1920</width><height>1080</height></samplecharacteristics></video></media>'
                f'</file>'
            )
        else:
            file_xml = f'<file id="{fid}"/>'

        speed_xml = ""
        if abs(speed - 1.0) > 1e-3:
            speed_xml = (
                f'<timeremap><speed>{speed * 100:.2f}</speed>'
                f'<frameblending>FALSE</frameblending></timeremap>'
            )

        video_items.append(f'''
      <clipitem id="v-{order}">
        <name>{escape(f"scene_{c['scene_id']}")}</name>
        <enabled>TRUE</enabled>
        <duration>{file_durations[str(sf.resolve())]}</duration>
        <rate><timebase>{FPS}</timebase><ntsc>FALSE</ntsc></rate>
        <start>{audio_start_f}</start>
        <end>{audio_end_f}</end>
        <in>{clip_start_f}</in>
        <out>{clip_end_f}</out>
        {file_xml}
        {speed_xml}
      </clipitem>''')

    audio_item = f'''
      <clipitem id="audio-music">
        <name>{escape(music_path.name)}</name>
        <enabled>TRUE</enabled>
        <duration>{total_frames}</duration>
        <rate><timebase>{FPS}</timebase><ntsc>FALSE</ntsc></rate>
        <start>0</start>
        <end>{total_frames}</end>
        <in>0</in>
        <out>{total_frames}</out>
        <file id="music-file">
          <name>{escape(music_path.name)}</name>
          <pathurl>{escape(_file_url(music_path))}</pathurl>
          <rate><timebase>{FPS}</timebase><ntsc>FALSE</ntsc></rate>
          <duration>{total_frames}</duration>
          <media><audio><samplecharacteristics>
            <samplerate>48000</samplerate><depth>16</depth>
          </samplecharacteristics><channelcount>2</channelcount></audio></media>
        </file>
        <sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>
      </clipitem>'''

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="5">
  <sequence id="motif-seq">
    <name>{escape(sequence_name)}</name>
    <duration>{total_frames}</duration>
    <rate><timebase>{FPS}</timebase><ntsc>FALSE</ntsc></rate>
    <media>
      <video>
        <format><samplecharacteristics>
          <width>1920</width><height>1080</height>
          <rate><timebase>{FPS}</timebase><ntsc>FALSE</ntsc></rate>
        </samplecharacteristics></format>
        <track>{"".join(video_items)}
        </track>
      </video>
      <audio>
        <format><samplecharacteristics><samplerate>48000</samplerate><depth>16</depth></samplecharacteristics></format>
        <track>{audio_item}
        </track>
      </audio>
    </media>
  </sequence>
</xmeml>
'''


def export_fcpxml(
    plan: list[dict],
    music_path: Path | str,
    output_xml: Path | str,
    sequence_name: str | None = None,
    stripped_dir: Path | str | None = None,
) -> Path:
    """将 render_plan 导出为 FCP7 XMEML 文件。

    Args:
        plan: render_plan 列表（每项是 RenderItem 的 dict 形式）。
        music_path: 音乐文件路径。
        output_xml: 输出 .xml 路径。
        sequence_name: 在 PR 里显示的 sequence 名，默认用 xml 文件名 stem。
        stripped_dir: 若提供，会自动把 plan 里的 source_file 映射到 stripped 版本。

    Returns:
        写入的 xml 路径。
    """
    music_path = Path(music_path)
    output_xml = Path(output_xml)
    stripped_dir = Path(stripped_dir) if stripped_dir else None
    sequence_name = sequence_name or output_xml.stem

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    xml = _build_xml(plan, music_path, sequence_name, stripped_dir)
    output_xml.write_text(xml, encoding="utf-8")
    return output_xml
