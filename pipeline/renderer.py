from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger

from models.plan import RenderItem


def render_video(
    render_plan: list[RenderItem],
    music_path: str,
    output_path: str,
    temp_dir: str = "output/clips",
) -> str:
    if not render_plan:
        raise ValueError("render_plan 为空，无法渲染")

    plan = sorted(render_plan, key=lambda r: r.audio_start)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for r in plan:
        clip_path = _extract_clip(r, temp_dir)
        if clip_path:
            clip_paths.append(clip_path)

    if not clip_paths:
        raise RuntimeError("所有片段截取失败")

    logger.info(f"[Renderer] 成功截取 {len(clip_paths)}/{len(plan)} 个片段")

    concat_path = str(Path(temp_dir) / "concat.mp4")
    _concat_with_transitions(clip_paths, plan, concat_path)
    _mix_audio(concat_path, music_path, output_path)

    logger.info(f"[Renderer] 渲染完成：{output_path}")
    return output_path


def _extract_clip(r: RenderItem, temp_dir: str) -> str | None:
    out_path = Path(temp_dir) / f"clip_{r.order:04d}.mp4"

    clip_duration = r.clip_end - r.clip_start
    if clip_duration <= 0:
        logger.warning(f"  clip {r.order} 时长为 0，跳过")
        return None

    source_path = Path(r.source_file)

    # 如果路径不存在，尝试多种补全策略
    if not source_path.exists():
        # 策略1：在常用素材目录下查找
        search_dirs = [
            Path("素材"),
            Path("test/videos"),
            Path("test/videos").parent,
        ]
        # 递归搜索文件名匹配的文件
        fname = source_path.name
        found = None
        for search_dir in search_dirs:
            if search_dir.exists():
                matches = list(search_dir.rglob(fname))
                if matches:
                    found = matches[0]
                    break
        if found:
            logger.info(f"  clip {r.order} 路径补全: {r.source_file} → {found}")
            source_path = found
        else:
            logger.warning(f"  clip {r.order} 找不到文件: {r.source_file}，跳过")
            return None

    # RIFE 插帧已禁用：精度损失导致每个慢放 clip 末尾有 0.2-0.8s 静帧"卡一下"，
    # 砍掉后慢放走纯 setpts，画面会有轻微粘连但时长精确无卡顿
    rife_clip_start = r.clip_start

    vf_parts = []
    if r.speed_factor != 1.0:
        vf_parts.append(f"setpts={1.0/r.speed_factor:.4f}*PTS")

    # 转场特效（作用于 clip 开头 transition_duration 秒）
    td = r.transition_duration if r.transition_duration > 0 else 0.15
    ct = r.cut_type

    if ct in ("flash_white", "flash_black"):
        color = "white" if ct == "flash_white" else "black"
        nb_frames = max(2, int(td * 24))
        vf_parts.append(f"fade=type=in:start_frame=0:nb_frames={nb_frames}:color={color}")

    elif ct in ("camera_shake_cut", "shake", "camera_shake"):
        # 相机抖动：全 clip 轻微偏移（crop 不支持 enable，只能持续）
        vf_parts.append("crop=iw-20:ih-20:15:15")

    elif ct in ("zoom_punch", "zoom_in"):
        # 冲击缩放：全 clip 轻微放大（scale 不支持 enable，只能持续）
        vf_parts.append("scale=iw*1.15:ih*1.15,crop=iw/1.15:ih/1.15:(iw-iw/1.15)/2:(ih-ih/1.15)/2")

    elif ct in ("zoom_out", "zoom_pull"):
        # 拉远缩放：全 clip 缩小 + 黑边
        vf_parts.append(
            "scale=iw*0.85:ih*0.85,"
            "pad=ceil(iw/0.85/2)*2:ceil(ih/0.85/2)*2:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    elif ct in ("rgb_split", "chromatic", "glitch"):
        # 色偏：全 clip 色相偏移（hue 支持 enable，但为统一改常量）
        vf_parts.append("hue=h=20:s=1.3")

    elif ct in ("radial_blur", "spin_blur"):
        # 径向模糊近似：开头 td 秒内叠加高斯模糊（常量 sigma，用 enable 时间开关）
        vf_parts.append(f"gblur=sigma=8:enable='lt(t,{td})'")

    elif ct in ("whip_pan", "motion_blur"):
        # 水平动态模糊：开头 td 秒内开 boxblur
        vf_parts.append(
            f"boxblur=luma_radius=20:luma_power=1:chroma_radius=0:enable='lt(t,{td})'"
        )

    elif ct in ("fade_in", "fade"):
        # 平滑淡入
        nb_frames = max(4, int(td * 24))
        vf_parts.append(f"fade=type=in:start_frame=0:nb_frames={nb_frames}:color=black")

    elif ct in ("dissolve",):
        # 单片 dissolve 近似成 fade_in（真 dissolve 需要两片 xfade，暂不做）
        nb_frames = max(4, int(td * 24))
        vf_parts.append(f"fade=type=in:start_frame=0:nb_frames={nb_frames}")

    # 输出帧数 = 音乐时长 × 24fps（强制精确，避免 ffmpeg 自行取整累积漂移）
    audio_duration = r.audio_end - r.audio_start
    nb_output_frames = max(1, round(audio_duration * 24))

    cmd = ["ffmpeg", "-y", "-ss", f"{rife_clip_start:.3f}", "-i", str(source_path)]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += [
        "-frames:v", str(nb_output_frames),
        "-vsync", "cfr",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", "-r", "24", str(out_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err_tail = result.stderr.decode("utf-8", errors="ignore").strip().splitlines()[-5:]
        logger.warning(f"  clip {r.order} 截取失败 | cmd={' '.join(cmd[:6])} ...")
        for line in err_tail:
            logger.warning(f"    ffmpeg: {line}")
        return None

    logger.info(
        f"  [{r.order:03d}] {source_path.name} {r.clip_start:.2f}s-{r.clip_end:.2f}s "
        f"→ 音乐 {r.audio_start:.2f}s-{r.audio_end:.2f}s "
        f"| scene {r.scene_id:03d} | {clip_duration:.2f}s"
    )

    return str(out_path)


def _probe_duration(path: str) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip()) if r.stdout.strip() else None
    except Exception:
        return None


def _concat_with_transitions(
    clip_paths: list[str],
    plan: list[RenderItem],
    output_path: str,
) -> None:
    """
    拼接片段。
    - 连续 dissolve 片段组内用 xfade filter_complex
    - 组间（hard_cut 边界）用 concat demuxer
    """
    if len(clip_paths) == 1:
        import shutil
        shutil.copy(clip_paths[0], output_path)
        return

    has_dissolve = any(
        r.cut_type == "dissolve" and r.transition_duration > 0
        for r in plan[1:len(clip_paths)]
    )
    if not has_dissolve:
        _concat_clips(clip_paths, output_path)
        return

    # ── 按 hard_cut 边界分组 ──────────────────────────────────────────────────
    # 每组是连续的 dissolve 片段（含组头，组头本身可以是 hard_cut 进入的）
    groups: list[list[tuple[str, RenderItem]]] = []
    current: list[tuple[str, RenderItem]] = [(clip_paths[0], plan[0])]

    for i in range(1, len(clip_paths)):
        r = plan[i]
        if r.cut_type == "dissolve" and r.transition_duration > 0:
            current.append((clip_paths[i], plan[i]))
        else:
            groups.append(current)
            current = [(clip_paths[i], plan[i])]
    groups.append(current)

    # ── 每组单独处理 ──────────────────────────────────────────────────────────
    temp_dir = Path(clip_paths[0]).parent
    group_paths: list[str] = []

    for gi, group in enumerate(groups):
        if len(group) == 1:
            group_paths.append(group[0][0])
            continue

        group_out = str(temp_dir / f"group_{gi:04d}.mp4")
        _xfade_group([(p, r) for p, r in group], group_out)
        group_paths.append(group_out)

    # ── 拼接各组 ─────────────────────────────────────────────────────────────
    if len(group_paths) == 1:
        import shutil
        shutil.copy(group_paths[0], output_path)
    else:
        _concat_clips(group_paths, output_path)

    logger.info(f"[Renderer] 拼接完成（{len(groups)} 组，含 dissolve 转场）：{output_path}")


def _xfade_group(group: list[tuple[str, "RenderItem"]], output_path: str) -> None:
    """对一组连续 dissolve 片段应用 xfade，输出单个文件。"""
    clip_paths = [g[0] for g in group]
    items = [g[1] for g in group]
    n = len(clip_paths)
    durations = [r.audio_end - r.audio_start for r in items]

    filter_parts = []
    cumulative_offset = durations[0]

    for i in range(1, n):
        r = items[i]
        in_label = "[0:v]" if i == 1 else f"[v{i-1}]"
        out_label = "[vout]" if i == n - 1 else f"[v{i}]"

        t = min(r.transition_duration, durations[i - 1] * 0.8, durations[i] * 0.8)
        t = max(t, 0.04)  # 最小 1 帧（24fps）
        offset = max(0.0, cumulative_offset - t)

        filter_parts.append(
            f"{in_label}[{i}:v]xfade=transition=fade"
            f":duration={t:.4f}:offset={offset:.4f}{out_label}"
        )
        cumulative_offset = offset + durations[i]

    filter_complex = "; ".join(filter_parts)
    input_args = []
    for p in clip_paths:
        input_args += ["-i", p]

    cmd = [
        "ffmpeg", "-y", *input_args,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "24",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[:300]
        logger.warning(f"[Renderer] xfade 组失败，回退 concat: {err}")
        _concat_clips(clip_paths, output_path)


def _concat_clips(clip_paths: list[str], output_path: str) -> None:
    list_path = str(Path(output_path).parent / "concat_list.txt")
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).absolute()}'\n")
    # 重编码而非 -c copy，避免各 clip 时间戳不一致导致丢帧（48 个 clip × 几十毫秒 ≈ 数秒缺失）
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "24",
        "-fflags", "+genpts",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"拼接失败: {err}")
    logger.info(f"[Renderer] 拼接完成：{output_path}")


def _mix_audio(video_path: str, music_path: str, output_path: str) -> None:
    # 视频末尾用 tpad 复制最后一帧 +10s 兜底，避免视频比音乐短导致 -shortest 截短音乐
    # 由于 -shortest 取短的流（此时是音乐），最终时长 = 音乐时长，画面不足部分用最后一帧填补
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", music_path,
        "-vf", "tpad=stop_mode=clone:stop_duration=10",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"音频合成失败: {err}")
    logger.info(f"[Renderer] 音频合成完成：{output_path}")