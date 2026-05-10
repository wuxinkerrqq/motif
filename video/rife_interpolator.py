"""
RIFE 光流插帧模块（可选功能）

用于慢动作片段的帧插值，使 setpts 降速后画面丝滑而非卡顿。
依赖：rife-ncnn-vulkan 命令行工具（需单独安装）

开关：config.py → DefaultConfig.RIFE_ENABLED = True
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger


def is_available() -> bool:
    from config import DefaultConfig
    exe = DefaultConfig.RIFE_EXE_PATH
    return Path(exe).is_file() if exe else shutil.which("rife-ncnn-vulkan") is not None


def _exe() -> str:
    from config import DefaultConfig
    return DefaultConfig.RIFE_EXE_PATH or "rife-ncnn-vulkan"


def needed_multiplier(speed_factor: float) -> int:
    """计算需要的插帧倍率（2 的幂次），使慢动作帧数足够。"""
    if speed_factor >= 1.0:
        return 1
    ratio = 1.0 / speed_factor
    exp = math.ceil(math.log2(ratio))
    return 2 ** max(1, exp)


def interpolate_clip(input_path: str, output_path: str, multiplier: int = 2) -> bool:
    """
    对视频片段做 RIFE 插帧，输出帧率为原始帧率 × multiplier。

    流程：
      1. FFmpeg 提取 PNG 帧序列
      2. rife-ncnn-vulkan 插帧
      3. FFmpeg 按新帧率重新编码

    Returns True on success, False on any failure（调用方回退到原始路径）。
    """
    if not is_available():
        logger.warning("[RIFE] rife-ncnn-vulkan 未找到，跳过插帧")
        return False

    # 探测源帧率
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        capture_output=True, text=True,
    )
    try:
        num, den = probe.stdout.strip().split("/")
        src_fps = float(num) / float(den)
    except Exception:
        src_fps = 24.0

    new_fps = src_fps * multiplier

    with tempfile.TemporaryDirectory(prefix="rife_") as tmp:
        frames_in  = Path(tmp) / "in"
        frames_out = Path(tmp) / "out"
        frames_in.mkdir()
        frames_out.mkdir()

        # Step 1: 提取帧
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, str(frames_in / "%08d.png")],
            capture_output=True,
        )
        if r.returncode != 0:
            logger.warning(f"[RIFE] 帧提取失败：{r.stderr.decode()[:120]}")
            return False

        frame_count = len(list(frames_in.glob("*.png")))
        if frame_count == 0:
            logger.warning("[RIFE] 未提取到任何帧")
            return False

        # Step 2: RIFE 插帧（-n 指定目标帧数）
        target_frames = frame_count * multiplier
        from config import DefaultConfig
        rife_cmd = [
            _exe(),
            "-i", str(frames_in),
            "-o", str(frames_out),
            "-n", str(target_frames),
        ]
        if DefaultConfig.RIFE_MODEL_PATH:
            rife_cmd += ["-m", DefaultConfig.RIFE_MODEL_PATH]
        r = subprocess.run(rife_cmd)
        if r.returncode != 0:
            logger.warning(f"[RIFE] 插帧失败：{r.stderr.decode()[:120]}")
            return False

        # Step 3: 重新编码
        r = subprocess.run(
            [
                "ffmpeg", "-y",
                "-r", f"{new_fps:.3f}",
                "-i", str(frames_out / "%08d.png"),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an",
                output_path,
            ],
            capture_output=True,
        )
        if r.returncode != 0:
            logger.warning(f"[RIFE] 重编码失败：{r.stderr.decode()[:120]}")
            return False

    logger.info(f"[RIFE] 插帧完成：{multiplier}x，{src_fps:.1f}fps → {new_fps:.1f}fps")
    return True
