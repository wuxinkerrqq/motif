from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger


def strip_audio(input_path: str, output_dir: str) -> str:
    """
    去除视频音轨，输出静音视频。
    使用 -c:v copy 直接复制视频流，不重新编码，速度极快。

    返回：输出文件路径
    """
    input_p = Path(input_path)
    output_p = Path(output_dir) / f"{input_p.stem}_stripped{input_p.suffix}"

    if output_p.exists():
        logger.info(f"  [preprocessor] 已存在，跳过：{output_p.name}")
        return str(output_p)

    output_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_p),
        "-an",           # 去掉音轨
        "-c:v", "copy",  # 视频流直接复制，不重新编码
        str(output_p),
    ]

    logger.info(f"  [preprocessor] 去音轨：{input_p.name} → {output_p.name}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 去音轨失败：{input_p.name}\n{result.stderr[-500:]}"
        )

    logger.info(f"  [preprocessor] 完成：{output_p.name}")
    return str(output_p)


def strip_audio_batch(
    video_paths: list[str],
    output_dir: str,
) -> list[str]:
    """
    批量去音轨，顺序处理。
    返回：去音轨后的文件路径列表（顺序与输入一致）
    """
    logger.info(f"[Preprocessor] 开始批量去音轨，共 {len(video_paths)} 个文件")
    stripped = []
    for path in video_paths:
        out = strip_audio(path, output_dir)
        stripped.append(out)
    logger.info(f"[Preprocessor] 全部完成")
    return stripped


def verify_no_audio(video_path: str) -> bool:
    """
    用 ffprobe 验证视频文件确实没有音轨。
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    return result.stdout.strip() == ""