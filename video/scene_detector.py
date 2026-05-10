from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
) -> list[dict]:
    """
    用 PySceneDetect 检测场景边界。
    threshold 越低切得越细，越高切得越粗，默认 27.0。

    返回：场景列表，每项包含 start/end/duration（秒）
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec
        scenes.append({
            "scene_id": i,
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(duration, 3),
            "scene_index": f"{i + 1}/{len(scene_list)}",
        })

    return scenes


def extract_keyframes(
    video_path: str,
    scene: dict,
    output_dir: str,
) -> list[str]:
    """
    从场景中提取关键帧图片。
    取帧策略：
        时长 < 1s → 取 1 帧（中间帧）
        时长 1-3s → 取 2 帧（1/3 和 2/3 处）
        时长 > 3s → 取 3 帧（1/4、1/2、3/4 处）
    不取首帧和尾帧，避免运动模糊和转场污染。

    返回：关键帧图片路径列表
    """
    duration = scene["duration"]
    start = scene["start"]
    scene_id = scene["scene_id"]

    if duration < 1.0:
        offsets = [0.5]
    elif duration < 3.0:
        offsets = [1 / 3, 2 / 3]
    else:
        offsets = [1 / 4, 1 / 2, 3 / 4]

    timestamps = [start + duration * offset for offset in offsets]

    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    frame_paths = []

    for i, ts in enumerate(timestamps):
        frame_path = output_p / f"{video_stem}_scene{scene_id:04d}_frame{i}.jpg"

        if frame_path.exists():
            frame_paths.append(str(frame_path))
            continue

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",         # 图片质量（2 最高，31 最低）
            str(frame_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode == 0 and frame_path.exists():
            frame_paths.append(str(frame_path))
        else:
            logger.warning(f"  关键帧提取失败：scene {scene_id} frame {i} @ {ts:.2f}s")

    return frame_paths


def process_video(
    video_path: str,
    frames_dir: str,
    threshold: float = 27.0,
) -> list[dict]:
    """
    对单个视频文件完成场景检测 + 关键帧提取。

    返回：scene 列表，每项包含物理元数据 + keyframes 路径
    """
    source_file = Path(video_path).name
    logger.info(f"  [scene_detector] 处理：{source_file}")

    scenes = detect_scenes(video_path, threshold=threshold)
    logger.info(f"    检测到 {len(scenes)} 个场景")

    for scene in scenes:
        frames = extract_keyframes(video_path, scene, frames_dir)
        scene["source_file"] = source_file
        scene["keyframes"] = frames

    return scenes