"""
motion_analyzer.py — 用 OpenCV Farneback 光流给每个场景算客观运动强度。

输入：视频路径 + 场景列表（含 start/end）
输出：dict {scene_id: motion_intensity}，motion_intensity ∈ [0, 1]

实现：
  - 每个场景在 [start, end] 内均匀采样 N_SAMPLES 对相邻帧（间隔 SAMPLE_GAP_SEC）
  - 用 cv2.calcOpticalFlowFarneback 算稠密光流
  - 取所有帧对的光流幅值均值
  - tanh(mean_mag / NORM_SCALE) 归一化到 [0, 1]
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

N_SAMPLES = 4         # 每个场景采样多少对相邻帧
SAMPLE_GAP_SEC = 0.1  # 每对帧的时间间隔（秒）
DOWNSCALE_W = 320     # 下采样到此宽度，加速光流计算
NORM_SCALE = 6.0      # tanh 归一化的 scale，6.0 大致让快剪战斗 → 0.85+


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    if w > DOWNSCALE_W:
        new_h = int(h * DOWNSCALE_W / w)
        frame = cv2.resize(frame, (DOWNSCALE_W, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _scene_motion(cap: cv2.VideoCapture, start: float, end: float) -> float:
    duration = max(end - start, 0.0)
    if duration < SAMPLE_GAP_SEC * 2:
        # 太短：取中间一对帧
        mid = (start + end) / 2.0
        sample_times = [max(start, mid - SAMPLE_GAP_SEC / 2)]
    else:
        # 在 [start+gap, end-gap] 内均匀取 N_SAMPLES 个起点
        usable_start = start + SAMPLE_GAP_SEC
        usable_end = end - SAMPLE_GAP_SEC
        n = max(1, min(N_SAMPLES, int(duration / SAMPLE_GAP_SEC)))
        if n == 1:
            sample_times = [(usable_start + usable_end) / 2]
        else:
            step = (usable_end - usable_start) / (n - 1)
            sample_times = [usable_start + i * step for i in range(n)]

    mags = []
    for t in sample_times:
        f1 = _read_frame_at(cap, t)
        f2 = _read_frame_at(cap, t + SAMPLE_GAP_SEC)
        if f1 is None or f2 is None:
            continue
        if f1.shape != f2.shape:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(mag.mean()))

    if not mags:
        return 0.0
    mean_mag = float(np.mean(mags))
    # tanh 归一化：mean_mag/NORM_SCALE → [0,1]，对极快剪辑饱和
    return float(math.tanh(mean_mag / NORM_SCALE))


def compute_motion_intensities(
    video_path: str,
    scenes: list[dict],
) -> dict[int, float]:
    """
    给一个视频里每个 scene 算 motion_intensity ∈ [0,1]，返回 {scene_id: value}。
    scene 必须含 scene_id / start / end。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"[motion] 无法打开视频：{video_path}")
        return {}

    result: dict[int, float] = {}
    try:
        for scene in scenes:
            sid = scene["scene_id"]
            try:
                m = _scene_motion(cap, float(scene["start"]), float(scene["end"]))
            except Exception as e:
                logger.warning(f"[motion] scene {sid} 光流计算失败：{e}")
                m = 0.0
            result[sid] = m
    finally:
        cap.release()

    if result:
        vals = list(result.values())
        logger.info(
            f"[motion] {Path(video_path).name}: {len(result)} 场景  "
            f"motion_intensity min={min(vals):.2f} max={max(vals):.2f} mean={np.mean(vals):.2f}"
        )
    return result
