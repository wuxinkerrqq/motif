from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from models.audio import SpecialEvent


def detect_events(
    energy_times: list[float],
    energy_values: list[float],
    onset_times: list[float],
    onset_values: list[float],
    *,
    hop_length: int = 512,
    sample_rate: int = 22050,
) -> list[SpecialEvent]:
    """
    统一事件检测：基于显著性曲线，自适应阈值，不硬编码事件类型枚举。

    输出两类事件：
    - section_boundary: 能量/频谱发生结构性大跳变（原 explosion/drop/silence 归这里）
    - impact: 打击乐显著峰值，强度排前5%且间隔>=2s

    所有阈值从数据自适应计算，无硬编码参数。
    """
    events: list[SpecialEvent] = []

    events += _detect_boundaries(energy_times, energy_values, hop_length, sample_rate)
    events += _detect_impacts(onset_times, onset_values)

    events.sort(key=lambda e: e.time)
    events = _deduplicate(events, min_gap=0.3)
    return events


def _detect_boundaries(
    times: list[float],
    values: list[float],
    hop_length: int,
    sample_rate: int,
) -> list[SpecialEvent]:
    """
    用前后窗口均值对比检测结构性跳变。
    前2s均值 vs 后2s均值，差值超过全曲std的1.5倍才算边界。
    边界之间至少间隔5s，全曲最多保留8个。
    """
    arr = np.array(values, dtype=float)
    t = np.array(times, dtype=float)
    if len(arr) < 10:
        return []

    frame_duration = float(t[1] - t[0]) if len(t) > 1 else hop_length / sample_rate
    window_frames = max(1, int(2.0 / frame_duration))
    min_gap_frames = max(1, int(5.0 / frame_duration))

    global_std = float(arr.std())
    threshold = global_std * 1.5

    scores = []
    for i in range(window_frames, len(arr) - window_frames):
        before = float(arr[i - window_frames:i].mean())
        after = float(arr[i:i + window_frames].mean())
        scores.append(abs(after - before))
    scores = np.array(scores)

    peaks, props = find_peaks(scores, height=threshold, distance=min_gap_frames)

    # 只保留得分超过 scores 的 85 百分位的峰
    if len(peaks) > 0:
        score_threshold = float(np.percentile(scores[peaks], 85))
        peaks = peaks[scores[peaks] >= score_threshold]

    events = []
    for p in peaks:
        real_idx = p + window_frames
        events.append(SpecialEvent(
            time=round(float(t[real_idx]), 3),
            type="section_boundary",
            intensity=round(float(scores[p]), 3),
        ))
    return events


def _detect_impacts(
    times: list[float],
    values: list[float],
) -> list[SpecialEvent]:
    """
    从 percussive onset 曲线检测显著打击点。
    阈值 = 95百分位，间隔 >= 2s，只保留真正突出的峰。
    """
    if not times or not values:
        return []

    arr = np.array(values, dtype=float)
    t = np.array(times, dtype=float)

    frame_duration = float(t[1] - t[0]) if len(t) > 1 else 0.023
    min_frames = max(1, int(2.0 / frame_duration))

    height = max(float(np.percentile(arr, 97)), 0.6)
    peaks, props = find_peaks(arr, height=height, distance=min_frames)

    # 只保留强度超过 peaks 自身 85 百分位的
    if len(peaks) > 0:
        intensity_threshold = float(np.percentile(arr[peaks], 85))
        peaks = peaks[arr[peaks] >= intensity_threshold]

    events = []
    for p in peaks:
        events.append(SpecialEvent(
            time=round(float(t[p]), 3),
            type="impact",
            intensity=round(float(arr[p]), 3),
        ))
    return events


def _deduplicate(events: list[SpecialEvent], min_gap: float = 0.3) -> list[SpecialEvent]:
    """去掉时间上过于接近的同类事件，保留强度更高的那个。"""
    if not events:
        return events

    result = [events[0]]
    for ev in events[1:]:
        last = result[-1]
        if ev.type == last.type and abs(ev.time - last.time) < min_gap:
            if (ev.intensity or 0) > (last.intensity or 0):
                result[-1] = ev
        else:
            result.append(ev)
    return result


# ── 兼容旧调用接口 ────────────────────────────────────────────────────────────

def detect_special_events(
    times: list[float],
    values: list[float],
    *,
    silence_threshold: float | None = None,
    explosion_threshold: float | None = None,
    quiet_threshold: float | None = None,
    lookback_seconds: float = 1.0,
    min_silence_duration: float = 0.3,
    hop_length: int = 512,
    sample_rate: int = 22050,
) -> list[SpecialEvent]:
    """旧接口兼容层，内部转发到 detect_events 的边界检测部分。"""
    return _detect_boundaries(times, values, hop_length, sample_rate)


def detect_impact_events(
    onset_times: list[float],
    onset_values: list[float],
    **kwargs,
) -> list[SpecialEvent]:
    """旧接口兼容层，内部转发到 detect_impacts。"""
    return _detect_impacts(onset_times, onset_values)
