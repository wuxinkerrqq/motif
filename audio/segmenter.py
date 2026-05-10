from __future__ import annotations

import numpy as np
from loguru import logger

from models.audio import AudioSegment


def compute_segments(
    energy_times: list[float],
    energy_values: list[float],
    downbeats: list[float],
    total_duration: float,
    onset_times: list[float] | None = None,
    onset_values: list[float] | None = None,
    spectral_flux_times: list[float] | None = None,
    spectral_flux_values: list[float] | None = None,
    min_segment_duration: float = 8.0,
    max_segment_duration: float = 60.0,
    target_segment_duration: float = 25.0,
) -> list[AudioSegment]:
    """
    代码驱动分段：基于多信号融合计算段落边界。

    信号源（权重）：
    1. 能量变化（前后窗口均值差）— 权重 1.0
    2. Onset strength（瞬变强度）— 权重 1.5，对瞬变更敏感
    3. Spectral flux（频谱变化率）— 权重 1.2，反映音色突变
    4. 特殊事件加成 — 固定加分

    前移修正：边界点向最近的 downbeat 前移（音乐结构变化通常发生在强拍上）。
    """
    arr = np.array(energy_values)
    times = np.array(energy_times)
    global_mean = float(arr.mean())
    global_std = float(arr.std())

    onset_arr = np.array(onset_values) if onset_values else None
    onset_times_arr = np.array(onset_times) if onset_times else None
    flux_arr = np.array(spectral_flux_values) if spectral_flux_values else None
    flux_times_arr = np.array(spectral_flux_times) if spectral_flux_times else None

    # ── Step 1: 计算每个 downbeat 的边界得分 ────────────────────────────────

    boundary_scores: dict[int, float] = {}

    for idx, db_time in enumerate(downbeats):
        score = 0.0

        # 信号1：能量变化得分（前后窗口均值差）
        score += _energy_change_score(db_time, arr, times, global_std)

        # 信号2：Onset strength 得分
        if onset_arr is not None and onset_times_arr is not None:
            score += 1.5 * _point_signal_score(db_time, onset_arr, onset_times_arr)

        # 信号3：Spectral flux 得分
        if flux_arr is not None and flux_times_arr is not None:
            score += 1.2 * _point_signal_score(db_time, flux_arr, flux_times_arr)

        if score > 0:
            boundary_scores[idx] = score

    # ── Step 2: 估算目标段落数，选取 Top-N 边界 ─────────────────────────────

    target_n = max(3, int(total_duration / target_segment_duration))
    sorted_boundaries = sorted(boundary_scores.items(), key=lambda x: x[1], reverse=True)

    selected_indices: list[int] = []
    for idx, score in sorted_boundaries:
        db_time = downbeats[idx]
        too_close = any(
            abs(db_time - downbeats[si]) < min_segment_duration
            for si in selected_indices
        )
        if too_close:
            continue
        selected_indices.append(idx)
        if len(selected_indices) >= target_n - 1:
            break

    selected_indices.sort()

    # ── Step 3: 前移修正 + 生成段落 ─────────────────────────────────────────

    boundaries = [0.0]
    for idx in selected_indices:
        t = downbeats[idx]
        if t > boundaries[-1] + min_segment_duration:
            t = _snap_backward(t, downbeats, min_step=0.2)
            if t > boundaries[-1] + min_segment_duration:
                boundaries.append(t)
    if boundaries[-1] < total_duration - 1.0:
        boundaries.append(total_duration)

    segments: list[AudioSegment] = []
    for i in range(len(boundaries) - 1):
        seg_start = round(boundaries[i], 3)
        seg_end = round(boundaries[i + 1], 3)

        seg_energy, seg_trend, seg_peak = _compute_segment_energy(
            arr, times, seg_start, seg_end, global_mean,
        )

        segments.append(AudioSegment(
            name=f"segment_{i + 1}",
            start=seg_start,
            end=seg_end,
            energy=seg_energy,
            energy_trend=seg_trend,
            energy_peak=seg_peak,
            mood="neutral",
            description="",
            visual_suggestion="",
        ))

    segments = _merge_short_segments(segments, min_segment_duration)
    segments = _split_long_segments(
        segments, arr, times, downbeats, global_mean,
        max_segment_duration, min_segment_duration,
    )

    for i, seg in enumerate(segments):
        seg.name = f"segment_{i + 1}"

    logger.info(f"  [segmenter] 分段完成: {len(segments)} 个段落")
    return segments


# ──────────────────────────────────────────────────────────────────────────────
# 得分计算辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _energy_change_score(
    db_time: float,
    arr: np.ndarray,
    times: np.ndarray,
    global_std: float,
) -> float:
    """能量变化得分：downbeat 前后窗口均值差 / 全局标准差。"""
    window_sec = 1.5
    frame_start = _time_to_frame(db_time - window_sec, times)
    frame_end = _time_to_frame(db_time + window_sec, times)
    frame_center = _time_to_frame(db_time, times)

    if frame_start < frame_center < frame_end < len(arr):
        before_mean = float(arr[frame_start:frame_center].mean())
        after_mean = float(arr[frame_center:frame_end].mean())
        delta = abs(after_mean - before_mean)
        if global_std > 0:
            return delta / global_std
    return 0.0


def _point_signal_score(
    db_time: float,
    signal_arr: np.ndarray,
    signal_times: np.ndarray,
) -> float:
    """
    Onset / Spectral flux 得分：取 downbeat 附近窗口内的峰值。
    使用峰值而非均值，因为瞬变是尖峰信号。
    """
    window_sec = 1.0
    frame_start = _time_to_frame(db_time - window_sec, signal_times)
    frame_end = _time_to_frame(db_time + window_sec, signal_times)

    if frame_start < frame_end < len(signal_arr):
        window = signal_arr[frame_start:frame_end]
        if len(window) > 0:
            return float(window.max())
    return 0.0


def _snap_backward(
    boundary_time: float,
    downbeats: list[float],
    min_step: float = 0.2,
) -> float:
    """
    前移修正：将边界点向最近的 downbeat 前移。
    音乐结构变化通常发生在强拍上，而非强拍之后。
    最多前移一个 downbeat 间隔，且至少 min_step 秒。
    """
    for i in range(1, len(downbeats)):
        if downbeats[i] >= boundary_time:
            prev_db = downbeats[i - 1]
            if boundary_time - prev_db >= min_step:
                return prev_db
            break
    return boundary_time


# ──────────────────────────────────────────────────────────────────────────────
# 段落能量计算
# ──────────────────────────────────────────────────────────────────────────────

def _time_to_frame(time_sec: float, times: list[float] | np.ndarray) -> int:
    if isinstance(times, np.ndarray):
        idx = int(np.searchsorted(times, time_sec))
    else:
        idx = 0
        for i, t in enumerate(times):
            if t >= time_sec:
                idx = i
                break
        else:
            idx = len(times) - 1
    return max(0, min(idx, len(times) - 1))


def _compute_segment_energy(
    arr: np.ndarray,
    times: np.ndarray,
    seg_start: float,
    seg_end: float,
    global_mean: float,
) -> tuple[int, str, int]:
    start_frame = _time_to_frame(seg_start, times)
    end_frame = _time_to_frame(seg_end, times)

    if start_frame >= end_frame:
        return 3, "stable", 3

    seg_data = arr[start_frame:end_frame]
    seg_mean = float(seg_data.mean())
    seg_max = float(seg_data.max())

    arr_max = float(arr.max()) if float(arr.max()) > 0 else 1.0
    energy = max(1, min(10, round(seg_mean / arr_max * 10)))
    energy_peak = max(1, min(10, round(seg_max / arr_max * 10)))

    mid = len(seg_data) // 2
    if mid == 0:
        return energy, "stable", energy_peak

    first_half_mean = float(seg_data[:mid].mean())
    second_half_mean = float(seg_data[mid:].mean())
    diff = second_half_mean - first_half_mean

    threshold = global_mean * 0.15

    if abs(diff) < threshold:
        if seg_mean > global_mean * 1.3:
            trend = "peak"
        else:
            trend = "stable"
    elif diff > 0:
        quarter = len(seg_data) // 4
        if quarter > 0:
            q1_mean = float(seg_data[:quarter].mean())
            q3_mean = float(seg_data[quarter * 3:].mean())
            if q1_mean < first_half_mean and q3_mean < second_half_mean:
                trend = "rise_then_fall"
            else:
                trend = "rising"
        else:
            trend = "rising"
    else:
        quarter = len(seg_data) // 4
        if quarter > 0:
            q1_mean = float(seg_data[:quarter].mean())
            q3_mean = float(seg_data[quarter * 3:].mean())
            if q1_mean > first_half_mean and q3_mean > second_half_mean:
                trend = "fall_then_rise"
            else:
                trend = "falling"
        else:
            trend = "falling"

    return energy, trend, energy_peak


# ──────────────────────────────────────────────────────────────────────────────
# 段落合并/拆分
# ──────────────────────────────────────────────────────────────────────────────

def _merge_short_segments(
    segments: list[AudioSegment],
    min_duration: float,
) -> list[AudioSegment]:
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if prev.duration < min_duration or seg.duration < min_duration:
            new_energy = max(prev.energy, seg.energy)
            new_peak = max(prev.energy_peak, seg.energy_peak)
            new_trend = prev.energy_trend if prev.duration >= seg.duration else seg.energy_trend
            merged[-1] = AudioSegment(
                name=prev.name, start=prev.start, end=seg.end,
                energy=new_energy, energy_trend=new_trend, energy_peak=new_peak,
                mood=prev.mood, description=prev.description, visual_suggestion=prev.visual_suggestion,
            )
        else:
            merged.append(seg)

    return merged


def _split_long_segments(
    segments: list[AudioSegment],
    arr: np.ndarray,
    times: np.ndarray,
    downbeats: list[float],
    global_mean: float,
    max_duration: float,
    min_duration: float,
) -> list[AudioSegment]:
    result: list[AudioSegment] = []

    for seg in segments:
        if seg.duration <= max_duration:
            result.append(seg)
            continue

        n_splits = int(seg.duration / max_duration)
        split_points: list[float] = []

        for _ in range(n_splits):
            best_score = 0.0
            best_time = None
            search_start = seg.start + min_duration
            search_end = seg.end - min_duration

            for db_time in downbeats:
                if db_time <= search_start or db_time >= search_end:
                    continue
                if any(abs(db_time - sp) < min_duration for sp in split_points):
                    continue

                frame = _time_to_frame(db_time, times)
                window = max(1, int(1.0 / (times[1] - times[0]) if len(times) > 1 and times[1] > times[0] else 10))
                f_start = max(0, frame - window)
                f_end = min(len(arr), frame + window)
                if f_start < frame < f_end:
                    before = float(arr[f_start:frame].mean())
                    after = float(arr[frame:f_end].mean())
                    score = abs(after - before)
                    if score > best_score:
                        best_score = score
                        best_time = db_time

            if best_time is not None:
                best_time = _snap_backward(best_time, downbeats, min_step=0.2)
                split_points.append(best_time)

        split_points.sort()

        boundaries = [seg.start] + split_points + [seg.end]
        for i in range(len(boundaries) - 1):
            s_start = round(boundaries[i], 3)
            s_end = round(boundaries[i + 1], 3)
            s_energy, s_trend, s_peak = _compute_segment_energy(
                arr, times, s_start, s_end, global_mean,
            )
            result.append(AudioSegment(
                name=seg.name, start=s_start, end=s_end,
                energy=s_energy, energy_trend=s_trend, energy_peak=s_peak,
                mood=seg.mood, description=seg.description, visual_suggestion=seg.visual_suggestion,
            ))

    return result
