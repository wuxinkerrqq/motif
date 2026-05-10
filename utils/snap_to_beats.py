from __future__ import annotations

import bisect


def snap_to_beat(
    cut_time: float,
    beat_array: list[float],
    tolerance: float = 0.15,
) -> tuple[float, float]:
    """
    将单个切点吸附到最近的 beat 时间戳。

    返回：(吸附后的时间, 偏移量)
    偏移量为正表示向后偏移，负表示向前偏移。
    超出容忍范围则不吸附，返回原时间和偏移 0.0。
    """
    if not beat_array:
        return cut_time, 0.0

    # 用二分查找找最近的 beat
    idx = bisect.bisect_left(beat_array, cut_time)

    candidates = []
    if idx > 0:
        candidates.append(beat_array[idx - 1])
    if idx < len(beat_array):
        candidates.append(beat_array[idx])

    nearest = min(candidates, key=lambda b: abs(b - cut_time))
    offset = nearest - cut_time

    if abs(offset) <= tolerance:
        return round(nearest, 3), round(offset, 3)

    return cut_time, 0.0


def snap_timeline(
    timeline: list[dict],
    beat_array: list[float],
    tolerance: float = 0.15,
) -> list[dict]:
    """
    对整个时间轴做 snap_to_beats 处理。
    修改每个条目的 audio_start，并同步前一片段的 audio_end，
    保证相邻片段连续（不出现空洞或重叠）。
    记录 beat_snap_offset 字段。
    """
    sorted_timeline = sorted(timeline, key=lambda x: x["audio_start"])

    snapped_starts: list[float] = []
    for item in sorted_timeline:
        snapped, offset = snap_to_beat(
            item["audio_start"], beat_array, tolerance
        )
        item["audio_start"] = snapped
        item["beat_snap_offset"] = offset
        snapped_starts.append(snapped)

    # 同步前一片段的 audio_end，让相邻片段保持连续
    for i in range(len(sorted_timeline) - 1):
        next_start = snapped_starts[i + 1]
        cur = sorted_timeline[i]
        # 仅当原 audio_end 与下一段的 audio_start 距离 < tolerance 时才同步
        # 防止把本就有空隙的片段强行拼接
        if abs(cur["audio_end"] - next_start) <= tolerance:
            cur["audio_end"] = round(next_start, 3)

    return sorted_timeline