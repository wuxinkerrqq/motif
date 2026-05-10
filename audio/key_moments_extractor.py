"""
Audio Analyzer v2 - L2 事件层

输入: analyze_v2 输出的 analysis dict
输出: dict（兼容 eval_model/audio/results/<name>/key_moments.json 格式）
  - bpm / total_duration
  - segments（带 energy_level / density_level / pacing_hint 等）
  - key_moments（候选打分 + 自适应阈值过滤）
  - tempo_density_curve（节奏密度曲线）
  - stats

核心思路:
  1. 极短段落合并（裁剪噪声）
  2. 候选生成（时间窗合并 dense_events）
  3. 候选打分（multi_stem + downbeat + strength + boundary + energy_jump）
  4. 段落能量分级（本歌 RMS_dB 分位数）
  5. 全曲难度系数 → 自适应阈值过滤
  6. 节奏密度曲线（events_per_beat 滑窗 + 本歌分位数分级）
  7. 段落汇总（pacing_hint）
"""
from __future__ import annotations

import numpy as np
from loguru import logger

# ── 配置 ──────────────────────────────────────────────────────────────────────

# 锚点候选时间窗口（秒）：判定多事件"同时"发生的容忍范围
COINCIDENCE_WINDOW = 0.15

# 节奏密度曲线滑窗（秒）
DENSITY_WINDOW = 4.0
DENSITY_STEP = 1.0

# 段落 importance 阈值的【基准 + 浮动】
KEEP_RATIO_BASE = {"low": 0.42, "medium": 0.32, "high": 0.28}
KEEP_RATIO_RANGE = {"low": 0.18, "medium": 0.20, "high": 0.22}

# difficulty 映射：density 0.5 → 0, density 3.0 → 1
DENSITY_MIN = 0.5
DENSITY_MAX = 3.0

# 极短段落合并阈值（秒）：裁剪不精产生的噪声段
MIN_SEGMENT_DURATION = 1.0

# Importance 评分权重
WEIGHT_MULTI_STEM = 0.25
WEIGHT_DOWNBEAT = 0.25
WEIGHT_STRENGTH = 0.30
WEIGHT_ENERGY_JUMP = 0.10
WEIGHT_BOUNDARY = 0.10
STRONG_ONSET_BONUS = 0.10


# ── 候选生成 ──────────────────────────────────────────────────────────────────

def build_candidates(
    dense_events: list[dict],
    coincidence_window: float = COINCIDENCE_WINDOW,
) -> list[dict]:
    if not dense_events:
        return []

    sorted_events = sorted(dense_events, key=lambda e: e["time"])

    candidates = []
    i = 0
    n = len(sorted_events)
    while i < n:
        anchor_time = sorted_events[i]["time"]
        window_events = [sorted_events[i]]
        j = i + 1
        while j < n and sorted_events[j]["time"] - anchor_time <= coincidence_window:
            window_events.append(sorted_events[j])
            j += 1

        stems_with_onset = set()
        max_strength = 0.0
        has_downbeat = False
        has_boundary = False
        boundary_label = None

        for ev in window_events:
            t = ev["type"]
            if t.endswith("_onset"):
                stem = t.replace("_onset", "")
                stems_with_onset.add(stem)
                max_strength = max(max_strength, ev.get("strength", 0.0))
            elif t == "downbeat":
                has_downbeat = True
            elif t == "segment_boundary":
                has_boundary = True
                boundary_label = ev.get("label")

        if not stems_with_onset and not has_downbeat and not has_boundary:
            i = j
            continue

        anchor_t = (window_events[0]["time"] + window_events[-1]["time"]) / 2

        candidates.append({
            "time": round(anchor_t, 3),
            "window_start": round(window_events[0]["time"], 3),
            "window_end": round(window_events[-1]["time"], 3),
            "stems": sorted(stems_with_onset),
            "stem_count": len(stems_with_onset),
            "max_strength": round(max_strength, 3),
            "has_downbeat": has_downbeat,
            "has_boundary": has_boundary,
            "boundary_label": boundary_label,
            "evidence": [
                ev["type"] + (f"@{ev.get('strength', 0):.2f}" if "strength" in ev else "")
                for ev in window_events
            ],
        })
        i = j

    return candidates


# ── 段落能量跃迁 ─────────────────────────────────────────────────────────────

def compute_energy_jumps(segments: list[dict]) -> dict[float, float]:
    jumps = {}
    for i in range(1, len(segments)):
        prev_db = segments[i - 1].get("rms_db", -60)
        curr_db = segments[i].get("rms_db", -60)
        diff = abs(curr_db - prev_db)
        normalized = min(diff / 10.0, 1.0)
        jumps[segments[i]["start"]] = normalized
    return jumps


# ── 候选打分 ──────────────────────────────────────────────────────────────────

def score_candidate(cand: dict, energy_jumps: dict[float, float]) -> float:
    score = 0.0

    if cand["stem_count"] > 0:
        stem_factor = 0.5 + 0.5 * (cand["stem_count"] - 1) / 3.0
        score += WEIGHT_MULTI_STEM * stem_factor

    if cand["has_downbeat"]:
        score += WEIGHT_DOWNBEAT

    score += WEIGHT_STRENGTH * cand["max_strength"]

    if cand["max_strength"] >= 0.7:
        score += STRONG_ONSET_BONUS

    if cand["has_boundary"]:
        score += WEIGHT_BOUNDARY
        jump = energy_jumps.get(cand["window_start"], 0)
        for t, j in energy_jumps.items():
            if abs(t - cand["time"]) < 1.0:
                jump = max(jump, j)
        score += WEIGHT_ENERGY_JUMP * jump

    return min(score, 1.0)


# ── anchor_type 推断 ──────────────────────────────────────────────────────────

def infer_anchor_type(cand: dict, energy_jumps: dict[float, float]) -> str:
    if cand["has_boundary"]:
        jump = max(
            (j for t, j in energy_jumps.items() if abs(t - cand["time"]) < 1.0),
            default=0,
        )
        if jump >= 0.5:
            return "section_drop" if cand["stem_count"] >= 2 else "section_change_with_jump"
        return f"section_change_to_{cand['boundary_label']}" if cand["boundary_label"] else "section_change"

    stems = set(cand["stems"])

    if cand["stem_count"] >= 3 and cand["has_downbeat"]:
        return "full_band_hit_on_downbeat"
    if cand["stem_count"] >= 3:
        return "full_band_hit"

    if {"drums", "bass"}.issubset(stems) and cand["has_downbeat"]:
        return "rhythmic_section_hit"

    if "vocals" in stems and cand["has_downbeat"]:
        return "vocal_phrase_on_downbeat"
    if "vocals" in stems:
        return "vocal_onset"

    if "other" in stems and cand["has_downbeat"]:
        return "melodic_hit_on_downbeat"

    if "drums" in stems and cand["has_downbeat"]:
        return "drum_hit_on_downbeat"
    if "drums" in stems:
        return "drum_hit"

    if cand["stem_count"] == 1:
        return f"{cand['stems'][0]}_solo_hit"

    if cand["has_downbeat"]:
        return "downbeat"

    return "weak_event"


# ── 难度系数 ──────────────────────────────────────────────────────────────────

def compute_difficulty(
    dense_events: list[dict], total_duration: float,
) -> tuple[float, float]:
    onset_count = sum(1 for e in dense_events if e["type"].endswith("_onset"))
    if total_duration <= 0:
        return 0.0, 0.0
    density = onset_count / total_duration
    diff = (density - DENSITY_MIN) / (DENSITY_MAX - DENSITY_MIN)
    diff = max(0.0, min(1.0, diff))
    return diff, density


def get_threshold(level: str, difficulty: float) -> float:
    base = KEEP_RATIO_BASE.get(level, 0.35)
    rng = KEEP_RATIO_RANGE.get(level, 0.20)
    return base + rng * difficulty


# ── 极短段落合并 ─────────────────────────────────────────────────────────────

def merge_short_segments(
    segments: list[dict],
    dense_events: list[dict],
    min_duration: float = MIN_SEGMENT_DURATION,
) -> tuple[list[dict], list[dict]]:
    if not segments:
        return segments, dense_events

    segs = [dict(s) for s in segments]
    kept: list[dict] = []
    removed_labels_with_start: list[tuple[str, float]] = []

    for i, seg in enumerate(segs):
        dur = seg["end"] - seg["start"]
        if dur >= min_duration:
            kept.append(seg)
            continue

        removed_labels_with_start.append((seg["label"], seg["start"]))

        if not kept:
            if i + 1 < len(segs):
                segs[i + 1]["start"] = seg["start"]
        else:
            kept[-1]["end"] = seg["end"]

    for seg in kept:
        seg["duration"] = round(seg["end"] - seg["start"], 3)

    removed_keys = set()
    for label, start in removed_labels_with_start:
        removed_keys.add((label, round(start, 3)))

    cleaned_events = []
    for ev in dense_events:
        if ev["type"] == "segment_boundary":
            key = (ev.get("label"), round(ev["time"], 3))
            if key in removed_keys:
                continue
        cleaned_events.append(ev)

    return kept, cleaned_events


# ── 段落能量分级 ─────────────────────────────────────────────────────────────

def classify_segment_energy(segments: list[dict]) -> dict[str, str]:
    valid = [s for s in segments if s["end"] - s["start"] >= 1.0]
    if not valid:
        return {}

    db_values = [s["rms_db"] for s in valid]
    p33 = float(np.percentile(db_values, 33))
    p66 = float(np.percentile(db_values, 66))

    levels = {}
    for s in segments:
        key = f"{s['label']}@{s['start']}"
        db = s["rms_db"]
        if db < p33:
            levels[key] = "low"
        elif db > p66:
            levels[key] = "high"
        else:
            levels[key] = "medium"
    return levels


def get_segment_for_time(t: float, segments: list[dict]) -> dict | None:
    for s in segments:
        if s["start"] <= t < s["end"]:
            return s
    return segments[-1] if segments else None


# ── 节奏密度曲线 ─────────────────────────────────────────────────────────────

def compute_tempo_density(
    dense_events: list[dict],
    bpm: float,
    total_duration: float,
    window: float = DENSITY_WINDOW,
    step: float = DENSITY_STEP,
) -> list[dict]:
    onset_times = sorted(
        e["time"] for e in dense_events if e["type"].endswith("_onset")
    )
    if not onset_times:
        return []

    onset_arr = np.array(onset_times)
    beats_per_sec = bpm / 60.0

    raw_curve = []
    t = 0.0
    while t + window <= total_duration:
        count = int(np.sum((onset_arr >= t) & (onset_arr < t + window)))
        events_per_sec = count / window
        events_per_beat = events_per_sec / beats_per_sec
        raw_curve.append({
            "time": round(t + window / 2, 2),
            "events_per_sec": round(events_per_sec, 2),
            "events_per_beat": round(events_per_beat, 2),
        })
        t += step

    if not raw_curve:
        return []

    epb_values = [c["events_per_beat"] for c in raw_curve]
    p33 = float(np.percentile(epb_values, 33))
    p66 = float(np.percentile(epb_values, 66))

    for c in raw_curve:
        epb = c["events_per_beat"]
        if epb < p33:
            c["level"] = "sparse"
        elif epb > p66:
            c["level"] = "dense"
        else:
            c["level"] = "medium"

    return raw_curve


# ── pacing hint ──────────────────────────────────────────────────────────────

def pacing_hint(energy_level: str, density_level: str) -> dict:
    table = {
        ("low", "sparse"):    ([8, 16], "低能稀疏：长镜头/慢节奏"),
        ("low", "medium"):    ([4, 8],  "低能中密度：中长镜头"),
        ("low", "dense"):     ([4, 6],  "低能密事件：中等镜头"),
        ("medium", "sparse"): ([4, 8],  "中能稀疏：中长镜头"),
        ("medium", "medium"): ([2, 4],  "中能中密度：标准节奏"),
        ("medium", "dense"):  ([2, 3],  "中能密事件：偏快节奏"),
        ("high", "sparse"):   ([2, 4],  "高能稀疏：标准镜头"),
        ("high", "medium"):   ([1, 2],  "高能中密度：快切"),
        ("high", "dense"):    ([1, 2],  "高能密事件：极速快切"),
    }
    rng, why = table.get((energy_level, density_level), ([2, 4], "默认中等节奏"))
    return {"beats_per_shot_range": rng, "rationale": why}


# ── 主入口 ────────────────────────────────────────────────────────────────────

def extract_key_moments(analysis: dict) -> dict:
    """
    L2 事件层主入口（纯函数，无 IO）

    Args:
        analysis: analyzer_v2.run_analysis_v2() 的输出 dict

    Returns:
        dict（同 eval_model key_moments.json 格式）
    """
    dense_events = analysis["dense_events"]
    segments = analysis["segments"]
    bpm = analysis["bpm"]
    total_duration = analysis["total_duration"]

    # 0. 合并极短噪声段落
    before_count = len(segments)
    segments, dense_events = merge_short_segments(segments, dense_events)
    merged_count = before_count - len(segments)
    if merged_count > 0:
        logger.info(f"  [L2] 合并 {merged_count} 个极短段落（<{MIN_SEGMENT_DURATION}s）")

    # 1. 候选生成
    candidates = build_candidates(dense_events)

    # 2. 段落能量跃迁 + 分级
    energy_jumps = compute_energy_jumps(segments)
    seg_energy_levels = classify_segment_energy(segments)

    # 3. 候选打分 + 推断 anchor_type
    for cand in candidates:
        cand["importance"] = round(score_candidate(cand, energy_jumps), 3)
        cand["anchor_type"] = infer_anchor_type(cand, energy_jumps)
        seg = get_segment_for_time(cand["time"], segments)
        if seg:
            cand["segment"] = seg["label"]
            seg_key = f"{seg['label']}@{seg['start']}"
            cand["segment_energy_level"] = seg_energy_levels.get(seg_key, "medium")
        else:
            cand["segment"] = None
            cand["segment_energy_level"] = "medium"

    # 4. 自适应阈值过滤
    difficulty, overall_density = compute_difficulty(dense_events, total_duration)
    thresholds = {
        level: round(get_threshold(level, difficulty), 3)
        for level in ("low", "medium", "high")
    }
    logger.info(
        f"  [L2] onset 密度={overall_density:.2f} events/s  difficulty={difficulty:.2f}  "
        f"阈值: low={thresholds['low']} medium={thresholds['medium']} high={thresholds['high']}"
    )

    key_moments = []
    for cand in candidates:
        threshold = thresholds.get(cand["segment_energy_level"], 0.5)
        if cand["importance"] >= threshold:
            key_moments.append(cand)

    # 5. 节奏密度曲线
    density_curve = compute_tempo_density(dense_events, bpm, total_duration)

    # 6. 段落汇总（pacing_hint）
    segment_summary = []
    for s in segments:
        seg_key = f"{s['label']}@{s['start']}"
        level = seg_energy_levels.get(seg_key, "medium")

        seg_density = [
            c for c in density_curve if s["start"] <= c["time"] < s["end"]
        ]
        if seg_density:
            avg_epb = np.mean([c["events_per_beat"] for c in seg_density])
            level_counter = {"sparse": 0, "medium": 0, "dense": 0}
            for c in seg_density:
                level_counter[c["level"]] += 1
            density_level = max(level_counter, key=level_counter.get)
        else:
            avg_epb = 0
            density_level = "sparse"

        pacing = pacing_hint(level, density_level)

        segment_summary.append({
            "label": s["label"],
            "start": s["start"],
            "end": s["end"],
            "duration": round(s["end"] - s["start"], 2),
            "beats_count": s["beats_count"],
            "rms_db": s["rms_db"],
            "energy_level": level,
            "density_level": density_level,
            "avg_events_per_beat": round(float(avg_epb), 2),
            "pacing_hint": pacing,
            "key_moments_count": sum(
                1 for k in key_moments if s["start"] <= k["time"] < s["end"]
            ),
        })

    logger.info(
        f"  [L2] 候选 {len(candidates)} → 保留 {len(key_moments)} 锚点 "
        f"({len(key_moments) / max(len(candidates), 1) * 100:.1f}%)"
    )

    return {
        "music_file": analysis["music_file"],
        "bpm": bpm,
        "total_duration": total_duration,
        "segments": segment_summary,
        "key_moments": key_moments,
        "tempo_density_curve": density_curve,
        "stats": {
            "total_candidates": len(candidates),
            "key_moments_kept": len(key_moments),
            "keep_ratio": round(len(key_moments) / max(len(candidates), 1), 3),
        },
    }
