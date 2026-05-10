from __future__ import annotations

import json
from pathlib import Path

from models.audio import (
    AudioMap,
    AudioSegment,
    EnergyKeypoint,
    KeyMoment,
    SpecialEvent,
)


def load_audio_map(json_path: str, music_path: str | None = None) -> AudioMap:
    """
    从缓存 JSON 加载 AudioMap。
    自动识别 v2 格式（含 narrative_summary）和 v1 格式（含 r1_gemini_output）。
    beat_array / downbeats 优先从 JSON 读取，缺失时才重新提取。
    """
    data = json.load(open(json_path, encoding="utf-8"))

    is_v2 = "narrative_summary" in data or "key_moments_v2" in data

    if is_v2:
        return _load_v2(data, json_path, music_path)
    return _load_v1(data, music_path)


# ── v2 加载 ──────────────────────────────────────────────────────────────────

def _load_v2(data: dict, json_path: str, music_path: str | None) -> AudioMap:
    segments: list[AudioSegment] = []
    for s in data.get("segments", []):
        rms_db = float(s.get("rms_db", -30.0))
        energy_int = _rms_db_to_energy(rms_db)

        segments.append(AudioSegment(
            name=s.get("label") or s.get("name", "unknown"),
            start=float(s["start"]),
            end=float(s["end"]),
            energy=energy_int,
            energy_trend="stable",
            energy_peak=energy_int,
            mood=str(s.get("mood") or "neutral"),
            description=str(s.get("description") or ""),
            visual_suggestion=str(s.get("visual_suggestion") or ""),
            energy_level=s.get("energy_level"),
            density_level=s.get("density_level"),
            pacing_hint=s.get("pacing_hint"),
            visual_profile=s.get("visual_profile"),
        ))

    key_moments_v2: list[KeyMoment] = []
    for k in data.get("key_moments_v2", []):
        try:
            key_moments_v2.append(KeyMoment(
                time=float(k["time"]),
                importance=float(k.get("importance", 0.5)),
                tier=k.get("tier", "rhythmic_hit"),
                anchor_type=str(k.get("anchor_type", "unknown")),
                description=str(k.get("description") or ""),
                visual_profile=k.get("visual_profile") or {},
                transition_recommendation=str(
                    k.get("transition_recommendation") or "hard_cut"
                ),
                evidence=list(k.get("evidence") or []),
                segment=k.get("segment"),
                segment_energy_level=k.get("segment_energy_level"),
            ))
        except Exception as e:
            print(f"  [cache] 解析 KeyMoment 失败: {e}")

    return AudioMap(
        bpm=float(data["bpm"]),
        total_duration=float(data["total_duration"]),
        beat_array=[round(float(t), 3) for t in data.get("beat_array", [])],
        downbeats=[round(float(t), 3) for t in data.get("downbeats", [])],
        segments=segments,
        energy_keypoints=[],
        r1_understanding="",
        key_moments=[],
        key_moments_v2=key_moments_v2,
        narrative_summary=str(data.get("narrative_summary", "")),
        mood_arc=list(data.get("mood_arc", [])),
        tempo_density_curve=list(data.get("tempo_density_curve", [])),
    )


# ── v1 加载（向后兼容历史 JSON）──────────────────────────────────────────────

def _load_v1(data: dict, music_path: str | None) -> AudioMap:
    segments = []
    for s in data.get("segments", []):
        segments.append(AudioSegment(
            name=s["name"],
            start=float(s["start"]),
            end=float(s["end"]),
            energy=int(s.get("energy", 5)),
            energy_trend=s.get("energy_trend", "stable"),
            energy_peak=int(s.get("energy_peak", 5)),
            mood=s.get("mood", "neutral"),
            description=s.get("description", ""),
            visual_suggestion=s.get("visual_suggestion", ""),
        ))

    energy_keypoints = []
    for kp in data.get("energy_keypoints", []):
        energy_keypoints.append(EnergyKeypoint(
            time=float(kp["time"]),
            value=float(kp["value"]),
        ))

    beat_array = data.get("beat_array", [])
    downbeats = data.get("downbeats", [])

    if not beat_array or not downbeats:
        if music_path:
            print("  [cache] beat_array/downbeats 缺失，从音频重新提取...")
            from audio.extractor import (
                extract_downbeats,
                extract_energy_keypoints,
                extract_librosa_features,
            )
            features = extract_librosa_features(music_path)
            beat_array = features["beat_times"]
            if not energy_keypoints:
                energy_keypoints = extract_energy_keypoints(
                    features["energy_curve_times"],
                    features["energy_curve_values"],
                )
            downbeats = extract_downbeats(music_path)
            if not downbeats:
                downbeats = beat_array[::4]
            print(f"  [cache] 提取完成: {len(beat_array)} beats, {len(downbeats)} downbeats")
        else:
            print("  [cache] 警告：beat_array 缺失且未提供 music_path")

    return AudioMap(
        bpm=float(data["bpm"]),
        total_duration=float(data["total_duration"]),
        beat_array=[round(float(t), 3) for t in beat_array],
        downbeats=[round(float(t), 3) for t in downbeats],
        segments=segments,
        energy_keypoints=energy_keypoints,
        r1_understanding=data.get("r1_gemini_output", ""),
        key_moments=data.get("r1_key_moments", []),
    )


# ── 辅助 ─────────────────────────────────────────────────────────────────────

def _rms_db_to_energy(rms_db: float) -> int:
    if rms_db <= -40:
        return 1
    if rms_db <= -25:
        return max(1, min(5, int(2 + (rms_db + 40) / 5)))
    if rms_db <= -10:
        return max(5, min(8, int(5 + (rms_db + 25) / 5)))
    return max(8, min(10, int(8 + (rms_db + 10) / 5)))
