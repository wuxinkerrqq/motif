"""
io_loader.py — 把 eval_model/audio 和 eval_model/video 的输出加载成 motif 主流程能用的对象。

eval_model 的 audio_map.json 字段名跟 motif 主流程的 AudioMap pydantic 模型有些差异：
  - segment 用 "label"，主流程用 "name"
  - 没有 "beat_array" / "downbeats" / "energy_keypoints"
  - 锚点放在顶层 "key_moments"，含 tier / visual_profile / transition_recommendation
    （和主流程的 key_moments_v2 同 schema）

这个 loader 把它们对齐，让现有 planner_tools 的 get_music_overview / get_special_events 直接能用。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.audio import AudioMap, AudioSegment, EnergyKeypoint, KeyMoment
from models.video import SceneItem


def load_audio_map(audio_map_path: Path) -> AudioMap:
    raw = json.loads(audio_map_path.read_text(encoding="utf-8"))

    segments: list[AudioSegment] = []
    for s in raw["segments"]:
        segments.append(AudioSegment(
            name=s.get("label") or s.get("name", "unknown"),
            start=s["start"],
            end=s["end"],
            energy=_energy_level_to_int(s.get("energy_level", "medium")),
            energy_trend=s.get("energy_trend", "stable"),
            energy_peak=_energy_level_to_int(s.get("energy_level", "medium")),
            mood=s.get("mood", "neutral"),
            description=s.get("description", ""),
            visual_suggestion="",
            energy_level=s.get("energy_level"),
            density_level=s.get("density_level"),
            pacing_hint=s.get("pacing_hint"),
            visual_profile=s.get("visual_profile"),
        ))

    key_moments_v2: list[KeyMoment] = []
    # 兼容两种格式：eval_model 用 key_moments，server 写的用 key_moments_v2
    raw_kms = raw.get("key_moments_v2") or raw.get("key_moments", [])
    for k in raw_kms:
        key_moments_v2.append(KeyMoment(
            time=k["time"],
            importance=k.get("importance", 0.5),
            tier=k.get("tier", "rhythmic_hit"),
            anchor_type=k.get("anchor_type", "unknown"),
            description=k.get("description", ""),
            visual_profile=k.get("visual_profile", {}),
            transition_recommendation=k.get("transition_recommendation", "hard_cut"),
            evidence=k.get("evidence", []),
            segment=k.get("segment"),
            segment_energy_level=k.get("segment_energy_level"),
        ))

    return AudioMap(
        bpm=raw["bpm"],
        total_duration=raw["total_duration"],
        beat_array=raw.get("beat_array", []),
        downbeats=raw.get("downbeats", []),
        segments=segments,
        energy_keypoints=[EnergyKeypoint(**ep) for ep in raw.get("energy_keypoints", [])],
        narrative_summary=raw.get("narrative_summary", ""),
        mood_arc=raw.get("mood_arc", []),
        key_moments_v2=key_moments_v2,
    )


def _energy_level_to_int(level: str) -> int:
    return {"low": 3, "medium": 5, "high": 8}.get(level, 5)


def load_scene_table(scene_table_path: Path) -> list[SceneItem]:
    raw = json.loads(scene_table_path.read_text(encoding="utf-8"))
    items: list[SceneItem] = []
    for d in raw:
        items.append(SceneItem(
            scene_id=d["scene_id"],
            source_file=d["source_file"],
            source_dir=d.get("source_dir", ""),
            start=d["start"],
            end=d["end"],
            duration=d["duration"],
            scene_index=d["scene_index"],
            keyframes=d.get("keyframes", []),
            scene_description=d.get("scene_description", ""),
            characters=d.get("characters", []),
            mood=d.get("mood", "neutral"),
            visual_profile=d.get("visual_profile", {}),
            is_outro_material=bool(d.get("is_outro_material", False)),
            is_climax_material=bool(d.get("is_climax_material", False)),
        ))
    return items


def load_clip_index(
    clip_emb_path: Path,
    clip_ids_path: Path,
) -> tuple[np.ndarray, list[int]]:
    embeddings = np.load(str(clip_emb_path))
    ids = json.loads(clip_ids_path.read_text(encoding="utf-8"))
    return embeddings, ids
