"""
Audio Analyzer Agent (v2)

调度三层音乐分析 pipeline，输出 AudioMap：
  L1 (analyzer_v2):           All-In-One + Demucs + per-stem onset + RMS
  L2 (key_moments_extractor): 候选打分 + 自适应阈值 + pacing_hint + density curve
  L3 (audio_l3_enricher):     LLM VAD 标注 + tier/transition 硬规则 + 叙事弧线

对外接口签名保持兼容：
  async def run_audio_analyzer(music_path, ...) -> (AudioMap, config_patch)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loguru import logger

from audio.analyzer_v2 import run_analysis_v2
from audio.audio_l3_enricher import enrich_audio_map
from audio.key_moments_extractor import extract_key_moments
from config import AdaptiveConfig, load_config
from models.audio import AudioMap, AudioSegment, EnergyKeypoint, KeyMoment


async def run_audio_analyzer(
    music_path: str,
    background_info: str | None = None,
    editing_style: str = "visual_driven",
    save_dir: str | None = None,
) -> tuple[AudioMap, dict]:
    """
    Audio Analyzer 主入口（v2 三层 pipeline）。

    流程：
      L1: All-In-One + Demucs + onset + RMS  → analysis dict
      L2: 候选打分 + 自适应阈值 + pacing_hint → key_moments dict
      L3: LLM VAD 标注 + 叙事弧线          → audio_map dict

    Returns:
        (AudioMap, config_patch)
        AudioMap 含原有字段 + 新增字段（key_moments_v2 / narrative_summary 等）
        旧字段（key_moments / r1_understanding）置空，向后兼容
    """
    music_path_obj = Path(music_path)
    logger.info(f"[Audio Analyzer v2] 开始分析：{music_path}")

    # 计算保存目录和 stem 输出目录
    if save_dir is None:
        save_dir = str(music_path_obj.parent)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    music_stem = music_path_obj.stem
    stems_dir = save_path / f"{music_stem}_stems"

    # ── L1: 信号层 ───────────────────────────────────────────────────────────
    logger.info("[Audio Analyzer] L1：All-In-One + Demucs + onset + RMS")
    analysis = await asyncio.to_thread(run_analysis_v2, music_path, stems_dir)
    logger.info(
        f"  [L1] BPM={analysis['bpm']}  时长={analysis['total_duration']}s  "
        f"段落数={len(analysis['segments'])}  beats={len(analysis['beats'])}  "
        f"downbeats={len(analysis['downbeats'])}  dense_events={len(analysis['dense_events'])}"
    )

    # ── L2: 事件层 ───────────────────────────────────────────────────────────
    logger.info("[Audio Analyzer] L2：候选打分 + 自适应阈值 + pacing_hint")
    l2_data = extract_key_moments(analysis)

    # ── L3: 语义层 ───────────────────────────────────────────────────────────
    logger.info("[Audio Analyzer] L3：LLM VAD 标注 + 叙事弧线")
    audio_map_data = await enrich_audio_map(l2_data)
    logger.info(f"  [L3] 叙事概要：{audio_map_data.get('narrative_summary', '')}")
    logger.info(f"  [L3] 情绪轨迹：{' → '.join(audio_map_data.get('mood_arc', []))}")

    # ── 组装 AudioMap ────────────────────────────────────────────────────────
    audio_map = _build_audio_map(audio_map_data, analysis)

    # ── config_patch ──────────────────────────────────────────────────────────
    adaptive = AdaptiveConfig()
    # 从 L1 的 analysis 取 RMS 能量曲线数据用于 thresholds（暂用 segment rms 列表近似）
    seg_rms_values = [s.get("rms", 0) for s in analysis["segments"]]
    config_patch = adaptive.patch_runtime_config(
        load_config(editing_style),
        bpm=analysis["bpm"],
        energy_values=seg_rms_values if seg_rms_values else None,
    )

    # ── 持久化 ────────────────────────────────────────────────────────────────
    _save_audio_map_json(
        music_path=music_path,
        audio_map_data=audio_map_data,
        analysis=analysis,
        l2_data=l2_data,
        config_patch=config_patch,
        save_dir=save_dir,
    )

    logger.info("[Audio Analyzer v2] 分析完成")
    return audio_map, config_patch


# ──────────────────────────────────────────────────────────────────────────────
# AudioMap 组装
# ──────────────────────────────────────────────────────────────────────────────

def _build_audio_map(audio_map_data: dict, analysis: dict) -> AudioMap:
    """
    把 L3 输出的 audio_map_data 组装成 AudioMap pydantic 实例。
    保留向后兼容字段（segments 仍然有 energy/energy_trend/mood 等）。
    """
    segments: list[AudioSegment] = []
    for s in audio_map_data["segments"]:
        # 从 RMS_dB 推断 energy 1-10 整数值（兼容老字段）
        rms_db = s.get("rms_db", -30.0)
        energy_int = _rms_db_to_energy(rms_db)

        segments.append(AudioSegment(
            name=s["label"],
            start=float(s["start"]),
            end=float(s["end"]),
            energy=energy_int,
            energy_trend="stable",
            energy_peak=energy_int,
            mood=str(s.get("mood") or "neutral"),
            description=str(s.get("description") or ""),
            visual_suggestion="",
            energy_level=s.get("energy_level"),
            density_level=s.get("density_level"),
            pacing_hint=s.get("pacing_hint"),
            visual_profile=s.get("visual_profile"),
        ))

    key_moments_v2: list[KeyMoment] = []
    for k in audio_map_data.get("key_moments", []):
        try:
            key_moments_v2.append(KeyMoment(
                time=float(k["time"]),
                importance=float(k.get("importance", 0.5)),
                tier=k.get("tier", "rhythmic_hit"),
                anchor_type=str(k.get("anchor_type", "unknown")),
                description=str(k.get("description") or ""),
                visual_profile=k.get("visual_profile") or {},
                transition_recommendation=str(k.get("transition_recommendation") or "hard_cut"),
                evidence=list(k.get("evidence") or []),
                segment=k.get("segment"),
                segment_energy_level=k.get("segment_energy_level"),
            ))
        except Exception as e:
            logger.warning(f"  解析 KeyMoment 失败: {e}, 跳过 {k}")

    return AudioMap(
        bpm=float(audio_map_data["bpm"]),
        total_duration=float(audio_map_data["total_duration"]),
        beat_array=[round(float(t), 3) for t in analysis["beats"]],
        downbeats=[round(float(t), 3) for t in analysis["downbeats"]],
        segments=segments,
        energy_keypoints=[],  # v2 不再生成（保留字段以兼容）
        r1_understanding="",  # v2 不再产生 R1 文本
        key_moments=[],       # 旧字段置空
        key_moments_v2=key_moments_v2,
        narrative_summary=audio_map_data.get("narrative_summary", ""),
        mood_arc=list(audio_map_data.get("mood_arc", [])),
        tempo_density_curve=list(audio_map_data.get("tempo_density_curve", [])),
    )


def _rms_db_to_energy(rms_db: float) -> int:
    """
    把 RMS dB 映射到 1-10 整数 energy（兼容老字段）。
    -30 dB 以下 → 1-3，-30 ~ -15 → 4-6，-15 以上 → 7-10。
    """
    if rms_db <= -40:
        return 1
    if rms_db <= -25:
        return max(1, min(5, int(2 + (rms_db + 40) / 5)))
    if rms_db <= -10:
        return max(5, min(8, int(5 + (rms_db + 25) / 5)))
    return max(8, min(10, int(8 + (rms_db + 10) / 5)))


# ──────────────────────────────────────────────────────────────────────────────
# JSON 持久化
# ──────────────────────────────────────────────────────────────────────────────

def _save_audio_map_json(
    music_path: str,
    audio_map_data: dict,
    analysis: dict,
    l2_data: dict,
    config_patch: dict,
    save_dir: str,
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    music_stem = Path(music_path).stem
    json_path = save_path / f"{music_stem}.json"

    # 主 audio_map.json：planner 直接读这个
    result = {
        # ── 基本信息 ──
        "music_file": Path(music_path).name,
        "bpm": audio_map_data["bpm"],
        "total_duration": audio_map_data["total_duration"],
        "beat_array": [round(float(t), 3) for t in analysis["beats"]],
        "downbeats": [round(float(t), 3) for t in analysis["downbeats"]],
        # ── L3 输出 ──
        "narrative_summary": audio_map_data.get("narrative_summary", ""),
        "mood_arc": audio_map_data.get("mood_arc", []),
        "segments": audio_map_data["segments"],
        "key_moments_v2": audio_map_data.get("key_moments", []),
        "tempo_density_curve": audio_map_data.get("tempo_density_curve", []),
        # ── 调试 / 兜底 ──
        "rms_quantiles": analysis.get("rms_quantiles", {}),
        "stem_onsets": analysis.get("stem_onsets", {}),
        "config_patch": config_patch,
    }

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  分析结果已保存: {json_path}")
