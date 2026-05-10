"""
Audio Analyzer v2 - L3 语义增强层

输入: key_moments_extractor.extract_key_moments() 的 dict
输出: 完整的 audio_map dict（兼容 eval_model audio_map.json 格式）
  - 每段加 mood / description / visual_profile
  - 每锚点加 description / visual_profile / tier / transition_recommendation
  - 全曲加 narrative_summary / mood_arc

实现说明:
  - A 方案：tier / transition_recommendation 由代码硬规则决定，不让 LLM 决定
  - C 方案：锚点分小批次（每批 5 个），注入所在段落 VAD 基线，迫使 LLM 差异化
  - 术语词典注入所有 prompt（修正 downbeat 错译等）
  - 失败重试：每个 LLM 调用最多 3 次，指数退避
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger

from config import DefaultConfig
from utils.clients import GPT_5_5, QWEN_MAX, get_openai_client, get_qwen_client
from utils.prompt_loader import load_and_render

BATCH_SIZE = 5  # C 方案：锚点每批处理的数量

VISUAL_PROFILE_SCHEMA = '''{
  "valence": float[0-1],
  "arousal": float[0-1],
  "dominance": float[0-1],
  "motion_intensity": float[0-1],
  "grain": "detail|mid|broad",
  "temporal_pattern": "accelerating|decelerating|stable|pulsing"
}'''


# ── LLM 调用（带重试）─────────────────────────────────────────────────────────

async def _call_llm_json(
    prompt: str,
    backend: str,
    max_tokens: int = 4000,
    max_retries: int = 3,
) -> Any:
    """
    根据 backend 调用 GPT-5.5 或 Qwen Max，强制 JSON 输出，带指数退避重试。
    """
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            if backend == "gpt":
                client = get_openai_client()
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=GPT_5_5,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.35,
                        max_tokens=max_tokens,
                    ),
                    timeout=180.0,
                )
            elif backend == "qwen":
                client = get_qwen_client()
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=QWEN_MAX,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.35,
                        max_tokens=max_tokens,
                    ),
                    timeout=180.0,
                )
            else:
                raise ValueError(f"未知 L3_LLM_BACKEND: {backend}")

            raw = response.choices[0].message.content
            clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            return json.loads(clean)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"  [L3 retry {attempt + 1}/{max_retries}] {type(e).__name__}: "
                    f"{str(e)[:80]} - 等待 {wait}s"
                )
                await asyncio.sleep(wait)
            else:
                raise last_err


# ── Prompt 构造 ──────────────────────────────────────────────────────────────

def _build_segments_prompt(l2_data: dict) -> str:
    segments_brief = []
    for s in l2_data["segments"]:
        segments_brief.append(
            f"  [{s['label']:10s}] {s['start']:6.1f}-{s['end']:6.1f}s  "
            f"dur={s['duration']:5.1f}s  "
            f"energy={s['energy_level']}({s['rms_db']}dB)  "
            f"density={s['density_level']}({s['avg_events_per_beat']}/beat)  "
            f"key_moments={s['key_moments_count']}"
        )
    segments_text = "\n".join(segments_brief)

    return load_and_render(
        "audio/l3_segments.md",
        bpm=l2_data["bpm"],
        total_duration=l2_data["total_duration"],
        segments_text=segments_text,
        visual_profile_schema=VISUAL_PROFILE_SCHEMA,
    )


def _build_key_moments_batch_prompt(
    batch: list[dict],
    segment_baseline: dict[str, dict],
    global_offset: int,
) -> str:
    lines = []
    for i, k in enumerate(batch):
        evidence = " + ".join(k["evidence"])
        baseline = segment_baseline.get(k.get("_seg_key")) or {}
        baseline_str = (
            f"V={baseline.get('valence', 0.5):.2f} "
            f"A={baseline.get('arousal', 0.5):.2f} "
            f"D={baseline.get('dominance', 0.5):.2f} "
            f"motion={baseline.get('motion_intensity', 0.5):.2f}"
            if baseline else "(无基线)"
        )
        lines.append(
            f"  #{global_offset + i + 1}  time={k['time']:.2f}s  "
            f"importance={k['importance']:.2f}  "
            f"type={k['anchor_type']}  "
            f"segment={k['segment']}({k['segment_energy_level']})  "
            f"evidence: {evidence}\n"
            f"      段落基线: {baseline_str}"
        )
    batch_text = "\n".join(lines)

    return load_and_render(
        "audio/l3_key_moments.md",
        batch_size=len(batch),
        batch_text=batch_text,
        visual_profile_schema=VISUAL_PROFILE_SCHEMA,
    )


def _build_narrative_prompt(
    l2_data: dict, enriched_segments: list[dict],
) -> str:
    seg_summary = []
    for s, e in zip(l2_data["segments"], enriched_segments):
        vp = e.get("visual_profile") or {}
        seg_summary.append(
            f"  [{s['label']}] {s['start']:.1f}-{s['end']:.1f}s  "
            f"mood={e.get('mood', '?')}  "
            f"V={vp.get('valence', 0):.2f} "
            f"A={vp.get('arousal', 0):.2f} "
            f"D={vp.get('dominance', 0):.2f}"
        )
    seg_text = "\n".join(seg_summary)
    return load_and_render("audio/l3_narrative.md", seg_text=seg_text)


# ── 硬规则（A 方案）──────────────────────────────────────────────────────────

def assign_tier(km: dict) -> str:
    """按 importance + anchor_type 强制分类 tier"""
    imp = km["importance"]
    atype = km.get("anchor_type", "")
    if imp >= 0.65 or "section_drop" in atype or "section_change_with_jump" in atype:
        return "narrative_anchor"
    if "section_change_to_" in atype and imp >= 0.55:
        return "narrative_anchor"
    if imp >= 0.45:
        return "section_beat"
    return "rhythmic_hit"


def assign_transition(km: dict, vp: dict) -> str:
    """按 anchor_type + visual_profile 强制生成 transition_recommendation"""
    atype = km.get("anchor_type", "")
    arousal = vp.get("arousal", 0.5) if vp else 0.5
    pattern = vp.get("temporal_pattern", "stable") if vp else "stable"

    if "section_change_to_outro" in atype or pattern == "decelerating":
        return "dissolve@0.3s"

    if any(k in atype for k in [
        "section_drop", "full_band_hit", "section_change_to_chorus",
        "section_change_with_jump",
    ]) and arousal >= 0.65:
        return "flash_white@0.08s"

    valence = vp.get("valence", 0.5) if vp else 0.5
    if "section_change_to_" in atype and valence < 0.35:
        return "flash_black@0.08s"

    return "hard_cut"


def enforce_vad_differentiation(
    km: dict,
    baseline: dict | None,
) -> dict:
    """
    检查锚点 VAD 是否与段落基线有差异；若完全一致（LLM 偷懒），
    根据 anchor_type 强制注入差异化扰动。
    """
    vp = km.get("visual_profile")
    if not vp or not baseline:
        return vp or {}

    same = all(
        abs(vp.get(k, 0) - baseline.get(k, 0)) < 0.015
        for k in ("valence", "arousal", "dominance")
    )
    if not same:
        return vp

    atype = km.get("anchor_type", "")
    adjusted = dict(vp)

    if any(k in atype for k in ["full_band_hit", "section_drop", "rhythmic_section_hit"]):
        adjusted["arousal"] = min(1.0, baseline["arousal"] + 0.15)
        adjusted["dominance"] = min(1.0, baseline["dominance"] + 0.10)
        adjusted["motion_intensity"] = min(1.0, baseline.get("motion_intensity", 0.5) + 0.15)
    elif "drum_hit" in atype:
        adjusted["arousal"] = min(1.0, baseline["arousal"] + 0.10)
        adjusted["motion_intensity"] = min(1.0, baseline.get("motion_intensity", 0.5) + 0.10)
    elif "vocal" in atype:
        adjusted["arousal"] = min(1.0, baseline["arousal"] + 0.05)
    elif "melodic_hit" in atype:
        adjusted["arousal"] = min(1.0, baseline["arousal"] + 0.05)
    elif "bass_solo" in atype:
        adjusted["arousal"] = min(1.0, baseline["arousal"] + 0.08)
    elif "section_change_to_outro" in atype:
        adjusted["arousal"] = max(0.0, baseline["arousal"] - 0.20)
        adjusted["temporal_pattern"] = "decelerating"

    return adjusted


# ── 主入口 ───────────────────────────────────────────────────────────────────

async def enrich_audio_map(
    l2_data: dict,
    backend: str | None = None,
) -> dict:
    """
    L3 语义增强主入口

    Args:
        l2_data: extract_key_moments() 的输出 dict
        backend: "gpt" 或 "qwen"，None 时用 config.DefaultConfig.L3_LLM_BACKEND

    Returns:
        完整 audio_map dict（同 eval_model audio_map.json 格式）
    """
    if backend is None:
        backend = DefaultConfig.L3_LLM_BACKEND

    logger.info(f"  [L3] 后端: {backend} ({GPT_5_5 if backend == 'gpt' else QWEN_MAX})")

    # Step 1: 段落标注
    logger.info(f"  [L3 1/3] 段落语义标注（{len(l2_data['segments'])} 段）")
    seg_prompt = _build_segments_prompt(l2_data)
    enriched_segs = await _call_llm_json(seg_prompt, backend, max_tokens=3000)
    if len(enriched_segs) != len(l2_data["segments"]):
        logger.warning(
            f"  [L3] 段落输出数 {len(enriched_segs)} != 输入 {len(l2_data['segments'])}"
        )
        while len(enriched_segs) < len(l2_data["segments"]):
            enriched_segs.append({})

    # 构建段落 VAD 基线表
    segment_baseline: dict[str, dict] = {}
    for orig, enriched in zip(l2_data["segments"], enriched_segs):
        key = f"{orig['label']}@{orig['start']}"
        segment_baseline[key] = enriched.get("visual_profile") or {}

    # Step 2: 锚点标注（批量）
    km_list = l2_data["key_moments"]
    for km in km_list:
        km["_seg_key"] = None
        for s in l2_data["segments"]:
            if s["start"] <= km["time"] < s["end"]:
                km["_seg_key"] = f"{s['label']}@{s['start']}"
                km["segment_start"] = s["start"]
                break

    n_batches = (len(km_list) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        f"  [L3 2/3] 锚点语义标注（{len(km_list)} 锚点，分 {n_batches} 批）"
    )

    enriched_moments: list[dict] = []
    for batch_i in range(n_batches):
        start = batch_i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(km_list))
        batch = km_list[start:end]
        logger.info(f"    批 {batch_i + 1}/{n_batches}：锚点 #{start + 1}-#{end}")
        batch_prompt = _build_key_moments_batch_prompt(batch, segment_baseline, start)
        try:
            batch_result = await _call_llm_json(batch_prompt, backend, max_tokens=3000)
        except Exception as e:
            logger.error(f"    [L3] 批 {batch_i + 1} 失败: {e}，填充空值")
            batch_result = [{} for _ in batch]
        if len(batch_result) != len(batch):
            logger.warning(
                f"    [L3] 批 {batch_i + 1} 输出数 {len(batch_result)} != 输入 {len(batch)}"
            )
            while len(batch_result) < len(batch):
                batch_result.append({})
        enriched_moments.extend(batch_result)

    # Step 3: 整体叙事弧线
    logger.info("  [L3 3/3] 整体叙事弧线")
    narr_prompt = _build_narrative_prompt(l2_data, enriched_segs)
    narrative = await _call_llm_json(narr_prompt, backend, max_tokens=500)

    # 组装 segments
    segments_out = []
    for orig, enriched in zip(l2_data["segments"], enriched_segs):
        merged = dict(orig)
        merged.update({
            "mood": enriched.get("mood"),
            "description": enriched.get("description"),
            "visual_profile": enriched.get("visual_profile"),
        })
        segments_out.append(merged)

    # 组装 key_moments，A 方案后处理
    moments_out = []
    for orig, enriched in zip(km_list, enriched_moments):
        baseline = segment_baseline.get(orig.get("_seg_key")) or {}
        vp = enforce_vad_differentiation(
            {**orig, "visual_profile": enriched.get("visual_profile")},
            baseline,
        )
        merged = {k: v for k, v in orig.items() if not k.startswith("_")}
        merged.update({
            "description": enriched.get("description"),
            "visual_profile": vp,
            "tier": assign_tier(orig),
            "transition_recommendation": assign_transition(orig, vp),
        })
        moments_out.append(merged)

    return {
        "music_file": l2_data["music_file"],
        "bpm": l2_data["bpm"],
        "total_duration": l2_data["total_duration"],
        "narrative_summary": narrative.get("narrative_summary", ""),
        "mood_arc": narrative.get("mood_arc", []),
        "segments": segments_out,
        "key_moments": moments_out,
        "tempo_density_curve": l2_data.get("tempo_density_curve", []),
    }
