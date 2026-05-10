from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class SpecialEvent(BaseModel):
    """音乐中的特殊事件，如突然静音、爆发、drop 等"""

    time: float = Field(..., description="事件发生的时间戳（秒）")
    type: Literal[
        "sudden_silence",
        "explosion_after_silence",
        "fade_to_silence",
        "drop",
        "riser",
        "breakdown",
        "key_change",
        "section_boundary",
        "impact",
    ] = Field(..., description="事件类型")
    duration: float | None = Field(None, description="事件持续时长（秒），静音类事件适用")
    intensity: float | None = Field(None, description="事件强度 0-1，爆发类事件适用")


class AudioSegment(BaseModel):
    """音乐中的一个段落"""

    name: str = Field(..., description="段落名称，如 intro / verse_1 / chorus / drop / outro")
    start: float = Field(..., description="段落开始时间戳（秒）")
    end: float = Field(..., description="段落结束时间戳（秒）")
    energy: int = Field(..., ge=1, le=10, description="段落平均能量等级 1-10")
    energy_trend: Literal["rising", "falling", "stable", "peak", "rise_then_fall", "fall_then_rise"] = Field(
        "stable",
        description="段落内能量趋势：rising=上升, falling=下降, stable=平稳, peak=持续高能, "
                    "rise_then_fall=先升后降, fall_then_rise=先降后升",
    )
    energy_peak: int = Field(..., ge=1, le=10, description="段落内峰值能量等级 1-10")
    mood: str = Field("neutral", description="情绪标签，如 somber / tense / epic / peaceful")
    description: str = Field("", description="段落的自然语言描述，包含音乐特征、情感走向、人声特征等")
    visual_suggestion: str = Field("", description="该段落适合搭配的视觉画面建议，包含场景、风格、剪辑节奏")

    # ── L2/L3 新增字段（可选，向后兼容）─────────────────────────────────────
    energy_level: Literal["low", "medium", "high"] | None = Field(
        None, description="段落能量等级（基于本歌 RMS_dB 分位数）",
    )
    density_level: Literal["sparse", "medium", "dense"] | None = Field(
        None, description="段落事件密度等级（基于本歌 events_per_beat 分位数）",
    )
    pacing_hint: dict | None = Field(
        None, description="剪辑节奏建议：{beats_per_shot_range: [min, max], rationale: str}",
    )
    visual_profile: dict | None = Field(
        None,
        description="抽象视觉需求：{valence, arousal, dominance, motion_intensity, grain, temporal_pattern}",
    )

    @property
    def duration(self) -> float:
        return self.end - self.start


class EnergyKeypoint(BaseModel):
    """能量曲线关键点（自适应阈值筛选后的结果）"""

    time: float = Field(..., description="时间戳（秒）")
    value: float = Field(..., ge=0.0, le=1.0, description="能量值 0-1")


class KeyMoment(BaseModel):
    """L2+L3 提炼的关键剪辑锚点"""

    time: float = Field(..., description="锚点时间戳（秒）")
    importance: float = Field(..., ge=0.0, le=1.0, description="锚点重要性 0-1")
    tier: Literal["narrative_anchor", "section_beat", "rhythmic_hit"] = Field(
        ...,
        description="锚点级别：narrative_anchor=叙事骨架/必须踩, section_beat=段内重音, rhythmic_hit=节奏卡点",
    )
    anchor_type: str = Field(
        ..., description="锚点类型：section_drop / full_band_hit / vocal_phrase_on_downbeat 等",
    )
    description: str = Field("", description="这一瞬间发生的音乐事件描述")
    visual_profile: dict = Field(
        default_factory=dict,
        description="该锚点对画面的抽象需求（同 segment.visual_profile）",
    )
    transition_recommendation: str = Field(
        "hard_cut",
        description="建议转场：hard_cut / flash_white@0.08s / flash_black@0.08s / dissolve@0.3s",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="锚点的信号层证据（从 dense_events 收集）",
    )
    segment: str | None = Field(None, description="锚点所在段落标签")
    segment_energy_level: str | None = Field(None, description="所在段落的能量等级")


class AudioMap(BaseModel):
    """Audio Analyzer 的完整输出，音乐的结构化地图"""

    bpm: float = Field(..., description="每分钟拍数")
    total_duration: float = Field(..., description="音乐总时长（秒）")

    beat_array: list[float] = Field(
        ..., description="每一拍的精确时间戳数组（秒），由 madmom 提供"
    )
    downbeats: list[float] = Field(
        ..., description="强拍（每小节第一拍）时间戳数组（秒），由 madmom 提供"
    )

    segments: list[AudioSegment] = Field(
        ..., description="音乐段落列表，由代码分段 + LLM 语义标注"
    )
    energy_keypoints: list[EnergyKeypoint] = Field(
        ..., description="能量曲线关键点，自适应阈值筛选后的结果"
    )

    r1_understanding: str = Field(
        default="",
        description="第一轮 Gemini 感性理解的原始文本，供调试使用",
    )

    key_moments: list[dict] = Field(
        default_factory=list,
        description="Gemini R1 识别的关键时刻列表（旧字段，保留兼容；新方案用 key_moments_v2）",
    )

    # ── L2/L3 新增字段（可选，向后兼容）────────────────────────────────────
    key_moments_v2: list[KeyMoment] = Field(
        default_factory=list,
        description="L2+L3 提炼的关键锚点（带 tier / VAD / transition_recommendation）",
    )
    narrative_summary: str = Field(
        "",
        description="整曲叙事弧线一句话总结（来自 L3）",
    )
    mood_arc: list[str] = Field(
        default_factory=list,
        description="按段落顺序的情绪标签数组（来自 L3）",
    )
    tempo_density_curve: list[dict] = Field(
        default_factory=list,
        description="节奏密度曲线，滑窗算 events_per_beat + 本歌分位数分级",
    )

