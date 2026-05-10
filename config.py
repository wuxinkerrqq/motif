from __future__ import annotations

import numpy as np
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# 全局默认参数
# 所有硬编码数值统一在这里管理，不要散落在各模块代码中
# ──────────────────────────────────────────────────────────────────────────────

class DefaultConfig:

    # ── 预检阈值 ────────────────────────────────────────────────────────────────
    VIDEO_RATIO_WARN: float = 0.8       # 视频总时长 < 音乐时长 × 0.8 → 警告
    VIDEO_RATIO_CRITICAL: float = 0.3   # 视频总时长 < 音乐时长 × 0.3 → 强烈警告
    VIDEO_SIZE_LIMIT_GB: float = 2.0    # 视频素材总大小上限
    MUSIC_DURATION_WARN: float = 480.0  # 音乐时长超过 8 分钟 → 警告

    # ── 音频分析 ────────────────────────────────────────────────────────────────
    ENERGY_THRESHOLD_SIGMA: float = 1.0     # 自适应阈值：mean + N*std
    SILENCE_RATIO: float = 0.1              # 低于 mean * ratio 认定为静音
    EXPLOSION_SIGMA: float = 2.0            # 超过 mean + N*std 认定为爆发
    FADE_LOOKBACK_SECONDS: float = 0.23     # 渐弱检测回溯窗口（秒）

    # L3 语义增强后端："gpt" = GPT-5.5（jiekou.ai 中转），"qwen" = Qwen Max
    L3_LLM_BACKEND: Literal["gpt", "qwen"] = "gpt"

    # ── 视频打标签 ──────────────────────────────────────────────────────────────
    FRAME_SHORT_THRESHOLD: float = 1.0      # 场景时长 < 1s → 取 1 帧
    FRAME_MEDIUM_THRESHOLD: float = 3.0     # 场景时长 1-3s → 取 2 帧；>3s → 取 3 帧
    TAGGER_MAX_RETRY: int = 2               # 轻量 Reviewer 校验失败最大重试次数
    TAGGER_CONCURRENCY: int = 5             # Gemini 并发调用数量上限

    # ── 卡点吸附 ────────────────────────────────────────────────────────────────
    BEAT_SNAP_TOLERANCE: float = 0.15       # snap_to_beats 容忍范围（秒）

    # ── 变速 ────────────────────────────────────────────────────────────────────
    SPEED_FACTOR_MIN: float = 0.3           # 变速倍率下限
    SPEED_FACTOR_MAX: float = 3.0           # 变速倍率上限
    SPEED_EXTREME_RATIO: float = 3.0        # 时长差距超过此倍数 → 换素材而非变速

    # ── RIFE 插帧（可选） ────────────────────────────────────────────────────────
    RIFE_ENABLED: bool = True               # 慢动作片段启用光流插帧
    RIFE_EXE_PATH: str = r"D:\Python_Programes\rife-ncnn-vulkan-20221029-windows\rife-ncnn-vulkan.exe"
    RIFE_MODEL_PATH: str = r"D:\Python_Programes\rife-ncnn-vulkan-20221029-windows\rife-v4.6"

    # ── 镜头匹配 ────────────────────────────────────────────────────────────────
    DENSITY_MATCH_TOLERANCE: int = 2        # action_density 匹配容忍范围
    DENSITY_GAP_MAX: int = 5               # 相邻镜头 action_density 最大跨度
    DURATION_FILL_TOLERANCE: float = 0.5   # 时长填充允许的最大误差（秒）

    # ── Reviewer 执行质量 ───────────────────────────────────────────────────────
    BEAT_SNAP_ERROR_MAX: float = 0.15       # 卡点误差上限（秒），超过则报错
    EDITOR_MAX_RETRY: int = 2              # Editor Agent 最大重试次数

    # ── 反思循环 ────────────────────────────────────────────────────────────────
    PLANNER_MAX_RETRY: int = 2             # Edit Planner 最大反思次数

    # ── 情绪矛盾对 ──────────────────────────────────────────────────────────────
    # 以下情绪对被认为直接相邻会造成视觉割裂
    CONTRADICTORY_MOOD_PAIRS: list[tuple[str, str]] = [
        ("joyful", "somber"),
        ("tense", "peaceful"),
        ("epic", "intimate"),
        ("triumphant", "melancholic"),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 风格预设
# 不同剪辑风格对应不同的参数组合，覆盖全局默认值
# ──────────────────────────────────────────────────────────────────────────────

STYLE_PRESETS: dict[str, dict] = {
    "visual_driven": {
        "BEAT_SNAP_TOLERANCE": 0.15,        # 卡点要求适中（原 0.08 太严，导致大量切点未吸附）
        "DENSITY_GAP_MAX": 7,               # 允许更大的密度跳跃
        "DENSITY_MATCH_TOLERANCE": 3,
        "EDITOR_MAX_RETRY": 3,
    },
    "story_driven": {
        "BEAT_SNAP_TOLERANCE": 0.25,        # 卡点要求宽松（原 0.20）
        "DENSITY_GAP_MAX": 3,               # 密度变化要平缓
        "DENSITY_MATCH_TOLERANCE": 2,
        "ENERGY_THRESHOLD_SIGMA": 0.8,      # 多识别一些特殊事件
    },
    "emotion_driven": {
        "BEAT_SNAP_TOLERANCE": 0.35,        # 几乎不强制卡拍（原 0.30）
        "DENSITY_GAP_MAX": 4,
        "SILENCE_RATIO": 0.15,              # 更敏感的静音检测
        "DENSITY_MATCH_TOLERANCE": 3,
    },
}


def load_config(
    editing_style: Literal["visual_driven", "story_driven", "emotion_driven"]
) -> dict:
    """
    加载风格预设，合并默认值。
    返回一个扁平的 dict，供注入 GraphState.runtime_config 使用。
    """
    base = {
        k: v for k, v in vars(DefaultConfig).items()
        if not k.startswith("_") and not callable(v)
    }
    preset = STYLE_PRESETS.get(editing_style, {})
    return {**base, **preset}


# ──────────────────────────────────────────────────────────────────────────────
# 自适应参数
# 运行时根据实际输入动态计算，覆盖默认值
# 分阶段调用：音频分析完成后补充音频相关参数，视频分析完成后补充素材密度参数
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveConfig:

    def get_beat_snap_tolerance(self, bpm: float) -> float:
        """
        BPM 越高，相邻 beat 间距越小，容忍范围必须动态缩小。
        公式：beat_interval * 0.35，上限 0.25s，下限 0.08s。
        """
        beat_interval = 60.0 / bpm
        tolerance = beat_interval * 0.35
        return round(max(0.08, min(0.25, tolerance)), 3)

    def get_silence_threshold(self, energy_values: list[float]) -> float:
        """
        用 10 百分位数替代 mean * 0.1，对响度战争压限的音乐更鲁棒。
        """
        return float(np.percentile(energy_values, 10))

    def get_explosion_threshold(self, energy_values: list[float]) -> float:
        """
        爆发阈值：75 百分位数，比固定的 mean + 2*std 对极端值更稳健。
        """
        return float(np.percentile(energy_values, 75))

    def get_density_gap_tolerance(
        self, scene_count: int, music_duration: float
    ) -> int:
        """
        素材越少，密度匹配越宽松，避免因素材不足而频繁失败。
        """
        scenes_per_minute = scene_count / (music_duration / 60.0)
        if scenes_per_minute < 5:
            return 7    # 素材极少，大幅放宽
        elif scenes_per_minute < 10:
            return 5    # 素材一般
        else:
            return 3    # 素材充足，严格匹配

    def get_video_ratio_warn(
        self,
        editing_style: Literal["visual_driven", "story_driven", "emotion_driven"],
    ) -> float:
        """
        情绪向剪辑允许更多素材重用，阈值可以低一点。
        """
        if editing_style == "emotion_driven":
            return 0.5
        return DefaultConfig.VIDEO_RATIO_WARN

    def patch_runtime_config(
        self,
        config: dict,
        *,
        bpm: float | None = None,
        energy_values: list[float] | None = None,
        scene_count: int | None = None,
        music_duration: float | None = None,
    ) -> dict:
        """
        分阶段更新 runtime_config。
        音频分析完成后传入 bpm + energy_values；
        视频分析完成后传入 scene_count + music_duration。
        只更新能计算的字段，其他字段保持不变。
        """
        updated = dict(config)

        if bpm is not None:
            updated["BEAT_SNAP_TOLERANCE"] = self.get_beat_snap_tolerance(bpm)

        if energy_values is not None:
            updated["SILENCE_THRESHOLD"] = self.get_silence_threshold(energy_values)
            updated["EXPLOSION_THRESHOLD"] = self.get_explosion_threshold(energy_values)

        if scene_count is not None and music_duration is not None:
            updated["DENSITY_GAP_MAX"] = self.get_density_gap_tolerance(
                scene_count, music_duration
            )

        return updated


# ──────────────────────────────────────────────────────────────────────────────
# 全局单例，供各模块直接 import 使用
# ──────────────────────────────────────────────────────────────────────────────

adaptive_config = AdaptiveConfig()