from __future__ import annotations

from pydantic import BaseModel, Field


class SceneItem(BaseModel):
    """Video Tagger 输出的单个场景记录"""

    scene_id: int = Field(..., description="场景唯一 ID，全局递增")
    source_file: str = Field(..., description="来源视频文件名")
    source_dir: str = Field("", description="来源视频所在目录")

    start: float = Field(..., description="场景开始时间戳（秒）")
    end: float = Field(..., description="场景结束时间戳（秒）")
    duration: float = Field(..., description="场景时长（秒）")
    scene_index: str = Field(..., description="场景在来源文件中的位置，如 3/38")

    keyframes: list[str] = Field(
        ..., description="关键帧图片路径列表，取自场景中间段"
    )

    scene_description: str = Field(..., description="场景的自然语言描述")
    characters: list[str] = Field(
        default_factory=list, description="画面中出现的人物列表"
    )

    mood: str = Field(
        "neutral",
        description="情绪标签（粗筛用），如 somber/lonely/tense/determined/epic/triumphant/peaceful/intimate/excited/melancholic/anxious/calm/joyful/nostalgic",
    )

    visual_profile: dict = Field(
        default_factory=dict,
        description="抽象视觉档案 {valence, arousal, dominance, motion_intensity, grain, temporal_pattern}，"
                    "与音频侧 visual_profile 同 schema，用于规划器做向量距离匹配",
    )

    is_outro_material: bool = Field(
        False, description="是否是结局素材，True 时只能用于 outro 段落"
    )
    is_climax_material: bool = Field(
        False, description="是否是高潮素材，True 时只能用于 drop/chorus 段落"
    )
