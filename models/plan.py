from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class RenderItem(BaseModel):
    """Render Plan 中的单条渲染指令"""

    order: int = Field(..., description="渲染顺序，从 1 开始")

    audio_start: float = Field(..., description="对应音乐的开始时间戳（秒）")
    audio_end: float = Field(..., description="对应音乐的结束时间戳（秒）")

    scene_id: int = Field(..., description="使用的场景 ID，对应 SceneItem.scene_id")
    source_file: str = Field(..., description="来源视频文件路径")
    clip_start: float = Field(..., description="从视频中截取的开始时间戳（秒）")
    clip_end: float = Field(..., description="从视频中截取的结束时间戳（秒）")

    speed_factor: float = Field(
        1.0, ge=0.3, le=3.0, description="变速倍率，1.0 为原速，<1 慢动作，>1 加速"
    )
    beat_snap_offset: float = Field(
        0.0, description="卡点吸附后的时间偏移量（秒），正数表示向后偏移"
    )
    cut_type: str = Field(
        "hard_cut", description="切入方式：hard_cut 硬切，flash_white/black 闪白/闪黑，dissolve 溶解，或 LLM 自定义"
    )
    transition_duration: float = Field(
        0.0, ge=0.0, le=0.5, description="转场/特效时长（秒），flash 建议 0.05-0.15，dissolve 建议 0.2-0.4"
    )

    @property
    def clip_duration(self) -> float:
        return self.clip_end - self.clip_start

    @property
    def audio_duration(self) -> float:
        return self.audio_end - self.audio_start