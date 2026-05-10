from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from loguru import logger

from utils.clients import QWEN_PLUS, get_qwen_client

if TYPE_CHECKING:
    from models.video import SceneItem


async def check_continuity(
    scene_a: "SceneItem",
    scene_b: "SceneItem",
    cut_type: str = "hard_cut",
) -> dict:
    """
    判断两个相邻场景的切换是否合理。
    返回 {pass: bool, reason: str, suggestion: str}
    """
    client = get_qwen_client()

    prompt = f"""你是一位经验丰富的视频剪辑师，请判断以下两个相邻镜头的切换是否合理。

## 上一个镜头
- 描述：{scene_a.scene_description}
- 情绪：{scene_a.mood}
- visual_profile：V/A/D={(scene_a.visual_profile or {}).get('valence',0):.2f}/{(scene_a.visual_profile or {}).get('arousal',0):.2f}/{(scene_a.visual_profile or {}).get('dominance',0):.2f}  motion={(scene_a.visual_profile or {}).get('motion_intensity',0):.2f}  grain={(scene_a.visual_profile or {}).get('grain','?')}

## 下一个镜头
- 描述：{scene_b.scene_description}
- 情绪：{scene_b.mood}
- visual_profile：V/A/D={(scene_b.visual_profile or {}).get('valence',0):.2f}/{(scene_b.visual_profile or {}).get('arousal',0):.2f}/{(scene_b.visual_profile or {}).get('dominance',0):.2f}  motion={(scene_b.visual_profile or {}).get('motion_intensity',0):.2f}  grain={(scene_b.visual_profile or {}).get('grain','?')}

## 切换方式
{cut_type}

## 判断标准
1. 情绪跳跃是否过大（如从极度悲伤直接切到欢快）
2. 视觉风格是否严重冲突（如色调、亮度极端反差且无音乐支撑）
3. 动作密度跳跃是否合理（如从静止特写直接切到高速追逐）
4. 切换方式是否与内容匹配

注意：AMV 剪辑允许一定程度的跳跃和对比，不要过于严格。只有明显不合理的切换才判定为不通过。

严格输出 JSON，不要有多余文字：
{{"pass": true或false, "reason": "简短说明（20字以内）", "suggestion": "如果不通过，建议换什么类型的场景（15字以内），通过则留空"}}"""

    try:
        response = await client.chat.completions.create(
            model=QWEN_PLUS,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150,
        )
        raw = response.choices[0].message.content
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        result = json.loads(clean)
        return {
            "pass": bool(result.get("pass", True)),
            "reason": result.get("reason", ""),
            "suggestion": result.get("suggestion", ""),
        }
    except Exception as e:
        logger.warning(f"[Continuity] 审查失败: {e}，默认通过")
        return {"pass": True, "reason": "审查失败，默认通过", "suggestion": ""}
