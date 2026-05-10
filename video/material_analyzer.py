from __future__ import annotations

import base64
import json
import random
import re
from pathlib import Path

from loguru import logger

from utils.clients import GEMINI_FLASH, get_gemini_client


async def analyze_material_type(
    scene_frames: list[str],
    background_info: str | None = None,
) -> dict:
    """
    第一阶段：让 Gemini 看样本帧，自动判断素材类型，
    生成适合这类素材的标签维度和叙事阶段。

    参数：
        scene_frames: 所有场景的关键帧路径列表
        background_info: 用户提供的背景信息

    返回：
        {
            "material_type": "格斗动漫",
            "narrative_stages": ["进场", "对峙", ...],
            "key_dimensions": ["叙事位置", "动作类型", ...],
            "tagging_hints": "打标签时注意...",
            "tagger_extra_prompt": "注入到 tagger prompt 的额外说明"
        }
    """
    from google.genai import types

    client = get_gemini_client()

    # 随机抽取最多 12 帧作为样本（太多超过 token 限制）
    sample_frames = random.sample(scene_frames, min(12, len(scene_frames)))
    logger.info(f"[MaterialAnalyzer] 抽取 {len(sample_frames)} 帧分析素材类型")

    parts = []
    for frame_path in sample_frames:
        try:
            img_bytes = Path(frame_path).read_bytes()
            parts.append(types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64encode(img_bytes).decode()
                )
            ))
        except Exception as e:
            logger.warning(f"  读取帧失败: {frame_path}: {e}")

    if not parts:
        logger.warning("[MaterialAnalyzer] 无有效帧，使用默认配置")
        return _default_config()

    background_section = f"用户提供的背景信息：{background_info}" if background_info else "未提供背景信息"

    prompt = f"""
{background_section}

请看以上视频帧样本，完成以下分析：

1. 判断这批素材的内容类型（如：格斗动漫、情感剧情、科幻动作、日常生活、风景纪录等）
2. 根据内容类型，提取这类素材的叙事阶段（即这类内容通常会经历哪些叙事节点）
3. 提出打标签时最需要关注的维度
4. 给出打标签的具体提示

严格输出 JSON，不要有任何多余文字：

{{
  "material_type": "素材类型（简短描述）",
  "narrative_stages": ["阶段1", "阶段2", "阶段3", "...（按叙事顺序排列，5-8个）"],
  "key_dimensions": ["维度1", "维度2", "维度3", "维度4"],
  "tagging_hints": "打标签时最需要注意的事项（1-3句话）",
  "tagger_extra_prompt": "注入到场景打标签 prompt 里的额外说明（50-100字，指导 Gemini 如何描述这类素材的场景）"
}}
"""

    parts.append(types.Part(text=prompt))

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_FLASH,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            ),
        )

        raw = response.text
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        result = json.loads(clean)

        logger.info(f"[MaterialAnalyzer] 素材类型: {result.get('material_type')}")
        logger.info(f"[MaterialAnalyzer] 叙事阶段: {result.get('narrative_stages')}")

        return result

    except Exception as e:
        logger.warning(f"[MaterialAnalyzer] 分析失败: {e}，使用默认配置")
        return _default_config()


def _default_config() -> dict:
    """默认配置，分析失败时使用。"""
    return {
        "material_type": "通用影视素材",
        "narrative_stages": ["开始", "发展", "高潮", "结尾"],
        "key_dimensions": ["情绪", "动作强度", "人物状态", "场景氛围"],
        "tagging_hints": "请客观描述画面内容，注意人物情绪和动作状态。",
        "tagger_extra_prompt": "请在 scene_description 中描述画面的叙事位置和情绪状态。",
    }


def build_material_type_section(analysis: dict) -> str:
    """
    根据素材分析结果，生成注入到 tagger_system.md 的动态段落。
    """
    material_type = analysis.get("material_type", "通用影视素材")
    narrative_stages = analysis.get("narrative_stages", [])
    key_dimensions = analysis.get("key_dimensions", [])
    tagging_hints = analysis.get("tagging_hints", "")
    extra_prompt = analysis.get("tagger_extra_prompt", "")

    stages_str = "、".join(narrative_stages) if narrative_stages else "无"
    dimensions_str = "、".join(key_dimensions) if key_dimensions else "无"

    return f"""## 素材类型特殊说明（自动分析生成）

**素材类型：** {material_type}

**叙事阶段参考（按顺序）：** {stages_str}

**重点标注维度：** {dimensions_str}

**打标签注意事项：** {tagging_hints}

**scene_description 额外要求：** {extra_prompt}
在描述场景时，请明确指出这个场景属于哪个叙事阶段（如：【进场】【对峙】【变身】等），
并描述人物的具体状态和动作。
"""