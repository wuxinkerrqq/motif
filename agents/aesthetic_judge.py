"""
aesthetic_judge.py — 从候选场景中做审美判断，选出最匹配且视觉最多样的场景

设计依据：
- "Multimodal LLMs Can Reason about Aesthetics in Zero-Shot"（2025）
  → 给 LLM 明确的审美维度，zero-shot 即可接近人类判断
- ICIP 2023 RL-DiVTS
  → 审美质量和视觉多样性需要同时优化
- 候选集内做相对比较比绝对打分更稳定
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from loguru import logger

from utils.clients import QWEN_PLUS, get_qwen_client

if TYPE_CHECKING:
    from models.video import SceneItem


async def judge_best_scene(
    candidates: list["SceneItem"],
    candidate_scores: list[float],
    audio_segment: dict,
    committed_scenes: list["SceneItem"],
) -> tuple[int, str]:
    """
    从候选场景中选出审美上最合适的一个。

    Args:
        candidates:       search_scenes 返回的候选 SceneItem 列表
        candidate_scores: 对应的 embedding 相似度分数
        audio_segment:    当前音乐段落信息，包含 mood/energy/energy_trend/description
        committed_scenes: 已经 commit 的场景列表，用于多样性判断

    Returns:
        (scene_id, reason) — 选中的场景 ID 和理由
    """
    if not candidates:
        raise ValueError("候选场景为空")
    if len(candidates) == 1:
        return candidates[0].scene_id, "唯一候选"

    client = get_qwen_client()

    # 构建候选描述
    candidates_text = ""
    for i, (scene, score) in enumerate(zip(candidates, candidate_scores), start=1):
        vp = scene.visual_profile or {}
        candidates_text += (
            f"\n候选{i}（scene_id={scene.scene_id}，相似度={score:.3f}）\n"
            f"  描述：{scene.scene_description}\n"
            f"  情绪：{scene.mood}  V/A/D={vp.get('valence',0):.2f}/{vp.get('arousal',0):.2f}/{vp.get('dominance',0):.2f}  motion={vp.get('motion_intensity',0):.2f}\n"
            f"  grain={vp.get('grain','?')}  pattern={vp.get('temporal_pattern','?')}\n"
            f"  时长：{scene.duration:.1f}s\n"
        )

    # 构建已用场景摘要（最近5个，避免 prompt 过长）
    recent_committed = committed_scenes[-5:] if len(committed_scenes) > 5 else committed_scenes
    committed_text = ""
    if recent_committed:
        committed_text = "\n## 最近已使用的场景（避免视觉重复）\n"
        for scene in recent_committed:
            vp = scene.visual_profile or {}
            committed_text += (
                f"  scene_id={scene.scene_id}：{scene.mood}  "
                f"V/A/D={vp.get('valence',0):.2f}/{vp.get('arousal',0):.2f}/{vp.get('dominance',0):.2f}  grain={vp.get('grain','?')}\n"
            )
    else:
        committed_text = "\n## 最近已使用的场景\n  （暂无，这是开头部分）\n"

    prompt = f"""你是一位专业的 AMV 剪辑师，具备敏锐的审美判断力。
请从以下候选场景中，选出最适合当前音乐段落的一个。

## 当前音乐段落
- 情绪：{audio_segment.get('mood', '未知')}
- 能量：{audio_segment.get('energy', '未知')}/10
- 能量趋势：{audio_segment.get('energy_trend', '未知')}
- 描述：{audio_segment.get('description', '未提供')}
{committed_text}
## 候选场景
{candidates_text}

## 审美判断标准（按优先级）
1. **情绪契合** — 场景情绪与音乐情绪是否呼应（允许反差，但要有意图）
2. **能量匹配** — 场景动作密度/视觉张力与音乐能量是否协调
3. **视觉多样性** — 与已使用场景相比，色调/构图/镜头运动是否足够不同
4. **创作价值** — 相似度分数高不代表最好，有时反差或意外感更有冲击力

注意：这是创作判断，不是做题，没有标准答案。相似度只是参考，不是决定因素。

严格输出 JSON，不要有多余文字：
{{"scene_id": 选中的scene_id整数, "reason": "选择理由（30字以内，说明情绪/视觉/多样性哪点打动你）"}}"""

    try:
        response = await client.chat.completions.create(
            model=QWEN_PLUS,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100,
        )
        raw = response.choices[0].message.content
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        result = json.loads(clean)
        scene_id = int(result["scene_id"])
        reason = result.get("reason", "")

        # 验证返回的 scene_id 在候选列表中
        valid_ids = {s.scene_id for s in candidates}
        if scene_id not in valid_ids:
            logger.warning(f"[AestheticJudge] 返回了无效 scene_id={scene_id}，回退到相似度第一")
            return candidates[0].scene_id, "judge返回无效，回退相似度第一"

        logger.debug(f"[AestheticJudge] 选中 scene_id={scene_id}：{reason}")
        return scene_id, reason

    except Exception as e:
        logger.warning(f"[AestheticJudge] 判断失败: {e}，回退到相似度第一")
        return candidates[0].scene_id, "judge失败，回退相似度第一"
