from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from config import load_config
from models.state import GraphState


async def parse_video_intent(
    filenames: list[str],
    user_description: str,
) -> dict:
    """
    将用户对视频素材的自然语言描述解析为结构化元数据。

    Args:
        filenames: 视频文件名列表，如 ["clip_001.mp4", "clip_002.mp4"]
        user_description: 用户描述，如 "第1-2个是《普罗梅亚》高潮，第3个是城市夜景"

    Returns:
        {filename: {"source": str, "episode": str|None, "context": str}}
        未被提及的文件不出现在返回值中。
    """
    from utils.clients import QWEN_MAX, get_qwen_client

    numbered_list = "\n".join(f"#{i+1} {name}" for i, name in enumerate(filenames))

    prompt = f"""你是一个视频素材管理助手。

用户有以下视频文件（按编号列出）：
{numbered_list}

用户的描述：
"{user_description}"

请将用户描述解析为如下 JSON 格式，严格只输出 JSON，不要任何其他文字：
{{
  "文件名.mp4": {{"source": "作品名称", "episode": null, "context": "这段素材的具体内容"}},
  ...
}}

字段说明：
- source：动漫/影视作品名称，若是个人拍摄/录制则填"个人拍摄"
- episode：集数、桥段或具体场景（如"第3集高潮"、"Galo初次变身"），没有则为 null
- context：这段素材具体描述的是什么内容

注意：
- 用"第X个"、"第X-Y个"等序号引用时，对应上方文件列表的编号
- 用户未提及的文件不出现在 JSON 里
- 只输出 JSON，不要 markdown 代码块"""

    client = get_qwen_client()
    response = await client.chat.completions.create(
        model=QWEN_MAX,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(f"[Manager] parse_video_intent JSON 解析失败，原始输出：\n{raw[:300]}")
        return {}

    # 只保留文件名在 filenames 里的条目，防止模型幻觉
    valid = {k: v for k, v in result.items() if k in filenames}
    logger.info(f"[Manager] 意图解析完成，命中 {len(valid)}/{len(filenames)} 个文件")
    return valid


async def handle_audio_edit_instruction(
    instructions: str,
    segments,
) -> list[dict]:
    """
    Manager 将用户对音频段落的自然语言修改指令转发给音频分析模块，
    解析为结构化编辑列表。

    Args:
        instructions: 用户描述，如 "把 pre_chorus 能量调高至 6"
        segments: AudioSegment 列表

    Returns:
        [{"name": seg_name, "field": "energy"|"mood", "value": ...}]
    """
    from utils.clients import QWEN_MAX, get_qwen_client

    seg_list = "\n".join(f"- {s.name}: energy={s.energy}, mood={s.mood}" for s in segments)
    prompt = f"""你是音频段落编辑助手（Manager 转发的用户指令）。

当前段落列表：
{seg_list}

用户的修改请求：
{instructions}

请将修改请求解析为 JSON 数组，每项包含：
- name: 段落名（必须是上方列表中的名称，严格匹配）
- field: "energy"（能量，整数 1-10）或 "mood"（情绪，字符串）
- value: 新值

只输出 JSON 数组，不要任何其他文字。
示例：[{{"name": "pre_chorus", "field": "energy", "value": 6}}, {{"name": "chorus", "field": "mood", "value": "euphoric"}}]"""

    client = get_qwen_client()
    response = await client.chat.completions.create(
        model=QWEN_MAX,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    result = json.loads(clean)

    valid_names = {s.name for s in segments}
    edits = [
        e for e in result
        if isinstance(e, dict)
        and e.get("name") in valid_names
        and e.get("field") in ("energy", "mood")
    ]
    logger.info(f"[Manager] 音频编辑指令解析完成，有效编辑 {len(edits)} 条")
    return edits


async def handle_plan_feedback(
    feedback: str,
    render_plan,
    scene_table=None,
) -> str:
    """
    Manager 将用户对剪辑方案的自然语言反馈转发给剪辑规划师，
    返回适合注入 current_errors 的结构化描述。

    Args:
        feedback: 用户反馈，如 "第3段太快了，感觉和情绪不搭"
        render_plan: 当前 RenderItem 列表
        scene_table: 可选的 SceneItem 列表，用于提供上下文

    Returns:
        注入 current_errors 的字符串描述
    """
    from utils.clients import QWEN_MAX, get_qwen_client

    plan_summary = "\n".join(
        f"#{r.order}: 音频 {r.audio_start:.1f}-{r.audio_end:.1f}s | 场景#{r.scene_id} | "
        f"素材 {r.clip_start:.1f}-{r.clip_end:.1f}s | 速度 {r.speed_factor:.1f}x"
        for r in render_plan
    ) if render_plan else "（无规划数据）"

    prompt = f"""你是剪辑方案评审助手（Manager 转发的用户反馈）。

当前剪辑方案：
{plan_summary}

用户反馈：
{feedback}

请将用户反馈转化为剪辑规划师能理解的具体修改要求，简洁明确（1-3 条），
直接输出文字，不要 JSON 或其他格式。"""

    client = get_qwen_client()
    response = await client.chat.completions.create(
        model=QWEN_MAX,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.2,
    )
    result = response.choices[0].message.content.strip()
    logger.info(f"[Manager] 规划反馈解析完成：{result[:80]}")
    return result


async def precheck_node(state: GraphState) -> dict:
    """
    预检节点：验证输入文件存在，初始化 runtime_config。
    MVP 阶段不触发用户交互，只打印警告。
    """
    import subprocess

    music_path = state["music_path"]
    video_paths = state["video_paths"]
    editing_style = state.get("editing_style", "visual_driven")

    logger.info(f"[Manager] 预检开始")
    logger.info(f"  音乐: {music_path}")
    logger.info(f"  视频: {len(video_paths)} 个文件")

    # 检查文件存在
    if not Path(music_path).exists():
        raise FileNotFoundError(f"音乐文件不存在: {music_path}")
    for vp in video_paths:
        if not Path(vp).exists():
            raise FileNotFoundError(f"视频文件不存在: {vp}")

    # 获取时长
    def get_duration(path: str) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    music_duration = get_duration(music_path)
    video_duration = sum(get_duration(vp) for vp in video_paths)

    logger.info(f"  音乐时长: {music_duration:.1f}s，视频素材总时长: {video_duration:.1f}s")

    ratio = video_duration / music_duration if music_duration > 0 else 0
    if ratio < 0.3:
        logger.warning(f"  ⚠️ 素材严重不足（{ratio:.1%}），效果可能较差")
    elif ratio < 0.8:
        logger.warning(f"  ⚠️ 素材略显不足（{ratio:.1%}），建议补充素材")

    # 初始化 runtime_config
    runtime_config = load_config(editing_style)  # type: ignore

    return {
        "runtime_config": runtime_config,
    }


async def deliver_node(state: GraphState) -> dict:
    """
    交付节点：任务完成，输出结果路径。
    """
    output_path = state.get("output_video_path", "")
    logger.info("=" * 60)
    logger.info(f"[Manager] ✅ 混剪完成：{output_path}")
    logger.info("=" * 60)
    return {}