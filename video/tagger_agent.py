from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from config import DefaultConfig
from models.video import SceneItem
from utils.clients import GEMINI_FLASH, QWEN_VL, get_gemini_client, get_qwen_client
from utils.prompt_loader import load_and_render, load_prompt
from video.format_reviewer import validate_scene_table
from video.motion_analyzer import compute_motion_intensities
from video.scene_detector import process_video

TaggerBackend = Literal["gemini", "qwen"]

# Qwen3-VL 视频理解 pass 最多送多少帧（整体理解用，不是逐场景）
_UNDERSTANDING_MAX_FRAMES = 32


def _build_background_section(
    source_file: str,
    global_background: str | None,
    video_metadata: dict | None,
    video_understanding: str | None = None,
) -> str:
    """构建逐场景的背景信息段落：全局背景 + 视频意图描述 + 视频整体理解。"""
    base = (
        f"作品背景：\n{global_background}" if global_background
        else "无背景信息，请根据画面内容客观描述。"
    )
    if video_metadata:
        filename = Path(source_file).name
        meta = video_metadata.get(filename)
        if meta:
            base += f"\n\n【此视频素材的具体来源】\n来源：{meta.get('source', '')}"
            if meta.get("episode"):
                base += f"\n桥段：{meta['episode']}"
            base += f"\n内容：{meta.get('context', '')}"
    if video_understanding:
        base += f"\n\n【视频整体理解（AI预分析）】\n{video_understanding}"
    return base


async def _understand_video_qwen(
    source_video: str,
    video_scenes: list[dict],
    background_info: str | None,
    video_metadata: dict | None,
) -> str:
    """
    Qwen3-VL 整体看一遍视频：把该视频所有关键帧按时序排列送入模型，
    获取叙事层面的整体理解，用于后续逐场景打标签时的上下文注入。
    """
    client = get_qwen_client()
    filename = Path(source_video).name

    # 收集所有帧，按场景时序排列
    all_frames: list[str] = []
    for scene in sorted(video_scenes, key=lambda s: s["start"]):
        all_frames.extend(scene.get("keyframes", []))

    if not all_frames:
        return ""

    # 均匀采样，最多 _UNDERSTANDING_MAX_FRAMES 帧
    if len(all_frames) > _UNDERSTANDING_MAX_FRAMES:
        step = len(all_frames) / _UNDERSTANDING_MAX_FRAMES
        all_frames = [all_frames[int(i * step)] for i in range(_UNDERSTANDING_MAX_FRAMES)]

    bg_section = _build_background_section(source_video, background_info, video_metadata)

    prompt = f"""请整体观看这段视频（按时序排列的帧序列），理解其叙事内容。

背景信息：
{bg_section}

文件名：{filename}

请输出该视频的整体理解（100-200字），包含：
1. 主要角色与人物关系
2. 核心叙事主题（觉醒/战斗/情感/转折等）
3. 视觉风格特点
4. 情绪变化节点（从低到高还是持续高能等）

直接输出文字，无需 JSON。"""

    content: list[dict] = []
    for frame_path in all_frames:
        try:
            b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        except Exception as e:
            logger.warning(f"[Video Understanding] {filename} 读取帧失败：{e}")

    if not content:
        return ""

    content.append({"type": "text", "text": prompt})

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=QWEN_VL,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.2,
            )
            understanding = response.choices[0].message.content.strip()
            logger.info(f"[Video Understanding] {filename}：{understanding[:80]}…")
            return understanding
        except Exception as e:
            err_str = str(e)
            if ("data_inspection_failed" in err_str or "DataInspectionFailed" in err_str) and attempt == 0:
                # 内容审核拦截：缩减为单帧重试
                logger.warning(f"[Video Understanding] {filename} 内容审核拦截，缩减为单帧重试")
                content = [c for c in content if c["type"] == "text"]
                if all_frames:
                    try:
                        b64 = base64.b64encode(Path(all_frames[0]).read_bytes()).decode()
                        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}] + content
                    except Exception:
                        pass
                continue
            logger.warning(f"[Video Understanding] {filename} 整体理解失败（跳过）：{e}")
            return ""
    return ""


async def run_video_tagger(
    video_paths: list[str],
    stripped_video_paths: list[str],
    frames_dir: str,
    background_info: str | None = None,
    video_metadata: dict | None = None,
    concurrency: int = DefaultConfig.TAGGER_CONCURRENCY,
    scene_table_save_path: str | None = None,
    backend: TaggerBackend = "qwen",
) -> tuple[list[SceneItem], np.ndarray, list[int]]:
    logger.info(f"[Video Tagger] 开始处理 {len(stripped_video_paths)} 个视频（backend={backend}）")

    # ── 第一步：PySceneDetect 切场景 ─────────────────────────────────────────
    all_scenes: list[dict] = []
    global_scene_id = 0

    for orig_path, stripped_path in zip(video_paths, stripped_video_paths):
        scenes = process_video(stripped_path, frames_dir)
        for scene in scenes:
            scene["scene_id"] = global_scene_id
            scene["source_file"] = str(orig_path)
            scene["source_dir"] = str(Path(orig_path).parent)
            global_scene_id += 1
        all_scenes.extend(scenes)

    logger.info(f"[Video Tagger] 共检测到 {len(all_scenes)} 个场景")

    # ── 第 1.5 步：光流计算 motion_intensity（客观值）─────────────────────────
    motion_map: dict[int, float] = {}
    scenes_by_video_for_motion: dict[str, list[dict]] = {}
    for scene in all_scenes:
        scenes_by_video_for_motion.setdefault(scene["source_file"], []).append(scene)
    for src, v_scenes in scenes_by_video_for_motion.items():
        try:
            motion_map.update(compute_motion_intensities(src, v_scenes))
        except Exception as e:
            logger.warning(f"[Video Tagger] {Path(src).name} 光流计算失败：{e}")

    # ── 第二步：Qwen 整体视频理解 pass（每个源视频一次） ─────────────────────
    video_understandings: dict[str, str] = {}

    if backend == "qwen":
        # 按源视频分组
        scenes_by_video: dict[str, list[dict]] = {}
        for scene in all_scenes:
            src = scene["source_file"]
            scenes_by_video.setdefault(src, []).append(scene)

        logger.info(f"[Video Tagger] 开始整体视频理解（{len(scenes_by_video)} 个视频）")
        understand_tasks = {
            src: _understand_video_qwen(src, v_scenes, background_info, video_metadata)
            for src, v_scenes in scenes_by_video.items()
        }
        results = await asyncio.gather(*understand_tasks.values(), return_exceptions=True)
        for src, result in zip(understand_tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"[Video Tagger] {Path(src).name} 理解失败：{result}")
                video_understandings[src] = ""
            else:
                video_understandings[src] = result or ""

    # ── 第四步：并发打标签 ────────────────────────────────────────────────────
    prompt_file = "video/tagger_system_qwen.md" if backend == "qwen" else "video/tagger_system.md"
    system_prompt = load_prompt(prompt_file)

    logger.info(f"[Video Tagger] 开始并发打标签（backend={backend}）")
    semaphore = asyncio.Semaphore(concurrency)

    async def tag_scene(scene: dict) -> dict | None:
        src = scene["source_file"]
        background_section = _build_background_section(
            source_file=src,
            global_background=background_info,
            video_metadata=video_metadata,
            video_understanding=video_understandings.get(src),
        )
        motion_value = motion_map.get(scene["scene_id"])
        async with semaphore:
            if backend == "qwen":
                return await _tag_single_scene_qwen(
                    scene=scene,
                    system_prompt=system_prompt,
                    background_section=background_section,
                    motion_intensity=motion_value,
                )
            else:
                return await _tag_single_scene_gemini(
                    scene=scene,
                    system_prompt=system_prompt,
                    background_section=background_section,
                    motion_intensity=motion_value,
                )

    tasks = [tag_scene(scene) for scene in all_scenes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── 第五步：格式校验 ──────────────────────────────────────────────────────
    raw_table = []
    for scene, result in zip(all_scenes, results):
        if isinstance(result, Exception):
            logger.warning(f"  scene {scene['scene_id']} 打标签异常: {result}")
            continue
        if result is None:
            continue
        raw_table.append(result)

    _, valid_table = validate_scene_table(raw_table)
    logger.info(f"[Video Tagger] 完成，有效场景数：{len(valid_table)}/{len(all_scenes)}")

    scene_items = [_dict_to_scene_item(d) for d in valid_table]

    # ── 第六步：保存 scene_table ──────────────────────────────────────────────
    if scene_table_save_path:
        Path(scene_table_save_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            [s.model_dump() for s in scene_items],
            open(scene_table_save_path, "w", encoding="utf-8"),
            ensure_ascii=False, indent=2,
        )
        logger.info(f"[Video Tagger] scene_table 已保存: {scene_table_save_path}")

    # 不再预计算文本 embedding；下游统一用 CLIP（向量空间一致，省一次 API 调用）
    scene_ids = [s.scene_id for s in scene_items]
    scene_embeddings = np.zeros((0, 0), dtype=np.float32)

    return scene_items, scene_embeddings, scene_ids


# ── Gemini 路径 ───────────────────────────────────────────────────────────────

async def _tag_single_scene_gemini(
    scene: dict,
    system_prompt: str,
    background_section: str,
    motion_intensity: float | None = None,
) -> dict | None:
    from google.genai import types

    client = get_gemini_client()
    scene_id = scene["scene_id"]
    keyframes = scene.get("keyframes", [])

    if not keyframes:
        logger.warning(f"  scene {scene_id} 无关键帧，跳过")
        return None

    motion_hint = f"{motion_intensity:.2f}" if motion_intensity is not None else "未知"
    user_prompt = load_and_render(
        "video/tagger_user.md",
        background_section=background_section,
        scene_index=scene["scene_index"],
        duration=f"{scene['duration']:.2f}",
        source_file=Path(scene["source_file"]).name,
        motion_hint=motion_hint,
    )

    parts = []
    for frame_path in keyframes:
        try:
            img_bytes = Path(frame_path).read_bytes()
            parts.append(types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64encode(img_bytes).decode()
                )
            ))
        except Exception as e:
            logger.warning(f"  scene {scene_id} 读取帧失败：{e}")

    if not parts:
        return None

    parts.append(types.Part(text=user_prompt))

    for attempt in range(DefaultConfig.TAGGER_MAX_RETRY + 1):
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_FLASH,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=2048,
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
            parsed = _parse_tag_response(raw)
            if parsed:
                _inject_motion(parsed, motion_intensity)
                parsed.update({
                    "scene_id": scene_id,
                    "source_file": scene["source_file"],
                    "source_dir": scene.get("source_dir", ""),
                    "start": scene["start"],
                    "end": scene["end"],
                    "duration": scene["duration"],
                    "scene_index": scene["scene_index"],
                    "keyframes": keyframes,
                })
                return parsed
            else:
                logger.warning(f"  scene {scene_id} Gemini 第 {attempt+1} 次解析失败")
        except Exception as e:
            if attempt < DefaultConfig.TAGGER_MAX_RETRY:
                logger.warning(f"  scene {scene_id} Gemini 第 {attempt+1} 次请求异常，重试：{e}")
                await asyncio.sleep(1)
            else:
                logger.error(f"  scene {scene_id} Gemini 最终失败：{e}")

    return None


# ── Qwen3-VL 路径 ─────────────────────────────────────────────────────────────

async def _tag_single_scene_qwen(
    scene: dict,
    system_prompt: str,
    background_section: str,
    motion_intensity: float | None = None,
) -> dict | None:
    client = get_qwen_client()
    scene_id = scene["scene_id"]
    keyframes = scene.get("keyframes", [])

    if not keyframes:
        logger.warning(f"  scene {scene_id} 无关键帧，跳过")
        return None

    motion_hint = f"{motion_intensity:.2f}" if motion_intensity is not None else "未知"
    user_prompt = load_and_render(
        "video/tagger_user.md",
        background_section=background_section,
        scene_index=scene["scene_index"],
        duration=f"{scene['duration']:.2f}",
        source_file=Path(scene["source_file"]).name,
        motion_hint=motion_hint,
    )

    def _build_content(frames: list[str]) -> list[dict]:
        content: list[dict] = []
        for frame_path in frames:
            try:
                b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            except Exception as e:
                logger.warning(f"  scene {scene_id} 读取帧失败：{e}")
        content.append({"type": "text", "text": user_prompt})
        return content

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": _build_content(keyframes)},
    ]

    for attempt in range(DefaultConfig.TAGGER_MAX_RETRY + 1):
        try:
            response = await client.chat.completions.create(
                model=QWEN_VL,
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
            )
            raw = response.choices[0].message.content
            parsed = _parse_tag_response(raw)
            if parsed:
                _inject_motion(parsed, motion_intensity)
                parsed.update({
                    "scene_id": scene_id,
                    "source_file": scene["source_file"],
                    "source_dir": scene.get("source_dir", ""),
                    "start": scene["start"],
                    "end": scene["end"],
                    "duration": scene["duration"],
                    "scene_index": scene["scene_index"],
                    "keyframes": keyframes,
                })
                return parsed
            else:
                logger.warning(f"  scene {scene_id} Qwen3-VL 第 {attempt+1} 次解析失败，原始：{raw[:100]}")
        except Exception as e:
            err_str = str(e)
            # 内容审核拦截：缩减为单帧重试，不降级到 Gemini
            if "data_inspection_failed" in err_str or "DataInspectionFailed" in err_str:
                if len(keyframes) > 1:
                    logger.warning(f"  scene {scene_id} 内容审核拦截，缩减为单帧重试")
                    messages[1]["content"] = _build_content(keyframes[:1])
                    continue
                else:
                    logger.warning(f"  scene {scene_id} 内容审核拦截且只有单帧，跳过")
                    return None
            if attempt < DefaultConfig.TAGGER_MAX_RETRY:
                logger.warning(f"  scene {scene_id} Qwen3-VL 第 {attempt+1} 次请求异常，重试：{e}")
                await asyncio.sleep(1)
            else:
                logger.error(f"  scene {scene_id} Qwen3-VL 最终失败：{e}")

    return None


# ── 公共工具 ──────────────────────────────────────────────────────────────────

def _inject_motion(parsed: dict, motion_intensity: float | None) -> None:
    """把客观光流值写入 visual_profile.motion_intensity（覆盖 LLM 估值）。"""
    if motion_intensity is None:
        return
    vp = parsed.get("visual_profile")
    if not isinstance(vp, dict):
        vp = {}
        parsed["visual_profile"] = vp
    vp["motion_intensity"] = float(motion_intensity)


def _parse_tag_response(raw: str) -> dict | None:
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        return json.loads(clean)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def _dict_to_scene_item(d: dict) -> SceneItem:
    vp = d.get("visual_profile") if isinstance(d.get("visual_profile"), dict) else {}
    # 兼容旧 scene_table：从 editing_metrics.emotion_mood 回填 mood
    mood = d.get("mood")
    if not mood:
        em = d.get("editing_metrics")
        if isinstance(em, dict):
            mood = em.get("emotion_mood", "neutral")
        else:
            mood = "neutral"
    return SceneItem(
        scene_id=d["scene_id"],
        source_file=d["source_file"],
        source_dir=d.get("source_dir", ""),
        start=d["start"],
        end=d["end"],
        duration=d["duration"],
        scene_index=d["scene_index"],
        keyframes=d.get("keyframes", []),
        scene_description=d.get("scene_description", ""),
        characters=d.get("characters", []),
        mood=mood,
        visual_profile=vp,
        is_outro_material=bool(d.get("is_outro_material", False)),
        is_climax_material=bool(d.get("is_climax_material", False)),
    )
