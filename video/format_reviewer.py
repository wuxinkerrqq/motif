from __future__ import annotations

from loguru import logger
from models.video import SceneItem


def validate_scene_item(data: dict) -> tuple[bool, list[str]]:
    """
    校验单个 scene_table 条目的格式合法性。
    纯代码，无 LLM。

    返回：(是否通过, 错误列表)
    """
    errors = []

    # 必填字段（与新版 SceneItem 对齐）
    required_fields = [
        "scene_id", "source_file", "start", "end", "duration",
        "scene_index", "keyframes", "scene_description",
        "mood", "visual_profile",
    ]
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"缺少必填字段：{field}")

    if errors:
        return False, errors

    # visual_profile 子字段校验
    vp = data.get("visual_profile", {})
    if not isinstance(vp, dict):
        errors.append("visual_profile 类型错误，必须是 dict")
        return False, errors
    for k in ("valence", "arousal", "dominance"):
        v = vp.get(k)
        if v is not None and not (0.0 <= float(v) <= 1.0):
            errors.append(f"visual_profile.{k} 超出 [0,1]：{v}")

    mood = data.get("mood")
    if not mood or not isinstance(mood, str):
        errors.append("mood 为空或类型错误")

    # 时间戳合法性
    start = data.get("start", 0)
    end = data.get("end", 0)
    if end <= start:
        errors.append(f"end ({end}) 必须大于 start ({start})")

    # keyframes 列表非空
    keyframes = data.get("keyframes", [])
    if not keyframes:
        errors.append("keyframes 列表为空")

    return len(errors) == 0, errors


def validate_scene_table(scene_table: list[dict]) -> tuple[bool, list[dict]]:
    """
    批量校验 scene_table，过滤掉不合格的条目。

    返回：(是否全部通过, 过滤后的合格条目列表)
    """
    valid = []
    invalid_count = 0

    for item in scene_table:
        ok, errors = validate_scene_item(item)
        if ok:
            valid.append(item)
        else:
            invalid_count += 1
            scene_id = item.get("scene_id", "?")
            logger.warning(
                f"  [format_reviewer] scene {scene_id} 格式不合格，跳过：{errors}"
            )

    all_passed = invalid_count == 0
    if not all_passed:
        logger.warning(
            f"  [format_reviewer] {invalid_count} 个场景格式不合格，已过滤"
        )

    return all_passed, valid