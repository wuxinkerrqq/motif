from __future__ import annotations

import json
import re

from loguru import logger

from models.audio import AudioMap
from models.plan import RenderItem
from models.video import SceneItem
from utils.clients import QWEN_MAX, get_qwen_client
from utils.prompt_loader import load_and_render, load_prompt

SCORE_PASS = 80
SCORE_RETRY = 40
MAX_RETRY = 3


class SegmentIssue:
    def __init__(
        self,
        segment_name: str,
        problem: str,
        suggested_scene_ids: list[int],
        action: str = "replace",
    ):
        self.segment_name = segment_name
        self.problem = problem
        self.suggested_scene_ids = suggested_scene_ids
        self.action = action  # "replace" | "rerank"


class ReviewResult:
    def __init__(
        self,
        score: int,
        pass_: bool,
        issues: list[str],
        suggestions: list[str],
        segment_issues: list[SegmentIssue],
        material_shortage: bool,
        material_shortage_reason: str,
        dimension_scores: dict,
        raw: str,
    ):
        self.score = score
        self.pass_ = pass_
        self.issues = issues
        self.suggestions = suggestions
        self.segment_issues = segment_issues
        self.material_shortage = material_shortage
        self.material_shortage_reason = material_shortage_reason
        self.dimension_scores = dimension_scores
        self.raw = raw

    def __repr__(self):
        return (
            f"ReviewResult(score={self.score}, pass={self.pass_}, "
            f"segment_issues={len(self.segment_issues)}, "
            f"material_shortage={self.material_shortage})"
        )


async def run_reviewer(
    render_plan: list[RenderItem],
    intents: list[dict],
    audio_map: AudioMap,
    scene_table: list[SceneItem],
    background_info: str | None,
    retry_count: int = 0,
) -> ReviewResult:
    # 强制 pass
    if retry_count >= MAX_RETRY:
        logger.warning(f"[Reviewer] 已重试 {retry_count} 次，强制 pass")
        return ReviewResult(
            score=retry_count * 10, pass_=True,
            issues=["已达最大重试次数，强制通过"],
            suggestions=[],
            segment_issues=[],
            material_shortage=False,
            material_shortage_reason="",
            dimension_scores={},
            raw="forced_pass",
        )

    # 基础指标
    total_video = sum(r.clip_end - r.clip_start for r in render_plan)
    coverage = total_video / audio_map.total_duration if audio_map.total_duration > 0 else 0
    total_material = sum(s.duration for s in scene_table)
    duration_ratio = total_material / audio_map.total_duration
    short_clips = [r for r in render_plan if r.clip_end - r.clip_start < 0.5]

    logger.info(
        f"[Reviewer] 基础指标 — 覆盖率: {coverage:.1%}, "
        f"素材/音乐比: {duration_ratio:.1%}, "
        f"极短片段: {len(short_clips)} 个"
    )

    scene_lookup = {s.scene_id: s for s in scene_table}
    retrieval_summary = _build_retrieval_summary(render_plan, intents, scene_lookup)

    system_prompt = load_prompt("planning/reviewer_system.md")
    user_prompt = load_and_render(
        "planning/reviewer_user.md",
        background_info=background_info or "未提供",
        r1_understanding=audio_map.r1_understanding[:1500] if audio_map.r1_understanding else "无",
        intents_json=json.dumps(intents, ensure_ascii=False, indent=2),
        retrieval_summary=json.dumps(retrieval_summary, ensure_ascii=False, indent=2),
        scene_count=len(scene_table),
        total_video_duration=total_video,
        music_duration=audio_map.total_duration,
        duration_ratio=duration_ratio,
        coverage=coverage,
    )

    logger.info("[Reviewer] 调用 qwen-max 进行审核...")
    client = get_qwen_client()

    response = await client.chat.completions.create(
        model=QWEN_MAX,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content
    result = _parse_review_result(raw, coverage, duration_ratio, retry_count)

    logger.info(
        f"[Reviewer] 评分: {result.score}/100 | "
        f"pass: {result.pass_} | "
        f"段落问题: {len(result.segment_issues)} 个"
    )
    if result.issues:
        for issue in result.issues:
            logger.info(f"  问题: {issue}")
    if result.segment_issues:
        for si in result.segment_issues:
            logger.info(
                f"  [段落] {si.segment_name}: {si.problem} "
                f"→ 建议替换为 scene_id {si.suggested_scene_ids}"
            )

    return result


def _build_retrieval_summary(
    render_plan: list[RenderItem],
    intents: list[dict],
    scene_lookup: dict,
) -> list[dict]:
    summary = []
    for intent in intents:
        seg_start = intent["audio_start"]
        seg_end = intent["audio_end"]

        seg_items = [
            r for r in render_plan
            if r.audio_start >= seg_start - 0.5 and r.audio_start < seg_end
        ]
        display_items = seg_items[:5]

        retrieved_scenes = []
        for r in display_items:
            scene = scene_lookup.get(r.scene_id)
            if scene:
                retrieved_scenes.append({
                    "scene_id": r.scene_id,
                    "desc": scene.scene_description[:40],
                    "mood": scene.mood,
                    "arousal": round(float((scene.visual_profile or {}).get("arousal", 0.0)), 2),
                    "duration": round(r.clip_end - r.clip_start, 2),
                })

        seg_coverage = sum(
            r.clip_end - r.clip_start for r in seg_items
        )
        expected = seg_end - seg_start

        summary.append({
            "segment": intent["name"],
            "intent": intent["intent"],
            "coverage": f"{seg_coverage:.1f}s/{expected:.1f}s ({seg_coverage/expected:.0%})" if expected > 0 else "N/A",
            "retrieved": retrieved_scenes,
        })
    return summary


def _parse_review_result(
    raw: str,
    coverage: float,
    duration_ratio: float,
    retry_count: int,
) -> ReviewResult:
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        data = json.loads(clean)

        score = int(data.get("score", 50))
        score = max(0, min(100, score))

        if coverage < 0.7:
            score = min(score, 45)
        elif coverage < 0.85:
            score = min(score, 58)

        pass_ = score >= SCORE_PASS

        # 解析精确的段落问题
        segment_issues = []
        for si in data.get("segment_issues", []):
            segment_issues.append(SegmentIssue(
                segment_name=si.get("segment_name", ""),
                problem=si.get("problem", ""),
                suggested_scene_ids=[int(x) for x in si.get("suggested_scene_ids", [])],
                action=si.get("action", "replace"),
            ))

        material_shortage = bool(data.get("material_shortage", False))
        if duration_ratio < 0.5 and coverage < 0.6:
            material_shortage = True

        return ReviewResult(
            score=score,
            pass_=pass_,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            segment_issues=segment_issues,
            material_shortage=material_shortage,
            material_shortage_reason=data.get("material_shortage_reason", ""),
            dimension_scores=data.get("dimension_scores", {}),
            raw=raw,
        )

    except Exception as e:
        logger.warning(f"[Reviewer] 评审结果解析失败: {e}")
        return ReviewResult(
            score=60, pass_=True,
            issues=[f"评审结果解析失败: {e}"],
            suggestions=[],
            segment_issues=[],
            material_shortage=duration_ratio < 0.5 and coverage < 0.6,
            material_shortage_reason="",
            dimension_scores={},
            raw=raw,
        )


def build_reflection_prompt(result: ReviewResult) -> str:
    """生成有方向性的反思 prompt。"""
    lines = [
        f"上一次规划评分 {result.score}/100，以下段落需要修改（其他段落保持不变）：",
        "",
    ]

    if result.segment_issues:
        lines.append("## 需要修改的段落（填空题：只改这些，其余不动）")
        for si in result.segment_issues:
            lines.append(f"\n### {si.segment_name}")
            lines.append(f"- 问题：{si.problem}")
            if si.suggested_scene_ids:
                lines.append(f"- 建议直接使用 scene_id：{si.suggested_scene_ids}")
            lines.append(f"- 动作：{si.action}")
        lines.append("")

    if result.suggestions:
        lines.append("## 其他建议")
        for sug in result.suggestions:
            lines.append(f"- {sug}")

    return "\n".join(lines)