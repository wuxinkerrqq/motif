"""Quantitative evaluation: m5 Beat-Cut Sync + m6 Energy Correspondence (DIRECT paper)."""
import json
import math
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent / "eval_videos" / "fate_gravitywall"
SIGMA = 0.1  # 100ms tolerance (DIRECT paper)


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def beat_cut_sync(cuts, beats):
    """m5: mean(exp(-Δt²/2σ²)) for each cut → nearest beat."""
    scores = []
    for c in cuts:
        dt = min(abs(c - b) for b in beats)
        scores.append(math.exp(-(dt ** 2) / (2 * SIGMA ** 2)))
    return sum(scores) / len(scores), scores


def rms_at_time(t, segments):
    for seg in segments:
        if seg["start"] <= t < seg["end"]:
            return seg["rms_db"]
    return segments[-1]["rms_db"]


def energy_correspondence(plan, scene_table, segments):
    scene_lookup = {s["scene_id"]: s for s in scene_table}
    visual_motion, audio_rms = [], []
    for clip in plan:
        scene = scene_lookup.get(clip["scene_id"])
        if not scene:
            continue
        m = scene.get("visual_profile", {}).get("motion_intensity")
        if m is None:
            continue
        t_mid = (clip["audio_start"] + clip["audio_end"]) / 2
        visual_motion.append(m)
        audio_rms.append(rms_at_time(t_mid, segments))
    rho, _ = spearmanr(visual_motion, audio_rms)
    return (rho + 1) / 2, rho, len(visual_motion)


def main():
    audio_map = load_json("gravityWall.json")
    plan = load_json("render_plan.json")
    scene_table = load_json("scene_table.json")

    cuts = [c["audio_start"] for c in plan if c["audio_start"] > 0.05]
    beats = audio_map["beat_array"]

    m5, per_cut = beat_cut_sync(cuts, beats)
    m6, rho, n = energy_correspondence(plan, scene_table, audio_map["segments"])

    direct_m5, direct_m6 = 0.9869, 0.8240

    report = f"""# Motif Quantitative Evaluation

**Test case**: Fate × GravityWall (Hiroyuki Sawano)
**Music duration**: {audio_map['total_duration']:.2f}s | **BPM**: {audio_map['bpm']}
**Cuts evaluated**: {len(cuts)} | **Beats**: {len(beats)} | **Scenes**: {len(scene_table)}

## m5 — Beat-Cut Sync (σ=100ms Gaussian)

Measures whether cut points fall on musical beats.

| Method | m5 |
|---|---|
| **Motif (this work)** | **{m5:.4f}** |
| DIRECT (paper baseline) | {direct_m5:.4f} |

## m6 — Energy Correspondence (Spearman ρ → [0,1])

Measures whether visual motion intensity tracks audio energy across clips.

| Method | m6 | Raw ρ |
|---|---|---|
| **Motif (this work)** | **{m6:.4f}** | {rho:.4f} |
| DIRECT (paper baseline) | {direct_m6:.4f} | — |

## Notes

- Formula (m5): `mean(exp(-Δt²/2σ²))`, σ=0.1s, Δt = min distance from cut to nearest beat.
- Formula (m6): `(spearmanr(motion_intensity, segment_rms_db) + 1) / 2`, n={n} clips.
- Cuts at t≈0 (first clip) excluded since they trivially align with the start.
- DIRECT baselines from arXiv:2604.04875v1 Table 2 (scaled to [0,1]).
"""
    out = ROOT / "eval_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
