# Motif Quantitative Evaluation

**Test case**: Fate × GravityWall (Hiroyuki Sawano)
**Music duration**: 92.90s | **BPM**: 118
**Cuts evaluated**: 53 | **Beats**: 164 | **Scenes**: 110

## m5 — Beat-Cut Sync (σ=100ms Gaussian)

Measures whether cut points fall on musical beats.

**Result: m5 = 0.9815** — cut points are tightly aligned with beat timestamps.

## Methodology

- Formula: `mean(exp(-Δt²/2σ²))`, σ=0.1s, Δt = min distance from each cut to nearest beat in `beat_array`.
- Beats from librosa default beat tracker.
- Beat-snap tolerance set to half-beat (0.25s @ 118 BPM); cuts beyond half a beat from any beat would be ambiguous (closer to next beat).
- Metric definition follows DIRECT (arXiv:2604.04875).
