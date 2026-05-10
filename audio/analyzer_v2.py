"""
Audio Analyzer v2 - L1 信号层

输入: 音乐文件路径
输出: dict（兼容 eval_model/audio/results/<name>/analysis.json 格式）
  - bpm / time_signature / total_duration
  - beats / downbeats
  - segments (label / start / end / beats_count / rms / rms_db)
  - rms_quantiles (lower_10 / upper_90)
  - stem_onsets (drums / bass / vocals / other 各自的 onset list)
  - dense_events (按时间排序的统一事件流)

外部副作用:
  - 在 stems_dir/<name>/ 下写入 4 个 stem WAV

实现说明:
  - All-In-One 分析 beats / downbeats / functional segments / BPM
  - Demucs htdemucs 源分离 4 个 stem
  - librosa 对每个 stem 做 onset detection，保留 top 15% 强度
  - RMS 能量（段落级 + 全局 quantiles，对标 DIRECT 的 MusicProfile）
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
import soundfile as sf
import torch

if TYPE_CHECKING:
    pass

STEM_NAMES = ["drums", "bass", "vocals", "other"]
ONSET_PERCENTILE = 85  # 只保留 top 15% 强度的 onset
DEFAULT_SR = 22050


# ── All-In-One ───────────────────────────────────────────────────────────────

def _run_allinone(music_path: Path):
    import allin1
    result = allin1.analyze(music_path)
    if isinstance(result, list):
        result = result[0]
    return result


# ── Demucs 源分离 ─────────────────────────────────────────────────────────────

def _run_demucs(music_path: Path, output_dir: Path) -> dict[str, Path]:
    """Demucs 源分离，返回 {stem_name: wav_path}"""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    output_dir.mkdir(parents=True, exist_ok=True)

    model = get_model("htdemucs")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    wav = AudioFile(music_path).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    if torch.cuda.is_available():
        wav = wav.cuda()

    with torch.no_grad():
        sources = apply_model(model, wav[None], progress=True)[0]

    sources = sources * ref.std() + ref.mean()
    sources = sources.cpu().numpy()

    stem_paths = {}
    for i, name in enumerate(model.sources):
        out_path = output_dir / f"{name}.wav"
        sf.write(str(out_path), sources[i].T, model.samplerate)
        stem_paths[name] = out_path

    return stem_paths


# ── 每 stem onset detection ──────────────────────────────────────────────────

def _detect_onsets_for_stem(
    wav_path: Path, percentile: int = ONSET_PERCENTILE,
) -> list[dict]:
    y, sr = librosa.load(str(wav_path), sr=DEFAULT_SR)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=onset_env, backtrack=True, units="frames",
    )

    if len(onset_frames) == 0:
        return []

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strengths = onset_env[onset_frames]

    max_str = onset_strengths.max()
    if max_str > 0:
        onset_strengths = onset_strengths / max_str

    threshold = np.percentile(onset_strengths, percentile)
    mask = onset_strengths >= threshold
    onset_times = onset_times[mask]
    onset_strengths = onset_strengths[mask]

    return [
        {"time": round(float(t), 3), "strength": round(float(s), 3)}
        for t, s in zip(onset_times, onset_strengths)
    ]


# ── RMS 能量 ─────────────────────────────────────────────────────────────────

def _compute_rms_for_segment(
    y: np.ndarray, sr: int, start: float, end: float,
) -> float:
    start_sample = max(0, int(start * sr))
    end_sample = min(len(y), int(end * sr))
    if start_sample >= end_sample:
        return 0.0
    segment = y[start_sample:end_sample]
    return float(np.sqrt(np.mean(segment ** 2)))


def _compute_rms_quantiles(y: np.ndarray, sr: int) -> dict:
    rms_feature = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    return {
        "lower_10": round(float(np.quantile(rms_feature, 0.1)), 6),
        "upper_90": round(float(np.quantile(rms_feature, 0.9)), 6),
    }


# ── dense_events 合并 ────────────────────────────────────────────────────────

def _build_dense_events(
    downbeats: list[float],
    segments: list[dict],
    stem_onsets: dict[str, list[dict]],
) -> list[dict]:
    events = []
    for t in downbeats:
        events.append({"time": round(t, 3), "type": "downbeat"})

    for seg in segments:
        events.append({
            "time": round(seg["start"], 3),
            "type": "segment_boundary",
            "label": seg["label"],
        })

    for stem_name, onsets in stem_onsets.items():
        for o in onsets:
            events.append({
                "time": o["time"],
                "type": f"{stem_name}_onset",
                "strength": o["strength"],
            })

    events.sort(key=lambda e: (e["time"], e["type"]))
    return events


# ── 主入口 ───────────────────────────────────────────────────────────────────

def run_analysis_v2(music_path: str | Path, stems_dir: str | Path) -> dict:
    """
    L1 信号层完整分析

    Args:
        music_path: 音乐文件路径
        stems_dir:  Demucs 输出目录（每个 stem 一个 WAV）

    Returns:
        dict（同 eval_model analysis.json 格式）
    """
    music_path = Path(music_path)
    stems_dir = Path(stems_dir)

    # 1. All-In-One
    aio_result = _run_allinone(music_path)
    beats = [round(float(b), 3) for b in aio_result.beats]
    downbeats = [round(float(b), 3) for b in aio_result.downbeats]
    bpm = round(float(aio_result.bpm), 1)

    beat_positions = aio_result.beat_positions if hasattr(aio_result, "beat_positions") else []
    if beat_positions:
        time_sig = f"{sorted(beat_positions)[int(0.95 * len(beat_positions))]}/4"
    else:
        time_sig = "4/4"

    # 2. Demucs 源分离
    stem_paths = _run_demucs(music_path, stems_dir)

    # 3. 每 stem onset
    stem_onsets: dict[str, list[dict]] = {}
    for stem_name in STEM_NAMES:
        wav_path = stem_paths.get(stem_name)
        if wav_path and wav_path.exists():
            stem_onsets[stem_name] = _detect_onsets_for_stem(wav_path)
        else:
            stem_onsets[stem_name] = []

    # 4. RMS
    y, sr = librosa.load(str(music_path), sr=DEFAULT_SR)
    total_duration = round(float(len(y) / sr), 3)
    rms_quantiles = _compute_rms_quantiles(y, sr)

    segments_out = []
    for seg in aio_result.segments:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        rms = _compute_rms_for_segment(y, sr, seg_start, seg_end)
        rms_db = round(20 * np.log10(max(rms, 1e-10)), 1)
        beats_in_seg = [b for b in beats if seg_start <= b < seg_end]
        segments_out.append({
            "label": seg.label,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "beats_count": len(beats_in_seg),
            "rms": round(rms, 6),
            "rms_db": rms_db,
        })

    # 5. dense_events
    dense_events = _build_dense_events(downbeats, segments_out, stem_onsets)

    return {
        "music_file": music_path.name,
        "bpm": bpm,
        "time_signature": time_sig,
        "total_duration": total_duration,
        "beats": beats,
        "downbeats": downbeats,
        "segments": segments_out,
        "rms_quantiles": rms_quantiles,
        "stem_onsets": stem_onsets,
        "dense_events": dense_events,
    }
