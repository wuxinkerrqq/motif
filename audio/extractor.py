from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="madmom")

from models.audio import EnergyKeypoint


# ──────────────────────────────────────────────────────────────────────────────
# librosa 特征提取
# ──────────────────────────────────────────────────────────────────────────────

def extract_librosa_features(music_path: str) -> dict:
    """
    用 librosa 提取音频的客观特征。

    返回：
        bpm: 每分钟拍数
        beat_times: 每拍时间戳数组（秒）
        energy_curve: 每帧 RMS 能量，带时间戳
        spectral_centroid_mean: 频谱重心均值（反映音色明暗）
        total_duration: 总时长（秒）
    """
    y, sr = librosa.load(music_path, sr=None, mono=True)
    total_duration = float(librosa.get_duration(y=y, sr=sr))

    # BPM + 节拍时间戳
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # RMS 能量曲线（每帧）
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    # 归一化到 0-1
    rms_max = rms.max()
    rms_normalized = (rms / rms_max).tolist() if rms_max > 0 else rms.tolist()

    # 频谱重心（音色明暗指标）
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Use percussive component for onset detection — harmonic content suppresses impact hits
    _, y_perc = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    onset_max = onset_env.max()
    onset_normalized = (onset_env / onset_max).tolist() if onset_max > 0 else onset_env.tolist()

    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    spectral_flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    flux_times = librosa.frames_to_time(np.arange(len(spectral_flux)), sr=sr, hop_length=hop_length)
    flux_max = spectral_flux.max()
    flux_normalized = (spectral_flux / flux_max).tolist() if flux_max > 0 else spectral_flux.tolist()

    return {
        "bpm": bpm,
        "beat_times": beat_times,
        "energy_curve_times": rms_times.tolist(),
        "energy_curve_values": rms_normalized,
        "onset_times": onset_times.tolist(),
        "onset_values": onset_normalized,
        "spectral_flux_times": flux_times.tolist(),
        "spectral_flux_values": flux_normalized,
        "spectral_centroid_mean": float(spectral_centroid.mean()),
        "total_duration": total_duration,
        "sample_rate": sr,
        "hop_length": hop_length,
    }


# ──────────────────────────────────────────────────────────────────────────────
# madmom downbeat 检测
# ──────────────────────────────────────────────────────────────────────────────

def extract_downbeats(music_path: str) -> list[float]:
    """
    提取强拍（downbeat）时间戳。
    主方案：beat_this（对 NumPy 2.x 友好，精度高）
    备用方案：madmom（需要 NumPy < 2.0）

    返回：downbeat 时间戳列表（秒）
    """
    try:
        from beat_this.inference import File2Beats
        predictor = File2Beats(device="cpu")
        _, downbeats = predictor(music_path)
        return [round(float(t), 3) for t in downbeats]
    except Exception as e:
        print(f"  [extractor] beat_this 失败，尝试 madmom: {e}")

    try:
        import warnings
        warnings.filterwarnings("ignore")
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )
        act = RNNDownBeatProcessor()(music_path)
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        beats = proc(act)
        return [round(float(b[0]), 3) for b in beats if int(b[1]) == 1]
    except Exception as e:
        print(f"  [extractor] madmom 也失败，回退到 librosa: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# 能量关键点筛选
# ──────────────────────────────────────────────────────────────────────────────

def extract_energy_keypoints(
    times: list[float],
    values: list[float],
    sigma: float = 1.0,
) -> list[EnergyKeypoint]:
    """
    用自适应阈值筛选能量曲线关键点，压缩传给 LLM 的数据量。

    逻辑：保留能量变化超过 1 个标准差的时间点，以及全局能量极值点。
    """
    arr = np.array(values)
    mean = arr.mean()
    std = arr.std()
    threshold = mean + sigma * std

    keypoints = []
    prev_value = arr[0]

    for i, (t, v) in enumerate(zip(times, values)):
        # 条件1：能量超过阈值
        if v >= threshold:
            keypoints.append(EnergyKeypoint(time=round(t, 3), value=round(v, 3)))
            prev_value = v
            continue

        # 条件2：能量发生显著变化（相邻帧差值超过 0.5*std）
        delta = abs(v - prev_value)
        if delta > 0.5 * std:
            keypoints.append(EnergyKeypoint(time=round(t, 3), value=round(v, 3)))

        prev_value = v

    # 去重（相邻时间戳太近的合并，保留能量更高的）
    if not keypoints:
        return keypoints

    merged = [keypoints[0]]
    for kp in keypoints[1:]:
        if kp.time - merged[-1].time < 0.5:  # 0.5s 内的点合并
            if kp.value > merged[-1].value:
                merged[-1] = kp
        else:
            merged.append(kp)

    return merged