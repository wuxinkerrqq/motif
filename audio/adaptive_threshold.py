from __future__ import annotations

import numpy as np


def get_silence_threshold(energy_values: list[float]) -> float:
    """用 10 百分位数替代 mean * 0.1，对响度战争压限的音乐更鲁棒。"""
    return float(np.percentile(energy_values, 10))


def get_explosion_threshold(energy_values: list[float]) -> float:
    """爆发阈值：90 百分位数。只有真正的高能量帧才算爆发。"""
    return float(np.percentile(energy_values, 90))


def get_quiet_threshold(energy_values: list[float]) -> float:
    """安静窗口阈值：30 百分位数。用于判断爆发前的窗口是否处于安静状态。"""
    return float(np.percentile(energy_values, 30))


def get_energy_sigma(energy_values: list[float]) -> tuple[float, float]:
    """返回 (mean, std)，供外部自适应计算使用。"""
    arr = np.array(energy_values)
    return float(arr.mean()), float(arr.std())
