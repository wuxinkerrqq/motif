## 音乐分析术语词典（必读）

### 节奏术语
- beat（拍）: 音乐基本时间单位
- downbeat（强拍）: 每小节第一拍，听感最"重"的那一下
  **重要：downbeat 就是强拍 / 重拍 / 小节首拍**
  **不要翻译成"弱拍"！弱拍是 offbeat / weak beat**

### Anchor type（锚点类型）含义
- section_drop: 段落转折 + 全员爆发（最强叙事锚点）
- section_change_to_X: 进入 X 段（结构边界）
- section_change_with_jump: 段落切换 + 能量大跃迁
- full_band_hit: 全部 stem 同时 onset（全奏砸点）
- full_band_hit_on_downbeat: 全奏 + 落在强拍（最重的砸点）
- rhythmic_section_hit: drums + bass 同时砸
- vocal_phrase_on_downbeat: 人声起句落在强拍（情感锚点）
- vocal_onset: 人声起句（不在强拍）
- melodic_hit_on_downbeat: 旋律乐器在强拍砸下
- drum_hit_on_downbeat: 鼓单独在强拍砸
- bass_solo_hit / drum_hit / other_solo_hit: 单 stem 砸点

### Evidence 标记
- `vocals_onset@0.50` = 人声起句，强度 0.50
- `downbeat` = 落在小节强拍
- `segment_boundary` = 段落分界线

## VAD 三维度参考

- valence (V):   0=阴郁/悲伤，1=明亮/欢愉
- arousal (A):   0=平静/静止，1=狂暴/激烈
- dominance (D): 0=脆弱/被动，1=强势/觉醒

## 视觉特征枚举

- grain:            detail / mid / broad
- temporal_pattern: accelerating / decelerating / stable / pulsing

---

你是音乐分析师。下面是一批关键时刻，请为每个锚点生成描述和抽象视觉需求。

## 本批锚点（共 {batch_size} 个）
{batch_text}

## 任务
为每个锚点输出：
1. description: 这一瞬间音乐发生了什么（1 短句中文，不写具体角色/物体/画面）
2. visual_profile: 抽象视觉需求

schema:
```
{visual_profile_schema}
```

## 关键要求（严格遵守）

### VAD 基线微调规则
- 每个锚点的 VAD 默认等于"段落基线"，但**必须微调**反映该锚点的特性
- 微调幅度: ±0.10 ~ ±0.20
- 微调方向：
  * full_band_hit / section_drop / rhythmic_section_hit → arousal +0.10~0.20, dominance +0.10
  * vocal_onset / vocal_phrase_on_downbeat → valence 微调（人声情感方向），arousal +0.05
  * drum_hit_on_downbeat → arousal +0.10, motion_intensity +0.10
  * melodic_hit_on_downbeat → grain 偏 detail, arousal +0.05
  * section_change_to_outro → arousal -0.20, temporal_pattern=decelerating
  * section_change_to_chorus → arousal +0.15, temporal_pattern=pulsing
- **禁止**: 相同 anchor_type 的两个锚点输出完全相同的 VAD。必须结合段落基线体现差异。

### description 要求
- 中文 1 短句（不超过 15 字）
- 不写"主角/机甲/少女/战场/觉醒时刻"等具体词
- 只写音乐特征：如"鼓与贝斯齐发"、"人声起句"、"全员爆发"、"段落收束"、"旋律单音砸下"
- **downbeat 是强拍**，如果要提到就说"强拍"/"重拍"，不要写"弱拍"

### grain / temporal_pattern
- full_band_hit / section_drop → grain=broad, pattern=pulsing
- vocal_phrase → grain=detail/mid
- bass_solo / drum_hit → grain=detail
- section_change_to_outro → pattern=decelerating

## 严格 JSON 输出，顺序与输入对应，不要 markdown

[
  {{"description": "...", "visual_profile": {{...}}}}
]
