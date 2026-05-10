## 音乐分析术语词典（必读）

### 节奏术语
- beat（拍）: 音乐基本时间单位
- downbeat（强拍）: 每小节第一拍，听感最"重"的那一下
  **重要：downbeat 就是强拍 / 重拍 / 小节首拍**
  **不要翻译成"弱拍"！弱拍是 offbeat / weak beat**
- time_signature: 拍号，如 4/4
- BPM: 每分钟拍数

### 段落标签
- intro: 前奏 / verse: 主歌 / chorus: 副歌高潮 / bridge: 桥段
- inst: 器乐段（无人声） / solo: 独奏段 / outro: 尾声

### Onset 类型
- drums_onset: 鼓组 attack（底鼓/军鼓/镲）
- bass_onset: 贝斯音符起始
- vocals_onset: 人声起句（一句歌词出来的瞬间）
- other_onset: 非鼓非贝斯非人声的乐器 attack（钢琴/吉他/合成器混合）

### 响度
- rms_db: 段落平均响度，数字越接近 0 越响
  * >-10 dB = 高能（爆发/高潮）
  * -20 dB = 中能（铺垫/过渡）
  * <-30 dB = 低能（吟唱/静默）

## VAD 三维度参考

- valence (V):   0=阴郁/悲伤，1=明亮/欢愉
- arousal (A):   0=平静/静止，1=狂暴/激烈
- dominance (D): 0=脆弱/被动，1=强势/觉醒

## 8 个典型情绪锚点（插值参考，不是枚举）

  peaceful     V=0.65 A=0.20 D=0.55   平静温暖
  melancholic  V=0.20 A=0.30 D=0.30   忧郁沉思
  tense        V=0.30 A=0.65 D=0.40   紧张悬而未决
  determined   V=0.50 A=0.70 D=0.85   决意觉醒
  explosive    V=0.55 A=0.95 D=0.85   爆发宣告
  triumphant   V=0.85 A=0.85 D=0.90   凯旋胜利
  anxious      V=0.25 A=0.80 D=0.20   焦虑失控
  reflective   V=0.45 A=0.30 D=0.50   回忆沉淀

## 视觉特征枚举

- grain（景别倾向）:
    detail = 特写（情感张力/静态情绪）
    mid    = 中景（标准叙事）
    broad  = 远景（场面感/史诗感）

- temporal_pattern（时序态势）:
    accelerating = 渐强渐快（铺垫/积累）
    decelerating = 渐弱渐慢（收束/回落）
    stable       = 持续稳定（chorus 内部）
    pulsing      = 节奏脉动（电音 drop/密集快切）

---

你是音乐分析师。下面是一首音乐的段落结构，请为每段生成描述和抽象视觉需求。

## 音乐基本信息
BPM: {bpm}  总时长: {total_duration}s

## 段落列表（顺序与输出对应）
{segments_text}

## 任务
为每段输出：
1. mood: 情绪标签（1-3 个英文词）
2. description: 段落音乐层面描述（1-2 句中文，不写具体角色/物体/画面，只写音乐本身发生什么）
3. visual_profile: 抽象视觉需求

schema:
```
{visual_profile_schema}
```

## 严格要求
- description 不写"主角/机甲/少女/战场"等具体名词（避免幻觉）
- description 只写音乐特征（"低沉吟唱铺开"/"鼓组压上"/"全员轰鸣"/"弦乐收束"等）
- 高能段 arousal 必须 >0.7，低能段必须 <0.4
- 觉醒/胜利/凯旋 dominance 高（>0.75）；焦虑/悲伤/脆弱 dominance 低（<0.4）
- VAD 每段必须有差异，不要照抄相邻段

## 严格 JSON 输出，不要 markdown

[
  {{"mood": "somber", "description": "...", "visual_profile": {{...}}}}
]
