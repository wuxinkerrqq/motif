## 导演意图

- 剪辑风格：{editing_style}
- 背景信息：{background_info}
- 用户反馈：{user_feedback}

---

## 音频地图

**基本信息**
- BPM：{bpm}
- 总时长：{total_duration} 秒

**段落结构**
{segments_json}

**特殊事件**
{special_events_json}

---

## 素材库

**源视频列表**
{source_files_info}

**整体统计**
- 总场景数：{scene_count} 个
- 情绪分布：{mood_distribution}
- 动感密度分布：{density_distribution}
- 时长统计：{duration_stats}

---

## 任务

请为以上每个音乐段落规划叙事意图，输出 JSON 数组。

注意：
1. audio_start 和 audio_end 必须和段落结构里的数据完全一致，不要自行修改
2. intent 要具体描述画面内容，不要只写情绪词，要说明动作、场景、氛围
3. 先确定 anchor=true 的锚点段落，再规划其余
4. 根据素材来源决定 temporal_mode 和 prefer_sources（见 system prompt 说明）
