# 同帧侧栏头像实名映射

`roster-avatar-identify` 面向 Discord Huddle 一类录屏：活动发言卡片有清晰的高亮边框，但卡片内没有可读姓名；同一帧的侧栏参会人列表同时显示姓名和头像。该命令把活动卡片中的头像与**同一帧**侧栏中的已命名头像匹配，并把通过校准的结果写入对应转写片段。

它不是固定坐标映射，不依据某个人常驻在某个格子，也不会把侧栏 OCR 文本直接当作当前说话人的名字。

## 证据边界

- 姓名只允许来自配置内的精确参会人白名单；`John` 与 `johnjr0507` 是两个不同身份，禁止模糊匹配。
- 每帧至少需要识别出三名不同的侧栏参会人，才允许形成头像候选集。
- 活动画面必须恰好有一个高亮框，并且该框内是紧凑头像，不是摄像头画面、屏幕共享或低信息默认头像。
- 匹配必须同时超过绝对相似度和第一、第二候选的分差阈值。
- 每个转写片段至少需要两帧同名支持，并满足配置的投票比例。
- 自动实名前必须有至少三个人工复核、时间分离且身份不同的锚点；每个锚点都必须复现正确姓名，否则整轮自动映射关闭。
- 默认只自动映射通过锚点验证的身份。侧栏中其他姓名即使出现为候选，也保留为待复核，直到各自拥有通过的锚点。
- 该来源固定标记为 `visual_roster_avatar_match`，不参加同会话声纹注册，也不向说话人聚类直接传播姓名。
- 已有更强的直接姓名牌、人工确认或声纹实名不被覆盖；不一致会写入冲突记录。
- 已校准的同帧侧栏证据可以纠正较弱的聚类传播或同会话声纹标签；后续动态姓名牌、静态视觉、头像模板和同会话声纹阶段在没有更强的直接姓名牌共识时都必须保留该同帧证据。
- 不同抽样请求若收敛到同一个毫秒级视频时间点，只作为一帧证据计票，不能重复满足片段内的多帧门槛。
- 已有通过门禁的映射重跑时，如果新一轮校准失败，旧的活动实名产物和转写保持不变；失败尝试写入独立的 `attempt` 审计文件。即使全局校准通过，某个片段没有达到新的多帧共识时也保留其旧实名。只有该片段的新证据完整通过后，才替换活动实名产物。

这条路径在没有足够侧栏姓名、活动高亮不唯一、画面布局变化或校准失败时保持匿名，而不是猜测实名。

## 配置

```json
{
  "participants": ["Billy", "John", "johnjr0507", "Xin"],
  "settings": {
    "samples_per_segment": 3,
    "minimum_supporting_frames": 2,
    "minimum_roster_identities": 3,
    "minimum_similarity": 0.20,
    "minimum_margin": 0.05,
    "minimum_segment_vote_share": 0.666,
    "minimum_ocr_confidence": 0.75,
    "min_active_score": 0.72,
    "minimum_anchor_seconds_separation": 15,
    "anchor_only_identities": true,
    "search_region": [0.20, 0.10, 0.98, 1.0]
  },
  "layouts": [
    {
      "name": "discord-huddle-left-roster",
      "start": 0,
      "end": 3600,
      "roster_region": [0.12, 0.56, 0.30, 0.90],
      "avatar": {
        "size_multiplier": 0.96,
        "gap_multiplier": 0.22,
        "vertical_offset_multiplier": 0.0
      }
    }
  ],
  "reviewed_anchors": [
    {"time": 120.0, "name": "Billy", "reviewed": true},
    {"time": 900.0, "name": "John", "reviewed": true},
    {"time": 1800.0, "name": "Xin", "reviewed": true}
  ]
}
```

示例中的相似度和分差只是配置格式说明，不能直接复制到其他录屏。`roster_region`、头像几何和阈值必须来自该录屏的人工画面复核与锚点重放；布局改变时应新增时间窗口，而不是沿用旧几何参数。三个抽帧且需要两帧支持时，投票比例应配置为不大于 `0.666`；`0.67` 会严格要求三帧全通过。

## 运行

先完成常规本地转写和说话人分离，再应用侧栏头像证据：

```bash
uv run meeting-minutes roster-avatar-identify \
  --output-dir /absolute/path/to/meeting-output \
  --roster-avatar-profile /absolute/path/to/roster-avatar-profile.json
```

输出文件：

- `roster_avatar_identity.json`：锚点、校准结论、逐帧候选、相似度、分差和弃权原因。
- `roster_avatar_identity_samples.json`：每个转写片段抽取的时间点。
- `roster_avatar_identity_anchors.json`：人工复核锚点的取帧请求。
- `roster_avatar_identity_ocr.json`：仅侧栏区域的 OCR 结果。
- `roster_avatar_identity_report.md`：校准门禁和实名映射汇总。
- `transcript.json` 与 `transcript.md`：仅在校准和片段共识都通过时写入实名。
- `roster_avatar_identity.attempt.json` 与 `roster_avatar_identity_attempt_report.md`：已有活动映射时，失败重跑的独立审计结果，不会替换活动产物。

## 复核要求

首次使用一种新的会议 UI 时，必须查看三张不同人的锚点原图，并检查侧栏姓名和活动头像在同一帧中的对应关系。校准报告只证明程序复现了这些锚点，不替代对布局变化、相机切换和共享屏幕的人工检查。
