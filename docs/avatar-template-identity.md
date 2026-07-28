# 基于头像模板的动态实名

`avatar-template-identify` 用于 Discord、Slack Huddle 等录屏中的动态网格会议。它不把某个屏幕位置当作某个人，而是从一张或多张已经显示卡内姓名牌的参考画面中提取头像模板，再只对同一帧中被绿色边框标记的发言卡片做匹配。

## 证据边界

- 参考模板必须来自卡片内部清晰可见的姓名牌，配置中的 `evidence` 固定为 `in_tile_nameplate`。
- 目标帧必须恰好有一个动态检测到的绿色高亮卡片。
- 每个匹配同时满足相似度和第一、第二候选的分差门槛。
- 多高亮、无高亮、无法提取头像、未通过门槛均明确弃权。
- 不沿说话人分离簇传播姓名。每个转写片段只使用自身时间范围内的画面证据。
- 直接姓名牌优先于头像模板。头像与已实名声纹不一致时保留声纹结果并写入冲突记录。
- 人工确认映射、已注册声纹、显式声纹注册以及以 `reviewed_`、`manual_`、`user_confirmed_` 开头的身份来源同样不能被头像模板覆盖；一致时只记录为佐证，不一致时写入冲突记录。

## 配置

```json
{
  "templates": [
    {
      "name": "Billy",
      "path": "/absolute/path/reference.jpg",
      "box": [0.60, 0.16, 0.94, 0.43],
      "evidence": "in_tile_nameplate"
    }
  ],
  "negative_samples": [
    {
      "description": "screen_share",
      "path": "/absolute/path/reference.jpg",
      "box": [0.32, 0.16, 0.60, 0.43]
    }
  ]
}
```

`box` 是相对于**该参考截图**的归一化左上坐标，不是后续会议画面的固定位置。每个自动实名模板至少需要三个参考帧以外的直接姓名牌验证帧。没有达到该条件的模板会显示为“仅审计”，不会自动写入转写结果。

负样本应使用同一录屏里的屏幕共享、非头像卡片或已知集合外画面。校准闸门只有在以下条件同时成立时才开放自动实名：

- 至少达到配置要求的直接姓名牌验证数量；
- 验证帧中没有错误接受；
- 负样本均未被错误接受；
- 参考帧从验证集中排除；
- 验证覆盖了与参考画面不同的卡片面积。

## 运行

先完成动态姓名牌阶段：

```bash
meeting-minutes dynamic-visual-identify \
  --output-dir /absolute/path/output \
  --dynamic-visual-profile /absolute/path/dynamic-profile.json
```

再执行头像模板阶段：

```bash
meeting-minutes avatar-template-identify \
  --output-dir /absolute/path/output \
  --avatar-template-profile /absolute/path/avatar-template-profile.json
```

输出文件：

- `avatar_template_identity.json`：逐帧得分、候选、分差、裁剪框和决定。
- `avatar_template_identity_report.md`：闸门、直接姓名牌验证、负样本和每个模板的自动实名资格。
- `transcript.json` 与 `transcript.md`：仅写入通过闸门、片段内一致的姓名。

## 当前限制

头像变化、视频画面取代头像、极低分辨率、多人同时高亮和没有姓名牌的参考画面都会降低覆盖率。系统会保持匿名或仅审计，而不会猜测实名。
