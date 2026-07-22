# 仅基于视频电话录制的会议纪要闭环方案

> Status: historical solution background. The current product contract is
> `docs/product-requirements.zh.md`, the engineering architecture is
> `docs/architecture.md`, and the implementation sequence is
> `docs/project-plan.md`.

## 目标

输入只有一份视频电话录屏，例如 `.mov` 或 `.mp4`。系统不依赖会议平台导出的参会者、聊天、字幕或说话人元数据。

输出应形成证据闭环：

- 逐字稿：每段发言包含时间戳、speaker/name、置信度。
- 会议纪要：每条结论、风险、Action Item 都能回跳到时间戳和关键帧。
- 关键帧：保留 PPT/页面变化、关键词片段附近画面、身份识别证据。
- 质量报告：列出 ASR、OCR、diarization、identity mapping 的状态和限制。
- 复核队列：低置信、未知说话人、混合 cluster、无视觉证据片段必须进入人工复核。

## 本地优先架构

1. 音视频预处理

   - 用 ffmpeg/AVFoundation 提取 16k mono WAV。
   - 生成低分辨率代理视频或定时抽帧。
   - 统一所有 artifact 的秒级时间轴。

2. ASR

   - 默认使用 `mlx-whisper` / Whisper large-v3 或 large-v3-turbo。
   - 输出 segment 级时间戳和文本。

3. 说话人分离

   - 优先使用本地 SpeechBrain ECAPA clustering。
   - 可选使用 pyannote，但需要 Hugging Face token 和模型授权。
   - diarization 只能区分声音，不能自动知道真实姓名。

4. 身份映射

   - 明确策略：不凭声音硬猜实名。
   - 真实姓名来源只能是 voice enrollment、人工确认 participant map、或视觉证据。
   - 视频电话 UI 中的姓名牌、头像名、活跃说话人高亮可作为视觉证据。
   - 若一个 cluster 被视觉证据证明是混合 cluster，不能整簇贴一个姓名。

5. 关键帧与 OCR

   - 定时抽帧加场景变化检测。
   - OCR 提取姓名牌、页面标题、MR/文档/看板等上下文。
   - 关键词如 decision、risk、deadline、owner、approve、merge、production 附近保留关键帧。

6. 摘要与纪要

   - 先生成 evidence-first extractive notes。
   - 可选用本地 LLM 润色，但不能脱离 transcript 和关键帧证据。
   - 输出必须明确质量边界和待复核片段。

## 身份判断规则

从强到弱：

1. Voice enrollment：已知某人的清晰非重叠语音样本。
2. Segment-level visual highlight：发言时间点对应画面中某个姓名牌/头像框被高亮。
3. Reviewed cluster fallback：多处视觉证据确认某个 voice cluster 主要对应某人。
4. Participant map：人工把 `Speaker N` 映射到姓名。
5. OCR candidates only：仅作为候选，不自动实名。

风险边界：

- cluster fallback 不是逐段证据，可能吞并极短插话或重叠说话。
- 屏幕共享时 presenter 边框不一定等于 active speaker，需要单独校准。
- 没有发言高亮的可见参会者只能说明在会，不能说明说过话。
- 真实姓名未被证明时，保留 `Speaker N` 并进入复核。

## 验收标准

- ASR 覆盖主要发言，并保留时间戳。
- 主要 speaker 能稳定区分；混合 cluster 被识别并标为复核。
- 有姓名牌/高亮时，实名映射必须带 frame evidence。
- 无姓名牌、无 voice enrollment、无人工确认时，不硬猜实名。
- 纪要每条结论、Action Item、风险都有时间戳和证据来源。

## 已验证的实战经验

- 四人会议中，KMeans 的四个 cluster 不一定等于四个真实人。
- 可能出现同一个人被拆成多个 cluster，或一个 cluster 混入多个人。
- 视觉高亮比单纯声纹更适合做实名锚点。
- 对重要会议，应该把 cluster fallback 和 segment-level visual evidence 分开统计。
