# 产品需求：仅基于视频电话录屏的会议纪要管线

## 权威边界

本文是中文产品契约，描述用户目标、验收边界和身份判断规则。工程实现细节以 `docs/architecture.md` 为准，实施顺序以 `docs/project-plan.md` 为准，验收追踪以 `docs/acceptance-matrix.md` 为准。

旧文件 `docs/solution.zh.md` 保留为早期方案背景，不再作为唯一事实源。

## 目标

输入只有一份视频电话录制文件，例如本地 `.mov` 或 `.mp4`。系统不依赖 Zoom、Google Meet、Teams、腾讯会议或其他会议平台导出的参会者列表、字幕、聊天记录、说话人标注或服务端元数据。

输出必须形成证据闭环：

- 逐字稿：每段发言包含开始时间、结束时间、speaker label、可选真实姓名、置信度和文本。
- 中文会议纪要：结论、风险、Action Item、负责人候选和待确认项必须能回跳到时间戳。
- 关键帧：保留页面切换、屏幕共享上下文、活跃说话人视觉证据和关键词附近画面。
- 质量报告：列出 ASR、说话人分离、OCR、视觉身份映射和摘要的状态与限制。
- 复核队列：低置信、未知说话人、混合声纹簇、缺少视觉证据的片段必须进入人工复核。

## 用户

- 需要从本地会议录屏生成会议纪要的人。
- 需要保留证据链，不能只要摘要的人。
- 录屏包含隐私内容，不希望默认上传到 SaaS 的人。
- 愿意在关键身份映射处做少量人工复核的人。

## 非目标

- 不做会议平台账号集成。
- 不把声音直接推断为真实姓名。
- 不把绿色单元测试解释为完整媒体管线质量保证。
- 不把云服务输出作为唯一事实来源。
- 不提交任何私人录屏、转写、截图、OCR、纪要或真实参会人名单。

## 功能需求

### R1 本地音视频预处理

系统应从录屏中提取 16 kHz mono WAV、媒体元数据、抽帧结果和统一秒级时间轴。所有后续 artifact 必须使用同一个时间基准。

### R2 ASR

系统应优先使用本机 `mlx-whisper` 路径进行中文、英文或中英混合转写。ASR 输出必须保留 segment 级时间戳。

### R3 说话人分离

系统应支持本地无 token 的 SpeechBrain ECAPA clustering，并保留 pyannote 作为可选后端。说话人分离只能产生 `Speaker N`、声纹簇或已验证姓名，不能单凭声音生成真实姓名。

### R4 身份映射

真实姓名只能来自以下证据：

- 明确的 voice enrollment 片段。
- 人工审核过的 participant map。
- segment 级视觉证据，例如姓名牌、头像名、活跃说话人高亮。
- 经过多处视觉证据支撑的 cluster fallback，并且必须标注它不是逐段证据。

如果证据不足，系统必须保留 `Speaker N` 或 `Speaker Unknown` 并写入复核队列。

### R5 视觉证据

系统应支持从关键帧或抽帧记录中识别会议 UI 的活跃说话人高亮。视觉高亮需要按会议 UI 布局校准，屏幕共享中的 presenter 边框不能默认等同于 active speaker。

### R6 纪要与输出

系统应输出：

- `minutes.md`
- `transcript.json`
- `transcript.md`
- `speaker_turns.json`
- `keyframes.json`
- `keyframes/`
- `ocr.json`
- `quality_report.md`
- `review_queue.md`
- `speaker_samples.md`

每条纪要候选应包含时间戳和文本证据；有视觉证据时应附 frame ref。

## 质量门槛

- 单元测试通过。
- 轻量 CI 不安装 Apple Silicon 专属依赖。
- 本机完整验证必须运行 `meeting-minutes doctor`。
- 私人会议 artifact 不得出现在 Git diff。
- 身份映射不能越过证据边界。
- 没有公开媒体 fixture 覆盖的核心 ASR、diarization 和视觉质量，必须在验收矩阵中标为人工验证。

## 平台支持

| 平台 | 支持范围 |
| --- | --- |
| Apple Silicon macOS | 完整本地管线目标平台，包含 `mlx-whisper`、SpeechBrain、抽帧、OCR 和报告生成。 |
| Linux CI | 只运行轻量单元测试，不安装 `mlx-whisper`、SpeechBrain 或 torchaudio。 |
| Linux GPU | 未来可作为可选后端，不是当前默认目标。 |
| SaaS API | 只可作为质量对照，不作为身份事实源。 |
