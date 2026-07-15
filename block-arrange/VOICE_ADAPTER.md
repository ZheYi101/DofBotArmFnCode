# 语音适配说明

## 当前结论

官方 I2C 语音识别模块通过 `0x0f` 地址读取识别结果，只会返回预先录入短语
对应的数字 ID。它适合做唤醒词、停止、确认等固定命令，不能直接识别自由组合的
多步骤中文句子。

`references/DOFBOT_python_source/voice/voice_web.py` 已经预留网络 ASR 接口，
但当前 `call_asr()` 仍固定返回“启动堆积木”，尚未接入真实语音识别服务。

因此当前实现把语音系统拆成两层：

1. ASR 层负责把麦克风录音转成完整中文文本。
2. `voice_adapter.py` 负责把文本解析成有顺序的机械臂动作。

## 执行模型

以下文本：

```text
你好，扫描下当前场景，然后把红色方块放到蓝色方块上面，然后把蓝色方块放到红色方块左边
```

首先解析为：

```text
1. resync
2. move red above blue
3. move blue left of red
```

第三步执行时，现有 `move` 逻辑发现蓝色方块被红色方块压住，会从历史中自动
撤销第二步，然后继续执行第三步。`undo` 是运行时根据真实场景状态产生的动作，
不需要语音解析器提前硬编码。

## 命令

只解析并显示计划，不操作机械臂：

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_block_arrange_over_ssh.ps1 voice --text "你好，扫描下当前场景，然后把红色方块放到蓝色方块上面，然后把蓝色方块放到红色方块左边"
```

确认识别文本和计划后执行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_block_arrange_over_ssh.ps1 voice --text "你好，扫描下当前场景，然后把红色方块放到蓝色方块上面，然后把蓝色方块放到红色方块左边" --execute
```

## 安全约束

- 默认只显示计划，必须显式添加 `--execute` 才会运动。
- 整句话会先完整解析；存在未知片段时，一步也不会执行。
- 明确要求小于 8cm 的水平距离时，会在扫描和移动前拒绝整个任务。
- 每个动作按顺序执行，任一步失败后不会继续执行后续动作。
- 自动撤销仍依赖 `scene_state.json` 和 `undo_history.json` 与真实场景一致。

## 后续接入 ASR

推荐先在性能更强的 Windows 控制端运行 Whisper、讯飞或百度 ASR，然后只把识别
文本通过 SSH 发送给树莓派。树莓派上的 I2C 模块可以保留为唤醒和紧急停止入口。
ASR 提供方只需要满足一个接口：输出 UTF-8 中文文本，不需要了解机械臂状态或
运动逻辑。
