# VoiceNote

低启动成本的电脑端语音记录工具 —— **想到什么直接说，后台自动记录并实时转成文字。**

不做语音输入法：那个场景微信输入法、豆包输入法已经足够好。这个工具只解决一件事 ——
把脑子里一闪而过的念头随手说出来，剩下的交给它。

## 它是怎么工作的

```
麦克风 ──▶ VAD 分段 ──▶ 语音识别 ──▶ 按日期落盘
           (本地 CPU)     (本地 GPU)     文本 + 音频
```

- **启动即监听，不需要点任何东西。** 托盘图标常驻，看到绿色就说明在听。
- **不说话时几乎零开销。** 本地 VAD 判断有没有人在说话，静音时的音频**直接丢弃** ——
  不占 GPU、不出文本、也不上传到任何地方。
- **停顿约 0.7 秒即出字。** 说一句、停一下，这句话就会出现在当天的记录里。
- **完全离线。** 语音不出本机，断网照常工作，没有调用成本。

## 环境要求

第一版只支持 Windows。本项目开发与验证所用的机器：

| | |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop（8GB） |
| CPU | i9-13900HX |
| 内存 | 16GB |
| Python | 3.13 |

> **必须用 NVIDIA GPU。** 实测 `large-v3-turbo` 在 CUDA 上 RTF 为 **0.18**（约 5.5 倍实时），
> 而同一台机器的 CPU 上 RTF 高达 **1.77** —— 也就是说 20 秒的话要花 35 秒识别，
> 队列只会越堆越长。CPU 不是"慢一点"，是**跑不动**。
> 如果哪天 CUDA 用不了，正确做法是把 `asr.model` 换成更小的模型（如 `small`），而不是改用 CPU。

## 快速开始

```bash
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

脚本会建虚拟环境、装依赖、生成 `config.toml` 并做一次环境体检。

然后启动：

```bash
.venv\Scripts\pythonw.exe -m voicenote
```

`pythonw.exe` 不带控制台，启动时不会闪黑窗。想开机自启，右键托盘图标勾选「开机自启」即可。

> 首次启动会下载约 1.6GB 的模型权重。国内网络拉不动的话，先执行
> `$env:HF_ENDPOINT = "https://hf-mirror.com"` 再启动。

## 记录存在哪

默认在 `%USERPROFILE%\VoiceNote`，想换盘就改 `config.toml` 里的 `general.data_dir`。

```
VoiceNote/
├── transcripts/2026-10-05.md          # 文本，一天一个文件
├── audio/2026-10-05/100211-838.flac   # 音频，按天分目录
├── models/silero_vad.onnx
└── logs/voicenote.log
```

记录长这样，点 ▶ 可以直接回放当时那段话：

```markdown
# 2026-10-05

- `09:42:13` 今天下午要把 VAD 阈值调一下，far-field 的时候老是断错。 [▶](../audio/2026-10-05/094213-123.flac)
```

音频用 FLAC 无损压缩，约 60MB/小时。不想要音频就把 `config.toml` 里的
`audio.enabled` 改成 `false`。

## 常用配置

完整说明见 `config.example.toml`。几个最可能想调的：

| 配置项 | 什么时候改它 |
|---|---|
| `vad.speech_threshold` | 环境吵、老是误触发 → 调高（如 0.6）；说话轻、老是漏 → 调低（如 0.35） |
| `vad.min_silence_ms` | 觉得断句太碎 → 调高（如 1000）；觉得出字太慢 → 调低（如 500） |
| `audio.device` | 想指定麦克风。先跑 `python -m voicenote --check` 看设备列表，填名字的一部分即可 |
| `asr.initial_prompt` | 人名、项目名、专业术语老识别错 → 写进去当提示词 |
| `asr.language` | 默认留空自动检测，中英混说体验最好。若输出繁体，可试填 `zh` |
| `general.data_dir` | 想把记录放到别的盘 |

## 验证与自检

```bash
# 单元测试：分段逻辑的边界（咳嗽被滤掉、长独白强制切分等）
.venv\Scripts\python.exe -m pytest tests\ -q

# 环境体检：打印配置、音频设备、CUDA DLL 情况
.venv\Scripts\python.exe -m voicenote --check

# 端到端自检：用一段公开语音跑完整链路，不需要麦克风
.venv\Scripts\python.exe scripts\e2e_selfcheck.py

# GPU 性能基线
.venv\Scripts\python.exe scripts\smoke_test.py
```

## 已知限制

- **不是字级实时上屏。** 说完整句、停顿之后才出字。要做到边说边出字需要流式模型，
  第一版没做。
- **强制切分不对齐低能量点。** 连续说超过 20 秒会硬切，切点可能落在词中间。
  20 秒一段的情况下很少见，暂不为它增加复杂度。
- **音频会持续累积**，约 60MB/小时。目前没有自动清理。
- **还没实现搜索界面。** 记录是纯 markdown，用任何编辑器或 `grep` 都能查。

## 后续可能的方向

如果这个基础功能确实用得上，再基于攒下来的音频和文本做搜索、整理、总结、知识库。
ASR 引擎也已经做成可替换的接口，可以用 `scripts/ab_compare.py` 拿自己的录音
对比 faster-whisper 和 Qwen3-ASR（中文 CER 更低），用数据决定要不要换。
