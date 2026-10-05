"""Qwen3-ASR 引擎 —— 占位，尚未实现。

为什么留这个空壳：先说清楚「将来打算怎么换引擎」，比事后重新设计接口便宜。
真正写实现之前，先跑 ``scripts/ab_compare.py`` 拿自己的录音对比 ——
Qwen3-ASR-0.6B 的中文 CER 更低，但工具链比 faster-whisper 新，
没验证过实际收益就换不划算。

实现要点（供将来参考）：
  - 模型：Qwen/Qwen3-ASR-0.6B（Apache 2.0，约 2GB 显存）
  - 依赖事实：Qwen3-ASR-0.6B 的官方推理走 transformers 或 vLLM，
    **不需要** ctranslate2，所以本文件的实现不会复用到上面那套 CUDA DLL 注册
    逻辑以外的东西。transformers 会自带一份 torch，届时内存占用要重新评估。
  - 入参出参与本文件之外的调用方完全一致：transcribe(float32 音频, 采样率) -> str
"""

from __future__ import annotations

from ..config import AsrConfig

_NOT_IMPLEMENTED = (
    "Qwen3-ASR 引擎尚未实现。请先用 config.toml 里的 asr.engine = \"faster_whisper\"，"
    "或跑 scripts/ab_compare.py 对比后再决定是否实现它。"
)


class Qwen3AsrEngine:
    def __init__(self, cfg: AsrConfig) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)
