"""ASR 引擎接口。

只定义一个极薄的协议，不做插件注册表 —— 目前只有一个真正实现的引擎
（faster-whisper），另一个（Qwen3-ASR）还是占位。等 A/B 对比有结论之后
再决定要不要引入更重的机制。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

SAMPLE_RATE = 16_000


@runtime_checkable
class ASREngine(Protocol):
    """把一段音频转成文本。"""

    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """audio 为 float32 单声道、取值 -1..1。返回文本，识别不出内容时返回空串。"""
        ...

    def warmup(self) -> None:
        """跑一次空推理，把首次推理的固定开销（cuDNN 内核选择、显存分配）提前付掉。

        不做预热的话，用户说的第一句话会明显变慢。
        """
        ...
