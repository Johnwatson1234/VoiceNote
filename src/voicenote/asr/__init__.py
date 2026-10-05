"""ASR 引擎选择。"""

from __future__ import annotations

from ..config import AsrConfig
from .base import ASREngine

__all__ = ["ASREngine", "create_engine"]


def create_engine(cfg: AsrConfig) -> ASREngine:
    if cfg.engine == "faster_whisper":
        from .faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine(cfg)

    # qwen3 的实现在 qwen3_engine.py，目前是占位。
    raise ValueError(f"未知的 ASR 引擎：{cfg.engine!r}")
