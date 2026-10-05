"""faster-whisper 引擎（ctranslate2 后端）。"""

from __future__ import annotations

import logging

import numpy as np

from ..config import AsrConfig
from .base import SAMPLE_RATE

log = logging.getLogger(__name__)


class FasterWhisperEngine:
    def __init__(self, cfg: AsrConfig) -> None:
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper/{cfg.model}"
        self._cfg = cfg

        log.info(
            "加载模型 %s（device=%s, compute_type=%s）…",
            cfg.model,
            cfg.device,
            cfg.compute_type,
        )
        self._model = WhisperModel(
            cfg.model, device=cfg.device, compute_type=cfg.compute_type
        )
        log.info("模型加载完成：%s", self.name)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"faster-whisper 只接受 {SAMPLE_RATE}Hz 音频，实际 {sample_rate}Hz")
        if audio.size == 0:
            return ""

        segments, _info = self._model.transcribe(
            audio,
            language=self._cfg.language or None,
            beam_size=self._cfg.beam_size,
            initial_prompt=self._cfg.initial_prompt or None,
            # 音频已经由我们自己的 VAD 切好了，再上一层 VAD 只会重复劳动。
            vad_filter=False,
            # 每段话之间是独立的，让模型参考上一段会导致它把话接歪、甚至陷入复读。
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()

    def warmup(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
