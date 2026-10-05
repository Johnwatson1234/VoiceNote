"""VAD 分段：silero-vad (ONNX) + 状态机，把连续音频流切成语音段。

为什么自己写 ONNX 封装而不用 pip 的 ``silero-vad`` 包：那个包依赖 torch，
会凭空引入约 250MB 安装体积和一份常驻内存。而 faster-whisper 走的是
ctranslate2，根本不需要 torch —— 在 16GB 内存的常驻应用上不值得为一个
5MB 的 VAD 背上一整个 torch。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path

import numpy as np

from .config import VadConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
# 调用方每次提供的采样点数（16kHz 下 32ms）。
WINDOW_SAMPLES = 512
# silero-vad v5 会在每个窗口前面额外拼接上一窗口的尾部作为上下文，
# 所以模型实际收到的是 512 + 64 = 576 个采样点。
# 少了这一步模型不会报错，但会对任何人都输出接近 0 的概率 —— 静默失效，
# 极难排查，务必保留。
CONTEXT_SAMPLES = 64


class SileroVad:
    """silero-vad ONNX 的最小封装。非线程安全。"""

    def __init__(self, model_path: Path, intra_op_threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # 每次推理的输入只有 576 个点，多线程只会增加调度开销。
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._state = self._empty_state()
        self._context = self._empty_context()

    @staticmethod
    def _empty_state() -> np.ndarray:
        return np.zeros((2, 1, 128), dtype=np.float32)

    @staticmethod
    def _empty_context() -> np.ndarray:
        return np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def reset(self) -> None:
        """清空内部状态。切完一段后调用，避免上一段的上下文渗到下一段。"""
        self._state = self._empty_state()
        self._context = self._empty_context()

    def probability(self, window: np.ndarray) -> float:
        """返回窗口内是人声的概率。window 必须正好 WINDOW_SAMPLES 个采样点。"""
        if window.shape[0] != WINDOW_SAMPLES:
            raise ValueError(
                f"VAD 窗口必须是 {WINDOW_SAMPLES} 个采样点，实际 {window.shape[0]}"
            )

        # 拼接上下文后送进模型，再把本窗口的尾部留作下一窗口的上下文。
        x = np.concatenate((self._context[0], window)).reshape(1, -1).astype(np.float32)
        out, self._state = self._session.run(
            ["output", "stateN"],
            {"input": x, "state": self._state, "sr": self._sr},
        )
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(out[0][0])


class State(Enum):
    IDLE = auto()
    SPEAKING = auto()


@dataclass(slots=True)
class Segment:
    """一段切好的语音。"""

    audio: np.ndarray  # float32, 单声道, SAMPLE_RATE
    started_at: datetime
    sample_rate: int = SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return self.audio.shape[0] / self.sample_rate


class Segmenter:
    """把连续音频块切成语音段。

    行为要点：
      - IDLE 时只维护一小段 pre-roll 环形缓冲，其余音频**直接丢弃**。
        不说话时既不占 GPU，也不把环境音送去识别。
      - 进入 SPEAKING 后累积音频；静音超过 ``min_silence_ms`` 或长度超过
        ``max_segment_s`` 就吐出一段。
      - 短于 ``min_speech_ms`` 的段丢弃，用来滤掉咳嗽、键盘声、桌椅响。
      - 句尾只保留 ``speech_pad_ms`` 的静音，不会把整段静音都带上。

    已知取舍：``max_segment_s`` 触发的强制切分切在当前窗口，不做低能量点
    对齐。20s 一段的情况下边界落在词中间的概率很低，且 whisper 对切边有
    一定容忍度，不值得为它引入额外的复杂度。
    """

    def __init__(
        self,
        vad: SileroVad,
        cfg: VadConfig,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._vad = vad
        self._cfg = cfg
        self._sr = sample_rate

        self._leftover = np.empty(0, dtype=np.float32)
        self._state = State.IDLE

        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._pre_roll_limit = int(cfg.pre_roll_ms / 1000 * sample_rate)

        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._silence_samples = 0
        self._segment_start: datetime | None = None

    @property
    def state(self) -> State:
        return self._state

    def push(self, chunk: np.ndarray) -> list[Segment]:
        """喂入任意长度的音频，返回本次切出的语音段（通常 0 个或 1 个）。"""
        if chunk.size == 0:
            return []

        data = (
            np.concatenate((self._leftover, chunk))
            if self._leftover.size
            else chunk.astype(np.float32, copy=False)
        )

        out: list[Segment] = []
        n_windows = data.shape[0] // WINDOW_SAMPLES
        for i in range(n_windows):
            window = data[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]
            segment = self._feed(window)
            if segment is not None:
                out.append(segment)

        self._leftover = data[n_windows * WINDOW_SAMPLES :].copy()
        return out

    def flush(self) -> Segment | None:
        """收尾时调用：把正在累积的段吐出来（比如退出前）。"""
        if self._state is State.SPEAKING:
            return self._emit(stay_speaking=False)
        return None

    def reset(self) -> None:
        """丢掉所有缓冲回到初始状态。

        恢复监听时调用 —— 否则暂停前残留的音频会和恢复后的第一句话接在一起。
        """
        self._leftover = np.empty(0, dtype=np.float32)
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._buffer = []
        self._buffer_samples = 0
        self._silence_samples = 0
        self._segment_start = None
        self._state = State.IDLE
        self._vad.reset()

    def _feed(self, window: np.ndarray) -> Segment | None:
        is_speech = self._vad.probability(window) >= self._cfg.speech_threshold

        if self._state is State.IDLE:
            if not is_speech:
                self._pre_roll.append(window)
                self._pre_roll_samples += window.shape[0]
                self._trim_pre_roll()
                return None

            # 开始说话：把 pre-roll 回捞进缓冲，避免吞掉句首。
            self._state = State.SPEAKING
            self._buffer = list(self._pre_roll)
            self._buffer_samples = self._pre_roll_samples
            self._pre_roll.clear()
            self._pre_roll_samples = 0
            self._silence_samples = 0
            # 段的起点定在"检测到说话的那一刻往前推 pre-roll 的长度"。
            # 如果改成在吐出这段时用 now - 段长 反推，时间戳会随段长漂移
            # —— 段越长标得越早，几段之间还会乱序。
            self._segment_start = datetime.now() - timedelta(
                seconds=self._buffer_samples / self._sr
            )

        self._buffer.append(window)
        self._buffer_samples += window.shape[0]
        self._silence_samples = 0 if is_speech else self._silence_samples + window.shape[0]

        if self._silence_samples / self._sr * 1000 >= self._cfg.min_silence_ms:
            return self._emit(stay_speaking=False)

        if self._buffer_samples / self._sr >= self._cfg.max_segment_s:
            log.debug("段长超过 %.0fs，强制切分", self._cfg.max_segment_s)
            return self._emit(stay_speaking=True)

        return None

    def _emit(self, stay_speaking: bool) -> Segment | None:
        audio = (
            np.concatenate(self._buffer)
            if self._buffer
            else np.empty(0, dtype=np.float32)
        )
        speech_samples = self._buffer_samples - self._silence_samples

        # 过短的段直接丢弃 —— 咳嗽、键盘敲击、椅子挪动都会落在这里。
        if speech_samples / self._sr * 1000 < self._cfg.min_speech_ms:
            log.debug(
                "丢弃过短段：语音 %.0fms < %dms",
                speech_samples / self._sr * 1000,
                self._cfg.min_speech_ms,
            )
            self._reset(stay_speaking=False)
            return None

        # 句尾只保留 speech_pad_ms，别把整段静音都交给识别模型。
        keep_silence = int(self._cfg.speech_pad_ms / 1000 * self._sr)
        trim = self._silence_samples - keep_silence
        if trim > 0:
            audio = audio[:-trim]

        start = self._segment_start or datetime.now()
        segment = Segment(audio=audio, started_at=start, sample_rate=self._sr)

        # 强制切分后仍要继续累积：下一段的起点接在本段末尾，保证时间戳单调。
        if stay_speaking:
            self._segment_start = start + timedelta(seconds=audio.shape[0] / self._sr)

        self._reset(stay_speaking=stay_speaking)
        return segment

    def _reset(self, stay_speaking: bool) -> None:
        self._buffer = []
        self._buffer_samples = 0
        self._silence_samples = 0
        if not stay_speaking:
            self._state = State.IDLE
            self._segment_start = None
            self._vad.reset()

    def _trim_pre_roll(self) -> None:
        while self._pre_roll_samples > self._pre_roll_limit and self._pre_roll:
            self._pre_roll_samples -= self._pre_roll.popleft().shape[0]
