"""VAD 分段逻辑的单元测试。

用一个假的 VAD 替身按脚本返回概率，这样分段边界完全可控 ——
比拿真模型跑音频可复现得多，也能精确覆盖"咳嗽被滤掉"这类边界。
"""

from __future__ import annotations

import numpy as np
import pytest

from voicenote.config import VadConfig
from voicenote.vad import WINDOW_SAMPLES, Segmenter, State


class FakeVad:
    """按预设脚本逐个窗口返回概率的 VAD 替身。"""

    def __init__(self, probabilities: list[float]) -> None:
        self._probs = probabilities
        self._index = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def probability(self, window: np.ndarray) -> float:
        if self._index >= len(self._probs):
            return 0.0
        value = self._probs[self._index]
        self._index += 1
        return value


def audio(n_windows: int) -> np.ndarray:
    return np.zeros(n_windows * WINDOW_SAMPLES, dtype=np.float32)


def speech(n: int, level: float = 0.9) -> list[float]:
    return [level] * n


def silence(n: int, level: float = 0.05) -> list[float]:
    return [level] * n


def make(cfg: VadConfig | None = None, probs: list[float] | None = None):
    cfg = cfg or VadConfig()
    vad = FakeVad(probs or [])
    return Segmenter(vad, cfg), vad


def test_single_utterance_emits_one_segment() -> None:
    # 320ms 说话（>250ms 门槛）+ 足够长的静音触发断句
    seg, _ = make(probs=speech(10) + silence(25))
    out = seg.push(audio(35))

    assert len(out) == 1
    # 断句发生在"静音累积到 700ms"那一刻，输出再裁到 200ms 尾垫，
    # 所以段长 = 320ms 语音 + 200ms 尾垫 = 520ms，不会把整段静音带上。
    assert out[0].duration_s == pytest.approx(0.52, abs=0.05)


def test_short_blip_is_discarded() -> None:
    """咳嗽、键盘声这种短促声响不该产生记录。"""
    seg, _ = make(probs=speech(3) + silence(25))  # 96ms < 250ms
    out = seg.push(audio(28))

    assert out == []


def test_two_utterances_emit_two_segments() -> None:
    probs = speech(10) + silence(25) + speech(10) + silence(25)
    seg, _ = make(probs=probs)
    out = seg.push(audio(len(probs)))

    assert len(out) == 2


def test_pre_roll_prevents_clipped_onsets() -> None:
    """说话前留一点缓冲，句首才不会被切掉。"""
    base, _ = make(probs=speech(10) + silence(25))
    base_out = base.push(audio(35))
    assert len(base_out) == 1

    probs = silence(20) + speech(10) + silence(25)
    seg, _ = make(probs=probs)
    out = seg.push(audio(len(probs)))

    assert len(out) == 1
    # 与没有前置静音的情况相比，多出来的正好是 pre-roll：
    # 上限 300ms，按 32ms 的窗口对齐后为 9 个窗口 = 288ms。
    extra = out[0].duration_s - base_out[0].duration_s
    assert extra == pytest.approx(0.288, abs=0.05)


def test_max_segment_forces_split_during_continuous_speech() -> None:
    """连续说话不停顿也必须持续出字，否则用户会以为程序死了。"""
    cfg = VadConfig(max_segment_s=1)  # 1 秒即强制切分
    probs = speech(40)
    seg, _ = make(cfg=cfg, probs=probs)
    out = seg.push(audio(len(probs)))

    assert len(out) == 1
    assert out[0].duration_s >= 1.0
    # 强制切分后仍应停在 SPEAKING，继续累积下一段
    assert seg.state is State.SPEAKING


def test_segment_timestamps_are_monotonic() -> None:
    """段的起始时间必须单调递增，否则按日记录里的顺序是乱的。

    回归用：早先用 "now - 段长" 反推起点，导致段越长标得越早，
    连续说话被强制切分时几段之间会错序。
    """
    cfg = VadConfig(max_segment_s=1)
    probs = speech(100)  # 3.2s，会被切成 3 段以上
    seg, _ = make(cfg=cfg, probs=probs)
    out = seg.push(audio(len(probs)))

    assert len(out) >= 3
    starts = [s.started_at for s in out]
    assert starts == sorted(starts)


def test_pure_silence_produces_nothing() -> None:
    """没人说话时不该有任何输出 —— 这是"不说话就不占 GPU"的前提。"""
    seg, _ = make(probs=silence(300))
    out = seg.push(audio(300))

    assert out == []
    assert seg.state is State.IDLE


def test_reset_returns_to_idle_and_resets_vad() -> None:
    seg, vad = make(probs=speech(10) + silence(5))
    seg.push(audio(15))
    assert seg.state is State.SPEAKING

    before = vad.reset_count
    seg.reset()

    assert seg.state is State.IDLE
    assert vad.reset_count == before + 1


def test_flush_emits_partial_segment() -> None:
    """退出时正在说的半句话不能丢。"""
    seg, _ = make(probs=speech(10))
    seg.push(audio(10))
    assert seg.state is State.SPEAKING

    tail = seg.flush()

    assert tail is not None
    assert tail.duration_s > 0.2


def test_chunks_not_aligned_to_window_are_buffered() -> None:
    """音频块长度和 VAD 窗口不一致时必须内部缓冲，不能丢样本。"""
    probs = speech(10) + silence(25)
    seg, _ = make(probs=probs)

    total = 35 * WINDOW_SAMPLES
    out = []
    # 故意用 100 个采样点一块，和 512 的窗口长度不对齐
    for start in range(0, total, 100):
        size = min(100, total - start)
        out.extend(seg.push(np.zeros(size, dtype=np.float32)))

    assert len(out) == 1
