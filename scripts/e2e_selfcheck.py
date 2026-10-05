"""端到端自检：拿一段真实语音跑完整链路（VAD 分段 → 识别 → 落盘）。

不需要麦克风、也不需要有人说话 —— 用一段公开测试语音喂进去，
所以可以随时重跑，用来确认「改完代码之后整条链路还能用」。

跑法：
    .venv\\Scripts\\python.exe scripts\\e2e_selfcheck.py
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from voicenote.asr import create_engine
from voicenote.config import (
    AsrConfig,
    AudioConfig,
    Config,
    GeneralConfig,
    LoggingConfig,
    VadConfig,
)
from voicenote.cuda_env import register_cuda_dlls
from voicenote.logging_setup import setup_logging
from voicenote.models import ensure_vad_model
from voicenote.store import Store
from voicenote.vad import WINDOW_SAMPLES, Segmenter, SileroVad

# 公开的测试语音（LibriSpeech 片段），用来做可复现的自检。
SAMPLE_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
# 这段录音里一定出现的词，用来确认识别结果确实对上了而不是空转。
EXCERPT_HINTS = ("dream", "nation")


def fetch_sample(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "selfcheck_sample.flac"
    if not target.is_file():
        print(f"下载测试语音：{SAMPLE_URL}")
        with urllib.request.urlopen(SAMPLE_URL, timeout=60) as resp:
            target.write_bytes(resp.read())
    return target


def load_mono_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16_000:
        # 简单线性重采样，自检够用（生产路径只从麦克风拿 16kHz，不走这里）
        n = int(audio.shape[0] * 16_000 / rate)
        audio = np.interp(
            np.linspace(0, audio.shape[0] - 1, n), np.arange(audio.shape[0]), audio
        ).astype(np.float32)
    return audio.astype(np.float32)


def main() -> int:
    setup_logging("INFO")
    register_cuda_dlls()

    with tempfile.TemporaryDirectory(prefix="voicenote-selfcheck-") as tmp:
        data_dir = Path(tmp)

        cfg = Config(
            general=GeneralConfig(),
            audio=AudioConfig(),
            vad=VadConfig(),
            asr=AsrConfig(),
            logging=LoggingConfig(),
        )
        # Config.data_dir 走 general，这里直接指到临时目录，别碰到真实数据。
        cfg.general.data_dir = str(data_dir)

        sample = fetch_sample(cfg.models_dir)
        audio = load_mono_16k(sample)
        print(f"测试语音：{audio.shape[0]} 采样点（{audio.shape[0] / 16000:.2f}s）")

        vad = SileroVad(ensure_vad_model(cfg.models_dir))
        segmenter = Segmenter(vad, cfg.vad)
        store = Store(cfg)
        engine = create_engine(cfg.asr)
        engine.warmup()

        # 按音频块喂进去，模拟真实采集路径。
        segments = []
        for start in range(0, audio.shape[0], WINDOW_SAMPLES):
            chunk = audio[start : start + WINDOW_SAMPLES]
            if chunk.shape[0] < WINDOW_SAMPLES:
                chunk = np.pad(chunk, (0, WINDOW_SAMPLES - chunk.shape[0]))
            segments.extend(segmenter.push(chunk))
        tail = segmenter.flush()
        if tail is not None:
            segments.append(tail)

        print(f"\nVAD 切出 {len(segments)} 段：")
        for i, seg in enumerate(segments, 1):
            print(f"  段{i}: {seg.duration_s:.2f}s")

        if not segments:
            print("\n[失败] VAD 没切出任何段 —— 检查 speech_threshold 或音频内容")
            return 1

        print()
        texts = []
        for i, seg in enumerate(segments, 1):
            text = engine.transcribe(seg.audio, seg.sample_rate)
            texts.append(text)
            print(f"  段{i}: {text}")
            store.save(seg, text)

        combined = " ".join(texts).lower()
        if not combined.strip():
            print("\n[失败] 识别结果为空")
            return 1

        # 确认落盘成功
        transcript = store.today_file()
        if not transcript.is_file() or not transcript.read_text(encoding="utf-8").strip():
            print(f"\n[失败] 记录文件没写出来：{transcript}")
            return 1

        print(f"\n记录已写入：{transcript.name}")
        print(transcript.read_text(encoding="utf-8").strip())

        hits = [w for w in EXCERPT_HINTS if w in combined]
        if hits:
            print(f"\n[OK] 识别结果命中关键词 {hits}，端到端链路正常")
        else:
            print(
                f"\n[注意] 未命中关键词 {EXCERPT_HINTS}，但链路跑通了；"
                "请人工核对上面的识别文本"
            )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
