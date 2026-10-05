"""端到端自检：拿一段公开语音把整条链路（VAD 分段 → 识别 → 落盘）跑一遍。

两个用途：

1. 改完代码确认链路没坏 —— ``python -m voicenote --selftest``
2. **验证打包产物** —— 打包出来的 exe 跑不了 pytest，这是唯一能证明
   "打包版里 VAD 和识别真的能用"的手段。只看到"进程起来了"是不够的。

不需要麦克风、也不需要有人说话，所以随时可以重跑。结果写在临时目录里，
不会污染真实记录。
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from .asr import create_engine
from .config import Config
from .cuda_env import register_cuda_dlls
from .models import ensure_vad_model
from .paths import bundle_root, is_frozen, project_root
from .store import Store
from .vad import WINDOW_SAMPLES, Segmenter, SileroVad

log = logging.getLogger(__name__)

# 公开的测试语音（LibriSpeech 片段，CC BY 4.0），保证自检可复现。
# 打包时会把这份一起带上，所以打包版自检不需要联网。
SAMPLE_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
# 这段录音里一定出现的词，用来确认识别结果是真对上了而不是空转。
EXCERPT_HINTS = ("dream", "nation")

TARGET_RATE = 16_000
SAMPLE_FILENAME = "selftest_sample.flac"


def _emit(line: str = "") -> None:
    """同时输出到日志和 stdout。

    打包版没有控制台（sys.stdout 是 None），日志文件是唯一能看到结果的地方，
    所以两处都要写。
    """
    if line:
        log.info("%s", line)
    if sys.stdout is not None:
        print(line)


def _locate_sample(cfg: Config) -> Path:
    """找到自检用的样例音频。

    优先用随包自带的那一份 —— 自检的职责是验证打包产物能不能用，
    不该因为网络不通而失败（那样就分不清是网络问题还是打包问题了）。
    只有源码运行、且本地还没缓存时才联网下载。
    """
    bundled = bundle_root() / SAMPLE_FILENAME
    if bundled.is_file() and bundled.stat().st_size > 0:
        return bundled

    # 源码运行：仓库里有同一份
    root = project_root()
    if root is not None:
        repo_copy = root / "packaging" / SAMPLE_FILENAME
        if repo_copy.is_file() and repo_copy.stat().st_size > 0:
            return repo_copy

    cached = cfg.models_dir / SAMPLE_FILENAME
    if cached.is_file() and cached.stat().st_size > 0:
        return cached

    _emit(f"本地没有样例音频，下载：{SAMPLE_URL}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SAMPLE_URL, timeout=60) as resp:
        cached.write_bytes(resp.read())
    return cached


def _load_mono_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if rate != TARGET_RATE:
        # 简单线性重采样，自检够用。生产路径只从麦克风拿 16kHz，不走这里。
        n = int(audio.shape[0] * TARGET_RATE / rate)
        audio = np.interp(
            np.linspace(0, audio.shape[0] - 1, n), np.arange(audio.shape[0]), audio
        )
    return audio.astype(np.float32)


def run_selfcheck(cfg: Config) -> int:
    register_cuda_dlls()

    with tempfile.TemporaryDirectory(prefix="voicenote-selftest-") as tmp:
        # 自检写临时目录，别把测试句子混进用户的真实记录里。
        cfg.general.data_dir = tmp

        _emit(f"VoiceNote 自检（{'打包版' if is_frozen() else '源码'}）")
        _emit(f"数据目录：{tmp}")

        try:
            sample = _locate_sample(cfg)
            audio = _load_mono_16k(sample)
        except Exception as exc:
            _emit(f"[失败] 无法准备测试语音：{type(exc).__name__}: {exc}")
            return 1

        duration = audio.shape[0] / TARGET_RATE
        _emit(f"测试语音：{audio.shape[0]} 采样点（{duration:.2f}s）")
        _emit()

        # ---- VAD 分段 ----
        try:
            vad = SileroVad(ensure_vad_model(cfg.models_dir))
            segmenter = Segmenter(vad, cfg.vad)
        except Exception as exc:
            _emit(f"[失败] VAD 初始化失败：{type(exc).__name__}: {exc}")
            return 1

        segments = []
        for start in range(0, audio.shape[0], WINDOW_SAMPLES):
            chunk = audio[start : start + WINDOW_SAMPLES]
            if chunk.shape[0] < WINDOW_SAMPLES:
                chunk = np.pad(chunk, (0, WINDOW_SAMPLES - chunk.shape[0]))
            segments.extend(segmenter.push(chunk))
        tail = segmenter.flush()
        if tail is not None:
            segments.append(tail)

        _emit(f"VAD 切出 {len(segments)} 段：")
        for i, seg in enumerate(segments, 1):
            _emit(f"  段{i}: {seg.duration_s:.2f}s")
        _emit()

        if not segments:
            _emit("[失败] VAD 没切出任何段 —— 检查 speech_threshold 或音频内容")
            return 1

        # ---- 识别 ----
        _emit(f"加载识别模型（device={cfg.asr.device}, compute_type={cfg.asr.compute_type}）…")
        try:
            t0 = time.perf_counter()
            engine = create_engine(cfg.asr)
            load_s = time.perf_counter() - t0
            engine.warmup()
        except Exception as exc:
            _emit(f"[失败] 识别引擎不可用：{type(exc).__name__}: {exc}")
            return 1
        _emit(f"模型加载 + 预热：{load_s:.1f}s")
        _emit()

        store = Store(cfg)
        texts: list[str] = []
        total_audio = 0.0
        total_infer = 0.0

        for i, seg in enumerate(segments, 1):
            t0 = time.perf_counter()
            text = engine.transcribe(seg.audio, seg.sample_rate)
            elapsed = time.perf_counter() - t0

            total_audio += seg.duration_s
            total_infer += elapsed
            texts.append(text)
            _emit(f"  段{i}（{seg.duration_s:.2f}s，耗时 {elapsed:.2f}s）：{text}")
            store.save(seg, text)

        _emit()

        combined = " ".join(texts).lower()
        if not combined.strip():
            _emit("[失败] 识别结果为空")
            return 1

        # ---- 落盘 ----
        transcript = store.today_file()
        if not transcript.is_file() or not transcript.read_text(encoding="utf-8").strip():
            _emit(f"[失败] 记录文件没写出来：{transcript}")
            return 1
        _emit(f"记录已写入：{transcript}")

        # ---- 结论 ----
        rtf = total_infer / total_audio if total_audio else float("inf")
        _emit()
        _emit(f"合计：{total_audio:.2f}s 音频，识别耗时 {total_infer:.2f}s，RTF {rtf:.3f}")

        if rtf >= 1.0:
            # RTF >= 1 意味着跟不上实时输入，队列会持续堆积。
            # 最常见的原因就是 GPU 没被用上，静默回落到了 CPU。
            _emit(
                f"[警告] RTF {rtf:.2f} >= 1，慢于实时。"
                f"若 device=cuda 则很可能是 CUDA DLL 没被正确打包，已回落到 CPU。"
            )
            return 1

        hits = [w for w in EXCERPT_HINTS if w in combined]
        if hits:
            _emit(f"[通过] 识别命中关键词 {hits}，端到端链路正常")
            return 0

        _emit(f"[注意] 链路跑通但未命中关键词 {EXCERPT_HINTS}，请人工核对上面的识别文本")
        return 0
