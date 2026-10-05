"""Step 0 冒烟测试 —— 验证 faster-whisper 能否在 CUDA 上真正跑起来。

这一步不做任何应用逻辑，只回答一个问题：GPU 推理到底行不行。
背景：本机 torch 是 CPU-only 构建，且系统里原本没有任何 cuDNN/cuBLAS DLL。
      ctranslate2.get_cuda_device_count() 返回 1 只说明驱动可见，
      不代表能推理 —— 所以必须实测。

跑法：
    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL = "large-v3-turbo"
SAMPLE_RATE = 16_000
AUDIO_SECONDS = 10.0


def register_cuda_dlls() -> list[Path]:
    """把 nvidia-* 包里的 DLL 目录注册进 Windows DLL 搜索路径。

    必须在 import ctranslate2 之前调用，否则 ct2 找不到 cudnn64_9.dll /
    cublas64_12.dll，会直接抛缺 DLL 的错误。
    """
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.is_dir():
        return []

    registered: list[Path] = []
    for pkg_dir in sorted(nvidia_root.iterdir()):
        bin_dir = pkg_dir / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            registered.append(bin_dir)
    return registered


def gpu_memory_mib() -> tuple[int, int] | None:
    """返回 (已用显存, 总显存)，单位 MiB。取不到就返回 None。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        used, total = (int(x) for x in out.splitlines()[0].split(","))
        return used, total
    except Exception:
        return None


def make_test_audio() -> np.ndarray:
    """合成一段测试音频。

    冒烟测试只关心「推理能不能跑通、多快」，不关心识别准不准。
    用带包络的噪声，避免 whisper 在纯静音上走最短路径导致计时失真。
    """
    n = int(SAMPLE_RATE * AUDIO_SECONDS)
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.05, n).astype(np.float32)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 0.7 * np.arange(n) / SAMPLE_RATE))
    return (noise * envelope).astype(np.float32)


def bench(device: str, compute_type: str) -> dict:
    from faster_whisper import WhisperModel

    result: dict = {"device": device, "compute_type": compute_type}
    before = gpu_memory_mib()

    t0 = time.perf_counter()
    model = WhisperModel(MODEL, device=device, compute_type=compute_type)
    result["load_s"] = time.perf_counter() - t0

    audio = make_test_audio()

    # 预热：首次推理包含 cuDNN 算法选择/内核编译的开销，不能计入稳态耗时
    list(model.transcribe(audio, language="zh", beam_size=1)[0])

    t0 = time.perf_counter()
    segments, info = model.transcribe(audio, language="zh", beam_size=1)
    text = "".join(s.text for s in segments)
    elapsed = time.perf_counter() - t0

    after = gpu_memory_mib()
    result["infer_s"] = elapsed
    result["audio_s"] = AUDIO_SECONDS
    result["rtf"] = elapsed / AUDIO_SECONDS
    result["text"] = text.strip()
    result["vram_delta_mib"] = (
        after[0] - before[0] if (after and before) else None
    )
    result["vram_after_mib"] = after[0] if after else None
    result["vram_total_mib"] = after[1] if after else None
    return result


def report(r: dict) -> None:
    print(f"\n--- device={r['device']} compute_type={r['compute_type']} ---")
    print(f"  模型加载        : {r['load_s']:.1f}s")
    print(f"  推理 {r['audio_s']:.0f}s 音频 : {r['infer_s']:.2f}s")
    print(f"  RTF             : {r['rtf']:.3f}  (<1 表示快于实时)")
    if r["vram_delta_mib"] is not None:
        print(
            f"  显存占用        : +{r['vram_delta_mib']} MiB"
            f"  (整卡 {r['vram_after_mib']}/{r['vram_total_mib']} MiB)"
        )
    print(f"  输出            : {r['text'][:80]!r}")


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}  ({sys.executable})")

    dirs = register_cuda_dlls()
    print(f"CUDA DLL: 已注册 {len(dirs)} 个目录")
    for d in dirs:
        print(f"          {d}")

    import ctranslate2

    print(f"ct2     : {ctranslate2.__version__}, 可见 CUDA 设备 {ctranslate2.get_cuda_device_count()}")

    print(f"\n开始测试 {MODEL}（首次运行需下载约 1.6GB 权重）...")

    try:
        gpu = bench("cuda", "int8_float16")
        report(gpu)
        verdict = "[OK] GPU 可用 —— 按计划走 CUDA 路径"
    except Exception as exc:
        print(f"\n[失败] CUDA 路径不可用：{type(exc).__name__}: {exc}")
        verdict = "[!!] GPU 不可用 —— 需要换更小的模型跑 CPU"

    print("\n测试 CPU 回退路径（作为对照）...")
    try:
        report(bench("cpu", "int8"))
    except Exception as exc:
        print(f"[失败] CPU 路径也不可用：{type(exc).__name__}: {exc}")

    print(f"\n{'=' * 60}\n结论：{verdict}\n{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
