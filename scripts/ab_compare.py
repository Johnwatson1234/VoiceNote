"""A/B 对比 ASR 引擎，用数据决定要不要换引擎。

本来打算先用 faster-whisper 跑通、再拿真实录音对比 Qwen3-ASR-0.6B 决定去留，
这个脚本就是那一步。Qwen3 引擎目前还是占位，脚本会如实说明并只跑已有的引擎 ——
等你实现之后再跑一次，就能直接看到两者对比。

用法：
    .venv\\Scripts\\python.exe scripts\\ab_compare.py <音频目录> [选项]

选项：
    --limit N        最多取 N 个文件（默认 20）
    --refs PATH      参考文本 JSON：{"文件名.wav": "人工转写内容"}。
                     提供后会计算 CER（字错率），否则只并排展示各引擎输出。
    --model NAME     覆盖 faster-whisper 的模型名
    --out PATH       报告输出路径（默认 ab_report.md）

音频目录可以直接指向 VoiceNote 自己的 audio/ 目录，用真实使用数据来评估。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from voicenote.config import AsrConfig
from voicenote.cuda_env import register_cuda_dlls
from voicenote.logging_setup import setup_logging

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
TARGET_RATE = 16_000


def levenshtein(a: str, b: str) -> int:
    """字符级编辑距离，用来算 CER。"""
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,        # 删除
                    cur[j - 1] + 1,     # 插入
                    prev[j - 1] + (ca != cb),  # 替换
                )
            )
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    reference = normalize(reference)
    if not reference:
        return float("nan")
    return levenshtein(reference, normalize(hypothesis)) / len(reference)


def normalize(text: str) -> str:
    """比对前统一格式：去空白、去标点，避免标点差异虚增错率。"""
    drop = set(" \t\n\r，。、！？；：""''（）《》…—,.!?;:\"'()[]{}")
    return "".join(ch for ch in text if ch not in drop).lower()


def load_audio(path: Path) -> np.ndarray | None:
    import soundfile as sf

    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        print(f"  [跳过] 读不出来 {path.name}: {exc}")
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != TARGET_RATE:
        n = int(audio.shape[0] * TARGET_RATE / rate)
        audio = np.interp(
            np.linspace(0, audio.shape[0] - 1, n), np.arange(audio.shape[0]), audio
        ).astype(np.float32)
    return audio.astype(np.float32)


def build_engines(args: argparse.Namespace) -> list:
    engines = []

    fw_cfg = AsrConfig(model=args.model or "large-v3-turbo")
    from voicenote.asr.faster_whisper_engine import FasterWhisperEngine

    print(f"加载 faster-whisper/{fw_cfg.model} …")
    engines.append(FasterWhisperEngine(fw_cfg))

    # Qwen3 还是占位。这里显式说明，而不是静默跳过 ——
    # 否则会让人以为"已经比过了"。
    try:
        from voicenote.asr.qwen3_engine import Qwen3AsrEngine

        engines.append(Qwen3AsrEngine(AsrConfig(engine="qwen3")))
    except NotImplementedError as exc:
        print(f"[说明] Qwen3-ASR 未实现，本次只跑 faster-whisper：{exc}")

    return engines


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B 对比 ASR 引擎")
    parser.add_argument("audio_dir", type=Path, help="存放录音的目录")
    parser.add_argument("--limit", type=int, default=20, help="最多取多少个文件")
    parser.add_argument("--refs", type=Path, help="参考文本 JSON，用于计算 CER")
    parser.add_argument("--model", help="覆盖 faster-whisper 模型名")
    parser.add_argument("--out", type=Path, default=Path("ab_report.md"))
    args = parser.parse_args()

    setup_logging("WARNING")
    register_cuda_dlls()

    if not args.audio_dir.is_dir():
        print(f"不是目录：{args.audio_dir}", file=sys.stderr)
        return 2

    files = sorted(
        p for p in args.audio_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES
    )[: args.limit]
    if not files:
        print(f"{args.audio_dir} 下没找到音频文件", file=sys.stderr)
        return 2

    refs: dict[str, str] = {}
    if args.refs and args.refs.is_file():
        refs = json.loads(args.refs.read_text(encoding="utf-8"))
        print(f"载入 {len(refs)} 条参考文本")

    engines = build_engines(args)
    for engine in engines:
        engine.warmup()

    print(f"\n共 {len(files)} 个文件，{len(engines)} 个引擎\n")

    results: list[dict] = []
    for path in files:
        audio = load_audio(path)
        if audio is None:
            continue

        row: dict = {"file": path.name, "duration_s": audio.shape[0] / TARGET_RATE}

        for engine in engines:
            t0 = time.perf_counter()
            try:
                text = engine.transcribe(audio, TARGET_RATE)
            except Exception as exc:
                text = f"<失败: {type(exc).__name__}>"
            elapsed = time.perf_counter() - t0

            row[engine.name] = {
                "text": text,
                "seconds": elapsed,
                "rtf": elapsed / max(row["duration_s"], 1e-6),
            }

        ref = refs.get(path.name)
        if ref:
            row["ref"] = ref
            for engine in engines:
                row[engine.name]["cer"] = cer(ref, row[engine.name]["text"])

        results.append(row)
        print(f"✓ {path.name}（{row['duration_s']:.1f}s）")

    lines = ["# ASR 引擎 A/B 对比", ""]
    lines.append(f"- 音频目录：`{args.audio_dir}`")
    lines.append(f"- 文件数：{len(results)}")
    lines.append(f"- 引擎：{', '.join(e.name for e in engines)}")
    lines.append("")

    for engine in engines:
        times = [r[engine.name]["seconds"] for r in results]
        rtfs = [r[engine.name]["rtf"] for r in results]
        lines.append(f"## {engine.name}")
        lines.append("")
        lines.append(f"- 总耗时 {sum(times):.1f}s")
        lines.append(f"- 平均 RTF {np.mean(rtfs):.3f}（越小越快，<1 表示快于实时）")
        if results and "ref" in results[0]:
            cers = [r[engine.name].get("cer", float("nan")) for r in results]
            lines.append(f"- 平均 CER {np.nanmean(cers):.4f}")
        lines.append("")

    lines.append("## 逐条结果")
    lines.append("")
    for row in results:
        lines.append(f"### {row['file']}（{row['duration_s']:.1f}s）")
        lines.append("")
        if "ref" in row:
            lines.append(f"- **参考**：{row['ref']}")
        for engine in engines:
            data = row[engine.name]
            cer_txt = f"，CER {data['cer']:.3f}" if "cer" in data else ""
            lines.append(
                f"- **{engine.name}**（{data['seconds']:.2f}s，RTF {data['rtf']:.3f}{cer_txt}）：{data['text']}"
            )
        lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
