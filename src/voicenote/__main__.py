"""VoiceNote 入口。

    python -m voicenote            # 启动（托盘常驻）
    python -m voicenote --check    # 只体检环境并打印配置，不启动
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import APP_NAME, __version__
from .config import Config, ConfigError, default_data_dir, load
from .cuda_env import register_cuda_dlls
from .logging_setup import setup_logging

log = logging.getLogger(__name__)

CONFIG_NAME = "config.toml"


def project_root() -> Path:
    """源码树根目录（src/ 的上一级）。"""
    return Path(__file__).resolve().parents[2]


def find_config(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser()
    local = project_root() / CONFIG_NAME
    if local.is_file():
        return local
    return default_data_dir() / CONFIG_NAME


def _describe_audio_devices(cfg: Config) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        print(f"音频设备：sounddevice 未安装（{exc}）")
        return

    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
    except Exception as exc:
        print(f"音频设备：枚举失败：{exc}")
        return

    print(f"音频设备（默认输入 #{default_in}）：")
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        marker = "*" if idx == default_in else " "
        print(
            f"  {marker} #{idx:<3} {dev['name']}"
            f"  ({dev['max_input_channels']}ch, {int(dev['default_samplerate'])}Hz)"
        )
    if cfg.audio.device:
        matches = [d for d in devices if cfg.audio.device.lower() in d["name"].lower()]
        print(
            f"  配置的 device={cfg.audio.device!r} 匹配到 {len(matches)} 个输入设备"
        )


def run_check(cfg: Config) -> int:
    print(f"{APP_NAME} {__version__}")
    print(f"Python       : {sys.version.split()[0]}  ({sys.executable})")
    print(f"数据目录      : {cfg.data_dir}")
    print(f"配置文件      : {find_config(None) or '(默认)'}")

    registered = register_cuda_dlls()
    print(f"CUDA DLL 目录 : {len(registered)} 个")
    for d in registered:
        print(f"               {d}")

    try:
        import ctranslate2

        print(f"ctranslate2  : {ctranslate2.__version__}（可见 CUDA 设备 {ctranslate2.get_cuda_device_count()}）")
    except ImportError as exc:
        print(f"ctranslate2  : 未安装（{exc}）")

    print()
    _describe_audio_devices(cfg)

    print()
    print("ASR 配置     :")
    print(f"  engine      = {cfg.asr.engine}")
    print(f"  model       = {cfg.asr.model}")
    print(f"  device      = {cfg.asr.device}")
    print(f"  compute_type= {cfg.asr.compute_type}")
    print(f"  language    = {cfg.asr.language or '(自动检测)'}")

    print()
    print("VAD 分段      :")
    print(f"  阈值 {cfg.vad.speech_threshold}，静音 {cfg.vad.min_silence_ms}ms 断句，"
          f"最长 {cfg.vad.max_segment_s}s 强制切分")
    print(f"  最短语音 {cfg.vad.min_speech_ms}ms（更短的丢弃）")

    print()
    print("音频留存      :", "开启" if cfg.audio.enabled else "关闭")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voicenote",
        description="低启动成本的电脑端语音记录工具",
    )
    parser.add_argument("--config", help="配置文件路径（默认项目根目录的 config.toml）")
    parser.add_argument("--check", action="store_true", help="只体检环境并打印配置，不启动")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    args = parser.parse_args(argv)

    try:
        cfg = load(find_config(args.config))
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    setup_logging(cfg.logging.level, cfg.data_dir / "logs" / "voicenote.log")

    if args.check:
        return run_check(cfg)

    from .app import run

    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
