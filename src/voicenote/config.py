"""配置加载：TOML 文件 → 带默认值的数据类。

未知的配置项会被忽略并记一条警告，而不是直接报错 —— 这样旧配置文件
在新增/改名配置项后仍然能用。
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from . import APP_NAME

log = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigError(Exception):
    """配置文件本身有问题（类型不对、值非法）。"""


def default_data_dir() -> Path:
    return Path.home() / APP_NAME


@dataclass(slots=True)
class GeneralConfig:
    data_dir: str = ""
    # 模型下载源。默认走国内镜像 —— 官方 huggingface.co 在国内经常不通，
    # 而打包版第一次启动必须下 1.6GB 权重。填 "" 即走官方源。
    hf_endpoint: str = "https://hf-mirror.com"

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser() if self.data_dir else default_data_dir()


@dataclass(slots=True)
class AudioConfig:
    enabled: bool = True
    device: str = ""
    sample_rate: int = 16_000
    # 32ms = 512 个采样点，正好是 silero-vad 在 16kHz 下的固定窗口长度。
    # 对齐这个值可以省掉一层重新分块的缓冲逻辑。
    block_ms: int = 32

    @property
    def block_samples(self) -> int:
        return self.sample_rate * self.block_ms // 1000


@dataclass(slots=True)
class VadConfig:
    speech_threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 700
    max_segment_s: int = 20
    pre_roll_ms: int = 300
    speech_pad_ms: int = 200


@dataclass(slots=True)
class AsrConfig:
    engine: str = "faster_whisper"
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = ""
    beam_size: int = 1
    initial_prompt: str = ""


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(slots=True)
class Config:
    general: GeneralConfig
    audio: AudioConfig
    vad: VadConfig
    asr: AsrConfig
    logging: LoggingConfig

    @property
    def data_dir(self) -> Path:
        return self.general.resolved_data_dir()

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


def _section(cls: type[T], raw: dict[str, Any], name: str) -> T:
    section = raw.get(name, {})
    if not isinstance(section, dict):
        raise ConfigError(f"[{name}] 必须是一个配置表，实际是 {type(section).__name__}")

    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    for key in section:
        if key not in allowed:
            log.warning("未知配置项 [%s].%s，已忽略", name, key)
    return cls(**{k: v for k, v in section.items() if k in allowed})  # type: ignore[call-arg]


def load(path: Path | None = None) -> Config:
    """读取配置文件。路径不存在时使用全默认配置（只记一条警告）。"""
    raw: dict[str, Any] = {}
    if path is not None:
        if path.is_file():
            with path.open("rb") as fh:
                try:
                    raw = tomllib.load(fh)
                except tomllib.TOMLDecodeError as exc:
                    raise ConfigError(f"配置文件解析失败：{path}\n{exc}") from exc
        else:
            log.warning("配置文件不存在：%s，使用默认配置", path)

    cfg = Config(
        general=_section(GeneralConfig, raw, "general"),
        audio=_section(AudioConfig, raw, "audio"),
        vad=_section(VadConfig, raw, "vad"),
        asr=_section(AsrConfig, raw, "asr"),
        logging=_section(LoggingConfig, raw, "logging"),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.audio.sample_rate <= 0:
        raise ConfigError("audio.sample_rate 必须为正数")
    if cfg.audio.block_samples <= 0:
        raise ConfigError("audio.block_ms 太小，算出来每个音频块不足 1 个采样点")
    if not 0.0 < cfg.vad.speech_threshold < 1.0:
        raise ConfigError("vad.speech_threshold 必须在 0 和 1 之间")
    if cfg.vad.min_silence_ms <= 0:
        raise ConfigError("vad.min_silence_ms 必须为正数")
    if cfg.vad.max_segment_s <= 0:
        raise ConfigError("vad.max_segment_s 必须为正数")
    if cfg.asr.device not in {"cuda", "cpu"}:
        raise ConfigError(f"asr.device 只能是 cuda 或 cpu，实际是 {cfg.asr.device!r}")
    if cfg.asr.engine != "faster_whisper":
        raise ConfigError(
            f"asr.engine 目前只支持 faster_whisper，实际是 {cfg.asr.engine!r}"
        )
