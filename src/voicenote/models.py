"""模型文件的下载与定位。

目前只有 VAD 模型需要在本地准备；Whisper 权重由 faster-whisper 自己
从 HuggingFace 拉取并缓存。
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

VAD_MODEL_FILENAME = "silero_vad.onnx"
VAD_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)


def ensure_vad_model(models_dir: Path) -> Path:
    """返回 VAD 模型路径，不存在则下载。"""
    target = models_dir / VAD_MODEL_FILENAME
    if target.is_file() and target.stat().st_size > 0:
        return target

    models_dir.mkdir(parents=True, exist_ok=True)
    log.info("下载 VAD 模型：%s", VAD_MODEL_URL)

    # 先下到临时文件再改名，避免下载中断留下半个文件被当成可用模型。
    tmp = target.with_suffix(".onnx.part")
    try:
        with urllib.request.urlopen(VAD_MODEL_URL, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    log.info("VAD 模型就绪：%s（%.1f MB）", target, target.stat().st_size / 1e6)
    return target
