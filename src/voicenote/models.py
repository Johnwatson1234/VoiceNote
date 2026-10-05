"""模型文件的下载与定位。

VAD 模型我们自己在本地准备；Whisper 权重交给 faster-whisper 从 HuggingFace
拉取并缓存，这里只负责判断"要不要下"。
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

_WEIGHT_SUFFIXES = {".bin", ".safetensors", ".npz"}


def whisper_model_cached(model: str) -> bool:
    """判断 faster-whisper 的权重是否已经缓存在本地。

    打包版第一次启动要下约 1.6GB，得先知道该不该提示用户"正在下载" ——
    否则一个无控制台的图标静静卡十几分钟，用户只会以为程序坏了。

    判断不出来时返回 False（当作没缓存）。最坏后果只是多弹一次提示，
    比"该提示时没提示"要好。
    """
    try:
        from faster_whisper.utils import _MODELS
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = _MODELS.get(model, model)
        snapshots = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots"
        if not snapshots.is_dir():
            return False

        for snap in snapshots.iterdir():
            if not snap.is_dir():
                continue
            if any(
                f.is_file() and f.suffix in _WEIGHT_SUFFIXES for f in snap.iterdir()
            ):
                return True
        return False
    except Exception as exc:
        log.debug("无法判断模型缓存状态：%s", exc)
        return False


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
