"""注册 nvidia-* pip 包提供的 CUDA DLL 目录。

ctranslate2 在 Windows 上不自带 cuDNN / cuBLAS，也不去自己的包目录里找。
若不在 import ctranslate2 之前把 nvidia/cudnn/bin 和 nvidia/cublas/bin 加进
DLL 搜索路径，ct2 会直接抛「找不到 cudnn64_9.dll」之类的错误。

注意：ctranslate2.get_cuda_device_count() 返回非 0 只说明驱动可见，
不代表能推理 —— 缺 DLL 时它照样返回 1。别拿它当 GPU 可用的证据。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .paths import bundle_root

log = logging.getLogger(__name__)

_registered: list[Path] = []
_scanned = False


def register_cuda_dlls() -> list[Path]:
    """把 nvidia/*/bin 加入 DLL 搜索路径，返回已注册的目录列表。

    幂等：重复调用只会真正注册一次。非 Windows 平台直接返回空列表。
    """
    global _scanned
    if _scanned:
        return _registered
    _scanned = True

    if sys.platform != "win32":
        return _registered

    nvidia_root = bundle_root() / "nvidia"
    if not nvidia_root.is_dir():
        log.debug("未找到 nvidia 包目录：%s", nvidia_root)
        return _registered

    for pkg_dir in sorted(nvidia_root.iterdir()):
        bin_dir = pkg_dir / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            _registered.append(bin_dir)

    # 用 info 而不是 debug：打包版排查问题时，"CUDA DLL 到底有没有找到"
    # 是第一个要看的东西 —— 找不到就会静默回落到 CPU。
    log.info("已注册 %d 个 CUDA DLL 目录", len(_registered))
    return _registered
