"""日志初始化：同时输出到控制台和文件。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def harden_console_encoding() -> None:
    """别让一个编码不出来的字符把输出搞崩。

    Windows 控制台默认是 GBK 代码页，遇到 GBK 之外的字符（emoji、某些符号）
    会抛 UnicodeEncodeError。日志和体检输出不该因此失败，降级成替代字符即可。
    文件日志不受影响，始终是 UTF-8。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # 被重定向到不支持 reconfigure 的对象
            continue
        try:
            reconfigure(errors="replace")
        except Exception:
            pass


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    harden_console_encoding()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
