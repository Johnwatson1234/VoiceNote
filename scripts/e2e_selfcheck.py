"""端到端自检（薄封装）。

真正的实现在 ``src/voicenote/selfcheck.py`` —— 刻意放进包里，是为了让**打包产物**
也能跑（``VoiceNote.exe --selftest``），那是验证打包版的唯一手段。

用法：
    .venv\\Scripts\\python.exe scripts\\e2e_selfcheck.py

等价于：
    .venv\\Scripts\\python.exe -m voicenote --selftest
"""

from __future__ import annotations

from voicenote.__main__ import find_config
from voicenote.config import load
from voicenote.logging_setup import setup_logging
from voicenote.selfcheck import run_selfcheck


def main() -> int:
    cfg = load(find_config(None))
    setup_logging(cfg.logging.level)
    return run_selfcheck(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
