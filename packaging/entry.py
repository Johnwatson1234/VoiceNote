"""PyInstaller 的入口脚本。

不能直接拿 ``voicenote/__main__.py`` 当入口：它用的是相对导入
（``from . import ...``），被当成顶层脚本执行时会直接 ImportError。
这里改用绝对导入绕开。
"""

from __future__ import annotations

import sys

from voicenote.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
