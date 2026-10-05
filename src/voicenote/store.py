"""落盘：音频写 FLAC，文本按日期追加到 markdown。

目录结构：
    <data_dir>/
    ├── transcripts/2026-10-05.md
    └── audio/2026-10-05/094213-123.flac
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from .config import Config
from .vad import Segment

log = logging.getLogger(__name__)


class Store:
    """写盘。内部加锁，可以从任意线程调用。"""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        cfg.transcripts_dir.mkdir(parents=True, exist_ok=True)
        if cfg.audio.enabled:
            cfg.audio_dir.mkdir(parents=True, exist_ok=True)

    def save(self, segment: Segment, text: str) -> Path | None:
        """存下这一段。返回音频路径；未留存音频或文本为空时返回 None。

        文本为空说明 VAD 判成了语音但模型没听出内容（多半是环境噪声），
        这种段不写进记录，免得 markdown 里全是空条目。
        """
        text = text.strip()
        if not text:
            log.debug("识别结果为空，跳过该段（%.1fs）", segment.duration_s)
            return None

        with self._lock:
            day = segment.started_at.strftime("%Y-%m-%d")
            stamp = segment.started_at.strftime("%H:%M:%S")
            audio_path = self._write_audio(segment, day, stamp)
            self._append_line(day, stamp, text, audio_path)
        return audio_path

    def _write_audio(self, segment: Segment, day: str, stamp: str) -> Path | None:
        if not self._cfg.audio.enabled:
            return None

        import soundfile as sf  # 延迟导入：不开音频留存就完全不需要它

        day_dir = self._cfg.audio_dir / day
        day_dir.mkdir(parents=True, exist_ok=True)

        # 文件名精确到毫秒，避免同一秒内两段互相覆盖
        millis = segment.started_at.microsecond // 1000
        path = day_dir / f"{stamp.replace(':', '')}-{millis:03d}.flac"

        try:
            sf.write(path, segment.audio, segment.sample_rate, format="FLAC")
        except Exception as exc:
            # 音频存不下来不该连文本一起丢，所以这里只记日志不往上抛。
            log.error("音频写入失败 %s：%s", path, exc)
            return None
        return path

    def _append_line(
        self, day: str, stamp: str, text: str, audio_path: Path | None
    ) -> None:
        path = self._cfg.transcripts_dir / f"{day}.md"
        is_new = not path.exists()

        link = ""
        if audio_path is not None:
            rel = os.path.relpath(audio_path, path.parent).replace(os.sep, "/")
            link = f" [▶]({rel})"

        with path.open("a", encoding="utf-8") as fh:
            if is_new:
                fh.write(f"# {day}\n\n")
            fh.write(f"- `{stamp}` {text}{link}\n")

    def today_file(self) -> Path:
        """今日记录文件路径（可能还不存在）。托盘菜单「打开今日记录」用它。"""
        from datetime import date

        return self._cfg.transcripts_dir / f"{date.today():%Y-%m-%d}.md"
