"""托盘图标与菜单。

状态用图标颜色表达，一眼能看出当前在不在听：
    绿 = 监听中   蓝 = 识别中   灰 = 已暂停   红 = 出错
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol

import pystray
from PIL import Image, ImageDraw

from . import autostart

log = logging.getLogger(__name__)


class IconState(Enum):
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    PAUSED = "paused"
    ERROR = "error"


_COLORS: dict[IconState, tuple[int, int, int]] = {
    IconState.LISTENING: (46, 160, 67),
    IconState.TRANSCRIBING: (31, 111, 235),
    IconState.PAUSED: (130, 130, 130),
    IconState.ERROR: (207, 34, 46),
}

_TITLES: dict[IconState, str] = {
    IconState.LISTENING: "VoiceNote — 监听中",
    IconState.TRANSCRIBING: "VoiceNote — 识别中",
    IconState.PAUSED: "VoiceNote — 已暂停",
    IconState.ERROR: "VoiceNote — 出错了",
}


class Controller(Protocol):
    """托盘需要的应用能力。"""

    @property
    def paused(self) -> bool: ...

    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def open_today(self) -> None: ...
    def shutdown(self) -> None: ...


def _render(state: IconState) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((10, 10, size - 10, size - 10), fill=_COLORS[state])
    return img


class Tray:
    def __init__(self, controller: Controller) -> None:
        self._controller = controller
        self._state = IconState.LISTENING
        self._icon = pystray.Icon(
            "voicenote",
            icon=_render(self._state),
            title=_TITLES[self._state],
            menu=self._build_menu(),
        )

    def _build_menu(self) -> pystray.Menu:
        c = self._controller
        return pystray.Menu(
            pystray.MenuItem(
                lambda _item: "恢复监听" if c.paused else "暂停监听",
                self._on_toggle,
                default=True,  # 双击托盘图标 = 暂停/恢复
            ),
            pystray.MenuItem("打开今日记录", self._on_open_today),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "开机自启",
                self._on_toggle_autostart,
                checked=lambda _item: autostart.is_enabled(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._on_quit),
        )

    def run(self) -> None:
        """阻塞运行托盘消息循环。只能在主线程调用。"""
        self._icon.run()

    def stop(self) -> None:
        try:
            self._icon.stop()
        except Exception as exc:
            log.debug("停止托盘失败：%s", exc)

    def set_state(self, state: IconState) -> None:
        if state is self._state:
            return
        self._state = state
        try:
            self._icon.icon = _render(state)
            self._icon.title = _TITLES[state]
        except Exception as exc:
            # 托盘图标没更新成功不值得让整个应用挂掉。
            log.debug("更新托盘状态失败：%s", exc)

    def notify(self, message: str) -> None:
        try:
            self._icon.notify(message, "VoiceNote")
        except Exception as exc:
            log.debug("托盘通知失败：%s", exc)

    def _on_toggle(self, _icon=None, _item=None) -> None:
        if self._controller.paused:
            self._controller.resume()
        else:
            self._controller.pause()

    def _on_open_today(self, _icon=None, _item=None) -> None:
        self._controller.open_today()

    def _on_toggle_autostart(self, _icon=None, _item=None) -> None:
        if autostart.is_enabled():
            autostart.disable()
        else:
            autostart.enable()

    def _on_quit(self, _icon=None, _item=None) -> None:
        self._controller.shutdown()
