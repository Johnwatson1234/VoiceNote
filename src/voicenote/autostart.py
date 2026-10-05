"""开机自启：读写 HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run。

用命令行方式：python -m voicenote.autostart [enable|disable|status]
"""

from __future__ import annotations

import logging
import sys

from .paths import executable_path, is_frozen

log = logging.getLogger(__name__)

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "VoiceNote"


def autostart_command() -> str:
    """构造自启命令行。

    冻结版就是 VoiceNote.exe 本身 —— 它是无控制台子系统，开机不会闪黑窗，
    而且**不能**再拼 ``-m voicenote``（那是给 python.exe 用的，exe 不认）。
    源码运行则用 pythonw.exe 加模块名。
    """
    exe = executable_path()
    if is_frozen():
        return f'"{exe}"'
    return f'"{exe}" -m voicenote'


def is_enabled() -> bool:
    try:
        import winreg
    except ImportError:
        return False

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_QUERY_VALUE
        ) as key:
            value, _ = winreg.QueryValueEx(key, VALUE_NAME)
            return bool(value)
    except FileNotFoundError:
        return False


def enable() -> None:
    import winreg

    command = autostart_command()
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
    ) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
    log.info("已开启开机自启：%s", command)


def disable() -> None:
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, VALUE_NAME)
        log.info("已关闭开机自启")
    except FileNotFoundError:
        log.info("开机自启本来就是关闭的")


def main(argv: list[str] | None = None) -> int:
    action = (argv or sys.argv[1:] or ["status"])[0].lower()

    if action == "enable":
        enable()
    elif action == "disable":
        disable()
    elif action == "status":
        pass
    else:
        print("用法：python -m voicenote.autostart [enable|disable|status]")
        return 2

    print("开机自启：" + ("已开启" if is_enabled() else "已关闭"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
