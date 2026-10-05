"""麦克风采集：sounddevice InputStream → 队列。"""

from __future__ import annotations

import logging
import queue
from typing import Any

import numpy as np

from .config import AudioConfig

log = logging.getLogger(__name__)


class AudioCapture:
    """把麦克风音频块推进队列。

    用回调模式而不是阻塞 read：sounddevice 的回调跑在独立的音频线程里，
    由驱动直接触发，抖动最小。回调里**只做入队**，任何耗时操作都会造成丢帧。
    """

    def __init__(self, cfg: AudioConfig, out_queue: queue.Queue[np.ndarray]) -> None:
        self._cfg = cfg
        self._queue = out_queue
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd

        device = self._resolve_device(sd)
        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=self._cfg.sample_rate,
            blocksize=self._cfg.block_samples,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "开始采集：device=%s, %dHz, 每块 %d 采样点",
            device,
            self._cfg.sample_rate,
            self._cfg.block_samples,
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:  # 设备已消失时关闭会抛异常，不值得因此中断退出流程
            log.debug("关闭音频流时出错：%s", exc)
        finally:
            self._stream = None

    def restart(self) -> None:
        """重建音频流。休眠唤醒、切换默认设备导致流失效后调用。"""
        self.stop()
        self.start()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            log.warning("音频流状态异常：%s", status)
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # 下游跟不上。丢掉最旧的一块保住最新的 —— 否则队列会一直涨，
            # 延迟越滚越大，最后内存爆掉。
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(indata[:, 0].copy())
            except (queue.Empty, queue.Full):
                pass

    def _resolve_device(self, sd) -> int | None:  # noqa: ANN001
        if not self._cfg.device:
            return None  # None = 系统默认输入设备

        want = self._cfg.device.lower()
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] >= 1 and want in dev["name"].lower():
                log.info("按 %r 匹配到输入设备 #%d：%s", self._cfg.device, idx, dev["name"])
                return idx

        log.warning("没有输入设备匹配 %r，改用系统默认设备", self._cfg.device)
        return None
