"""应用装配：把采集 → 分段 → 识别 → 落盘串起来，并挂上托盘。

线程模型（三段 + 托盘）：
    音频回调线程   sounddevice 驱动触发，只做入队
    segment 线程   消费音频队列，跑 VAD 分段，产出语音段
    asr 线程       消费语音段，跑识别，写盘
    主线程         pystray 消息循环，阻塞在 tray.run()

选线程而不是多进程：模型只加载一次，ctranslate2 推理时会释放 GIL，
而 16GB 内存下多进程会各存一份模型（约 1.5GB），不划算。
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .asr import create_engine
from .asr.base import ASREngine
from .audio import AudioCapture
from .config import Config
from .cuda_env import register_cuda_dlls
from .models import ensure_vad_model, whisper_model_cached
from .store import Store
from .tray import IconState, Tray
from .vad import Segment, Segmenter, SileroVad

log = logging.getLogger(__name__)

# 队列深度都留得比较宽：宁可多缓冲几秒，也不要因为瞬时抖动丢用户的话。
AUDIO_QUEUE_MAX = 64  # 64 × 32ms ≈ 2 秒
SEGMENT_QUEUE_MAX = 8

JOIN_TIMEOUT_S = 30.0


class SingleInstanceError(Exception):
    """已经有实例在跑了。"""


class _InstanceLock:
    """文件锁，防止同时跑两个实例。

    两个实例会各开一路麦克风、各往同一天的 markdown 里追加，
    记录会重复而且很难察觉，所以必须挡住。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None

    def acquire(self) -> None:
        import msvcrt

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a+")
        try:
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise SingleInstanceError(
                f"已经有一个 VoiceNote 在运行了（锁文件：{self._path}）"
            ) from exc

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            import msvcrt

            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception as exc:
            log.debug("释放锁失败：%s", exc)
        finally:
            self._fh.close()
            self._fh = None


class Application:
    """实现 tray.Controller 协议。"""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

        self._lock = _InstanceLock(cfg.state_dir / "session.lock")
        self._paused = threading.Event()
        self._stopping = threading.Event()
        self._shutdown_done = threading.Event()
        self._engine_ready = threading.Event()
        self._asr_lock = threading.Lock()

        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
        self._segment_q: queue.Queue[Segment] = queue.Queue(maxsize=SEGMENT_QUEUE_MAX)

        self._audio: AudioCapture | None = None
        self._segmenter: Segmenter | None = None
        self._engine: ASREngine | None = None
        self._store: Store | None = None
        self._tray: Tray | None = None
        self._threads: list[threading.Thread] = []

    # ---------- 生命周期 ----------

    def start(self) -> None:
        """启动全部组件，然后阻塞在托盘消息循环里，直到 shutdown()。"""
        self._lock.acquire()

        # 必须在任何 ctranslate2 调用之前注册 DLL 目录。
        register_cuda_dlls()

        vad_path = ensure_vad_model(self._cfg.models_dir)
        self._store = Store(self._cfg)
        self._segmenter = Segmenter(
            SileroVad(vad_path), self._cfg.vad, self._cfg.audio.sample_rate
        )
        self._audio = AudioCapture(self._cfg.audio, self._audio_q)

        # 托盘必须早于模型加载建好。模型加载（首次运行还要下 1.6GB）放在 asr
        # 线程里，图标才能立刻可见并弹出下载提示 —— 否则无控制台的打包版就是
        # 一个十几分钟毫无反应的图标，用户只会以为程序坏了。
        self._tray = Tray(self)

        self._threads = [
            threading.Thread(target=self._segment_loop, name="segment", daemon=True),
            threading.Thread(target=self._asr_loop, name="asr", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

        self._audio.start()
        log.info("VoiceNote 已启动，正在准备识别模型…（托盘图标：黄=准备中，绿=监听中）")

        self._tray.run()  # 阻塞，直到托盘被 stop()

        # 托盘循环结束了。shutdown() 可能是菜单「退出」在 pystray 线程里调用的，
        # 那个线程还要把最后一段话 flush 完 —— 直接返回会让主线程退出进程、
        # 把用户最后说的话截掉，所以这里必须等它收完尾。
        self.shutdown()
        self._shutdown_done.wait(timeout=JOIN_TIMEOUT_S)

    def shutdown(self) -> None:
        if self._stopping.is_set():
            return
        log.info("正在退出…")

        # 顺序很重要：先停采集（不再产生新音频），再把半段交出去，
        # 最后才放 worker 退出 —— 反过来的话最后一段话会被丢掉。
        if self._audio is not None:
            self._audio.stop()

        if self._segmenter is not None:
            try:
                tail = self._segmenter.flush()
                if tail is not None:
                    self._enqueue_segment(tail)
            except Exception:
                log.exception("收尾时处理最后一段失败")

        self._stopping.set()

        for thread in self._threads:
            thread.join(timeout=JOIN_TIMEOUT_S)

        if self._tray is not None:
            self._tray.stop()
        self._lock.release()
        self._shutdown_done.set()
        log.info("已退出")

    # ---------- tray.Controller ----------

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> None:
        if self.paused:
            return
        self._paused.set()
        if self._audio is not None:
            self._audio.stop()  # 真把麦克风关掉，而不是只忽略数据
        log.info("已暂停监听")
        self._set_state(IconState.PAUSED)

    def resume(self) -> None:
        if not self.paused:
            return
        self._paused.clear()

        # 丢掉暂停期间可能残留的音频，并清空分段器，
        # 否则恢复后第一段会带上暂停前的内容。
        self._drain(self._audio_q)
        if self._segmenter is not None:
            self._segmenter.reset()

        if self._audio is not None:
            try:
                self._audio.restart()
            except Exception:
                log.exception("恢复采集失败，可能需要重启程序")
                self._set_state(IconState.ERROR)
                return

        log.info("已恢复监听")
        self._set_state(IconState.LISTENING)

    def open_today(self) -> None:
        if self._store is None:
            return
        path = self._store.today_file()
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {path.stem}\n\n", encoding="utf-8")
            os.startfile(path)  # noqa: S606 — 用系统默认程序打开，符合预期
        except Exception:
            log.exception("打开今日记录失败：%s", path)

    # ---------- worker 线程 ----------

    def _segment_loop(self) -> None:
        assert self._segmenter is not None
        while not self._stopping.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                for segment in self._segmenter.push(chunk):
                    if self._engine_ready.is_set():
                        self._enqueue_segment(segment)
                    else:
                        # 引擎还在加载（首次运行要下模型）。这时说的话没法识别，
                        # 与其塞满队列、等模型好了再补识别一堆十几分钟前的过期内容，
                        # 不如直接丢掉。
                        log.debug(
                            "识别引擎尚未就绪，丢弃一段 %.1fs", segment.duration_s
                        )
            except Exception:
                log.exception("分段出错，跳过这一块音频")

    def _asr_loop(self) -> None:
        if not self._load_engine():
            return

        while True:
            # 退出时把队列里剩下的段处理完再走，别丢用户最后几句话。
            if self._stopping.is_set() and self._segment_q.empty():
                return
            try:
                segment = self._segment_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._transcribe(segment)

    def _load_engine(self) -> bool:
        """加载识别模型（首次运行会先下载约 1.6GB）。失败返回 False。

        放在 asr 线程里而不是 start()：托盘这时已经在跑了，下载提示才弹得出来。
        """
        first_run = not whisper_model_cached(self._cfg.asr.model)
        if first_run:
            source = self._cfg.general.hf_endpoint or "huggingface.co"
            log.info(
                "本地没有 %s 的缓存，开始下载（约 1.6GB，来源 %s）",
                self._cfg.asr.model,
                source,
            )
            self._notify(
                f"首次运行，正在下载识别模型（约 1.6GB）\n"
                f"来源：{source}\n下载期间无法记录，请稍候…"
            )
            self._set_state(IconState.LOADING)

        try:
            t0 = time.perf_counter()
            self._engine = create_engine(self._cfg.asr)
            log.info("模型加载完成，用时 %.1fs", time.perf_counter() - t0)
        except Exception:
            log.exception("ASR 引擎加载失败，本次运行无法识别")
            self._set_state(IconState.ERROR)
            return False

        try:
            t0 = time.perf_counter()
            self._engine.warmup()
            log.info("模型预热完成，用时 %.1fs", time.perf_counter() - t0)
        except Exception:
            # 预热失败只是慢，不该拦住整个应用。
            log.exception("模型预热失败，首次识别会明显变慢")

        self._engine_ready.set()
        if first_run:
            self._notify("模型准备完成，已开始监听")
        self._set_state(IconState.PAUSED if self.paused else IconState.LISTENING)
        return True

    def _transcribe(self, segment: Segment) -> None:
        assert self._engine is not None and self._store is not None

        self._set_state(IconState.TRANSCRIBING)
        t0 = time.perf_counter()
        try:
            # ctranslate2 的 WhisperModel 不支持并发调用同一个实例，
            # 这里加锁兜住（shutdown 路径也可能触发识别）。
            with self._asr_lock:
                text = self._engine.transcribe(segment.audio, segment.sample_rate)
        except Exception:
            log.exception("识别失败（段长 %.1fs）", segment.duration_s)
            return
        finally:
            self._set_state(
                IconState.PAUSED if self.paused else IconState.LISTENING
            )

        elapsed = time.perf_counter() - t0
        if not text:
            log.debug("识别结果为空（段长 %.1fs，耗时 %.1fs）", segment.duration_s, elapsed)
            return

        self._store.save(segment, text)
        log.info("[%.0fms] %s", elapsed * 1000, text)

    # ---------- 辅助 ----------

    def _enqueue_segment(self, segment: Segment) -> None:
        log.info("切出一段：%.1fs", segment.duration_s)
        try:
            self._segment_q.put(segment, timeout=1.0)
        except queue.Full:
            log.warning("识别队列已满，丢弃这一段（%.1fs）", segment.duration_s)

    def _set_state(self, state: IconState) -> None:
        if self._tray is not None:
            self._tray.set_state(state)

    def _notify(self, message: str) -> None:
        if self._tray is not None:
            self._tray.notify(message)

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return


def run(cfg: Config) -> int:
    app = Application(cfg)
    try:
        app.start()
    except SingleInstanceError as exc:
        log.error("%s", exc)
        return 3
    except KeyboardInterrupt:
        app.shutdown()
        return 0
    except Exception:
        log.exception("启动失败")
        return 1
    return 0
