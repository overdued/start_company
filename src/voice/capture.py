#!/usr/bin/env python3
"""capture.py — 四通道麦克风采集模块

HK MIC (USB-Audio): ALSA 暴露为 2ch @ 48kHz S16_LE
处理链路: 2ch@48k → 立体声降混 mono → 重采样 16kHz/16bit PCM

支持 sounddevice (首选) 和 arecord 子进程 (兜底)
"""

from __future__ import annotations

import subprocess
import threading
import queue
import numpy as np
from typing import Optional, Callable

SAMPLE_RATE = 48000       # 硬件原生采样率
TARGET_RATE = 16000       # 目标采样率 (ASR/KWS/声纹统一)
CHANNELS = 2              # 硬件声道数
BLOCK_MS = 30             # 每块毫秒数


def _to_mono_16k(pcm: np.ndarray, src_rate: int = SAMPLE_RATE) -> np.ndarray:
    """立体声降混 + 重采样到 16kHz float32 [-1, 1]"""
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if src_rate != TARGET_RATE:
        # 线性插值重采样 (48k → 16k，3:1 比例足够)
        ratio = src_rate / TARGET_RATE
        indices = np.arange(0, len(pcm), ratio)
        indices = indices[indices < len(pcm)]
        pcm = np.interp(indices, np.arange(len(pcm)), pcm)
    return pcm.astype(np.float32)


class MicCapture:
    """麦克风采集器（sounddevice 首选，arecord 兜底）"""

    def __init__(self, device: Optional[str] = None, on_block: Optional[Callable[[np.ndarray], None]] = None):
        self.device = device
        self.on_block = on_block
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._mode = None  # 'sounddevice' | 'arecord'

    def _detect_device(self) -> Optional[int]:
        """自动检测 HK MIC 设备索引"""
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0 and ('HK' in d['name'] or 'USB' in d['name']):
                    return i
            # fallback: 默认输入
            default = sd.default.device[0]
            if default is not None and default >= 0:
                return default
        except Exception:
            pass
        return None

    def start(self) -> bool:
        """启动采集，返回是否成功"""
        if self._running:
            return True
        try:
            import sounddevice as sd
            dev = self._detect_device()
            self._stream = sd.InputStream(
                device=dev,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype='float32',
                blocksize=int(SAMPLE_RATE * BLOCK_MS / 1000),
                callback=self._sd_callback,
            )
            self._stream.start()
            self._running = True
            self._mode = 'sounddevice'
            return True
        except Exception as e:
            # arecord 兜底
            try:
                self._proc = subprocess.Popen(
                    ["arecord", "-D", "plughw:0,0", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", str(CHANNELS), "-t", "raw"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                self._mode = 'arecord'
                self._running = True
                self._thread = threading.Thread(target=self._arecord_loop, daemon=True)
                self._thread.start()
                return True
            except Exception:
                return False

    def _sd_callback(self, indata, frames, time_info, status):
        mono = _to_mono_16k(indata.copy())
        if self.on_block:
            self.on_block(mono)
        else:
            try:
                self._queue.put_nowait(mono)
            except queue.Full:
                pass

    def _arecord_loop(self):
        bytes_per_block = int(SAMPLE_RATE * BLOCK_MS / 1000) * CHANNELS * 2
        while self._running and self._proc and self._proc.stdout:
            data = self._proc.stdout.read(bytes_per_block)
            if not data:
                break
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            pcm = pcm.reshape(-1, CHANNELS)
            mono = _to_mono_16k(pcm)
            if self.on_block:
                self.on_block(mono)
            else:
                try:
                    self._queue.put_nowait(mono)
                except queue.Full:
                    pass

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """从队列读取一个音频块"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def record_seconds(self, seconds: float) -> np.ndarray:
        """采集指定秒数的音频并返回完整波形 (16kHz mono float32)"""
        blocks = []
        n_blocks = int(seconds * 1000 / BLOCK_MS)
        for _ in range(n_blocks * 2):  # 留余量
            blk = self.read(timeout=0.5)
            if blk is not None:
                blocks.append(blk)
            if len(blocks) >= n_blocks:
                break
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)

    def stop(self):
        self._running = False
        if self._mode == 'sounddevice' and hasattr(self, '_stream'):
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
        elif self._mode == 'arecord' and hasattr(self, '_proc'):
            try:
                self._proc.terminate(); self._proc.wait(timeout=2)
            except Exception:
                pass

    @property
    def mode(self) -> str:
        return self._mode or 'none'
