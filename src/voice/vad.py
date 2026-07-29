#!/usr/bin/env python3
"""vad.py — 语音活动检测 (FSMN-VAD)

使用 FunASR 内置 FSMN-VAD 模型检测语音段。
降级方案: 能量阈值 VAD（无模型时可用）
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

TARGET_RATE = 16000


class EnergyVAD:
    """能量阈值 VAD — 无需模型的降级方案"""

    def __init__(self, threshold: float = 0.01, silence_ms: int = 500, min_speech_ms: int = 300):
        self.threshold = threshold
        self.silence_blocks = silence_ms // 30  # 30ms 一块
        self.min_speech_blocks = min_speech_ms // 30
        self._speech_active = False
        self._silence_count = 0
        self._speech_count = 0

    def is_speech(self, block: np.ndarray) -> bool:
        """判断一个 30ms 块是否为语音"""
        if len(block) == 0:
            return False
        energy = float(np.sqrt(np.mean(block ** 2)))
        if energy > self.threshold:
            self._speech_count += 1
            self._silence_count = 0
            if self._speech_count >= self.min_speech_blocks or self._speech_active:
                self._speech_active = True
        else:
            if self._speech_active:
                self._silence_count += 1
                if self._silence_count >= self.silence_blocks:
                    self._speech_active = False
                    self._speech_count = 0
            else:
                self._speech_count = 0
        return self._speech_active

    def reset(self):
        self._speech_active = False
        self._silence_count = 0
        self._speech_count = 0


class FunASRVAD:
    """FunASR FSMN-VAD — 精确的语音段检测"""

    def __init__(self, model_dir: Optional[str] = None):
        from funasr import AutoModel
        kwargs = {"model": "fsmn-vad", "device": "cpu", "disable_update": True}
        if model_dir:
            kwargs["model"] = model_dir
        self._model = AutoModel(**kwargs)

    def detect_segments(self, wav: np.ndarray) -> List[Tuple[int, int]]:
        """检测完整波形中的语音段，返回 [(start_ms, end_ms), ...]"""
        if len(wav) == 0:
            return []
        try:
            res = self._model.generate(input=wav, cache={}, is_final=True)
            if res and len(res) > 0 and 'value' in res[0]:
                return res[0]['value']
        except Exception:
            pass
        # fallback: 全段
        return [(0, int(len(wav) / TARGET_RATE * 1000))]


def create_vad(prefer_model: bool = True) -> object:
    """创建 VAD 实例，优先 FunASR 模型，失败降级能量 VAD"""
    if prefer_model:
        try:
            import os
            model_dir = os.path.expanduser("~/.cache/modelscope/hub/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch")
            if os.path.exists(model_dir):
                return FunASRVAD(model_dir=model_dir)
            return FunASRVAD()
        except Exception:
            pass
    return EnergyVAD()
