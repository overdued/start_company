#!/usr/bin/env python3
"""asr.py — 语音识别 (SenseVoiceSmall via FunASR)

将唤醒后的语音转写为文字。
过滤规则: <0.5s 片段、<2字 转写结果直接丢弃
"""

from __future__ import annotations

import os
import re
import numpy as np
from typing import Optional, Tuple

MIN_DURATION_S = 0.5
MIN_TEXT_LEN = 2
# 语气词/单字黑名单（过滤无意义输出）
_FILLER_WORDS = {"嗯", "啊", "哦", "好", "是", "对", "。", "！", "？", "，", "哈", "呵", "额"}


class SenseVoiceASR:
    """SenseVoiceSmall 语音识别"""

    def __init__(self, device: str = "cpu", model_dir: Optional[str] = None):
        from funasr import AutoModel
        kwargs = {
            "model": "iic/SenseVoiceSmall",
            "vad_model": "fsmn-vad",
            "device": device,
            "disable_update": True,
        }
        if model_dir:
            kwargs["model"] = model_dir
        self._model = AutoModel(**kwargs)

    def transcribe(self, wav: np.ndarray) -> Tuple[str, float]:
        """转写 16kHz mono float32 波形

        Returns:
            (文本, 时长秒)；无效返回 ("", 0.0)
        """
        duration = len(wav) / 16000.0
        if duration < MIN_DURATION_S:
            return "", duration
        try:
            res = self._model.generate(
                input=wav,
                cache={},
                language="zh",
                use_itn=True,
                batch_size_s=60,
            )
            if not res:
                return "", duration
            text = res[0].get("text", "")
            # SenseVoice 输出含 <|zh|><|HAPPY|><|Speech|> 等标签，清理
            text = re.sub(r"<\|[^|]+\|>", "", text).strip()
            if len(text) < MIN_TEXT_LEN or text in _FILLER_WORDS:
                return "", duration
            return text, duration
        except Exception as e:
            print(f"[WARN] ASR 转写失败: {e}")
            return "", duration


def create_asr(device: str = "cpu") -> Optional[SenseVoiceASR]:
    """创建 ASR 实例，模型缺失返回 None"""
    try:
        return SenseVoiceASR(device=device)
    except Exception as e:
        print(f"[WARN] ASR 初始化失败: {e}")
        return None
