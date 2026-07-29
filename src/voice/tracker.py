#!/usr/bin/env python3
"""tracker.py — 唤醒词追踪与多人场景检测

- 唤醒锚点: 唤醒时刻提取声纹，作为本次对话的声纹锚点
- 跟踪窗口: 唤醒后 5-30 秒内的语音与锚点比对，持续追踪同一说话人
- 多人检测: 滑窗声纹差异分析，连续 N 窗口差异超阈值 → 多人场景提示
"""

from __future__ import annotations

import time
import numpy as np
from typing import Any, Dict, List, Optional

WINDOW_SECONDS = 15.0           # 跟踪窗口默认 15 秒（可配 5-30）
MULTI_SPEAKER_THRESHOLD = 0.4   # 滑窗声纹差异阈值
CONSECUTIVE_WINDOWS = 3         # 连续 N 个窗口判多人


class WakeTracker:
    """唤醒词追踪器"""

    def __init__(self, window_seconds: float = WINDOW_SECONDS):
        self.window = window_seconds
        self.anchor_emb: Optional[np.ndarray] = None
        self.anchor_user: Optional[Dict[str, Any]] = None
        self.anchor_ts: float = 0.0
        self.active: bool = False

    def start(self, anchor_emb: np.ndarray, user: Optional[Dict[str, Any]] = None):
        """开始追踪：记录唤醒锚点声纹"""
        self.anchor_emb = anchor_emb.copy()
        self.anchor_user = user
        self.anchor_ts = time.time()
        self.active = True

    def is_within_window(self) -> bool:
        """当前是否在跟踪窗口内"""
        if not self.active:
            return False
        return (time.time() - self.anchor_ts) <= self.window

    def check_continuation(self, emb: np.ndarray, threshold: float = 0.35) -> bool:
        """检查新语音是否为锚点说话人（同一人继续说话）"""
        if not self.is_within_window() or self.anchor_emb is None:
            return False
        sim = float(np.dot(emb, self.anchor_emb))
        return sim >= threshold

    def refresh(self, emb: Optional[np.ndarray] = None):
        """每次成功交互后刷新窗口"""
        self.anchor_ts = time.time()
        if emb is not None and self.anchor_emb is not None:
            # 缓慢融合更新锚点
            self.anchor_emb = self.anchor_emb * 0.8 + emb * 0.2
            self.anchor_emb = self.anchor_emb / np.linalg.norm(self.anchor_emb)

    def stop(self):
        self.active = False
        self.anchor_emb = None
        self.anchor_user = None


class MultiSpeakerDetector:
    """滑窗多人场景检测器（本期不做分离，仅检测提示）"""

    def __init__(self, window_sec: float = 2.0, shift_sec: float = 0.8,
                 threshold: float = MULTI_SPEAKER_THRESHOLD,
                 consecutive: int = CONSECUTIVE_WINDOWS):
        self.window = int(window_sec * 16000)
        self.shift = int(shift_sec * 16000)
        self.threshold = threshold
        self.consecutive = consecutive

    def detect(self, wav: np.ndarray, extractor) -> bool:
        """检测波形中是否存在多人说话

        Args:
            wav: 16kHz mono float32
            extractor: CAMPlusExtractor 实例

        Returns:
            True = 检测到多人场景
        """
        if len(wav) < self.window * 2:
            return False
        embeddings = []
        pos = 0
        while pos + self.window <= len(wav):
            seg = wav[pos:pos + self.window]
            # 能量过滤：跳过静音窗
            if np.sqrt(np.mean(seg ** 2)) > 0.01:
                embeddings.append(extractor.extract(seg))
            pos += self.shift
        if len(embeddings) < 2:
            return False
        # 相邻窗口声纹差异
        diffs = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            diffs.append(1.0 - sim)
        # 连续多个窗口差异超阈值
        count = 0
        for d in diffs:
            if d > self.threshold:
                count += 1
                if count >= self.consecutive:
                    return True
            else:
                count = 0
        return False
