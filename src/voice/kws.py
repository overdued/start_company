#!/usr/bin/env python3
"""kws.py — 唤醒词检测 (sherpa-onnx Zipformer KWS)

唤醒词: "小鲲小鲲" 和 "你好小鲲"
拼音模糊匹配: 肖鲲/小框/肖坤 等同音变体也可唤醒（忽略声调）

使用 sherpa-onnx KeywordSpotter + pypinyin 应用层模糊匹配
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# 唤醒词同音/近音变体表（忽略声调）
WAKE_WORDS: List[List[str]] = [
    # 主唤醒词 "小鲲小鲲" 的变体
    ["xiao kun xiao kun", "xiao kun", "xiao kuo xiao kuo", "xiao kuang xiao kuang"],
    # "你好小鲲" 的变体
    ["ni hao xiao kun", "ni hao xiao kuang", "ni hao xiao kun xiao kun", "ni hao xiao"],
]
# 上述每行代表一组等价发音，命中任意一个即唤醒

KEYWORDS_RAW = """小鲲小鲲 @小鲲小鲲
你好小鲲 @你好小鲲
"""


def _char_to_pinyin(text: str) -> str:
    """汉字 → 无声调拼音（用 sherpa-onnx 或 pypinyin）"""
    try:
        from pypinyin import lazy_pinyin, Style
        return " ".join(lazy_pinyin(text, style=Style.NORMAL))
    except Exception:
        return text


def fuzzy_match_wake(decoded_text: str) -> Optional[str]:
    """对 KWS 解码输出做拼音模糊匹配

    Args:
        decoded_text: sherpa-onnx 解码的 token 序列（可能是拼音）

    Returns:
        匹配到的唤醒词规范形式，未匹配返回 None
    """
    if not decoded_text:
        return None
    text = decoded_text.lower().replace(" ", "")
    # 直接包含检查
    for group in WAKE_WORDS:
        for variant in group:
            v = variant.replace(" ", "")
            if v in text:
                return group[0].split()[1] if len(group[0].split()) > 1 else group[0]
    # 规范化对比（去掉空格后全等）
    norm = " ".join(decoded_text.lower().split())
    for group in WAKE_WORDS:
        if norm in group:
            return group[0]
    return None


class KeywordSpotter:
    """sherpa-onnx 唤醒词检测器"""

    def __init__(
        self,
        model_dir: str = "models/sherpa-kws",
        keywords_file: Optional[str] = None,
        threshold: float = 0.25,
        num_threads: int = 2,
    ):
        import sherpa_onnx

        model_dir = Path(model_dir)
        encoder = model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
        decoder = model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
        joiner = model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
        tokens = model_dir / "tokens.txt"

        if keywords_file is None:
            keywords_file = model_dir / "keywords.txt"
            if not Path(keywords_file).exists():
                # 尝试从 raw 生成
                raw = model_dir / "keywords_raw.txt"
                if raw.exists():
                    self._generate_keywords(str(raw), str(tokens), str(keywords_file))

        self._spotter = sherpa_onnx.KeywordSpotter(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            keywords_file=str(keywords_file),
            num_threads=num_threads,
            keywords_threshold=threshold,
            provider="cpu",
        )
        self._stream = self._spotter.create_stream()

    @staticmethod
    def _generate_keywords(raw_path: str, tokens_path: str, out_path: str) -> None:
        """从 keywords_raw.txt 生成 keywords.txt（ppinyin tokens）"""
        import subprocess
        try:
            r = subprocess.run(
                ["sherpa-onnx-cli", "text2token", "--tokens", tokens_path,
                 "--tokens-type", "ppinyin", raw_path],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0 and r.stdout.strip():
                Path(out_path).write_text(r.stdout, encoding="utf-8")
        except FileNotFoundError:
            # sherpa-onnx-cli 不存在时，简单拷贝 raw
            Path(out_path).write_text(Path(raw_path).read_text(encoding="utf-8"), encoding="utf-8")

    def accept_waveform(self, wav) -> Optional[str]:
        """喂入 16kHz float32 音频块，返回检测到的唤醒词或 None

        Args:
            wav: numpy float32 数组，16kHz mono

        Returns:
            唤醒词字符串（如 "小鲲小鲲"）或 None
        """
        import numpy as np
        wav = np.asarray(wav, dtype=np.float32)
        self._stream.accept_waveform(16000, wav.tolist())
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
        result = self._spotter.get_result(self._stream)
        if result:
            # 命中后重置流，避免重复触发
            self._spotter.reset_stream(self._stream)
            matched = fuzzy_match_wake(result)
            return matched or result
        return None

    def reset(self):
        try:
            self._spotter.reset_stream(self._stream)
        except Exception:
            self._stream = self._spotter.create_stream()


def create_spotter(model_dir: str = "models/sherpa-kws", **kwargs) -> Optional[KeywordSpotter]:
    """创建唤醒词检测器，模型缺失时返回 None"""
    try:
        return KeywordSpotter(model_dir=model_dir, **kwargs)
    except Exception as e:
        print(f"[WARN] KWS 初始化失败: {e}")
        return None
