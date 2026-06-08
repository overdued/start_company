#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kimi Chat Engine - KunPeng-Cortex 对话引擎

为 KunPeng-Cortex 提供基于 Kimi API 的智能对话能力，
支持多轮对话、情感化回复、硬件控制意图识别。

用法:
    from engines.kimi_chat import KimiChatEngine

    engine = KimiChatEngine(api_key="...", base_url="...")
    response = await engine.chat("帮我把客厅灯打开")
"""

from .engine import KimiChatEngine, ChatMessage, ChatResponse

__version__ = "1.0.0"
__all__ = ["KimiChatEngine", "ChatMessage", "ChatResponse"]
