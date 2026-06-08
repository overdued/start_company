#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py - KunPeng-Cortex Kimi 对话引擎

提供基于 Kimi API 的智能对话能力，专为养老场景优化：
    - 多轮对话上下文管理
    - 情感化、温柔的回复风格
    - 硬件控制意图识别与分发
    - 流式输出支持
    - 与 Orchestrator 深度集成

用法:
    >>> engine = KimiChatEngine(api_key="sk-...")
    >>> await engine.initialize()
    >>> response = await engine.chat("帮我把客厅灯打开")
    >>> print(response.content)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("kunpeng_cortex.kimi_chat.engine")


# =============================================================================
# 数据模型
# =============================================================================


@dataclasses.dataclass
class ChatMessage:
    """对话消息数据类。"""
    role: str
    content: str
    timestamp: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclasses.dataclass
class ChatResponse:
    """对话响应数据类。"""
    content: str
    reasoning: str = ""
    intent: str = ""
    confidence: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    model: str = ""


# =============================================================================
# KimiChatEngine 主类
# =============================================================================


class KimiChatEngine:
    """Kimi 对话引擎 - KunPeng-Cortex 的 LLM 能力核心。"""

    DEFAULT_SYSTEM_PROMPT = """你是 KunPeng-Cortex（小鲲），一位运行在 OrangePi Kunpeng Pro (RK3588) 上的智能家居养老助手。

你的核心使命：让老年人的生活更安全、更舒适、更不孤单。

交流风格要求：
1. 语气温柔、亲切、有耐心，像贴心的晚辈或护理员
2. 使用"您"称呼用户，避免命令式语气
3. 句子简短清晰，避免复杂技术术语
4. 表达关心和陪伴的意愿
5. 回复控制在 200 字以内，除非用户要求详细说明

你能提供的帮助：
- 智能家居控制（灯光、空调、窗帘等）
- 用药提醒和健康管理
- 天气查询和出行建议
- 紧急呼叫联系家人
- 陪伴聊天、播报新闻戏曲
- 生活起居辅助

当用户请求控制硬件设备时，请在回复末尾添加一行特殊的 [ACTION] 标记，格式如下：
[ACTION:设备类型:操作:参数]
例如：
[ACTION:light:turn_on:客厅]
[ACTION:arm:move:水杯]
[ACTION:emergency:call:儿子]

如果没有硬件操作请求，则不需要添加 [ACTION] 标记。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.kimi.com/coding",
        model: str = "kimi-for-coding",
        system_prompt: Optional[str] = None,
        max_history: int = 20,
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> None:
        self.api_key: str = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.max_history: int = max_history
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.system_prompt: str = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        self._client: Optional[httpx.AsyncClient] = None
        self._headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "claude-code/0.1.0",
            "X-Client-Name": "claude-code",
            "X-Client-Version": "0.1.0",
        }

        self._history: List[ChatMessage] = []
        self._initialized: bool = False

        logger.info(
            "KimiChatEngine created | model=%s, base_url=%s, max_history=%d",
            model, base_url, max_history,
        )

    async def initialize(self) -> bool:
        """异步初始化引擎，验证 API 可用性。"""
        if not self.api_key:
            logger.error("API Key not set, cannot initialize")
            return False

        try:
            self._client = httpx.AsyncClient(
                headers=self._headers,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )

            response = await self._client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_ids = [m.get("id") for m in models]
                logger.info("API verified | available models: %s", model_ids)
                self._initialized = True
                return True
            else:
                logger.error("API verification failed | status=%d, %s", response.status_code, response.text[:200])
                return False

        except Exception as e:
            logger.error("Initialization failed: %s", e)
            return False

    async def shutdown(self) -> None:
        """关闭引擎，释放资源。"""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        logger.info("KimiChatEngine shutdown")

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()
        logger.info("History cleared")

    def get_history(self) -> List[ChatMessage]:
        """获取当前对话历史。"""
        return list(self._history)

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """构建发送给 API 的消息列表。"""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        for msg in self._history[-self.max_history:]:
            messages.append(msg.to_dict())
        messages.append({"role": "user", "content": user_input})
        return messages

    def _parse_action(self, content: str) -> Optional[Dict[str, str]]:
        """从回复内容中解析硬件操作指令。"""
        match = re.search(r"\[ACTION:([^:]+):([^:]+):([^\]]+)\]", content)
        if match:
            return {
                "device": match.group(1).strip(),
                "action": match.group(2).strip(),
                "param": match.group(3).strip(),
            }
        return None

    def _extract_intent(self, user_input: str) -> Tuple[str, float]:
        """简单意图识别。"""
        text = user_input.lower()

        if any(kw in text for kw in ["开", "关", "打开", "关闭", "调", "设置"]):
            if any(kw in text for kw in ["灯", "光", "照明"]):
                return "control_light", 0.9
            elif any(kw in text for kw in ["空调", "温度", "暖气", "冷气"]):
                return "control_ac", 0.9
            elif any(kw in text for kw in ["窗帘", "窗"]):
                return "control_curtain", 0.9
            elif any(kw in text for kw in ["电视", "tv"]):
                return "control_tv", 0.9
            else:
                return "control_device", 0.7

        if any(kw in text for kw in ["天气", "温度", "下雨", "晴"]):
            return "query_weather", 0.9
        if any(kw in text for kw in ["时间", "几点", "日期", "星期"]):
            return "query_time", 0.9
        if any(kw in text for kw in ["药", "吃药", "用药"]):
            return "query_medicine", 0.9

        if any(kw in text for kw in ["救命", "急救", "医院", "120", "不舒服", "难受"]):
            return "emergency", 0.95

        if any(kw in text for kw in ["孤单", "寂寞", "无聊", "陪", "聊天", "说话"]):
            return "chat_companion", 0.85
        if any(kw in text for kw in ["谢谢", "感谢"]):
            return "chat_gratitude", 0.9
        if any(kw in text for kw in ["你好", "您好", "在吗", "小鲲"]):
            return "chat_greeting", 0.9

        return "chat_general", 0.6

    async def chat(self, user_input: str) -> ChatResponse:
        """发送消息并获取回复（非流式）。"""
        if not self._initialized or not self._client:
            raise RuntimeError("Engine not initialized, please call initialize() first")

        start_time = time.monotonic()
        intent, confidence = self._extract_intent(user_input)
        messages = self._build_messages(user_input)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content", "").strip()
            reasoning = message.get("reasoning_content", "")

            action = self._parse_action(content)
            if action:
                logger.info("Hardware action detected: %s", action)
                content = re.sub(r"\[ACTION:[^\]]+\]", "", content).strip()

            latency_ms = (time.monotonic() - start_time) * 1000

            self._history.append(ChatMessage(role="user", content=user_input))
            self._history.append(ChatMessage(role="assistant", content=content))

            if len(self._history) > self.max_history * 2:
                self._history = self._history[-self.max_history * 2:]

            usage = data.get("usage", {})

            return ChatResponse(
                content=content,
                reasoning=reasoning,
                intent=intent,
                confidence=confidence,
                tokens_used=usage.get("total_tokens", 0),
                latency_ms=latency_ms,
                model=data.get("model", self.model),
            )

        except httpx.HTTPStatusError as e:
            logger.error("API request failed: %s - %s", e.response.status_code, e.response.text[:200])
            return ChatResponse(
                content="抱歉，我暂时无法连接服务器，请稍后再试。",
                intent=intent,
                confidence=confidence,
            )
        except Exception as e:
            logger.error("Request exception: %s", e)
            return ChatResponse(
                content="抱歉，出了一点小问题，让我再试一次。",
                intent=intent,
                confidence=confidence,
            )

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """发送消息并获取流式回复。"""
        if not self._initialized or not self._client:
            raise RuntimeError("Engine not initialized")

        intent, confidence = self._extract_intent(user_input)
        messages = self._build_messages(user_input)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        full_content = ""

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            continue

        except Exception as e:
            logger.error("Stream request exception: %s", e)
            yield "抱歉，连接出现了问题。"

        if full_content:
            self._history.append(ChatMessage(role="user", content=user_input))
            self._history.append(ChatMessage(role="assistant", content=full_content))

    async def health_check(self) -> Tuple[bool, str]:
        """健康检查。"""
        if not self._initialized or not self._client:
            return False, "Not initialized"

        try:
            response = await self._client.get(
                f"{self.base_url}/v1/models",
                timeout=10.0,
            )
            if response.status_code == 200:
                return True, "API connection normal"
            else:
                return False, f"API error: {response.status_code}"
        except Exception as e:
            return False, f"Connection failed: {e}"
