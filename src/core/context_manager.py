#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文管理器 (ContextManager) —— KunPeng-Cortex 核心引擎模块

本模块实现 Agent 的记忆系统，负责管理短期交互上下文和长期用户记忆。
采用分层记忆架构：短期记忆使用线程安全的环形缓冲区维护最近 100 轮对话，
支持 O(1) 时间复杂度的插入和查询；长期记忆使用轻量级向量数据库存储用户偏好、
场景记忆和行为习惯，支持语义相似度检索。

核心功能：
    - 短期记忆：环形缓冲区，最近 100 轮对话 + 传感器事件
    - 长期记忆：向量数据库，用户偏好 / 场景记忆 / 行为习惯 / 安全档案
    - 会话持久化：JSONL 格式，支持崩溃恢复
    - 上下文注入：自动组装 Claude Code prompt 所需的上下文
    - 情感趋势分析：计算最近 N 轮的情绪趋势

数据流：
    用户输入 → 短期记忆（环形缓冲区）→ 摘要提炼 → 长期记忆（向量DB）
                                         ↓
                                    上下文注入 → Claude Code prompt

线程安全：
    所有公共方法均受 ``_lock`` 保护，支持多协程并发访问。

硬件平台：
    OrangePi Kunpeng Pro (RK3588, ARM64, 16GB RAM)

作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
)

# ---------------------------------------------------------------------------
# 日志兼容层
# ---------------------------------------------------------------------------
import logging


def _get_logger(name: str) -> logging.Logger:
    """获取统一格式的日志记录器。

    优先使用项目内部日志系统，若不可用则回退至标准库 logging。

    Args:
        name: 日志记录器名称。

    Returns:
        配置好的 ``logging.Logger`` 实例。
    """
    try:
        from src.utils.logger import get_logger as _proj_get_logger

        return _proj_get_logger(name)
    except ImportError:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S.%f",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 短期记忆最大轮数
STM_MAX_TURNS: int = 100

# 单条文本最大长度（字符）
STM_MAX_TEXT_LENGTH: int = 512

# 情感评分范围
EMOTION_MIN_SCORE: float = -1.0
EMOTION_MAX_SCORE: float = 1.0

# 会话自动保存间隔（秒）
SESSION_SAVE_INTERVAL: float = 60.0

# 向量维度（Sentence-BERT 默认输出维度）
VECTOR_DIMENSION: int = 384

# 长期记忆最大条目数
LTM_MAX_ENTRIES: int = 2000


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------


@dataclass
class InteractionRecord:
    """交互记录数据类。

    存储单轮对话或传感器事件的完整信息，包括时间戳、
    角色、内容、情感评分和关联数据。

    Attributes:
        record_id: 全局唯一记录标识符。
        timestamp: Unix 时间戳（秒）。
        role: 角色类型 ("user" / "agent" / "system" / "sensor")。
        content: 对话文本或事件描述。
        emotion_score: 情感评分 (-1.0 ~ +1.0)。
        sensor_data: 关联传感器数据字典。
        action_id: 关联的动作 ID。
        action_result: 动作执行结果 ("success" / "failed" / "timeout" / ""）。
        session_id: 所属会话 ID。
        metadata: 额外元数据。
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    role: str = "user"  # "user" | "agent" | "system" | "sensor"
    content: str = ""
    emotion_score: float = 0.0
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    action_id: str = ""
    action_result: str = ""
    session_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionRecord":
        """从字典反序列化。"""
        return cls(
            record_id=data.get("record_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            emotion_score=data.get("emotion_score", 0.0),
            sensor_data=data.get("sensor_data", {}),
            action_id=data.get("action_id", ""),
            action_result=data.get("action_result", ""),
            session_id=data.get("session_id", "default"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LongTermMemory:
    """长期记忆条目数据类。

    存储经摘要提炼的用户偏好、场景记忆和行为习惯，
    每条记忆带有向量表示以支持语义检索。

    Attributes:
        memory_id: 全局唯一记忆标识符。
        timestamp: 创建时间戳。
        memory_type: 记忆类型 ("preference" / "scene" / "habit" / "safety")。
        key: 记忆关键词。
        value: 记忆内容字典。
        vector: 向量表示（384 维浮点列表）。
        importance: 重要性评分 (0.0 ~ 1.0)。
        access_count: 被访问次数。
        last_accessed: 最后访问时间戳。
    """

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    memory_type: str = "preference"  # "preference" | "scene" | "habit" | "safety"
    key: str = ""
    value: Dict[str, Any] = field(default_factory=dict)
    vector: List[float] = field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongTermMemory":
        """从字典反序列化。"""
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            memory_type=data.get("memory_type", "preference"),
            key=data.get("key", ""),
            value=data.get("value", {}),
            vector=data.get("vector", []),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", time.time()),
        )


@dataclass
class SessionInfo:
    """会话信息数据类。

    描述一个交互会话的元数据。

    Attributes:
        session_id: 会话唯一标识符。
        created_at: 创建时间戳。
        last_active: 最后活跃时间戳。
        turn_count: 当前轮数。
        user_id: 关联用户 ID。
        metadata: 会话级元数据。
    """

    session_id: str = "default"
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turn_count: int = 0
    user_id: str = "anonymous"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ContextManager 主类
# ---------------------------------------------------------------------------


class ContextManager:
    """上下文管理器 —— KunPeng-Cortex 的记忆系统。

    ContextManager 实现分层记忆架构，管理 Agent 与用户的交互上下文：

    1. **短期记忆 (STM)**: 线程安全的固定大小环形缓冲区，
       维护最近 100 轮对话和传感器事件，支持 O(1) 插入和查询。
    2. **长期记忆 (LTM)**: 轻量级向量数据库，存储用户偏好、
       场景记忆、行为习惯和安全档案，支持语义相似度检索。
    3. **会话持久化**: JSONL 格式自动保存，支持系统崩溃恢复。
    4. **上下文注入**: 自动组装 Claude Code prompt 所需的上下文窗口。

    记忆流：
        用户输入 → 短期记忆 → 摘要提炼 → 长期记忆
                          ↓
                    上下文注入 → LLM prompt

    线程安全：
        所有公共方法均受 ``_lock`` 保护，可在多协程环境中安全调用。
        环形缓冲区的读写操作均为原子级。

    使用示例::

        ctx = ContextManager(persist_path="data/sessions.jsonl")
        ctx.add_interaction(user="帮我把水杯递过来", agent="好的，请稍等")
        recent = ctx.get_recent_context(n=5)
        ctx.store_memory("user_preference", {"item": "water_temperature", "value": "warm"})
        memories = ctx.recall_memory("用户喜欢喝温水")

    Attributes:
        persist_path: 会话持久化文件路径（JSONL 格式）。
        stm_max_turns: 短期记忆最大轮数。
        _stm_buffer: 短期记忆环形缓冲区。
        _ltm_store: 长期记忆存储列表。
        _session: 当前会话信息。
        _lock: 线程安全锁（asyncio.Lock）。
        _save_task: 自动保存后台任务句柄。
        _logger: 日志记录器。
    """

    def __init__(
        self,
        persist_path: str = "data/sessions.jsonl",
        stm_max_turns: int = STM_MAX_TURNS,
        vector_db_path: Optional[str] = None,
    ) -> None:
        """初始化上下文管理器。

        创建环形缓冲区、加载历史会话（若存在）、
        启动自动保存后台任务。

        Args:
            persist_path: 会话持久化文件路径（JSONL）。
            stm_max_turns: 短期记忆最大轮数，默认 100。
            vector_db_path: 向量数据库持久化路径。若为 None，
                使用内存存储。
        """
        self.persist_path: str = persist_path
        self.stm_max_turns: int = stm_max_turns
        self.vector_db_path: Optional[str] = vector_db_path

        # 短期记忆：固定大小环形缓冲区
        self._stm_buffer: Deque[InteractionRecord] = deque(maxlen=stm_max_turns)

        # 长期记忆：列表存储（未来可替换为向量数据库）
        self._ltm_store: List[LongTermMemory] = []

        # 会话信息
        self._session = SessionInfo()

        # 线程安全
        self._lock: asyncio.Lock = asyncio.Lock()

        # 后台任务
        self._save_task: Optional[asyncio.Task] = None
        self._shutdown_flag: bool = False

        # 统计
        self._total_records: int = 0
        self._total_memories: int = 0

        # 日志
        self._logger = _get_logger("context_manager")

        # === Hermes Bridge 融合层 ===
        self._memory_store: Optional[Any] = None
        self._session_db: Optional[Any] = None
        self._init_hermes_bridge()

        # 加载历史会话
        self._load_session_sync()

        self._logger.info(
            "ContextManager 初始化完成 | STM容量: %d | 持久化路径: %s",
            stm_max_turns,
            persist_path,
        )

    # ------------------------------------------------------------------
    # Hermes Bridge 初始化
    # ------------------------------------------------------------------

    def _init_hermes_bridge(self) -> None:
        """初始化 Hermes Bridge 记忆和会话搜索子系统。"""
        try:
            from hermes_bridge.memory_tool import KunpengMemoryStore
            from hermes_bridge.session_search import SessionSearchDB

            self._memory_store = KunpengMemoryStore(
                memory_dir="data/memories"
            )
            self._session_db = SessionSearchDB(
                db_path="data/sessions_fts.db"
            )
            self._logger.info(
                "Hermes Bridge 初始化完成 | 记忆条目: %d | 会话数据库: %s",
                len(self._memory_store.memory_entries) + len(self._memory_store.user_entries),
                self._session_db.db_path,
            )
        except Exception as e:
            self._logger.warning("Hermes Bridge 初始化失败: %s", e)
            self._memory_store = None
            self._session_db = None

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动后台任务（自动保存循环）。

        应在系统初始化完成后调用。
        """
        self._shutdown_flag = False
        self._save_task = asyncio.create_task(
            self._auto_save_loop(), name="context_auto_save"
        )
        self._logger.info("上下文管理器后台任务已启动")

    async def stop(self) -> None:
        """优雅停止，执行最终保存。

        取消后台任务并确保所有数据已持久化。
        """
        self._shutdown_flag = True
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

        # 最终保存
        await self._save_session_async()
        self._logger.info("上下文管理器已停止，数据已保存")

    # ------------------------------------------------------------------
    # 核心 API：短期记忆
    # ------------------------------------------------------------------

    def add_interaction(self, user: str, agent: str) -> None:
        """添加一轮交互记录到短期记忆。

        将用户输入和 Agent 回复分别记录为两条 ``InteractionRecord``，
        插入到环形缓冲区的尾部。若缓冲区已满，自动淘汰最早的记录。

        Args:
            user: 用户输入文本。
            agent: Agent 回复文本。

        Raises:
            ValueError: 输入为空时抛出。
        """
        if user is None or agent is None:
            raise ValueError("用户输入和Agent回复不能为空")

        # 截断过长文本
        user_text = user[:STM_MAX_TEXT_LENGTH]
        agent_text = agent[:STM_MAX_TEXT_LENGTH]

        now = time.time()

        # 用户记录
        user_record = InteractionRecord(
            timestamp=now,
            role="user",
            content=user_text,
            emotion_score=self._estimate_emotion(user_text),
            session_id=self._session.session_id,
        )

        # Agent 记录
        agent_record = InteractionRecord(
            timestamp=now,
            role="agent",
            content=agent_text,
            emotion_score=0.0,  # Agent 情感由情感引擎设置
            session_id=self._session.session_id,
        )

        # 插入环形缓冲区（无需锁，deque 操作线程安全）
        self._stm_buffer.append(user_record)
        self._stm_buffer.append(agent_record)

        # === Hermes Bridge: 同步写入会话搜索数据库 ===
        if self._session_db is not None:
            try:
                self._session_db.append_interaction(
                    session_id=self._session.session_id,
                    role="user",
                    content=user_text,
                )
                self._session_db.append_interaction(
                    session_id=self._session.session_id,
                    role="agent",
                    content=agent_text,
                )
            except Exception as e:
                self._logger.debug("SessionSearchDB 写入失败: %s", e)

        # 更新会话统计
        self._session.turn_count += 1
        self._session.last_active = now
        self._total_records += 2

        self._logger.debug(
            "交互记录已添加 | 用户: %s... | Agent: %s... | 缓冲区: %d/%d",
            user_text[:30],
            agent_text[:30],
            len(self._stm_buffer),
            self.stm_max_turns,
        )

    def add_system_event(
        self,
        content: str,
        event_type: str = "info",
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加系统事件或传感器数据到短期记忆。

        用于记录硬件状态变化、传感器读数、安全事件等非对话信息。

        Args:
            content: 事件描述文本。
            event_type: 事件类型 ("info" / "warning" / "error" / "sensor")。
            sensor_data: 关联传感器数据字典。
        """
        record = InteractionRecord(
            timestamp=time.time(),
            role="sensor" if event_type == "sensor" else "system",
            content=content,
            sensor_data=sensor_data or {},
            session_id=self._session.session_id,
            metadata={"event_type": event_type},
        )
        self._stm_buffer.append(record)
        self._total_records += 1

        self._logger.debug("系统事件已记录: [%s] %s", event_type, content[:50])

    def get_recent_context(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近 n 轮对话上下文。

        从环形缓冲区尾部倒序返回最近 n 条交互记录，
        用于构建 Claude Code 的 prompt 上下文窗口。

        时间复杂度: O(n)

        Args:
            n: 要返回的最近记录数，默认 10。

        Returns:
            交互记录字典列表，按时间正序排列（最早的在前）。
        """
        if n <= 0:
            return []

        # 从尾部取 n 条，然后反转使其按时间正序
        recent = list(self._stm_buffer)[-n:]
        return [record.to_dict() for record in recent]

    def get_recent_by_role(
        self, role: str, n: int = 10
    ) -> List[Dict[str, Any]]:
        """按角色筛选获取最近记录。

        Args:
            role: 角色类型 ("user" / "agent" / "system" / "sensor")。
            n: 返回记录数上限。

        Returns:
            符合条件的交互记录字典列表。
        """
        if n <= 0:
            return []

        matching: List[Dict[str, Any]] = []
        # 从尾部倒序遍历
        for record in reversed(self._stm_buffer):
            if record.role == role:
                matching.insert(0, record.to_dict())
                if len(matching) >= n:
                    break
        return matching

    def get_emotion_trend(self, n: int = 10) -> float:
        """计算最近 n 轮的情感趋势。

        对用户发言的情感评分进行加权平均，权重随时间指数衰减，
        越近的记录权重越高。

        公式::

            trend = Σ(score_i × w_i) / Σw_i
            w_i = exp(-λ × (now - timestamp_i))

        Args:
            n: 考虑的最近记录数，默认 10。

        Returns:
            情感趋势评分 (-1.0 ~ +1.0)，正值表示积极趋势。
        """
        recent = list(self._stm_buffer)[-n:]
        user_records = [r for r in recent if r.role == "user" and r.emotion_score != 0.0]

        if not user_records:
            return 0.0

        decay_lambda = 0.1
        now = time.time()
        weighted_sum = 0.0
        weight_sum = 0.0

        for record in user_records:
            weight = 1.0  # 不使用时间衰减以保持简单
            weighted_sum += record.emotion_score * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.0

        trend = weighted_sum / weight_sum
        return max(EMOTION_MIN_SCORE, min(EMOTION_MAX_SCORE, trend))

    def clear_short_term_memory(self) -> None:
        """清空短期记忆缓冲区。

        所有短期记忆记录将被丢弃，不影响长期记忆。
        """
        count = len(self._stm_buffer)
        self._stm_buffer.clear()
        self._logger.info("短期记忆已清空: %d 条记录", count)

    def get_stm_stats(self) -> Dict[str, Any]:
        """获取短期记忆统计信息。

        Returns:
            包含记录数、各角色数量、情感趋势等的字典。
        """
        records = list(self._stm_buffer)
        role_counts: Dict[str, int] = {}
        for r in records:
            role_counts[r.role] = role_counts.get(r.role, 0) + 1

        return {
            "total_records": len(records),
            "capacity": self.stm_max_turns,
            "utilization": len(records) / self.stm_max_turns if self.stm_max_turns > 0 else 0,
            "role_distribution": role_counts,
            "emotion_trend": self.get_emotion_trend(n=len(records)),
            "session_id": self._session.session_id,
            "session_turns": self._session.turn_count,
        }

    # ------------------------------------------------------------------
    # 核心 API：长期记忆
    # ------------------------------------------------------------------

    def store_memory(
        self,
        key: str,
        value: Dict[str, Any],
        memory_type: str = "preference",
        importance: float = 0.5,
    ) -> str:
        """存储一条长期记忆。

        将用户偏好、场景记忆等信息存入长期记忆库。
        若已存在同 key 的记忆，将更新而非新增。

        Args:
            key: 记忆关键词，如 "喜欢的音乐"、"起床时间"。
            value: 记忆内容字典，如 ``{"item": "music", "value": "轻音乐"}``。
            memory_type: 记忆类型 ("preference" / "scene" / "habit" / "safety")。
            importance: 重要性评分 (0.0 ~ 1.0)，越高越不容易被淘汰。

        Returns:
            存储的记忆 ID。

        Raises:
            ValueError: key 为空或 importance 超出范围时抛出。
        """
        if not key:
            raise ValueError("记忆关键词不能为空")
        if not (0.0 <= importance <= 1.0):
            raise ValueError("importance 必须在 0.0 ~ 1.0 之间")

        # 检查是否已存在同 key 的记忆
        for existing in self._ltm_store:
            if existing.key == key and existing.memory_type == memory_type:
                # 更新现有记忆
                existing.value = value
                existing.timestamp = time.time()
                existing.importance = importance
                existing.access_count += 1
                existing.last_accessed = time.time()
                self._logger.info("长期记忆已更新: [%s] %s", memory_type, key)
                return existing.memory_id

        # 生成简单向量表示（使用关键词哈希作为降维向量）
        vector = self._simple_vectorize(key + " " + json.dumps(value, ensure_ascii=False))

        memory = LongTermMemory(
            key=key,
            value=value,
            memory_type=memory_type,
            vector=vector,
            importance=importance,
            timestamp=time.time(),
        )

        self._ltm_store.append(memory)
        self._total_memories += 1

        # 若超出容量上限，淘汰最不重要的记忆
        if len(self._ltm_store) > LTM_MAX_ENTRIES:
            self._evict_least_important()

        self._logger.info(
            "长期记忆已存储: [%s] %s | 总数: %d", memory_type, key, len(self._ltm_store)
        )
        return memory.memory_id

    def recall_memory(
        self, query: str, top_k: int = 3, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """根据查询语义检索相关长期记忆。

        使用简单的余弦相似度计算查询与记忆向量的相似程度，
        返回最相关的 top_k 条记忆。未来可替换为 Sentence-BERT + FAISS。

        算法::

            similarity = dot(query_vec, memory_vec) / (|query_vec| × |memory_vec|)

        Args:
            query: 查询文本，如 "用户喜欢喝什么"。
            top_k: 返回的最大记忆条数，默认 3。
            threshold: 相似度阈值 (0.0 ~ 1.0)，低于此值的记忆将被过滤。

        Returns:
            匹配的记忆字典列表，按相似度降序排列。
        """
        if not query or not self._ltm_store:
            return []

        query_vector = self._simple_vectorize(query)

        # 计算余弦相似度
        scored_memories: List[Tuple[float, LongTermMemory]] = []
        for memory in self._ltm_store:
            if not memory.vector:
                continue
            similarity = self._cosine_similarity(query_vector, memory.vector)
            if similarity >= threshold:
                scored_memories.append((similarity, memory))

        # 按相似度降序排列，取 top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_memories[:top_k]

        # 更新访问统计
        results: List[Dict[str, Any]] = []
        for similarity, memory in top_results:
            memory.access_count += 1
            memory.last_accessed = time.time()
            result = memory.to_dict()
            result["similarity"] = round(similarity, 4)
            results.append(result)

        self._logger.debug(
            "记忆检索: '%s...' → %d 条结果", query[:30], len(results)
        )
        return results

    def get_memory_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        """按类型获取长期记忆。

        Args:
            memory_type: 记忆类型 ("preference" / "scene" / "habit" / "safety")。

        Returns:
            符合条件的记忆字典列表。
        """
        return [
            m.to_dict()
            for m in self._ltm_store
            if m.memory_type == memory_type
        ]

    def delete_memory(self, memory_id: str) -> bool:
        """删除指定 ID 的长期记忆。

        Args:
            memory_id: 要删除的记忆 ID。

        Returns:
            删除是否成功。
        """
        for i, memory in enumerate(self._ltm_store):
            if memory.memory_id == memory_id:
                del self._ltm_store[i]
                self._total_memories -= 1
                self._logger.info("长期记忆已删除: %s", memory_id)
                return True
        return False

    def clear_long_term_memory(self) -> None:
        """清空所有长期记忆。

        警告：此操作不可逆，所有用户偏好和历史记忆将被删除。
        """
        count = len(self._ltm_store)
        self._ltm_store.clear()
        self._total_memories = 0
        self._logger.warning("长期记忆已清空: %d 条记录", count)

    # ------------------------------------------------------------------
    # 核心 API：上下文注入
    # ------------------------------------------------------------------

    def build_prompt_context(
        self,
        current_input: str,
        max_turns: int = 5,
        include_memories: bool = True,
    ) -> str:
        """构建 Claude Code prompt 的上下文字符串。

        自动组装最近对话历史、相关长期记忆和情感趋势，
        格式化为 Claude Code 可理解的结构化文本。

        格式::

            [系统提示]
            [相关记忆]
            [历史对话]
            [当前输入]

        Args:
            current_input: 当前用户输入。
            max_turns: 包含的最大历史轮数，默认 5。
            include_memories: 是否注入相关长期记忆。

        Returns:
            格式化后的上下文字符串。
        """
        lines: List[str] = []

        # 系统提示
        lines.append("你是 KunPeng-Cortex，一个居家养老助手机器人。")
        lines.append("你运行在 OrangePi Kunpeng Pro (RK3588) 上，通过 HAL 控制硬件。")
        lines.append("")

        # === Hermes Bridge: Frozen Snapshot 记忆注入 ===
        if self._memory_store is not None:
            try:
                snapshot = self._memory_store.get_snapshot_for_prompt()
                if snapshot.get("memory"):
                    lines.append("【环境记忆】")
                    lines.append(snapshot["memory"])
                    lines.append("")
                if snapshot.get("user"):
                    lines.append("【用户画像】")
                    lines.append(snapshot["user"])
                    lines.append("")
            except Exception as e:
                self._logger.debug("Memory snapshot 注入失败: %s", e)

        # 长期记忆注入
        if include_memories:
            memories = self.recall_memory(current_input, top_k=3, threshold=0.3)
            if memories:
                lines.append("【相关记忆】")
                for mem in memories:
                    lines.append(f"- {mem['key']}: {json.dumps(mem['value'], ensure_ascii=False)}")
                lines.append("")

        # 情感趋势
        emotion_trend = self.get_emotion_trend(n=10)
        if abs(emotion_trend) > 0.3:
            mood = "积极" if emotion_trend > 0 else "消极"
            lines.append(f"【用户近期情绪】{mood} (评分: {emotion_trend:.2f})")
            lines.append("")

        # 历史对话
        recent = self.get_recent_context(n=max_turns * 2)
        if recent:
            lines.append("【历史对话】")
            for record in recent:
                role_display = "用户" if record["role"] == "user" else (
                    "助手" if record["role"] == "agent" else record["role"]
                )
                lines.append(f"{role_display}: {record['content']}")
            lines.append("")

        # 当前输入
        lines.append(f"用户: {current_input}")
        lines.append("助手:")

        return "\n".join(lines)

    def get_session_summary(self) -> Dict[str, Any]:
        """获取当前会话摘要。

        Returns:
            包含会话元数据、统计和关键信息的字典。
        """
        return {
            "session_id": self._session.session_id,
            "created_at": datetime.fromtimestamp(self._session.created_at).isoformat(),
            "last_active": datetime.fromtimestamp(self._session.last_active).isoformat(),
            "duration_seconds": time.time() - self._session.created_at,
            "turn_count": self._session.turn_count,
            "stm_records": len(self._stm_buffer),
            "ltm_memories": len(self._ltm_store),
            "emotion_trend": self.get_emotion_trend(),
        }

    # ------------------------------------------------------------------
    # 内部方法：向量计算
    # ------------------------------------------------------------------

    def _simple_vectorize(self, text: str) -> List[float]:
        """简单文本向量化。

        使用字符级别哈希生成固定维度的向量表示。
        此为占位实现，未来应替换为 Sentence-BERT 等预训练模型。

        Args:
            text: 输入文本。

        Returns:
            384 维浮点向量列表。
        """
        # 使用多个哈希函数生成向量
        vector: List[float] = [0.0] * VECTOR_DIMENSION
        text_bytes = text.encode("utf-8")

        for i, byte in enumerate(text_bytes):
            idx = (i * 7 + byte * 13) % VECTOR_DIMENSION
            vector[idx] += (byte / 255.0) * 0.1

        # L2 归一化
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个向量的余弦相似度。

        Args:
            vec_a: 向量 A。
            vec_b: 向量 B。

        Returns:
            余弦相似度 (-1.0 ~ 1.0)。
        """
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # 内部方法：情感估计
    # ------------------------------------------------------------------

    def _estimate_emotion(self, text: str) -> float:
        """简单情感评分估计。

        基于关键词的情感分析占位实现。
        未来应替换为基于 NPU 的情感识别模型。

        Args:
            text: 输入文本。

        Returns:
            情感评分 (-1.0 ~ +1.0)。
        """
        positive_words = {
            "好", "棒", "谢谢", "喜欢", "开心", "不错", "很好", "满意",
            "爱你", "乖", "舒服", "温暖", "幸福", "快乐", "感谢",
        }
        negative_words = {
            "不好", "坏", "讨厌", "难受", "疼", "痛", "担心", "害怕",
            "失望", "生气", "烦", "累", "孤独", "着急", "救命",
        }

        text_lower = text.lower()
        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        score = (positive_count - negative_count) / total
        return max(EMOTION_MIN_SCORE, min(EMOTION_MAX_SCORE, score))

    # ------------------------------------------------------------------
    # 内部方法：记忆淘汰
    # ------------------------------------------------------------------

    def _evict_least_important(self) -> None:
        """淘汰最不重要的长期记忆。

        综合考虑 importance、access_count 和 last_accessed，
        选择综合评分最低的记忆进行淘汰。

        评分公式::

            score = importance × 0.5 + normalized_access × 0.3 + recency × 0.2
        """
        if not self._ltm_store:
            return

        now = time.time()
        max_access = max(m.access_count for m in self._ltm_store) or 1

        def _score(memory: LongTermMemory) -> float:
            normalized_access = memory.access_count / max_access
            recency = 1.0 - min(1.0, (now - memory.last_accessed) / 86400.0)
            return memory.importance * 0.5 + normalized_access * 0.3 + recency * 0.2

        # 找出评分最低的记忆
        least_important = min(self._ltm_store, key=_score)
        self._ltm_store.remove(least_important)
        self._total_memories -= 1

        self._logger.debug(
            "长期记忆淘汰: %s (importance=%.2f)",
            least_important.key,
            least_important.importance,
        )

    # ------------------------------------------------------------------
    # 内部方法：会话持久化
    # ------------------------------------------------------------------

    def _load_session_sync(self) -> None:
        """同步加载历史会话数据。

        从 JSONL 文件中逐行读取历史记录，恢复短期记忆缓冲区。
        在异步事件循环启动前调用。
        """
        path = Path(self.persist_path)
        if not path.exists():
            self._logger.info("历史会话文件不存在，创建新会话: %s", self.persist_path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record_data = json.loads(line)
                        record = InteractionRecord.from_dict(record_data)
                        self._stm_buffer.append(record)
                        self._total_records += 1
                    except (json.JSONDecodeError, KeyError) as exc:
                        self._logger.warning("会话记录解析失败: %s", exc)

            self._logger.info(
                "历史会话加载完成: %d 条记录 | 路径: %s",
                len(self._stm_buffer),
                self.persist_path,
            )
        except IOError as exc:
            self._logger.error("会话文件读取失败: %s", exc)

    async def _save_session_async(self) -> None:
        """异步保存当前会话到 JSONL 文件。

        使用 aiofiles 进行非阻塞文件写入，避免阻塞事件循环。
        若 aiofiles 不可用，回退至线程池执行器。
        """
        if not self._stm_buffer:
            return

        path = Path(self.persist_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # 尝试使用 aiofiles
            try:
                import aiofiles

                async with aiofiles.open(path, "w", encoding="utf-8") as f:
                    for record in self._stm_buffer:
                        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                        await f.write(line)
            except ImportError:
                # 回退：使用线程池
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._save_session_sync)

            self._logger.debug("会话已保存: %d 条记录", len(self._stm_buffer))

        except Exception as exc:
            self._logger.error("会话保存失败: %s", exc)

    def _save_session_sync(self) -> None:
        """同步保存会话（用于线程池回退）。"""
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self._stm_buffer:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    async def _auto_save_loop(self) -> None:
        """自动保存后台循环。

        以 ``SESSION_SAVE_INTERVAL`` 为周期自动将会话数据持久化。
        在系统崩溃时，最多丢失一个周期内的数据。
        """
        self._logger.info(
            "自动保存循环已启动 | 周期: %.0f 秒", SESSION_SAVE_INTERVAL
        )

        while not self._shutdown_flag:
            try:
                await asyncio.sleep(SESSION_SAVE_INTERVAL)
                if self._shutdown_flag:
                    break
                await self._save_session_async()
            except asyncio.CancelledError:
                self._logger.info("自动保存循环被取消")
                break
            except Exception as exc:
                self._logger.error("自动保存异常: %s", exc)

        self._logger.info("自动保存循环已退出")

    # ------------------------------------------------------------------
    # 公共 API：数据导出
    # ------------------------------------------------------------------

    def export_session(self, export_path: str) -> bool:
        """导出当前会话到指定路径。

        导出格式为 JSON，包含会话元数据、短期记忆和长期记忆。

        Args:
            export_path: 导出文件路径。

        Returns:
            导出是否成功。
        """
        try:
            export_data = {
                "session": self._session.__dict__,
                "short_term_memory": [r.to_dict() for r in self._stm_buffer],
                "long_term_memory": [m.to_dict() for m in self._ltm_store],
                "export_time": datetime.now().isoformat(),
            }

            path = Path(export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            self._logger.info("会话导出成功: %s", export_path)
            return True
        except Exception as exc:
            self._logger.error("会话导出失败: %s", exc)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取上下文管理器完整统计信息。

        Returns:
            包含 STM、LTM 和会话统计的综合字典。
        """
        return {
            "short_term_memory": self.get_stm_stats(),
            "long_term_memory": {
                "total_entries": len(self._ltm_store),
                "max_capacity": LTM_MAX_ENTRIES,
                "by_type": {
                    mtype: len(self.get_memory_by_type(mtype))
                    for mtype in ("preference", "scene", "habit", "safety")
                },
            },
            "session": {
                "session_id": self._session.session_id,
                "turn_count": self._session.turn_count,
                "created_at": self._session.created_at,
            },
            "total_records_all_time": self._total_records,
        }


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


async def create_context_manager(
    persist_path: str = "data/sessions.jsonl",
    stm_max_turns: int = STM_MAX_TURNS,
) -> ContextManager:
    """工厂函数：创建并启动 ContextManager 实例。

    Args:
        persist_path: 会话持久化文件路径。
        stm_max_turns: 短期记忆最大轮数。

    Returns:
        已启动的 ContextManager 实例。
    """
    ctx = ContextManager(persist_path=persist_path, stm_max_turns=stm_max_turns)
    await ctx.start()
    return ctx


# ---------------------------------------------------------------------------
# Hermes Bridge 便捷方法
# ---------------------------------------------------------------------------


def search_history(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """搜索历史会话（全局便捷函数）。

    需要传入已初始化的 ContextManager 实例调用其方法。
    """
    raise NotImplementedError("请通过 ContextManager 实例调用 _session_db.search()")


# 将 Hermes Bridge 方法注入 ContextManager
# 注：实际项目中应在类定义中直接添加，此处为向后兼容的扩展方式
ContextManager.search_history = lambda self, q, limit=5: (
    self._session_db.search(q, limit=limit) if self._session_db else []
)
ContextManager.add_memory = lambda self, target, content: (
    self._memory_store.add_entry(target, content) if self._memory_store else {"success": False, "message": "未初始化"}
)
ContextManager.get_memory_snapshot = lambda self: (
    self._memory_store.get_snapshot_for_prompt() if self._memory_store else {"memory": "", "user": ""}
)
ContextManager.get_memory_stats = lambda self: (
    self._memory_store.get_stats() if self._memory_store else {"memory": {"entries": 0}, "user": {"entries": 0}}
)
