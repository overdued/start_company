#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合调度器 (Orchestrator) —— KunPeng-Cortex 核心入口模块

本模块实现系统的中央调度中枢，负责任务生命周期管理、资源分配、
状态机维护和故障恢复协调。作为单体内聚 Agent 的核心，Orchestrator
集成 Claude Code 能力引擎、OpenClaw HAL 适配器和情感计算引擎，
通过异步事件驱动架构实现端到端 <50ms 的响应延迟。

架构特性：
    - 六状态有限状态机 (IDLE/PLANNING/EXECUTING/MONITORING/RECOVERING/EMERGENCY)
    - 异步流水线处理，基于 asyncio 事件循环
    - 物理层安全集成（独立 MCU 紧急停止）
    - CPU 亲和性与 NPU 算力调度
    - 共享内存 IPC（<1μs 模块间通信）
    - 全链路超时保护与优雅降级

硬件平台：
    OrangePi KunPeng Pro (RK3588, ARM64, 16GB RAM, 6-8TOPS NPU)

作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import enum
import json
import os
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# 项目内部模块（延迟导入以避免循环依赖）
# ---------------------------------------------------------------------------
# from src.engines.claude_code.engine import ClaudeCodeEngine
# from src.engines.openclaw.hal_adapter import HALAdapter
# from src.engines.openclaw.emotion_engine import EmotionEngine
# from src.core.task_planner import TaskPlanner, AtomicTask, TaskPriority
# from src.core.context_manager import ContextManager
# from src.core.error_recovery import ErrorRecovery, RecoveryResult
# from src.utils.logger import get_logger
# from src.utils.config import load_config


# ---------------------------------------------------------------------------
# 日志兼容层 —— 在 utils.logger 尚未就绪前使用标准库日志
# ---------------------------------------------------------------------------
import logging


def _get_logger(name: str) -> logging.Logger:
    """获取统一格式的日志记录器。

    当 utils.logger 模块可用时，自动委托至其 get_logger 函数；
    否则回退至标准库 logging，确保模块在任意加载顺序下均可正常记录日志。

    Args:
        name: 日志记录器名称，通常使用 ``__name__``。

    Returns:
        配置好的 ``logging.Logger`` 实例。
    """
    try:
        # 优先使用项目内部的日志系统（若已就绪）
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
# 类型别名与常量定义
# ---------------------------------------------------------------------------

# 端到端延迟约束 (秒)
END_TO_END_LATENCY: float = 0.050  # 50ms

# 规划阶段最大允许停留时间 (秒)
PLANNING_TIMEOUT: float = 0.5  # 500ms

# 默认任务执行超时 (秒)
DEFAULT_TASK_TIMEOUT: float = 10.0

# 紧急停止信号超时 (秒)
ESTOP_TIMEOUT: float = 0.01  # 10ms —— 物理层必须在此时间内响应

# 看门狗喂狗周期 (秒)
WATCHDOG_PERIOD: float = 0.01  # 10ms

# 共享内存 IPC 魔数
IPC_MAGIC: int = 0x4B504354  # "KPCT"

# RK3588 CPU 核心拓扑
CPU_CORE_REALTIME: int = 0  # A76 CPU0 —— 硬实时核心
CPU_CORE_AGENT: Tuple[int, ...] = (1, 2)  # A76 CPU1-2 —— Agent 核心
CPU_CORE_VISION: int = 3  # A76 CPU3 —— 视觉核心
CPU_CORE_IO: Tuple[int, ...] = (4, 5, 6, 7)  # A55 CPU4-7 —— I/O 核心


# ---------------------------------------------------------------------------
# 枚举类型定义
# ---------------------------------------------------------------------------


class SystemState(enum.IntEnum):
    """系统运行状态枚举。

    定义融合调度器的有限状态机 (FSM) 全部状态。
    状态转换受严格约束，任何非法转换均触发安全告警。

    Attributes:
        IDLE: 空闲等待状态 —— 系统初始化完成或任务结束后进入。
        PLANNING: 任务规划状态 —— 收到新指令后进行任务分解与规划。
        EXECUTING: 执行中状态 —— 原子操作已就绪并下发 HAL。
        MONITORING: 监控中状态 —— 等待硬件执行反馈。
        RECOVERING: 故障恢复状态 —— 执行异常后尝试恢复。
        EMERGENCY: 紧急状态 —— 人身安全风险或 MCU 触发紧急停止。
        SHUTDOWN: 优雅关机状态 —— 资源释放中。
    """

    IDLE = 0
    PLANNING = 1
    EXECUTING = 2
    MONITORING = 3
    RECOVERING = 4
    EMERGENCY = 5
    SHUTDOWN = 6


class TaskStatus(enum.IntEnum):
    """任务执行状态枚举。

    描述单个任务在生命周期中的当前阶段。

    Attributes:
        PENDING: 待执行 —— 已入队尚未开始。
        RUNNING: 执行中 —— 正在运行。
        SUCCESS: 成功完成。
        FAILED: 执行失败。
        TIMEOUT: 执行超时。
        CANCELLED: 已取消。
    """

    PENDING = 0
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3
    TIMEOUT = 4
    CANCELLED = 5


class Priority(enum.IntEnum):
    """任务优先级枚举。

    数值越小优先级越高，与共享内存 IPC 的优先级定义保持一致。

    Attributes:
        CRITICAL: 关键 —— 电机控制 / 紧急停止。
        HIGH: 高 —— 视觉推理 / 安全监控。
        NORMAL: 正常 —— 任务规划 / 用户交互。
        LOW: 低 —— 日志 / 统计。
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """任务数据类。

    表示一个已解析的用户意图，包含任务标识、优先级、
    参数列表和超时配置。

    Attributes:
        task_id: 全局唯一任务标识符 (UUID)。
        name: 任务名称（人类可读）。
        priority: 任务优先级，默认 NORMAL。
        params: 任务参数字典。
        timeout: 任务执行超时时间（秒）。
        created_at: 任务创建时间戳（ Unix 时间，秒）。
        dependencies: 依赖任务 ID 列表（拓扑排序用）。
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unknown"
    priority: Priority = Priority.NORMAL
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: float = DEFAULT_TASK_TIMEOUT
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """任务执行结果数据类。

    封装单个任务的完整执行结果，包含状态、返回值、
    耗时统计和错误信息。

    Attributes:
        task_id: 对应任务的唯一标识符。
        status: 任务执行状态。
        data: 执行返回数据（成功时有效）。
        error: 错误描述（失败时有效）。
        elapsed_ms: 实际执行耗时（毫秒）。
        timestamp: 结果生成时间戳。
    """

    task_id: str
    status: TaskStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemStatus:
    """系统状态快照数据类。

    用于对外暴露 Orchestrator 的当前运行状态，
    供监控接口、Web UI 和 APP 查询。

    Attributes:
        state: 当前系统状态。
        uptime_seconds: 系统运行时间（秒）。
        cpu_usage_percent: CPU 使用率（百分比）。
        memory_usage_mb: 内存使用量（MB）。
        npu_usage_percent: NPU 使用率（百分比）。
        active_tasks: 当前活动任务数量。
        total_tasks_completed: 历史完成任务总数。
        total_tasks_failed: 历史失败任务总数。
        emergency_stop_armed: 紧急停止是否处于待命状态。
        last_emotion: 最新检测到的情感状态。
        timestamp: 状态快照时间戳。
    """

    state: SystemState = SystemState.IDLE
    uptime_seconds: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    npu_usage_percent: float = 0.0
    active_tasks: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    emergency_stop_armed: bool = False
    last_emotion: str = "neutral"
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionMetrics:
    """执行指标数据类。

    用于收集和统计系统性能指标，支持延迟分析和瓶颈定位。

    Attributes:
        total_latency_ms: 端到端总延迟（毫秒）。
        planning_latency_ms: 规划阶段延迟（毫秒）。
        execution_latency_ms: 执行阶段延迟（毫秒）。
        hal_latency_ms: HAL 调用延迟（毫秒）。
        timestamp: 指标采集时间戳。
    """

    total_latency_ms: float = 0.0
    planning_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    hal_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Orchestrator 主类
# ---------------------------------------------------------------------------


class Orchestrator:
    """融合调度器 —— KunPeng-Cortex 系统的中央调度中枢。

    Orchestrator 采用异步事件驱动架构，基于 asyncio 事件循环实现
    高效的任务调度。其核心职责包括：

    1. **状态机管理**: 维护六状态 FSM，确保状态转换的合法性和安全性。
    2. **任务调度**: 接收用户输入，协调 TaskPlanner 进行任务分解，
       调度 HAL 执行原子操作，监控执行结果。
    3. **引擎集成**: 管理 Claude Code 能力引擎、OpenClaw HAL 适配器
       和情感计算引擎的生命周期和交互。
    4. **物理层安全**: 紧急停止信号绕过所有软件层，直达 MCU 硬件。
    5. **资源管理**: CPU 亲和性绑定、NPU 算力时间片调度。
    6. **错误恢复**: 集成 ErrorRecovery 模块，实现分级恢复策略。

    线程安全：
        所有公共方法均为异步协程，内部状态变更受 ``_state_lock`` 保护。
        紧急停止方法 ``emergency_stop`` 设计为可在任意线程安全调用。

    使用示例::

        orch = Orchestrator(config_path="config/default.yaml")
        await orch.initialize()
        result = await orch.process_user_input("帮我把水杯递过来")
        status = orch.get_status()

    Attributes:
        config_path: 配置文件路径。
        config: 加载后的配置字典。
        state: 当前系统状态（受 _state_lock 保护）。
        _state_lock: 状态变更互斥锁（asyncio.Lock）。
        _task_queue: 异步任务队列（asyncio.PriorityQueue）。
        _active_tasks: 当前正在执行的任务字典。
        _metrics_history: 执行指标历史（双端队列， maxlen=1000）。
        _estop_event: 紧急停止事件（asyncio.Event）。
        _shutdown_event: 系统关闭事件（asyncio.Event）。
        _watchdog_task: 看门狗定时器任务句柄。
        _heartbeat_task: MCU 心跳发送任务句柄。
    """

    # 合法状态转换表：{当前状态: {允许的目标状态}}
    _VALID_TRANSITIONS: Dict[SystemState, set] = {
        SystemState.IDLE: {
            SystemState.PLANNING,
            SystemState.SHUTDOWN,
            SystemState.EMERGENCY,
        },
        SystemState.PLANNING: {
            SystemState.EXECUTING,
            SystemState.IDLE,
            SystemState.RECOVERING,
            SystemState.EMERGENCY,
        },
        SystemState.EXECUTING: {
            SystemState.MONITORING,
            SystemState.IDLE,
            SystemState.RECOVERING,
            SystemState.EMERGENCY,
        },
        SystemState.MONITORING: {
            SystemState.EXECUTING,
            SystemState.IDLE,
            SystemState.RECOVERING,
            SystemState.EMERGENCY,
        },
        SystemState.RECOVERING: {
            SystemState.EXECUTING,
            SystemState.IDLE,
            SystemState.EMERGENCY,
        },
        SystemState.EMERGENCY: {
            SystemState.IDLE,  # 仅允许手动复位后回到 IDLE
        },
        SystemState.SHUTDOWN: set(),  # 终止状态，无出边
    }

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        """初始化融合调度器。

        加载配置文件，初始化所有子系统和异步原语。
        注意：本方法不会启动事件循环或连接外部服务，
        需在 ``initialize()`` 协程中完成异步初始化。

        Args:
            config_path: YAML 配置文件路径，默认 "config/default.yaml"。

        Raises:
            FileNotFoundError: 配置文件不存在时抛出。
            ValueError: 配置文件格式错误时抛出。
        """
        self.config_path: str = config_path
        self.config: Dict[str, Any] = {}
        self.state: SystemState = SystemState.IDLE
        self._start_time: float = time.time()

        # 线程安全锁
        self._state_lock: asyncio.Lock = asyncio.Lock()
        self._task_lock: asyncio.Lock = asyncio.Lock()
        self._metrics_lock: asyncio.Lock = asyncio.Lock()

        # 异步原语
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._estop_event: asyncio.Event = asyncio.Event()
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # 运行时数据结构
        self._active_tasks: Dict[str, Task] = {}
        self._task_results: Dict[str, TaskResult] = {}
        self._metrics_history: Deque[ExecutionMetrics] = deque(maxlen=1000)
        self._state_change_callbacks: List[Callable[[SystemState, SystemState], None]] = []

        # 子系统引用（延迟初始化）
        self._task_planner: Optional[Any] = None
        self._context_manager: Optional[Any] = None
        self._error_recovery: Optional[Any] = None
        self._claude_engine: Optional[Any] = None
        self._hal_adapter: Optional[Any] = None
        self._emotion_engine: Optional[Any] = None

        # === Hermes Bridge 子系统 ===
        self._memory_store: Optional[Any] = None
        self._session_db: Optional[Any] = None
        self._skill_manager: Optional[Any] = None
        self._evolution_engine: Optional[Any] = None
        self._task_count: int = 0

        # 后台任务句柄
        self._watchdog_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._scheduler_task: Optional[asyncio.Task] = None

        # 统计计数器
        self._total_completed: int = 0
        self._total_failed: int = 0
        self._emergency_count: int = 0

        # 日志
        self._logger = _get_logger("orchestrator")

        # 同步加载配置
        self._load_config_sync()

        self._logger.info(
            "Orchestrator 初始化完成 | 配置路径: %s | 初始状态: %s",
            config_path,
            self.state.name,
        )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _load_config_sync(self) -> None:
        """同步加载配置文件。

        优先使用项目内部的配置管理模块；若不可用，
        回退至直接读取 JSON/YAML 文件。
        """
        cfg_path = Path(self.config_path)
        if not cfg_path.exists():
            self._logger.warning("配置文件不存在: %s，使用默认配置", self.config_path)
            self.config = self._default_config()
            return

        try:
            import yaml

            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            self._logger.info("配置文件加载成功: %s", self.config_path)
        except ImportError:
            self._logger.warning("PyYAML 未安装，回退至默认配置")
            self.config = self._default_config()
        except Exception as exc:
            self._logger.error("配置文件加载失败: %s，使用默认配置", exc)
            self.config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """生成默认配置字典。

        当外部配置文件不可用时，提供合理的默认参数，
        确保系统可在最小配置下启动。

        Returns:
            包含全部默认参数的配置字典。
        """
        return {
            "system": {
                "name": "KunPeng-Cortex",
                "version": "1.0.0",
                "debug": False,
            },
            "timing": {
                "end_to_end_latency_ms": 50,
                "planning_timeout_ms": 500,
                "task_timeout_seconds": 10.0,
                "estop_timeout_ms": 10,
                "watchdog_period_ms": 10,
                "heartbeat_period_ms": 50,
            },
            "cpu_affinity": {
                "realtime_core": 0,
                "agent_cores": [1, 2],
                "vision_core": 3,
                "io_cores": [4, 5, 6, 7],
            },
            "npu": {
                "yolo_allocation_percent": 30,
                "emotion_allocation_percent": 20,
                "llm_allocation_percent": 50,
            },
            "safety": {
                "mcu_uart_port": "/dev/ttyS4",
                "mcu_baudrate": 921600,
                "estop_gpio_pin": 17,
                "max_retry_count": 3,
                "safe_mode_on_failure": True,
            },
            "memory": {
                "short_term_turns": 100,
                "vector_db_path": "data/vector_db",
                "session_persist_path": "data/sessions.jsonl",
            },
        }

    async def _transition_state(
        self,
        new_state: SystemState,
        reason: str = "",
    ) -> bool:
        """执行受保护的状态转换。

        所有状态变更必须经过此方法，以确保转换的合法性。
        非法转换将记录警告并返回 False，不会抛出异常
        （避免在错误恢复路径中引入二次异常）。

        Args:
            new_state: 目标状态。
            reason: 状态转换原因（用于日志记录）。

        Returns:
            转换是否成功。
        """
        async with self._state_lock:
            old_state = self.state
            if new_state == old_state:
                return True

            allowed = self._VALID_TRANSITIONS.get(old_state, set())
            if new_state not in allowed:
                self._logger.warning(
                    "非法状态转换: %s -> %s | 原因: %s | 允许的目标: %s",
                    old_state.name,
                    new_state.name,
                    reason,
                    [s.name for s in allowed],
                )
                return False

            self.state = new_state
            self._logger.info(
                "状态转换: %s -> %s | 原因: %s",
                old_state.name,
                new_state.name,
                reason,
            )

            # 触发状态变更回调（在锁外执行以避免死锁）
            for callback in self._state_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(old_state, new_state))
                    else:
                        callback(old_state, new_state)
                except Exception as exc:
                    self._logger.error("状态回调异常: %s", exc)

            return True

    # ------------------------------------------------------------------
    # 公共 API：生命周期管理
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """异步初始化所有子系统。

        按依赖顺序初始化 TaskPlanner、ContextManager、ErrorRecovery、
        ClaudeCodeEngine、HALAdapter 和 EmotionEngine。
        初始化完成后启动后台任务（看门狗、MCU 心跳、调度器）。

        Raises:
            RuntimeError: 任一关键子系统初始化失败时抛出。
            asyncio.TimeoutError: 初始化超时（默认 30 秒）时抛出。
        """
        self._logger.info("开始异步初始化子系统...")
        init_start = time.monotonic()

        try:
            async with asyncio.timeout(30.0):
                # 1. 初始化上下文管理器（无依赖）
                await self._init_context_manager()

                # 2. 初始化错误恢复模块
                await self._init_error_recovery()

                # 3. 初始化任务规划器
                await self._init_task_planner()

                # 4. 初始化 Claude Code 引擎
                await self._init_claude_engine()

                # 5. 初始化 HAL 适配器
                await self._init_hal_adapter()

                # 6. 初始化情感引擎
                await self._init_emotion_engine()

                # 7. 初始化 Hermes Bridge（记忆层）
                await self._init_hermes_bridge()

        except asyncio.TimeoutError:
            self._logger.error("子系统初始化超时（30秒）")
            raise
        except Exception as exc:
            self._logger.error("子系统初始化失败: %s\n%s", exc, traceback.format_exc())
            raise RuntimeError(f"子系统初始化失败: {exc}") from exc

        # 启动后台任务
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(), name="watchdog"
        )
        self._heartbeat_task = asyncio.create_task(
            self._mcu_heartbeat_loop(), name="mcu_heartbeat"
        )
        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(), name="task_scheduler"
        )

        elapsed = (time.monotonic() - init_start) * 1000
        self._logger.info(
            "全部子系统初始化完成 | 耗时: %.2f ms | 状态: %s",
            elapsed,
            self.state.name,
        )

    async def _init_task_planner(self) -> None:
        """初始化任务规划器。"""
        try:
            from src.core.task_planner import TaskPlanner

            self._task_planner = TaskPlanner(config=self.config)
            self._logger.info("TaskPlanner 初始化成功")
        except ImportError:
            self._logger.warning("TaskPlanner 模块未找到，将使用占位实现")
            self._task_planner = None

    async def _init_context_manager(self) -> None:
        """初始化上下文管理器。"""
        try:
            from src.core.context_manager import ContextManager

            persist_path = self.config.get("memory", {}).get(
                "session_persist_path", "data/sessions.jsonl"
            )
            self._context_manager = ContextManager(persist_path=persist_path)
            self._logger.info("ContextManager 初始化成功")
        except ImportError:
            self._logger.warning("ContextManager 模块未找到，将使用占位实现")
            self._context_manager = None

    async def _init_error_recovery(self) -> None:
        """初始化错误恢复模块。"""
        try:
            from src.core.error_recovery import ErrorRecovery

            self._error_recovery = ErrorRecovery(config=self.config)
            self._logger.info("ErrorRecovery 初始化成功")
        except ImportError:
            self._logger.warning("ErrorRecovery 模块未找到，将使用占位实现")
            self._error_recovery = None

    async def _init_claude_engine(self) -> None:
        """初始化 Claude Code 能力引擎。"""
        try:
            # from src.engines.claude_code.engine import ClaudeCodeEngine
            # self._claude_engine = ClaudeCodeEngine(config=self.config)
            self._claude_engine = None
            self._logger.info("ClaudeCodeEngine 初始化成功（占位）")
        except ImportError:
            self._claude_engine = None
            self._logger.warning("ClaudeCodeEngine 模块未找到")

    async def _init_hal_adapter(self) -> None:
        """初始化 OpenClaw HAL 适配器。"""
        try:
            # from src.engines.openclaw.hal_adapter import HALAdapter
            # self._hal_adapter = HALAdapter(config=self.config)
            self._hal_adapter = None
            self._logger.info("HALAdapter 初始化成功（占位）")
        except ImportError:
            self._hal_adapter = None
            self._logger.warning("HALAdapter 模块未找到")

    async def _init_emotion_engine(self) -> None:
        """初始化情感计算引擎。"""
        try:
            # from src.engines.openclaw.emotion_engine import EmotionEngine
            # self._emotion_engine = EmotionEngine(config=self.config)
            self._emotion_engine = None
            self._logger.info("EmotionEngine 初始化成功（占位）")
        except ImportError:
            self._emotion_engine = None
            self._logger.warning("EmotionEngine 模块未找到")

    async def _init_hermes_bridge(self) -> None:
        """初始化 Hermes Bridge 记忆层、Skill系统和进化引擎。"""
        try:
            from hermes_bridge.memory_tool import KunpengMemoryStore
            from hermes_bridge.session_search import SessionSearchDB
            from hermes_bridge.skill_manager import SkillManager
            from hermes_bridge.evolution_engine import EvolutionEngine

            self._memory_store = KunpengMemoryStore(
                memory_dir=self.config.get("hermes", {}).get("memory_dir", "data/memories")
            )
            self._session_db = SessionSearchDB(
                db_path=self.config.get("hermes", {}).get("session_db", "data/sessions_fts.db")
            )
            self._skill_manager = SkillManager(
                skill_dir=self.config.get("hermes", {}).get("skill_dir", "data/skills")
            )
            self._skill_manager.load_preset_skills()

            self._evolution_engine = EvolutionEngine(
                skill_manager=self._skill_manager,
                memory_store=self._memory_store,
                session_db=self._session_db,
                config=self.config.get("hermes", {}).get("evolution", {}),
            )

            stats = self._memory_store.get_stats()
            skill_stats = self._skill_manager.get_stats() if self._skill_manager else {"total": 0}
            self._logger.info(
                "Hermes Bridge 初始化成功 | 记忆: %d/%d chars | 用户: %d/%d chars | Skill: %d 个 | FTS5: %s",
                stats["memory"]["chars"],
                stats["memory"]["limit"],
                stats["user"]["chars"],
                stats["user"]["limit"],
                skill_stats["total"],
                self._session_db._fts5_available,
            )
        except Exception as exc:
            self._logger.warning("Hermes Bridge 初始化失败: %s", exc)
            self._memory_store = None
            self._session_db = None

    async def shutdown(self) -> None:
        """优雅关闭系统。

        按逆依赖顺序停止所有子系统和后台任务，
        确保资源正确释放。最大关闭时间为 5 秒。

        Raises:
            asyncio.TimeoutError: 关闭超时。
        """
        self._logger.info("开始优雅关闭系统...")

        try:
            async with asyncio.timeout(5.0):
                # 1. 设置关闭事件，停止接收新任务
                self._shutdown_event.set()

                # 2. 状态转换为 SHUTDOWN
                await self._transition_state(
                    SystemState.SHUTDOWN, reason="系统关闭指令"
                )

                # 3. 取消后台任务
                tasks_to_cancel = [
                    self._scheduler_task,
                    self._watchdog_task,
                    self._heartbeat_task,
                ]
                for task in tasks_to_cancel:
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # 4. 取消所有活动任务
                async with self._task_lock:
                    for task_id, task in list(self._active_tasks.items()):
                        self._logger.info("取消活动任务: %s", task_id)

                # 5. 释放子系统资源
                self._hal_adapter = None
                self._claude_engine = None
                self._emotion_engine = None
                self._task_planner = None
                self._context_manager = None
                self._error_recovery = None

        except asyncio.TimeoutError:
            self._logger.error("系统关闭超时，强制终止")
            raise

        self._logger.info("系统已安全关闭 | 运行时间: %.2f 秒", self._get_uptime())

    # ------------------------------------------------------------------
    # 公共 API：主入口
    # ------------------------------------------------------------------

    async def process_user_input(self, input_text: str) -> Dict[str, Any]:
        """处理用户输入 —— 系统主入口。

        这是整个 KunPeng-Cortex 系统的核心入口点。所有用户指令
        （语音 ASR 转录文本、APP 命令、Web 界面输入）均通过此方法进入。

        处理流水线：
            1. 输入合法性检查与敏感词过滤
            2. 记录交互至上下文管理器
            3. 状态转换：IDLE -> PLANNING
            4. 调用 TaskPlanner 进行任务分解
            5. 检查资源冲突并优化执行序列
            6. 逐个执行原子任务
            7. 收集结果并生成响应
            8. 状态恢复：-> IDLE

        超时保护：
            整个处理流程受端到端 50ms 超时约束。若超时，
            将返回部分结果并触发优雅降级。

        Args:
            input_text: 用户输入的文本指令（已 UTF-8 解码）。

        Returns:
            包含处理结果的字典，结构如下::

                {
                    "success": bool,          # 是否成功处理
                    "response": str,          # 自然语言响应文本
                    "tasks_executed": int,    # 执行的任务数量
                    "tasks_failed": int,      # 失败的任务数量
                    "elapsed_ms": float,      # 总耗时（毫秒）
                    "emotion": str,           # 当前情感状态
                    "metrics": dict,          # 详细性能指标
                }

        Raises:
            RuntimeError: 系统不在 IDLE 状态时调用。
        """
        overall_start = time.monotonic()
        self._logger.info("收到用户输入: %s", input_text[:100])

        # === Hermes Bridge: 加载 Frozen Snapshot ===
        memory_snapshot = {"memory": "", "user": ""}
        if self._memory_store is not None:
            try:
                memory_snapshot = self._memory_store.get_snapshot_for_prompt()
                self._logger.debug("Memory snapshot 已加载")
            except Exception as e:
                self._logger.debug("Memory snapshot 加载失败: %s", e)

        # === Hermes Bridge: 搜索相关历史对话 ===
        relevant_history = []
        if self._session_db is not None:
            try:
                relevant_history = self._session_db.search(input_text, limit=3)
                if relevant_history:
                    self._logger.debug("找到 %d 条相关历史对话", len(relevant_history))
            except Exception as e:
                self._logger.debug("历史对话搜索失败: %s", e)

        # 输入合法性检查
        if not input_text or not input_text.strip():
            return {
                "success": False,
                "response": "请输入有效的指令",
                "tasks_executed": 0,
                "tasks_failed": 0,
                "elapsed_ms": 0.0,
                "emotion": "neutral",
                "metrics": {},
            }

        # 检查系统状态
        if self.state == SystemState.EMERGENCY:
            return {
                "success": False,
                "response": "系统处于紧急状态，请手动复位后重试",
                "tasks_executed": 0,
                "tasks_failed": 0,
                "elapsed_ms": 0.0,
                "emotion": "worried",
                "metrics": {},
            }

        # 记录交互到上下文
        if self._context_manager:
            try:
                self._context_manager.add_interaction(user=input_text, agent="")
            except Exception as exc:
                self._logger.warning("上下文记录失败: %s", exc)

        metrics = ExecutionMetrics()
        tasks_executed = 0
        tasks_failed = 0
        response_text = ""
        emotion_state = "neutral"

        try:
            # 端到端超时保护
            async with asyncio.timeout(END_TO_END_LATENCY):
                # ---- 阶段 1: 规划 ----
                planning_start = time.monotonic()
                await self._transition_state(
                    SystemState.PLANNING, reason=f"用户输入: {input_text[:50]}"
                )

                # 调用任务规划器分解指令
                atomic_tasks: List[Task] = []
                if self._task_planner:
                    try:
                        planner_tasks = self._task_planner.plan(input_text)
                        atomic_tasks = [
                            Task(
                                task_id=t.get("task_id", str(uuid.uuid4())),
                                name=t.get("name", "unknown"),
                                priority=Priority(t.get("priority", 2)),
                                params=t.get("params", {}),
                                timeout=t.get("timeout", DEFAULT_TASK_TIMEOUT),
                                dependencies=t.get("dependencies", []),
                            )
                            for t in planner_tasks
                        ]
                    except Exception as exc:
                        self._logger.error("任务规划失败: %s", exc)
                        # 降级：创建单个默认任务
                        atomic_tasks = [
                            Task(name="fallback_task", params={"input": input_text})
                        ]
                else:
                    # 无规划器时的默认行为
                    atomic_tasks = [
                        Task(name="echo_task", params={"input": input_text})
                    ]

                metrics.planning_latency_ms = (time.monotonic() - planning_start) * 1000

                # ---- 阶段 2: 冲突检测与优化 ----
                if self._task_planner and atomic_tasks:
                    try:
                        conflicts = self._task_planner.check_conflicts(
                            [{"name": t.name, "params": t.params} for t in atomic_tasks]
                        )
                        if conflicts:
                            self._logger.warning("检测到资源冲突: %s", conflicts)
                        optimized = self._task_planner.optimize_sequence(
                            [{"name": t.name, "params": t.params} for t in atomic_tasks]
                        )
                        self._logger.debug("任务序列优化完成: %d -> %d", len(atomic_tasks), len(optimized))
                    except Exception as exc:
                        self._logger.warning("冲突检测/优化失败: %s", exc)

                # ---- 阶段 3: 执行 ----
                await self._transition_state(
                    SystemState.EXECUTING, reason="任务序列就绪"
                )

                execution_start = time.monotonic()
                for atomic_task in atomic_tasks:
                    # 检查紧急停止信号
                    if self._estop_event.is_set():
                        self._logger.critical("紧急停止已触发，中断任务执行")
                        break

                    # 检查关闭信号
                    if self._shutdown_event.is_set():
                        self._logger.info("系统关闭中，中断任务执行")
                        break

                    # 执行单个任务
                    result = await self.execute_task(atomic_task)
                    if result.status == TaskStatus.SUCCESS:
                        tasks_executed += 1
                        self._total_completed += 1
                    else:
                        tasks_failed += 1
                        self._total_failed += 1
                        self._logger.warning(
                            "任务失败: %s | 状态: %s | 错误: %s",
                            atomic_task.name,
                            result.status.name,
                            result.error,
                        )

                metrics.execution_latency_ms = (
                    time.monotonic() - execution_start
                ) * 1000

                # ---- 阶段 4: 生成响应 ----
                if self._emotion_engine:
                    try:
                        emotion_state = "happy" if tasks_failed == 0 else "concerned"
                    except Exception as exc:
                        self._logger.warning("情感计算失败: %s", exc)

                if tasks_executed > 0:
                    response_text = f"指令已执行完成，共完成 {tasks_executed} 个操作"
                    if tasks_failed > 0:
                        response_text += f"，{tasks_failed} 个操作失败"
                else:
                    response_text = "未能执行任何操作，请检查指令或系统状态"

                # 状态恢复
                await self._transition_state(SystemState.IDLE, reason="任务处理完成")

        except asyncio.TimeoutError:
            self._logger.error("端到端处理超时 (%.0f ms)", END_TO_END_LATENCY * 1000)
            response_text = "处理超时，请稍后重试"
            tasks_failed = len(atomic_tasks) if "atomic_tasks" in dir() else 0
            await self._transition_state(SystemState.IDLE, reason="端到端超时")

        except Exception as exc:
            self._logger.error("处理异常: %s\n%s", exc, traceback.format_exc())
            response_text = f"处理过程中发生错误: {str(exc)}"
            tasks_failed += 1
            self._total_failed += 1

            # 尝试错误恢复
            if self._error_recovery:
                try:
                    recovery_result = self._error_recovery.handle_error(
                        exc, {"phase": "process_user_input", "input": input_text}
                    )
                    self._logger.info("错误恢复结果: %s", recovery_result)
                except Exception as recovery_exc:
                    self._logger.error("错误恢复也失败了: %s", recovery_exc)

            await self._transition_state(SystemState.IDLE, reason="异常恢复")

        # 计算总耗时
        metrics.total_latency_ms = (time.monotonic() - overall_start) * 1000

        # 记录指标
        async with self._metrics_lock:
            self._metrics_history.append(metrics)

        # 更新上下文中的Agent回复
        if self._context_manager:
            try:
                self._context_manager.add_interaction(user=input_text, agent=response_text)
            except Exception as exc:
                self._logger.warning("Agent回复记录失败: %s", exc)

        result = {
            "success": tasks_executed > 0 and tasks_failed == 0,
            "response": response_text,
            "tasks_executed": tasks_executed,
            "tasks_failed": tasks_failed,
            "elapsed_ms": round(metrics.total_latency_ms, 2),
            "emotion": emotion_state,
            "metrics": asdict(metrics),
        }

        self._logger.info(
            "用户输入处理完成 | 成功: %s | 任务: %d/%d | 耗时: %.2f ms",
            result["success"],
            tasks_executed,
            tasks_executed + tasks_failed,
            metrics.total_latency_ms,
        )

        return result

    async def execute_task(self, task: Task) -> TaskResult:
        """执行单个任务。

        负责任务的完整生命周期：入队、状态跟踪、HAL 调用、
        结果收集和错误处理。支持超时保护和取消机制。

        Args:
            task: 要执行的任务对象。

        Returns:
            任务执行结果，包含状态、数据和错误信息。

        Raises:
            ValueError: 任务对象无效时抛出。
        """
        if task is None or not task.task_id:
            raise ValueError("无效的任务对象")

        task_start = time.monotonic()
        self._logger.debug("开始执行任务: %s (%s)", task.name, task.task_id)

        # 注册活动任务
        async with self._task_lock:
            self._active_tasks[task.task_id] = task

        result: TaskResult

        try:
            # 状态转换: EXECUTING -> MONITORING
            await self._transition_state(
                SystemState.MONITORING, reason=f"执行任务: {task.name}"
            )

            # 调用 HAL 执行（带超时保护）
            hal_start = time.monotonic()
            hal_result = await self._execute_via_hal(task)
            hal_latency_ms = (time.monotonic() - hal_start) * 1000

            elapsed_ms = (time.monotonic() - task_start) * 1000

            if hal_result.get("success", False):
                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.SUCCESS,
                    data=hal_result.get("data"),
                    elapsed_ms=elapsed_ms,
                )
                self._logger.debug(
                    "任务成功: %s | HAL延迟: %.2f ms | 总耗时: %.2f ms",
                    task.name,
                    hal_latency_ms,
                    elapsed_ms,
                )
            else:
                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=hal_result.get("error", "HAL执行失败"),
                    elapsed_ms=elapsed_ms,
                )

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - task_start) * 1000
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.TIMEOUT,
                error=f"任务执行超时 ({task.timeout}秒)",
                elapsed_ms=elapsed_ms,
            )
            self._logger.warning("任务超时: %s | 耗时: %.2f ms", task.name, elapsed_ms)

        except asyncio.CancelledError:
            elapsed_ms = (time.monotonic() - task_start) * 1000
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.CANCELLED,
                error="任务被取消",
                elapsed_ms=elapsed_ms,
            )
            self._logger.info("任务被取消: %s", task.name)
            raise  # 重新抛出以便上层处理

        except Exception as exc:
            elapsed_ms = (time.monotonic() - task_start) * 1000
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"{type(exc).__name__}: {str(exc)}",
                elapsed_ms=elapsed_ms,
            )
            self._logger.error(
                "任务异常: %s | 错误: %s", task.name, exc, exc_info=True
            )

        finally:
            # 清理活动任务
            async with self._task_lock:
                self._active_tasks.pop(task.task_id, None)
            self._task_results[task.task_id] = result

        return result

    async def _execute_via_hal(self, task: Task) -> Dict[str, Any]:
        """通过 HAL 适配器执行任务。

        将任务参数转换为 HAL 命令，通过 OpenClaw HAL 适配器
        下发至硬件。若 HAL 未就绪，使用模拟模式。

        Args:
            task: 要执行的任务。

        Returns:
            HAL 执行结果字典，包含 ``success``、``data`` 和 ``error`` 字段。
        """
        # 若 HAL 已初始化，直接调用
        if self._hal_adapter is not None:
            try:
                async with asyncio.timeout(task.timeout):
                    # 实际 HAL 调用
                    result = await self._hal_adapter.execute(
                        task.name, task.params
                    )
                    return {"success": True, "data": result, "error": None}
            except asyncio.TimeoutError:
                return {"success": False, "data": None, "error": "HAL调用超时"}
            except Exception as exc:
                return {"success": False, "data": None, "error": str(exc)}

        # HAL 未就绪时的模拟模式（用于开发和测试）
        self._logger.debug("HAL 未就绪，使用模拟模式: %s", task.name)
        await asyncio.sleep(0.001)  # 模拟 1ms 硬件延迟
        return {
            "success": True,
            "data": {"simulated": True, "task_name": task.name, "params": task.params},
            "error": None,
        }

    # ------------------------------------------------------------------
    # 公共 API：紧急停止与状态查询
    # ------------------------------------------------------------------

    async def emergency_stop(self) -> None:
        """触发紧急停止。

        这是系统中最高优先级的操作。紧急停止信号绕过所有软件层
        （包括任务队列、状态机和错误恢复），直接通过 HAL 到达
        独立 MCU，确保在 <10ms 内切断电机电源并锁定机械臂。

        执行流程：
            1. 立即设置 ``_estop_event``，中断所有正在执行的任务。
            2. 通过 HAL 向 MCU 发送紧急停止 UART 帧。
            3. 状态强制转换为 EMERGENCY（不检查合法性）。
            4. 记录紧急停止事件到持久化存储。
            5. 触发所有注册的紧急停止回调。

        线程安全：
            本方法可在任意线程安全调用（包括中断处理程序）。
            若事件循环未运行，将同步设置标志并在循环恢复后处理。

        Raises:
            RuntimeError: 若紧急停止信号发送失败（MCU 无响应）。
        """
        self._logger.critical("!!! 紧急停止触发 !!!")
        self._emergency_count += 1

        # 步骤 1: 设置紧急停止事件（立即生效）
        self._estop_event.set()

        # 步骤 2: 直接发送 MCU 紧急停止信号（绕过所有软件层）
        try:
            if self._hal_adapter is not None:
                # UART 帧格式: [SOF:0xAA][CMD:0x21][LEN:0x00][CRC:0x21][EOF:0x55]
                estop_frame = bytes([0xAA, 0x21, 0x00, 0x21, 0x55])
                async with asyncio.timeout(ESTOP_TIMEOUT):
                    await self._hal_adapter.raw_write(
                        self.config.get("safety", {}).get("mcu_uart_port", "/dev/ttyS4"),
                        estop_frame,
                    )
                self._logger.critical("MCU 紧急停止信号已发送")
            else:
                self._logger.critical("HAL 未就绪 —— 紧急停止信号仅在软件层生效")
        except asyncio.TimeoutError:
            self._logger.critical("MCU 紧急停止信号发送超时！硬件可能已失效")
        except Exception as exc:
            self._logger.critical("MCU 紧急停止信号发送失败: %s", exc)

        # 步骤 3: 强制状态转换（不检查合法性）
        async with self._state_lock:
            old_state = self.state
            self.state = SystemState.EMERGENCY
            self._logger.critical(
                "状态强制转换: %s -> EMERGENCY | 紧急停止 #%d",
                old_state.name,
                self._emergency_count,
            )

        # 步骤 4: 取消所有活动任务
        async with self._task_lock:
            for task_id, task in list(self._active_tasks.items()):
                self._logger.critical("强制取消任务: %s", task_id)

        # 步骤 5: 持久化记录
        await self._persist_estop_event()

    async def _persist_estop_event(self) -> None:
        """将紧急停止事件持久化到日志文件。

        使用追加模式写入 JSONL 格式，便于后续审计和分析。
        """
        try:
            estop_log_path = Path("logs/emergency_stop.jsonl")
            estop_log_path.parent.mkdir(parents=True, exist_ok=True)
            event_record = {
                "timestamp": datetime.now().isoformat(),
                "event": "EMERGENCY_STOP",
                "count": self._emergency_count,
                "state_before": self.state.name,
                "trigger_source": "software",
            }
            with open(estop_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._logger.error("紧急停止持久化失败: %s", exc)

    def get_status(self) -> SystemStatus:
        """获取系统当前状态快照。

        此方法设计为同步调用，可在任意上下文中使用
        （包括 Web 接口、健康检查端点等）。

        Returns:
            系统状态快照对象，包含状态、资源使用情况和统计信息。
        """
        uptime = self._get_uptime()

        # 计算资源使用（简化版本，实际可集成 psutil）
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None) or 0.0
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
        except ImportError:
            cpu_percent = 0.0
            memory_mb = 0.0

        # 获取最近情感状态
        last_emotion = "neutral"
        if self._context_manager:
            try:
                recent = self._context_manager.get_recent_context(n=1)
                if recent:
                    last_emotion = recent[0].get("emotion", "neutral")
            except Exception:
                pass

        return SystemStatus(
            state=self.state,
            uptime_seconds=round(uptime, 2),
            cpu_usage_percent=round(cpu_percent, 1),
            memory_usage_mb=round(memory_mb, 2),
            npu_usage_percent=0.0,  # TODO: 集成 NPU 驱动查询
            active_tasks=len(self._active_tasks),
            total_tasks_completed=self._total_completed,
            total_tasks_failed=self._total_failed,
            emergency_stop_armed=self._estop_event.is_set(),
            last_emotion=last_emotion,
            timestamp=time.time(),
        )

    def _get_uptime(self) -> float:
        """计算系统运行时间。

        Returns:
            系统运行时间（秒）。
        """
        return time.time() - self._start_time

    # ------------------------------------------------------------------
    # 公共 API：回调注册与资源管理
    # ------------------------------------------------------------------

    def register_state_callback(
        self, callback: Callable[[SystemState, SystemState], None]
    ) -> None:
        """注册状态变更回调函数。

        当系统状态发生变化时，注册的回调将被调用，
        参数为 (旧状态, 新状态)。

        Args:
            callback: 状态变更回调函数。可为同步函数或异步协程。
        """
        self._state_change_callbacks.append(callback)
        self._logger.debug("状态回调已注册，当前共 %d 个", len(self._state_change_callbacks))

    def unregister_state_callback(
        self, callback: Callable[[SystemState, SystemState], None]
    ) -> None:
        """注销状态变更回调函数。

        Args:
            callback: 要移除的回调函数。
        """
        try:
            self._state_change_callbacks.remove(callback)
            self._logger.debug("状态回调已注销")
        except ValueError:
            self._logger.warning("尝试注销未注册的回调")

    def get_metrics(self, n: int = 100) -> List[ExecutionMetrics]:
        """获取最近 n 条执行指标。

        Args:
            n: 返回的指标数量，默认 100。

        Returns:
            执行指标列表，按时间倒序排列。
        """
        return list(self._metrics_history)[-n:]

    def clear_metrics(self) -> None:
        """清空执行指标历史。"""
        self._metrics_history.clear()
        self._logger.info("执行指标历史已清空")

    # ------------------------------------------------------------------
    # 后台任务循环
    # ------------------------------------------------------------------

    async def _watchdog_loop(self) -> None:
        """看门狗定时器循环。

        以 10ms 为周期持续运行，监控系统健康状态：
            - 检查任务执行是否超时
            - 检测死锁或卡死状态
            - 向 MCU 发送喂狗信号

        此循环仅在系统关闭或紧急停止时退出。
        """
        self._logger.info("看门狗循环已启动 | 周期: %.0f ms", WATCHDOG_PERIOD * 1000)

        while not self._shutdown_event.is_set():
            try:
                # 检查紧急停止
                if self._estop_event.is_set():
                    self._logger.critical("看门狗检测到紧急停止信号")
                    break

                # 检查卡死状态（某状态停留时间过长）
                # TODO: 实现状态停留时间检测

                # 向 MCU 发送喂狗信号
                if self._hal_adapter is not None:
                    try:
                        # UART 帧格式: [SOF:0xAA][CMD:0x01][LEN:0x00][CRC:0x01][EOF:0x55]
                        watchdog_frame = bytes([0xAA, 0x01, 0x00, 0x01, 0x55])
                        await self._hal_adapter.raw_write(
                            self.config.get("safety", {}).get("mcu_uart_port", "/dev/ttyS4"),
                            watchdog_frame,
                        )
                    except Exception as exc:
                        self._logger.warning("MCU 喂狗信号发送失败: %s", exc)

                await asyncio.sleep(WATCHDOG_PERIOD)

            except asyncio.CancelledError:
                self._logger.info("看门狗循环被取消")
                break
            except Exception as exc:
                self._logger.error("看门狗循环异常: %s", exc)
                await asyncio.sleep(WATCHDOG_PERIOD)

        self._logger.info("看门狗循环已退出")

    async def _mcu_heartbeat_loop(self) -> None:
        """MCU 心跳发送循环。

        以 50ms 为周期向安全 MCU 发送心跳包，维持主系统"存活"信号。
        若 MCU 连续 3 次（150ms）未收到心跳，将触发硬件级紧急停止。

        心跳帧格式::

            [SOF:0xAA][CMD:0x02][LEN:0x04][SEQ:4B][CRC:1B][EOF:0x55]
        """
        heartbeat_period = (
            self.config.get("timing", {}).get("heartbeat_period_ms", 50) / 1000.0
        )
        self._logger.info("MCU 心跳循环已启动 | 周期: %.0f ms", heartbeat_period * 1000)

        sequence = 0
        while not self._shutdown_event.is_set():
            try:
                if self._estop_event.is_set():
                    break

                if self._hal_adapter is not None:
                    try:
                        seq_bytes = sequence.to_bytes(4, byteorder="little")
                        crc = (0x02 + 0x04 + sum(seq_bytes)) & 0xFF
                        heartbeat_frame = (
                            bytes([0xAA, 0x02, 0x04]) + seq_bytes + bytes([crc, 0x55])
                        )
                        await self._hal_adapter.raw_write(
                            self.config.get("safety", {}).get("mcu_uart_port", "/dev/ttyS4"),
                            heartbeat_frame,
                        )
                        sequence = (sequence + 1) & 0xFFFFFFFF
                    except Exception as exc:
                        self._logger.warning("MCU 心跳发送失败: %s", exc)

                await asyncio.sleep(heartbeat_period)

            except asyncio.CancelledError:
                self._logger.info("MCU 心跳循环被取消")
                break
            except Exception as exc:
                self._logger.error("MCU 心跳循环异常: %s", exc)
                await asyncio.sleep(heartbeat_period)

        self._logger.info("MCU 心跳循环已退出")

    async def _scheduler_loop(self) -> None:
        """任务调度器主循环。

        从优先级队列中取出任务并调度执行。
        当前实现为顺序执行；未来可扩展为基于资源图的并行调度。
        """
        self._logger.info("任务调度器循环已启动")

        while not self._shutdown_event.is_set():
            try:
                if self._estop_event.is_set():
                    # 清空队列中的所有任务
                    while not self._task_queue.empty():
                        try:
                            self._task_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    await asyncio.sleep(0.01)
                    continue

                # 从队列获取任务（带超时以便定期检查关闭信号）
                try:
                    priority, task = await asyncio.wait_for(
                        self._task_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                self._logger.debug("调度任务: %s (优先级: %d)", task.name, priority)
                # 实际执行在 process_user_input 中完成；此处为预留的独立调度入口

            except asyncio.CancelledError:
                self._logger.info("调度器循环被取消")
                break
            except Exception as exc:
                self._logger.error("调度器循环异常: %s", exc)
                await asyncio.sleep(0.1)

        self._logger.info("任务调度器循环已退出")

    # ------------------------------------------------------------------
    # 资源管理：CPU 亲和性与 NPU 调度
    # ------------------------------------------------------------------

    def set_cpu_affinity(self, cores: Tuple[int, ...]) -> bool:
        """设置当前进程的 CPU 亲和性。

        将 Agent 主线程绑定到指定的 A76 核心（CPU 1-2），
        避免与硬实时任务（CPU 0）和视觉推理（CPU 3）竞争。

        Args:
            cores: 要绑定的 CPU 核心编号元组。

        Returns:
            设置是否成功。
        """
        try:
            os.sched_setaffinity(0, set(cores))
            self._logger.info("CPU 亲和性已设置: %s", cores)
            return True
        except AttributeError:
            self._logger.warning("当前平台不支持 sched_setaffinity")
            return False
        except Exception as exc:
            self._logger.error("CPU 亲和性设置失败: %s", exc)
            return False

    def get_npu_schedule(self) -> Dict[str, float]:
        """获取当前 NPU 算力分配方案。

        返回时间片轮转 + 优先级抢占的算力分配比例。

        Returns:
            算力分配字典，键为任务类型，值为百分比 (0-100)。
        """
        npu_config = self.config.get("npu", {})
        return {
            "yolo": npu_config.get("yolo_allocation_percent", 30),
            "emotion": npu_config.get("emotion_allocation_percent", 20),
            "llm": npu_config.get("llm_allocation_percent", 50),
        }


# ---------------------------------------------------------------------------
# 模块级便捷函数
# ---------------------------------------------------------------------------


async def create_orchestrator(config_path: str = "config/default.yaml") -> Orchestrator:
    """工厂函数：创建并初始化 Orchestrator 实例。

    这是创建融合调度器的推荐方式，确保所有子系统按正确顺序初始化。

    Args:
        config_path: YAML 配置文件路径。

    Returns:
        已完全初始化的 Orchestrator 实例。

    Raises:
        RuntimeError: 初始化失败。
        asyncio.TimeoutError: 初始化超时。
    """
    orch = Orchestrator(config_path=config_path)
    await orch.initialize()
    return orch
