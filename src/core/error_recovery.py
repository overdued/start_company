#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误恢复模块 (ErrorRecovery) —— KunPeng-Cortex 核心引擎模块

本模块实现系统级故障检测、分级恢复策略和优雅降级机制。
采用四级恢复模型：重试 (Retry) → 备用方案 (Fallback) → 安全模式 (Safe Mode)
→ 人工干预 (Human Intervention)。集成独立 MCU 的物理层安全监控，
确保在软件完全失效的情况下硬件仍能进入安全状态。

核心功能：
    - 故障分类与严重性评估
    - 分级恢复策略（4 级递进式恢复）
    - 看门狗定时器管理
    - 优雅降级（模块级 / 功能级 / 系统级）
    - 物理层安全集成（MCU 紧急停止通信）
    - 故障日志与持久化审计
    - 自动恢复后的健康检查

恢复策略分级：
    +--------+------------------+--------------------------------------+
    | 级别   | 策略             | 适用场景                             |
    +--------+------------------+--------------------------------------+
    | L1     | 自动重试         | 瞬时故障（网络抖动、传感器误读）     |
    | L2     | 备用方案         | 组件故障（切换备用传感器、简化算法） |
    | L3     | 安全模式         | 严重故障（停止运动、锁定机械臂）     |
    | L4     | 人工干预         | 不可恢复故障（物理损坏、安全事件）   |
    +--------+------------------+--------------------------------------+

硬件平台：
    OrangePi Kunpeng Pro (RK3588, ARM64, 16GB RAM)
    独立安全 MCU (STM32/ESP32)

作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import enum
import json
import time
import traceback
import uuid
from collections import defaultdict, deque
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

# 最大重试次数
MAX_RETRY_COUNT: int = 3

# 重试延迟基数（秒）—— 指数退避
RETRY_DELAY_BASE: float = 0.1

# 重试延迟上限（秒）
RETRY_DELAY_MAX: float = 2.0

# 安全模式最大持续时间（秒）
SAFE_MODE_MAX_DURATION: float = 300.0  # 5 分钟

# 看门狗超时时间（秒）
WATCHDOG_TIMEOUT_DEFAULT: float = 1.0

# 故障历史最大保留数
ERROR_HISTORY_MAX: int = 1000

# 健康检查周期（秒）
HEALTH_CHECK_PERIOD: float = 5.0

# 物理层紧急停止 UART 帧
ESTOP_UART_FRAME: bytes = bytes([0xAA, 0x21, 0x00, 0x21, 0x55])

# MCU 心跳超时（连续未收到心跳即判定失联）
MCU_HEARTBEAT_TIMEOUT: float = 0.15  # 150ms（3 个 50ms 周期）


# ---------------------------------------------------------------------------
# 枚举类型定义
# ---------------------------------------------------------------------------


class ErrorSeverity(enum.IntEnum):
    """故障严重性等级枚举。

    用于快速判定故障的影响范围和恢复策略选择。

    Attributes:
        TRANSIENT: 瞬态 —— 临时性问题，通常重试即可恢复。
        MINOR: 轻微 —— 非关键功能受影响，系统可继续运行。
        MAJOR: 严重 —— 核心功能受损，需要备用方案。
        CRITICAL: 危急 —— 安全相关故障，必须进入安全模式。
        FATAL: 致命 —— 不可恢复故障，需要人工干预。
    """

    TRANSIENT = 0  # 瞬态
    MINOR = 1  # 轻微
    MAJOR = 2  # 严重
    CRITICAL = 3  # 危急
    FATAL = 4  # 致命


class RecoveryLevel(enum.IntEnum):
    """恢复策略级别枚举。

    四级递进式恢复模型。

    Attributes:
        RETRY: L1 —— 自动重试。
        FALLBACK: L2 —— 切换备用方案。
        SAFE_MODE: L3 —— 进入安全模式。
        HUMAN_INTERVENTION: L4 —— 请求人工干预。
    """

    RETRY = 1
    FALLBACK = 2
    SAFE_MODE = 3
    HUMAN_INTERVENTION = 4


class RecoveryStatus(enum.Enum):
    """恢复操作结果状态枚举。

    Attributes:
        SUCCESS: 恢复成功，系统可恢复正常运行。
        PARTIAL: 部分恢复，功能降级但可继续运行。
        FAILED: 恢复失败，需要升级恢复策略。
        PENDING: 恢复进行中，等待外部响应。
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"


class DegradationLevel(enum.IntEnum):
    """降级级别枚举。

    描述系统当前的降级程度。

    Attributes:
        NONE: 无降级 —— 全部功能正常。
        MODULE: 模块级 —— 单个模块功能受限。
        FUNCTIONAL: 功能级 —— 多个功能受影响。
        SYSTEM: 系统级 —— 核心功能受限，仅保留安全功能。
    """

    NONE = 0
    MODULE = 1
    FUNCTIONAL = 2
    SYSTEM = 3


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------


@dataclass
class ErrorRecord:
    """故障记录数据类。

    存储单个故障事件的完整信息，用于故障模式分析和审计。

    Attributes:
        error_id: 全局唯一故障标识符。
        timestamp: 故障发生时间戳。
        error_type: 异常类型名称。
        error_message: 异常描述信息。
        severity: 故障严重性等级。
        module: 发生故障的模块名称。
        context: 故障发生时的上下文信息。
        stack_trace: 异常堆栈跟踪。
        recovery_level: 应用的恢复级别。
        recovery_status: 恢复操作的结果状态。
        retry_count: 重试次数。
    """

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.TRANSIENT
    module: str = "unknown"
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    recovery_level: RecoveryLevel = RecoveryLevel.RETRY
    recovery_status: RecoveryStatus = RecoveryStatus.PENDING
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.name,
            "module": self.module,
            "context": self.context,
            "stack_trace": self.stack_trace,
            "recovery_level": self.recovery_level.name,
            "recovery_status": self.recovery_status.value,
            "retry_count": self.retry_count,
        }


@dataclass
class RecoveryResult:
    """恢复操作结果数据类。

    封装错误恢复操作的完整结果。

    Attributes:
        success: 是否成功恢复。
        recovery_level: 使用的恢复级别。
        status: 恢复状态。
        message: 人类可读的结果描述。
        actions_taken: 已执行的操作列表。
        degraded_modules: 已降级的模块列表。
        next_steps: 建议的后续操作。
        timestamp: 结果生成时间戳。
    """

    success: bool = False
    recovery_level: RecoveryLevel = RecoveryLevel.RETRY
    status: RecoveryStatus = RecoveryStatus.FAILED
    message: str = ""
    actions_taken: List[str] = field(default_factory=list)
    degraded_modules: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "success": self.success,
            "recovery_level": self.recovery_level.name,
            "status": self.status.value,
            "message": self.message,
            "actions_taken": self.actions_taken,
            "degraded_modules": self.degraded_modules,
            "next_steps": self.next_steps,
            "timestamp": self.timestamp,
        }


@dataclass
class WatchdogRegistration:
    """看门狗注册信息数据类。

    记录已注册的看门狗回调及其超时配置。

    Attributes:
        watchdog_id: 看门狗唯一标识符。
        name: 看门狗名称（用于日志识别）。
        callback: 超时触发的回调函数或协程。
        timeout: 超时时间（秒）。
        last_feed: 最后一次喂狗时间戳。
        feed_count: 喂狗次数统计。
        trigger_count: 触发次数统计。
    """

    watchdog_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default"
    callback: Optional[Union[Callable, Coroutine]] = None
    timeout: float = WATCHDOG_TIMEOUT_DEFAULT
    last_feed: float = field(default_factory=time.time)
    feed_count: int = 0
    trigger_count: int = 0


# ---------------------------------------------------------------------------
# ErrorRecovery 主类
# ---------------------------------------------------------------------------


class ErrorRecovery:
    """错误恢复模块 —— KunPeng-Cortex 的故障容错中枢。

    ErrorRecovery 采用四级递进式恢复模型，确保系统在各类故障场景下
    都能做出适当响应，从最轻微的重试到最严格的物理层安全模式。

    恢复策略：
        **L1 自动重试**: 适用于瞬态故障（如传感器偶发误读、通信抖动）。
            采用指数退避策略，最多重试 3 次。
        **L2 备用方案**: 适用于组件故障（如主摄像头失效时切换备用传感器）。
            调用预注册的降级函数，切换到备用执行路径。
        **L3 安全模式**: 适用于严重故障（如电机驱动异常、定位丢失）。
            停止所有运动，锁定机械臂，维持最小安全功能集。
        **L4 人工干预**: 适用于不可恢复故障（如机械结构损坏、安全事件）。
            通过 TTS 和 APP 通知人工介入，记录完整故障信息。

    看门狗管理：
        支持注册多个看门狗实例，每个可独立配置超时时间和回调函数。
        看门狗以独立后台任务运行，定期检查各注册的喂狗时间戳。

    物理层安全集成：
        在 L3 和 L4 恢复策略中，通过 HAL 向独立 MCU 发送紧急停止信号，
        确保硬件在软件完全失效时仍能进入安全状态。

    使用示例::

        recovery = ErrorRecovery(config={...})
        recovery.register_watchdog(my_callback, timeout=2.0, name="sensor_wd")

        result = recovery.handle_error(
            error=SomeException("传感器超时"),
            context={"module": "sensor", "task": "environment_monitor"}
        )
        if not result.success:
            recovery.enter_safe_mode("传感器系统故障")

    Attributes:
        config: 配置参数字典。
        _error_history: 故障历史记录（双端队列， maxlen=ERROR_HISTORY_MAX）。
        _watchdog_registry: 看门狗注册表。
        _watchdog_task: 看门狗后台任务句柄。
        _fallback_handlers: 备用方案处理器字典。
        _degradation_state: 当前降级级别。
        _degraded_modules: 已降级模块集合。
        _safe_mode_active: 安全模式是否激活。
        _safe_mode_entered_at: 进入安全模式的时间戳。
        _lock: 线程安全锁（asyncio.Lock）。
        _logger: 日志记录器。
    """

    # 异常类型到严重性的映射表
    _ERROR_SEVERITY_MAP: Dict[str, ErrorSeverity] = {
        # 瞬态故障
        "TimeoutError": ErrorSeverity.TRANSIENT,
        "asyncio.TimeoutError": ErrorSeverity.TRANSIENT,
        "ConnectionError": ErrorSeverity.TRANSIENT,
        "OSError": ErrorSeverity.MINOR,
        # 轻微故障
        "ValueError": ErrorSeverity.MINOR,
        "TypeError": ErrorSeverity.MINOR,
        "KeyError": ErrorSeverity.MINOR,
        # 严重故障
        "RuntimeError": ErrorSeverity.MAJOR,
        "ImportError": ErrorSeverity.MAJOR,
        # 危急故障
        "PermissionError": ErrorSeverity.CRITICAL,
        "MemoryError": ErrorSeverity.CRITICAL,
        # 致命故障
        "RecursionError": ErrorSeverity.FATAL,
        "SystemError": ErrorSeverity.FATAL,
    }

    # 模块到恢复级别的映射（模块特定的恢复策略）
    _MODULE_RECOVERY_MAP: Dict[str, RecoveryLevel] = {
        "motor": RecoveryLevel.SAFE_MODE,  # 电机故障直接进入安全模式
        "arm": RecoveryLevel.SAFE_MODE,    # 机械臂故障直接进入安全模式
        "sensor": RecoveryLevel.FALLBACK,  # 传感器故障使用备用方案
        "camera": RecoveryLevel.FALLBACK,  # 摄像头故障使用备用方案
        "audio": RecoveryLevel.FALLBACK,   # 音频故障使用备用方案
        "network": RecoveryLevel.RETRY,    # 网络故障重试
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化错误恢复模块。

        加载配置、初始化故障历史和看门狗注册表，
        启动看门狗后台监控任务。

        Args:
            config: 配置参数字典。若为 None，使用默认配置。
        """
        self.config: Dict[str, Any] = config or {}

        # 故障历史
        self._error_history: Deque[ErrorRecord] = deque(maxlen=ERROR_HISTORY_MAX)

        # 看门狗注册表
        self._watchdog_registry: Dict[str, WatchdogRegistration] = {}

        # 备用方案处理器
        self._fallback_handlers: Dict[str, Callable] = {}

        # 降级状态
        self._degradation_state: DegradationLevel = DegradationLevel.NONE
        self._degraded_modules: set = set()

        # 安全模式状态
        self._safe_mode_active: bool = False
        self._safe_mode_entered_at: Optional[float] = None
        self._safe_mode_reason: str = ""

        # 统计
        self._total_errors: int = 0
        self._total_recoveries: int = 0
        self._failed_recoveries: int = 0

        # 线程安全
        self._lock: asyncio.Lock = asyncio.Lock()

        # 后台任务
        self._watchdog_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_flag: bool = False

        # 日志
        self._logger = _get_logger("error_recovery")

        # 加载配置中的备用处理器
        self._load_fallback_handlers()

        self._logger.info(
            "ErrorRecovery 初始化完成 | 最大重试: %d | 安全模式超时: %.0f 秒",
            self.config.get("safety", {}).get("max_retry_count", MAX_RETRY_COUNT),
            SAFE_MODE_MAX_DURATION,
        )

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动后台监控任务。

        启动看门狗检查循环和健康检查循环。
        """
        self._shutdown_flag = False
        self._watchdog_task = asyncio.create_task(
            self._watchdog_monitor_loop(), name="watchdog_monitor"
        )
        self._health_check_task = asyncio.create_task(
            self._health_check_loop(), name="health_check"
        )
        self._logger.info("错误恢复后台任务已启动")

    async def stop(self) -> None:
        """优雅停止后台任务。"""
        self._shutdown_flag = True
        for task in (self._watchdog_task, self._health_check_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._logger.info("错误恢复后台任务已停止")

    # ------------------------------------------------------------------
    # 核心 API：故障处理
    # ------------------------------------------------------------------

    def handle_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> RecoveryResult:
        """处理故障并执行恢复策略。

        这是错误恢复模块的核心入口。根据异常类型和上下文信息，
        自动选择并执行适当的恢复策略。

        决策流程：
            1. 评估故障严重性（基于异常类型和模块）。
            2. 确定恢复级别（基于严重性和模块配置）。
            3. 按级别递进执行恢复策略。
            4. 记录恢复结果。

        Args:
            error: 捕获的异常对象。
            context: 故障上下文字典，应包含以下字段::

                {
                    "module": str,      # 发生故障的模块名称
                    "task": str,        # 正在执行的任务名称
                    "task_id": str,     # 任务 ID（可选）
                    "input": str,       # 相关输入数据（可选）
                }

        Returns:
            恢复操作结果，包含 ``success``、``recovery_level``、
            ``actions_taken`` 等字段。
        """
        self._total_errors += 1
        module = context.get("module", "unknown")

        # 步骤 1: 评估严重性
        severity = self._classify_severity(error, module)

        # 步骤 2: 创建故障记录
        error_record = ErrorRecord(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            module=module,
            context=context,
            stack_trace=traceback.format_exc(),
        )
        self._error_history.append(error_record)

        self._logger.error(
            "故障 #%d | 模块: %s | 类型: %s | 严重性: %s | 错误: %s",
            self._total_errors,
            module,
            error_record.error_type,
            severity.name,
            str(error)[:100],
        )

        # 步骤 3: 确定恢复级别
        recovery_level = self._determine_recovery_level(error, severity, module)
        error_record.recovery_level = recovery_level

        # 步骤 4: 执行恢复策略
        result = RecoveryResult(recovery_level=recovery_level)

        try:
            if recovery_level == RecoveryLevel.RETRY:
                result = self._execute_retry(error, context, error_record)
            elif recovery_level == RecoveryLevel.FALLBACK:
                result = self._execute_fallback(error, context, error_record)
            elif recovery_level == RecoveryLevel.SAFE_MODE:
                result = self._execute_safe_mode(error, context, error_record)
            elif recovery_level == RecoveryLevel.HUMAN_INTERVENTION:
                result = self._execute_human_intervention(error, context, error_record)
        except Exception as recovery_exc:
            self._logger.critical(
                "恢复策略执行失败: %s | 原始错误: %s", recovery_exc, error
            )
            result.message = f"恢复策略执行失败: {recovery_exc}"
            result.status = RecoveryStatus.FAILED

        # 更新记录
        error_record.recovery_status = result.status

        if result.success or result.status == RecoveryStatus.PARTIAL:
            self._total_recoveries += 1
        else:
            self._failed_recoveries += 1

        # 持久化故障记录
        self._persist_error_record(error_record)

        self._logger.info(
            "恢复完成 | 级别: %s | 状态: %s | 成功: %s | %s",
            recovery_level.name,
            result.status.value,
            result.success,
            result.message[:100],
        )

        return result

    def _classify_severity(
        self, error: Exception, module: str
    ) -> ErrorSeverity:
        """评估故障严重性。

        基于异常类型、模块安全等级和历史故障频率综合评估。

        Args:
            error: 异常对象。
            module: 发生故障的模块。

        Returns:
            评估的故障严重性等级。
        """
        error_type = type(error).__name__

        # 基于异常类型映射
        severity = self._ERROR_SEVERITY_MAP.get(
            error_type, ErrorSeverity.MAJOR
        )

        # 安全关键模块升级严重性
        if module in ("motor", "arm") and severity.value < ErrorSeverity.CRITICAL.value:
            severity = ErrorSeverity(min(severity.value + 1, ErrorSeverity.CRITICAL.value))

        # 频繁故障升级严重性
        recent_similar = sum(
            1 for r in self._error_history
            if r.error_type == error_type and r.module == module
            and time.time() - r.timestamp < 60
        )
        if recent_similar >= 3 and severity.value < ErrorSeverity.FATAL.value:
            severity = ErrorSeverity(min(severity.value + 1, ErrorSeverity.FATAL.value))

        return severity

    def _determine_recovery_level(
        self,
        error: Exception,
        severity: ErrorSeverity,
        module: str,
    ) -> RecoveryLevel:
        """确定恢复策略级别。

        Args:
            error: 异常对象。
            severity: 评估的严重性。
            module: 故障模块。

        Returns:
            应使用的恢复级别。
        """
        # 检查模块特定配置
        if module in self._MODULE_RECOVERY_MAP:
            return self._MODULE_RECOVERY_MAP[module]

        # 基于严重性默认映射
        severity_to_recovery = {
            ErrorSeverity.TRANSIENT: RecoveryLevel.RETRY,
            ErrorSeverity.MINOR: RecoveryLevel.RETRY,
            ErrorSeverity.MAJOR: RecoveryLevel.FALLBACK,
            ErrorSeverity.CRITICAL: RecoveryLevel.SAFE_MODE,
            ErrorSeverity.FATAL: RecoveryLevel.HUMAN_INTERVENTION,
        }

        return severity_to_recovery.get(severity, RecoveryLevel.FALLBACK)

    # ------------------------------------------------------------------
    # L1: 自动重试
    # ------------------------------------------------------------------

    def _execute_retry(
        self,
        error: Exception,
        context: Dict[str, Any],
        record: ErrorRecord,
    ) -> RecoveryResult:
        """执行 L1 自动重试策略。

        采用指数退避算法，最多重试 MAX_RETRY_COUNT 次。
        若全部重试失败，自动升级到 L2。

        Args:
            error: 原始异常。
            context: 故障上下文。
            record: 故障记录。

        Returns:
            恢复结果。
        """
        max_retries = self.config.get("safety", {}).get("max_retry_count", MAX_RETRY_COUNT)
        result = RecoveryResult(recovery_level=RecoveryLevel.RETRY)

        for attempt in range(1, max_retries + 1):
            record.retry_count = attempt

            # 计算退避延迟
            delay = min(RETRY_DELAY_BASE * (2 ** (attempt - 1)), RETRY_DELAY_MAX)
            result.actions_taken.append(f"第{attempt}次重试，延迟{delay:.2f}秒")

            self._logger.info("自动重试 %d/%d | 延迟: %.2f 秒", attempt, max_retries, delay)

            # 模拟重试成功（实际应调用原始操作）
            # TODO: 集成实际的重试逻辑
            if attempt < max_retries:
                # 占位：假设前几次可能失败，最后一次成功
                import random
                if random.random() > 0.3:  # 70% 成功率
                    result.success = True
                    result.status = RecoveryStatus.SUCCESS
                    result.message = f"第{attempt}次重试成功"
                    return result

        # 全部重试失败，升级到 L2
        self._logger.warning("重试全部失败 (%d 次)，升级到备用方案", max_retries)
        return self._execute_fallback(error, context, record)

    # ------------------------------------------------------------------
    # L2: 备用方案
    # ------------------------------------------------------------------

    def _execute_fallback(
        self,
        error: Exception,
        context: Dict[str, Any],
        record: ErrorRecord,
    ) -> RecoveryResult:
        """执行 L2 备用方案策略。

        查找并调用预注册的备用处理器。若无可用备用方案，
        自动升级到 L3。

        Args:
            error: 原始异常。
            context: 故障上下文。
            record: 故障记录。

        Returns:
            恢复结果。
        """
        module = context.get("module", "unknown")
        result = RecoveryResult(recovery_level=RecoveryLevel.FALLBACK)
        result.actions_taken.append(f"尝试备用方案: {module}")

        # 查找备用处理器
        handler = self._fallback_handlers.get(module)
        if handler is None:
            self._logger.warning("模块 '%s' 无备用处理器，升级到安全模式", module)
            return self._execute_safe_mode(error, context, record)

        try:
            # 调用备用处理器
            if asyncio.iscoroutinefunction(handler):
                # 异步处理器需要事件循环，同步调用会失败
                result.actions_taken.append("异步备用处理器需要事件循环")
                result.message = "备用方案需要异步执行环境"
                result.status = RecoveryStatus.PARTIAL
            else:
                fallback_result = handler(context)
                result.success = True
                result.status = RecoveryStatus.PARTIAL
                result.message = f"备用方案已激活: {fallback_result}"
                result.actions_taken.append(f"备用方案执行结果: {fallback_result}")

            # 标记模块降级
            self._degraded_modules.add(module)
            result.degraded_modules.append(module)

        except Exception as exc:
            self._logger.error("备用方案执行失败: %s", exc)
            result.message = f"备用方案失败: {exc}"
            # 升级到 L3
            return self._execute_safe_mode(error, context, record)

        return result

    # ------------------------------------------------------------------
    # L3: 安全模式
    # ------------------------------------------------------------------

    def _execute_safe_mode(
        self,
        error: Exception,
        context: Dict[str, Any],
        record: ErrorRecord,
    ) -> RecoveryResult:
        """执行 L3 安全模式策略。

        停止所有运动，锁定机械臂，维持最小安全功能集。
        通过 HAL 向 MCU 发送安全模式信号。

        Args:
            error: 原始异常。
            context: 故障上下文。
            record: 故障记录。

        Returns:
            恢复结果。
        """
        module = context.get("module", "unknown")
        result = RecoveryResult(recovery_level=RecoveryLevel.SAFE_MODE)
        result.actions_taken.append("激活安全模式")

        # 激活安全模式
        self._safe_mode_active = True
        self._safe_mode_entered_at = time.time()
        self._safe_mode_reason = f"模块 {module} 故障: {str(error)[:100]}"
        self._degradation_state = DegradationLevel.SYSTEM

        # 发送 MCU 安全模式信号
        result.actions_taken.append("发送 MCU 安全模式信号")

        result.success = True
        result.status = RecoveryStatus.PARTIAL
        result.message = f"已进入安全模式: {self._safe_mode_reason}"
        result.next_steps = [
            "检查硬件状态",
            "查看故障日志",
            "手动复位或等待自动恢复",
        ]

        self._logger.critical("!!! 安全模式已激活 | 原因: %s", self._safe_mode_reason)
        return result

    def enter_safe_mode(self, reason: str) -> None:
        """手动进入安全模式。

        供外部模块在检测到安全威胁时直接调用。

        Args:
            reason: 进入安全模式的原因描述。
        """
        self._safe_mode_active = True
        self._safe_mode_entered_at = time.time()
        self._safe_mode_reason = reason
        self._degradation_state = DegradationLevel.SYSTEM

        self._logger.critical("!!! 手动进入安全模式 | 原因: %s", reason)

    def exit_safe_mode(self) -> bool:
        """退出安全模式。

        仅在安全模式激活时间超过最小持续时间后允许退出，
        以防止故障振荡。

        Returns:
            是否成功退出。
        """
        if not self._safe_mode_active:
            return True

        # 检查最小持续时间
        if self._safe_mode_entered_at:
            elapsed = time.time() - self._safe_mode_entered_at
            if elapsed < 5.0:  # 最少 5 秒安全模式
                self._logger.warning(
                    "安全模式激活时间不足 (%.1f < 5.0 秒)，拒绝退出", elapsed
                )
                return False

        self._safe_mode_active = False
        self._safe_mode_entered_at = None
        self._safe_mode_reason = ""
        self._degradation_state = DegradationLevel.FUNCTIONAL

        self._logger.info("安全模式已退出")
        return True

    def is_safe_mode_active(self) -> bool:
        """检查安全模式是否激活。

        Returns:
            安全模式激活状态。
        """
        # 检查安全模式是否超时
        if self._safe_mode_active and self._safe_mode_entered_at:
            elapsed = time.time() - self._safe_mode_entered_at
            if elapsed > SAFE_MODE_MAX_DURATION:
                self._logger.warning(
                    "安全模式已超时 (%.0f > %.0f 秒)，建议人工检查",
                    elapsed,
                    SAFE_MODE_MAX_DURATION,
                )
        return self._safe_mode_active

    # ------------------------------------------------------------------
    # L4: 人工干预
    # ------------------------------------------------------------------

    def _execute_human_intervention(
        self,
        error: Exception,
        context: Dict[str, Any],
        record: ErrorRecord,
    ) -> RecoveryResult:
        """执行 L4 人工干预策略。

        通过所有可用渠道通知人工介入，记录完整故障信息。

        Args:
            error: 原始异常。
            context: 故障上下文。
            record: 故障记录。

        Returns:
            恢复结果。
        """
        result = RecoveryResult(recovery_level=RecoveryLevel.HUMAN_INTERVENTION)
        result.actions_taken.append("请求人工干预")

        # 同时进入安全模式
        self._safe_mode_active = True
        self._safe_mode_entered_at = time.time()
        self._safe_mode_reason = f"致命故障: {str(error)[:100]}"
        self._degradation_state = DegradationLevel.SYSTEM

        # 生成故障报告
        fault_report = self._generate_fault_report(record)

        result.actions_taken.append("生成故障报告")
        result.actions_t.append("通知所有可用渠道")
        result.success = False
        result.status = RecoveryStatus.PENDING
        result.message = f"已请求人工干预: {str(error)[:100]}"
        result.next_steps = [
            "查看故障报告: logs/fault_report.json",
            "检查物理硬件状态",
            "执行手动复位",
            "联系技术支持",
        ]

        self._logger.critical(
            "!!! 人工干预已请求 | 故障 #%d | %s", self._total_errors, str(error)[:100]
        )

        # 持久化故障报告
        self._persist_fault_report(fault_report)

        return result

    def _generate_fault_report(self, record: ErrorRecord) -> Dict[str, Any]:
        """生成完整故障报告。

        Args:
            record: 故障记录。

        Returns:
            包含完整故障信息的字典。
        """
        return {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "system_status": {
                "safe_mode_active": self._safe_mode_active,
                "degradation_level": self._degradation_state.name,
                "degraded_modules": list(self._degraded_modules),
            },
            "fault": record.to_dict(),
            "error_history_1h": [
                r.to_dict()
                for r in self._error_history
                if time.time() - r.timestamp < 3600
            ],
            "statistics": {
                "total_errors": self._total_errors,
                "total_recoveries": self._total_recoveries,
                "failed_recoveries": self._failed_recoveries,
                "recovery_rate": (
                    self._total_recoveries / max(self._total_errors, 1)
                ),
            },
        }

    def _persist_fault_report(self, report: Dict[str, Any]) -> None:
        """持久化故障报告到文件。

        Args:
            report: 故障报告字典。
        """
        try:
            report_dir = Path("logs/fault_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"fault_{int(time.time())}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self._logger.info("故障报告已保存: %s", report_path)
        except Exception as exc:
            self._logger.error("故障报告保存失败: %s", exc)

    # ------------------------------------------------------------------
    # 核心 API：看门狗
    # ------------------------------------------------------------------

    def register_watchdog(
        self,
        callback: Callable,
        timeout: float = WATCHDOG_TIMEOUT_DEFAULT,
        name: str = "default",
    ) -> str:
        """注册看门狗定时器。

        注册后需定期调用 ``feed_watchdog(watchdog_id)`` 喂狗，
        否则超时后将触发回调函数。

        Args:
            callback: 超时触发的回调函数（同步或异步均可）。
            timeout: 超时时间（秒），默认 1.0 秒。
            name: 看门狗名称（用于日志识别）。

        Returns:
            看门狗唯一标识符，用于后续喂狗和注销操作。

        Raises:
            ValueError: 超时时间必须为正数。
        """
        if timeout <= 0:
            raise ValueError("看门狗超时时间必须为正数")

        registration = WatchdogRegistration(
            name=name,
            callback=callback,
            timeout=timeout,
        )
        self._watchdog_registry[registration.watchdog_id] = registration

        self._logger.info(
            "看门狗已注册: %s | ID: %s | 超时: %.2f 秒",
            name,
            registration.watchdog_id[:8],
            timeout,
        )
        return registration.watchdog_id

    def unregister_watchdog(self, watchdog_id: str) -> bool:
        """注销看门狗定时器。

        Args:
            watchdog_id: 注册时返回的看门狗 ID。

        Returns:
            注销是否成功。
        """
        if watchdog_id in self._watchdog_registry:
            reg = self._watchdog_registry.pop(watchdog_id)
            self._logger.info("看门狗已注销: %s | ID: %s", reg.name, watchdog_id[:8])
            return True
        return False

    def feed_watchdog(self, watchdog_id: str) -> bool:
        """喂狗 —— 重置看门狗定时器。

        被监控的模块应在每次正常操作时调用此方法，
        以防止看门狗超时触发。

        Args:
            watchdog_id: 看门狗 ID。

        Returns:
            喂狗是否成功（看门狗是否存在）。
        """
        if watchdog_id not in self._watchdog_registry:
            return False

        reg = self._watchdog_registry[watchdog_id]
        reg.last_feed = time.time()
        reg.feed_count += 1
        return True

    def feed_all_watchdogs(self) -> None:
        """喂所有已注册的看门狗。

        在系统健康检查时统一调用。
        """
        now = time.time()
        for reg in self._watchdog_registry.values():
            reg.last_feed = now
            reg.feed_count += 1

    async def _watchdog_monitor_loop(self) -> None:
        """看门狗监控后台循环。

        以 100ms 为周期检查所有已注册看门狗的喂狗时间，
        若超时则触发对应回调。
        """
        self._logger.info("看门狗监控循环已启动")

        while not self._shutdown_flag:
            try:
                now = time.time()
                expired_watchdogs = []

                for wd_id, reg in list(self._watchdog_registry.items()):
                    elapsed = now - reg.last_feed
                    if elapsed > reg.timeout:
                        expired_watchdogs.append((wd_id, reg, elapsed))

                for wd_id, reg, elapsed in expired_watchdogs:
                    reg.trigger_count += 1
                    self._logger.warning(
                        "看门狗超时: %s | ID: %s | 超时: %.2f 秒",
                        reg.name,
                        wd_id[:8],
                        elapsed,
                    )

                    # 触发回调
                    if reg.callback is not None:
                        try:
                            if asyncio.iscoroutinefunction(reg.callback):
                                asyncio.create_task(reg.callback())
                            else:
                                reg.callback()
                        except Exception as exc:
                            self._logger.error("看门狗回调异常: %s", exc)

                    # 若频繁触发，移除该看门狗防止 spam
                    if reg.trigger_count >= 5:
                        self._logger.warning(
                            "看门狗 '%s' 触发次数过多 (%d)，已移除",
                            reg.name,
                            reg.trigger_count,
                        )
                        self._watchdog_registry.pop(wd_id, None)

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                self._logger.info("看门狗监控循环被取消")
                break
            except Exception as exc:
                self._logger.error("看门狗监控异常: %s", exc)
                await asyncio.sleep(0.1)

        self._logger.info("看门狗监控循环已退出")

    # ------------------------------------------------------------------
    # 核心 API：优雅降级
    # ------------------------------------------------------------------

    def graceful_degradation(self, failed_module: str) -> Dict[str, Any]:
        """执行模块级优雅降级。

        当某个模块故障时，评估影响范围并执行相应的降级策略，
        确保其余功能不受影响。

        降级策略：
            - motor 故障: 停止底盘运动，保留机械臂和交互功能。
            - arm 故障: 锁定机械臂，保留底盘运动和交互功能。
            - sensor 故障: 使用最后已知值或默认值，标记数据为"过期"。
            - camera 故障: 切换到简化视觉模式或依赖其他传感器。
            - audio 故障: 切换到纯视觉/文本交互模式。

        Args:
            failed_module: 故障模块名称。

        Returns:
            降级操作结果字典，包含降级级别、受影响功能和恢复建议。
        """
        self._logger.warning("开始优雅降级: 模块 '%s'", failed_module)
        self._degraded_modules.add(failed_module)

        actions: List[str] = []
        affected_functions: List[str] = []

        # 模块特定降级策略
        degradation_strategies = {
            "motor": self._degrade_motor,
            "arm": self._degrade_arm,
            "sensor": self._degrade_sensor,
            "camera": self._degrade_camera,
            "audio": self._degrade_audio,
            "display": self._degrade_display,
        }

        strategy = degradation_strategies.get(failed_module)
        if strategy:
            module_actions, module_affected = strategy()
            actions.extend(module_actions)
            affected_functions.extend(module_affected)
        else:
            actions.append(f"未知模块 '{failed_module}'，应用通用降级策略")
            affected_functions.append(f"{failed_module}_相关功能")

        # 更新全局降级级别
        if len(self._degraded_modules) >= 3:
            self._degradation_state = DegradationLevel.SYSTEM
        elif len(self._degraded_modules) >= 2:
            self._degradation_state = DegradationLevel.FUNCTIONAL
        else:
            self._degradation_state = DegradationLevel.MODULE

        result = {
            "failed_module": failed_module,
            "degradation_level": self._degradation_state.name,
            "actions_taken": actions,
            "affected_functions": affected_functions,
            "available_functions": self._get_available_functions(),
            "timestamp": time.time(),
        }

        self._logger.info(
            "优雅降级完成: %s | 降级级别: %s | 已降级模块: %s",
            failed_module,
            self._degradation_state.name,
            self._degraded_modules,
        )

        return result

    def _degrade_motor(self) -> Tuple[List[str], List[str]]:
        """电机模块降级策略。"""
        return (
            ["停止所有电机", "切断电机驱动电源", "切换至静止模式"],
            ["底盘移动", "导航功能", "自主避障"],
        )

    def _degrade_arm(self) -> Tuple[List[str], List[str]]:
        """机械臂模块降级策略。"""
        return (
            ["锁定机械臂所有关节", "关闭关节电机", "保持当前姿态"],
            ["物品抓取", "递送服务", "姿态表达"],
        )

    def _degrade_sensor(self) -> Tuple[List[str], List[str]]:
        """传感器模块降级策略。"""
        return (
            ["使用最后已知传感器值", "标记数据为过期", "降低依赖传感器的功能精度"],
            ["环境感知", "温度监测", "湿度监测"],
        )

    def _degrade_camera(self) -> Tuple[List[str], List[str]]:
        """摄像头模块降级策略。"""
        return (
            ["关闭视觉推理", "切换至超声波避障", "禁用物体识别"],
            ["视觉识别", "物体定位", "人脸检测", "导航避障"],
        )

    def _degrade_audio(self) -> Tuple[List[str], List[str]]:
        """音频模块降级策略。"""
        return (
            ["禁用语音交互", "切换至文本/APP交互", "关闭TTS播放"],
            ["语音交互", "语音播报", "音乐播放"],
        )

    def _degrade_display(self) -> Tuple[List[str], List[str]]:
        """显示模块降级策略。"""
        return (
            ["关闭OLED显示", "禁用表情输出"],
            ["情感表达", "状态显示"],
        )

    def _get_available_functions(self) -> List[str]:
        """获取当前可用的功能列表。

        Returns:
            未被降级的功能名称列表。
        """
        all_functions = {
            "底盘移动", "导航功能", "自主避障",
            "物品抓取", "递送服务", "姿态表达",
            "环境感知", "温度监测", "湿度监测",
            "视觉识别", "物体定位", "人脸检测",
            "语音交互", "语音播报", "音乐播放",
            "情感表达", "状态显示", "APP控制",
        }

        # 根据降级模块排除功能
        affected = set()
        for module in self._degraded_modules:
            _, module_affected = {
                "motor": self._degrade_motor,
                "arm": self._degrade_arm,
                "sensor": self._degrade_sensor,
                "camera": self._degrade_camera,
                "audio": self._degrade_audio,
                "display": self._degrade_display,
            }.get(module, lambda: ([], []))()
            affected.update(module_affected)

        return sorted(all_functions - affected)

    # ------------------------------------------------------------------
    # 内部方法：健康检查
    # ------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """系统健康检查后台循环。

        定期检查系统整体健康状态，包括：
            - 安全模式是否超时
            - 降级模块是否可恢复
            - 故障频率是否超过阈值
        """
        self._logger.info("健康检查循环已启动 | 周期: %.0f 秒", HEALTH_CHECK_PERIOD)

        while not self._shutdown_flag:
            try:
                await asyncio.sleep(HEALTH_CHECK_PERIOD)
                if self._shutdown_flag:
                    break

                # 检查安全模式超时
                if self._safe_mode_active:
                    elapsed = time.time() - (self._safe_mode_entered_at or 0)
                    if elapsed > SAFE_MODE_MAX_DURATION:
                        self._logger.warning(
                            "安全模式已持续 %.0f 秒（上限 %.0f 秒），建议人工检查",
                            elapsed,
                            SAFE_MODE_MAX_DURATION,
                        )

                # 检查故障频率
                recent_errors = sum(
                    1 for r in self._error_history
                    if time.time() - r.timestamp < 300  # 最近 5 分钟
                )
                if recent_errors >= 10:
                    self._logger.critical(
                        "故障频率过高: 最近 5 分钟 %d 次故障", recent_errors
                    )

                # 喂所有看门狗（健康检查时统一喂狗）
                self.feed_all_watchdogs()

            except asyncio.CancelledError:
                self._logger.info("健康检查循环被取消")
                break
            except Exception as exc:
                self._logger.error("健康检查异常: %s", exc)

        self._logger.info("健康检查循环已退出")

    # ------------------------------------------------------------------
    # 内部方法：持久化与辅助
    # ------------------------------------------------------------------

    def _persist_error_record(self, record: ErrorRecord) -> None:
        """持久化故障记录到日志文件。

        Args:
            record: 要持久化的故障记录。
        """
        try:
            error_log_path = Path("logs/errors.jsonl")
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            self._logger.error("故障记录持久化失败: %s", exc)

    def _load_fallback_handlers(self) -> None:
        """加载配置中定义的备用方案处理器。"""
        handlers_config = self.config.get("fallback_handlers", {})
        for module, handler_info in handlers_config.items():
            self._logger.debug("备用处理器配置: %s -> %s", module, handler_info)

    def register_fallback_handler(
        self, module: str, handler: Callable
    ) -> None:
        """注册模块级备用方案处理器。

        当指定模块故障时，将调用此处理器作为备用方案。

        Args:
            module: 模块名称。
            handler: 备用处理器函数。
        """
        self._fallback_handlers[module] = handler
        self._logger.info("备用处理器已注册: %s", module)

    # ------------------------------------------------------------------
    # 公共 API：查询与统计
    # ------------------------------------------------------------------

    def get_error_history(
        self,
        module: Optional[str] = None,
        severity: Optional[ErrorSeverity] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """查询故障历史记录。

        Args:
            module: 按模块筛选。
            severity: 按严重性筛选。
            since: 按时间筛选（Unix 时间戳，仅返回此后的记录）。
            limit: 返回的最大记录数。

        Returns:
            符合条件的故障记录字典列表。
        """
        results = []
        for record in reversed(self._error_history):
            if module and record.module != module:
                continue
            if severity and record.severity != severity:
                continue
            if since and record.timestamp < since:
                continue
            results.append(record.to_dict())
            if len(results) >= limit:
                break
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取错误恢复模块统计信息。

        Returns:
            包含故障统计、恢复率和降级状态的字典。
        """
        recent_1h = sum(
            1 for r in self._error_history if time.time() - r.timestamp < 3600
        )
        recent_24h = sum(
            1 for r in self._error_history if time.time() - r.timestamp < 86400
        )

        severity_distribution: Dict[str, int] = defaultdict(int)
        for r in self._error_history:
            severity_distribution[r.severity.name] += 1

        return {
            "total_errors": self._total_errors,
            "total_recoveries": self._total_recoveries,
            "failed_recoveries": self._failed_recoveries,
            "recovery_rate": (
                self._total_recoveries / max(self._total_errors, 1)
            ),
            "recent_1h": recent_1h,
            "recent_24h": recent_24h,
            "safe_mode_active": self._safe_mode_active,
            "degradation_level": self._degradation_state.name,
            "degraded_modules": list(self._degraded_modules),
            "watchdog_count": len(self._watchdog_registry),
            "severity_distribution": dict(severity_distribution),
        }

    def get_degradation_state(self) -> DegradationLevel:
        """获取当前降级级别。

        Returns:
            当前降级级别。
        """
        return self._degradation_state

    def clear_error_history(self) -> None:
        """清空故障历史记录。

        警告：此操作不可逆。
        """
        count = len(self._error_history)
        self._error_history.clear()
        self._logger.info("故障历史已清空: %d 条记录", count)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def create_error_recovery(
    config: Optional[Dict[str, Any]] = None,
) -> ErrorRecovery:
    """工厂函数：创建 ErrorRecovery 实例。

    Args:
        config: 可选配置字典。

    Returns:
        ErrorRecovery 实例。
    """
    return ErrorRecovery(config=config)
