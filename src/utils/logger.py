"""
日志系统模块

提供分级日志、文件轮转、彩色终端输出和结构化JSON日志功能。
适用于OrangePi Kunpeng Pro (RK3588)平台的KunPeng-Cortex项目。

功能特性:
    - 五级日志: DEBUG/INFO/WARNING/ERROR/CRITICAL
    - 文件轮转: 按大小和时间轮转
    - 彩色终端输出: 不同级别不同颜色
    - 结构化JSON日志: 可选机器可解析格式
    - 异步日志写入
    - 上下文追踪(request_id等)
    - 性能指标自动记录

颜色映射:
    DEBUG:    灰色
    INFO:     绿色
    WARNING:  黄色
    ERROR:    红色
    CRITICAL: 红底白字

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


class LogFormat(Enum):
    """日志格式枚举"""
    STANDARD = "standard"   # 标准文本格式
    COLORED = "colored"     # 彩色终端格式
    JSON = "json"           # JSON结构化格式
    DETAILED = "detailed"   # 详细格式(含文件名行号)


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# ANSI颜色代码
class ColorCode:
    """ANSI颜色代码常量"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # 亮前景色
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


# 日志级别颜色映射
LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: ColorCode.DIM + ColorCode.BRIGHT_BLACK,
    logging.INFO: ColorCode.BRIGHT_GREEN,
    logging.WARNING: ColorCode.BRIGHT_YELLOW,
    logging.ERROR: ColorCode.BRIGHT_RED,
    logging.CRITICAL: ColorCode.BG_RED + ColorCode.BRIGHT_WHITE,
}

LEVEL_NAMES_SHORT: dict[int, str] = {
    logging.DEBUG: "DBG",
    logging.INFO: "INF",
    logging.WARNING: "WRN",
    logging.ERROR: "ERR",
    logging.CRITICAL: "CRT",
}


@dataclass
class LogConfig:
    """日志配置参数

    属性:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志文件目录
        max_file_size: 单个日志文件最大大小(字节)
        backup_count: 保留的备份文件数量
        console_format: 控制台日志格式
        file_format: 文件日志格式
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        enable_json: 是否启用JSON格式文件
        enable_colored: 是否启用彩色输出
        log_filename: 日志文件名
        json_filename: JSON日志文件名
    """
    name: str = "kpcortex"
    level: str = "INFO"
    log_dir: str = "/var/log/kpcortex"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_format: LogFormat = LogFormat.COLORED
    file_format: LogFormat = LogFormat.STANDARD
    enable_console: bool = True
    enable_file: bool = True
    enable_json: bool = True
    enable_colored: bool = True
    log_filename: str = "kpcortex.log"
    json_filename: str = "kpcortex.jsonl"


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器

    为终端输出添加颜色,不同日志级别使用不同颜色。
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_colors: bool = True,
    ) -> None:
        """初始化彩色格式化器

        参数:
            fmt: 格式字符串
            datefmt: 日期格式
            use_colors: 是否使用颜色
        """
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors: bool = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录

        参数:
            record: 日志记录

        返回:
            str: 格式化后的日志字符串
        """
        # 保存原始值
        original_levelname = record.levelname
        original_name = record.name

        if self.use_colors and sys.stdout.isatty():
            color = LEVEL_COLORS.get(record.levelno, ColorCode.RESET)
            reset = ColorCode.RESET

            # 级别名着色
            short_name = LEVEL_NAMES_SHORT.get(
                record.levelno, record.levelname
            )
            record.levelname = f"{color}{ColorCode.BOLD}{short_name}{reset}"

            # 记录器名着色
            record.name = f"{ColorCode.BRIGHT_CYAN}{record.name}{reset}"

            # 消息着色
            if record.levelno >= logging.ERROR:
                record.msg = f"{color}{record.msg}{reset}"
        else:
            short_name = LEVEL_NAMES_SHORT.get(
                record.levelno, record.levelname
            )
            record.levelname = short_name

        result = super().format(record)

        # 恢复原始值
        record.levelname = original_levelname
        record.name = original_name

        return result


class JSONFormatter(logging.Formatter):
    """JSON日志格式化器

    将日志记录格式化为JSON Lines格式,便于机器解析和处理。
    """

    def __init__(
        self,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """初始化JSON格式化器

        参数:
            extra_fields: 额外的固定字段
        """
        super().__init__()
        self.extra_fields: dict[str, Any] = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """格式化为JSON

        参数:
            record: 日志记录

        返回:
            str: JSON字符串
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # 添加异常信息
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # 添加额外字段
        if hasattr(record, "context"):
            log_entry["context"] = record.context
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        # 合并固定额外字段
        log_entry.update(self.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class AsyncLogHandler(logging.Handler):
    """异步日志处理器

    在后台线程中异步写入日志,避免阻塞主线程。
    """

    def __init__(self, base_handler: logging.Handler) -> None:
        """初始化异步处理器

        参数:
            base_handler: 基础处理器
        """
        super().__init__()
        self.base_handler: logging.Handler = base_handler
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """启动异步处理循环"""
        self._task = asyncio.create_task(self._process_loop())

    async def _process_loop(self) -> None:
        """处理循环"""
        while not self._stop_event.is_set():
            try:
                record = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
                self.base_handler.emit(record)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"异步日志处理异常: {e}", file=sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        """发送日志记录到队列

        参数:
            record: 日志记录
        """
        try:
            if not self._queue.full():
                self._queue.put_nowait(record)
        except Exception:
            pass

    async def stop(self) -> None:
        """停止异步处理"""
        self._stop_event.set()

        # 排空队列
        while not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                self.base_handler.emit(record)
            except Exception:
                break

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class LoggerManager:
    """日志管理器类

    提供统一的日志管理功能,支持多级输出、格式化和轮转。

    示例:
        >>> lm = LoggerManager(LogConfig(name="kpcortex", level="DEBUG"))
        >>> await lm.initialize()
        >>> 
        >>> log = lm.get_logger("devices.camera")
        >>> log.info("摄像头已初始化")
        >>> log.warning("帧率低于预期: %d fps", 15)
        >>> 
        >>> # 带上下文的日志
        >>> log.info("处理请求", extra={"request_id": "req-001", "duration_ms": 45})
        >>> 
        >>> await lm.shutdown()

    属性:
        config: 日志配置
        _logger: 根日志器
    """

    def __init__(self, config: LogConfig | None = None) -> None:
        """初始化日志管理器

        参数:
            config: 日志配置,None则使用默认配置
        """
        self.config: LogConfig = config or LogConfig()

        # 根日志器
        self._logger: logging.Logger = logging.getLogger(self.config.name)
        self._logger.setLevel(self.config.level)

        # 处理器列表
        self._handlers: list[logging.Handler] = []
        self._async_handlers: list[AsyncLogHandler] = []

        # 状态
        self._initialized: bool = False

    async def initialize(self) -> bool:
        """初始化日志系统

        配置控制台、文件和JSON日志处理器。

        返回:
            bool: 初始化成功返回True
        """
        if self._initialized:
            return True

        try:
            # 清除已有处理器
            self._logger.handlers.clear()

            # 控制台处理器
            if self.config.enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.config.level)

                if self.config.enable_colored and sys.stdout.isatty():
                    formatter = ColoredFormatter(
                        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S",
                    )
                else:
                    formatter = logging.Formatter(
                        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )

                console_handler.setFormatter(formatter)
                self._logger.addHandler(console_handler)
                self._handlers.append(console_handler)

            # 文件处理器(轮转)
            if self.config.enable_file:
                log_dir = Path(self.config.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)

                log_path = log_dir / self.config.log_filename
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=str(log_path),
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(self.config.level)

                file_formatter = logging.Formatter(
                    fmt="%(asctime)s %(levelname)s [%(name)s] "
                        "[%(filename)s:%(lineno)d] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler.setFormatter(file_formatter)
                self._logger.addHandler(file_handler)
                self._handlers.append(file_handler)

            # JSON日志处理器
            if self.config.enable_json:
                log_dir = Path(self.config.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)

                json_path = log_dir / self.config.json_filename
                json_handler = logging.handlers.RotatingFileHandler(
                    filename=str(json_path),
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count,
                    encoding="utf-8",
                )
                json_handler.setLevel(self.config.level)
                json_handler.setFormatter(JSONFormatter())
                self._logger.addHandler(json_handler)
                self._handlers.append(json_handler)

            self._initialized = True
            self._logger.info("日志系统初始化成功")
            return True

        except Exception as e:
            # 回退到基本日志
            logging.basicConfig(level=logging.INFO)
            logging.error(f"日志系统初始化失败: {e}")
            return False

    def get_logger(self, name: str | None = None) -> logging.Logger:
        """获取日志器

        返回指定名称的子日志器,名称会自动添加根前缀。

        参数:
            name: 日志器名称,None则返回根日志器

        返回:
            logging.Logger: 日志器实例

        示例:
            >>> log = lm.get_logger("devices.camera")
            >>> log.info("摄像头初始化")
            # 输出: 2025-01-15 10:00:00 INF [kpcortex.devices.camera] 摄像头初始化
        """
        if name:
            return logging.getLogger(f"{self.config.name}.{name}")
        return self._logger

    def set_level(self, level: str | int) -> None:
        """设置日志级别

        参数:
            level: 日志级别("DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL"或int值)

        示例:
            >>> lm.set_level("DEBUG")
            >>> lm.set_level(logging.WARNING)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self._logger.setLevel(level)
        for handler in self._handlers:
            handler.setLevel(level)

        self._logger.info(f"日志级别已设置为: {logging.getLevelName(level)}")

    def add_context(
        self,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> logging.LoggerAdapter:
        """创建带上下文的日志适配器

        为日志记录添加请求ID等上下文信息。

        参数:
            request_id: 请求ID
            **kwargs: 其他上下文字段

        返回:
            logging.LoggerAdapter: 日志适配器

        示例:
            >>> ctx_log = lm.add_context(request_id="req-001")
            >>> ctx_log.info("处理请求")
            # JSON日志中会包含 request_id: "req-001"
        """
        extra: dict[str, Any] = {}
        if request_id:
            extra["request_id"] = request_id
        extra.update(kwargs)

        return logging.LoggerAdapter(self._logger, {"context": extra})

    @staticmethod
    def log_execution_time(
        logger: logging.Logger,
        level: int = logging.DEBUG,
    ) -> Callable:
        """函数执行时间日志装饰器

        自动记录函数执行时间。

        参数:
            logger: 日志器
            level: 日志级别

        返回:
            装饰器函数

        示例:
            >>> @LoggerManager.log_execution_time(log)
            ... def my_function():
            ...     time.sleep(1)
            ... 
            ... my_function()
            # 输出: my_function 执行时间: 1000.5ms
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed_ms = (time.time() - start) * 1000
                    logger.log(
                        level,
                        f"{func.__name__} 执行时间: {elapsed_ms:.1f}ms",
                        extra={"duration_ms": elapsed_ms},
                    )
            return wrapper
        return decorator

    async def shutdown(self) -> None:
        """关闭日志系统

        刷新并关闭所有日志处理器。
        """
        # 停止异步处理器
        for handler in self._async_handlers:
            await handler.stop()

        # 关闭所有处理器
        for handler in self._handlers:
            handler.close()

        self._handlers.clear()
        self._async_handlers.clear()

        self._initialized = False

    def __repr__(self) -> str:
        return (
            f"LoggerManager(name={self.config.name}, "
            f"level={self.config.level}, "
            f"handlers={len(self._handlers)})"
        )

    async def __aenter__(self) -> LoggerManager:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
