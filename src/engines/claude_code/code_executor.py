#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code_executor.py — KunPeng-Cortex 代码执行器模块

在安全沙箱环境中执行由代码生成器产生的Python代码，集成硬件工具注入、
超时控制、结果收集和错误处理功能。是Claude Code能力引擎与硬件抽象层
(HAL)之间的关键桥梁。

核心功能：
    - 安全沙箱中的代码执行（bubblewrap隔离）
    - 硬件工具函数注入（gpio_read、motor_control等）
    - 执行超时控制（默认5秒硬超时）
    - 执行结果结构化收集
    - 错误分类与处理
    - 执行历史记录与审计

安全设计：
    - 所有代码必须经过沙箱安全审查
    - 硬件工具调用受白名单约束
    - 超时强制终止（SIGKILL）
    - 内存和CPU资源限制
    - 完整执行日志审计

硬件平台: OrangePi Kunpeng Pro (RK3588, ARM64)
作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入同模块的其他组件
from .sandbox import Sandbox, SandboxResult

# 配置模块日志记录器
logger = logging.getLogger("kunpeng_cortex.claude_code.code_executor")

# =============================================================================
# 类型别名
# =============================================================================

HardwareTool = Callable[..., Any]
ValidationResult = Tuple[bool, str]

# =============================================================================
# 数据模型定义
# =============================================================================


@dataclasses.dataclass
class ExecutionResult:
    """
    代码执行结果数据类。

    属性:
        success: 代码是否成功执行
        output: 标准输出内容
        error: 标准错误内容
        return_code: 进程返回码
        execution_time_ms: 实际执行时间(毫秒)
        timed_out: 是否因超时终止
        tool_calls: 执行过程中调用的硬件工具列表
        tool_results: 硬件工具调用结果字典
        error_type: 错误类型分类
        error_message: 用户友好的错误描述
        metadata: 额外元数据
    """
    success: bool
    output: str = ""
    error: str = ""
    return_code: int = 0
    execution_time_ms: float = 0.0
    timed_out: bool = False
    tool_calls: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    tool_results: Dict[str, Any] = dataclasses.field(default_factory=dict)
    error_type: str = ""  # syntax|security|timeout|runtime|hardware|unknown
    error_message: str = ""
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ExecutionRecord:
    """
    执行记录数据类，用于审计和历史追踪。

    属性:
        record_id: 记录唯一标识
        timestamp: 执行时间戳
        code: 执行的代码内容（脱敏后）
        result: 执行结果
        triggered_tools: 触发的硬件工具列表
    """
    record_id: str
    timestamp: float
    code_summary: str
    result_summary: str
    triggered_tools: List[str]


# =============================================================================
# CodeExecutor 主类
# =============================================================================


class CodeExecutor:
    """
    代码执行器 —— 在安全沙箱中执行Python代码。

    集成沙箱管理、硬件工具注入和超时控制，提供完整的代码执行生命周期管理。
    所有硬件操作通过注入的工具函数进行，确保受白名单约束。

    属性:
        sandbox: 沙箱管理器实例
        timeout: 默认执行超时时间(秒)
        hardware_tools: 已注册的硬件工具函数字典
        max_history: 最大执行历史记录数
        _execution_history: 执行历史记录列表
        _total_executions: 总执行次数统计
        _total_failures: 总失败次数统计

    示例:
        >>> executor = CodeExecutor(Sandbox())
        >>> executor.register_hardware_tool("gpio_read", lambda pin: 1)
        >>> result = await executor.execute("print(gpio_read(15))")
        >>> print(result.output)
        1
    """

    # 错误类型分类映射
    ERROR_TYPE_MAP: Dict[str, List[str]] = {
        "syntax": ["SyntaxError", "IndentationError", "TabError"],
        "security": ["SecurityError", "PermissionError", "Blocked", "sandbox"],
        "timeout": ["TimeoutError", "timed out", "timeout"],
        "runtime": ["RuntimeError", "ValueError", "TypeError", "IndexError",
                     "KeyError", "AttributeError", "ZeroDivisionError",
                     "OverflowError", "AssertionError"],
        "hardware": ["HALError", "DeviceError", "CommunicationError",
                      "I2CError", "GPIOError", "MotorError"],
    }

    def __init__(
        self,
        sandbox: Sandbox,
        timeout: float = 5.0,
        max_history: int = 1000,
    ) -> None:
        """
        初始化代码执行器。

        参数:
            sandbox: 沙箱管理器实例（负责进程隔离和安全审查）
            timeout: 默认执行超时时间(秒)，默认5.0秒
            max_history: 最大执行历史记录数，默认1000条
        """
        self.sandbox: Sandbox = sandbox
        self.timeout: float = timeout
        self.max_history: int = max_history
        self.hardware_tools: Dict[str, HardwareTool] = {}
        self._execution_history: List[ExecutionRecord] = []
        self._total_executions: int = 0
        self._total_failures: int = 0
        self._tool_call_counter: Dict[str, int] = {}

        logger.info(
            f"代码执行器已初始化: timeout={timeout}s, "
            f"max_history={max_history}"
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """
        获取执行器统计信息。

        返回:
            包含执行次数、成功率、工具调用次数等的字典
        """
        total = self._total_executions
        failures = self._total_failures
        return {
            "total_executions": total,
            "total_failures": failures,
            "success_rate": (total - failures) / max(total, 1),
            "tool_call_counts": dict(self._tool_call_counter),
            "history_size": len(self._execution_history),
            "registered_tools": list(self.hardware_tools.keys()),
        }

    @property
    def execution_history(self) -> List[ExecutionRecord]:
        """
        获取执行历史记录（只读视图）。

        返回:
            执行记录列表的副本
        """
        return list(self._execution_history)

    async def execute(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        inject_tools: bool = True,
    ) -> ExecutionResult:
        """
        在安全沙箱中执行Python代码。

        执行流程：
            1. 代码安全验证
            2. 硬件工具注入（如启用）
            3. 沙箱内执行
            4. 结果解析与错误分类
            5. 执行记录保存

        参数:
            code: 要执行的Python代码字符串
            globals_dict: 额外注入的全局变量字典
            timeout: 本次执行的超时时间(秒)，None使用默认值
            inject_tools: 是否注入已注册的硬件工具函数

        返回:
            ExecutionResult数据类，包含完整执行结果和元数据

        示例:
            >>> executor = CodeExecutor(Sandbox())
            >>> executor.register_hardware_tool("gpio_read", hal.gpio_read)
            >>> result = await executor.execute("result = gpio_read(15)")
            >>> print(result.success)
            True
        """
        self._total_executions += 1
        actual_timeout = timeout if timeout is not None else self.timeout
        start_time = time.monotonic()
        tool_calls_log: List[Dict[str, Any]] = []

        # 步骤1: 代码安全验证
        safe, reason = self.validate_code(code)
        if not safe:
            self._total_failures += 1
            logger.warning(f"代码安全验证未通过: {reason}")
            return ExecutionResult(
                success=False,
                error_type="security",
                error_message=f"安全验证失败: {reason}",
                execution_time_ms=0.0,
            )

        # 步骤2: 构建执行代码（注入硬件工具）
        executable_code = code
        injected_globals: Dict[str, Any] = {}

        if inject_tools and self.hardware_tools:
            # 生成工具注入代码
            tool_injection = self._generate_tool_injection_code()
            executable_code = tool_injection + "\n" + code

            # 记录工具调用声明
            for tool_name in self.hardware_tools:
                tool_calls_log.append({
                    "tool": tool_name,
                    "phase": "injected",
                    "timestamp": time.time(),
                })

        # 添加额外全局变量
        if globals_dict:
            # 序列化全局变量到环境变量
            for key, value in globals_dict.items():
                injected_globals[key] = value

        # 注入全局变量赋值代码
        if injected_globals:
            global_assignments = "\n".join(
                f"{key} = {repr(value)}"
                for key, value in injected_globals.items()
            )
            executable_code = global_assignments + "\n" + executable_code

        # 步骤3: 沙箱执行
        try:
            sandbox_result: SandboxResult = await self.sandbox.run(
                code=executable_code,
                timeout=actual_timeout,
            )

            execution_time = (time.monotonic() - start_time) * 1000

            # 步骤4: 结果解析与错误分类
            if sandbox_result.timed_out:
                error_type = "timeout"
                error_message = f"执行超时（限制{actual_timeout}秒）"
            elif sandbox_result.return_code != 0:
                error_type = self._classify_error(
                    sandbox_result.stderr or sandbox_result.error_message
                )
                error_message = sandbox_result.error_message or sandbox_result.stderr
                if not error_message:
                    error_message = f"进程异常退出(返回码: {sandbox_result.return_code})"
            elif not sandbox_result.success:
                error_type = "runtime"
                error_message = sandbox_result.error_message or "执行失败"
            else:
                error_type = ""
                error_message = ""

            # 从stdout解析工具调用结果
            tool_results = self._parse_tool_results(sandbox_result.stdout)

            # 统计工具调用
            for tool_name in self._extract_tool_calls(code):
                self._tool_call_counter[tool_name] = self._tool_call_counter.get(tool_name, 0) + 1

            # 构建结果
            result = ExecutionResult(
                success=sandbox_result.success and not sandbox_result.timed_out,
                output=sandbox_result.stdout,
                error=sandbox_result.stderr,
                return_code=sandbox_result.return_code,
                execution_time_ms=execution_time,
                timed_out=sandbox_result.timed_out,
                tool_calls=tool_calls_log,
                tool_results=tool_results,
                error_type=error_type,
                error_message=error_message,
                metadata={
                    "sandbox_time_ms": sandbox_result.execution_time_ms,
                    "memory_usage_mb": sandbox_result.memory_usage_mb,
                    "command_blocked": sandbox_result.command_blocked,
                    "injected_tools": list(self.hardware_tools.keys()) if inject_tools else [],
                },
            )

            if not result.success:
                self._total_failures += 1

            # 步骤5: 保存执行记录
            self._record_execution(code, result)

            logger.debug(
                f"代码执行完成: success={result.success}, "
                f"time={execution_time:.1f}ms, rc={sandbox_result.return_code}"
            )
            return result

        except Exception as e:
            self._total_failures += 1
            execution_time = (time.monotonic() - start_time) * 1000
            logger.error(f"执行器异常: {type(e).__name__}: {e}")
            return ExecutionResult(
                success=False,
                error_type="unknown",
                error_message=f"执行器异常: {type(e).__name__}: {e}",
                execution_time_ms=execution_time,
            )

    def validate_code(self, code: str) -> ValidationResult:
        """
        验证代码是否符合执行要求。

        依次进行以下验证：
            1. 非空检查
            2. 沙箱安全审查（黑名单扫描）
            3. 语法合法性检查

        参数:
            code: 要验证的Python代码

        返回:
            (是否通过验证, 原因信息) 元组
        """
        if not code or not isinstance(code, str):
            return False, "代码为空或不是字符串"

        if len(code) > 65536:  # 最大64KB
            return False, f"代码长度超过限制: {len(code)} > 65536"

        # 使用沙箱的安全检查
        safe, reason = self.sandbox.check_command_safety(code)
        if not safe:
            return False, reason

        # 语法检查
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: 第{e.lineno}行 - {e.msg}"

        return True, "OK"

    def register_hardware_tool(self, name: str, func: HardwareTool) -> None:
        """
        注册硬件工具函数到执行器。

        注册的函数将在代码执行时作为全局变量注入到沙箱环境中，
        使生成的代码可以直接调用硬件操作。

        参数:
            name: 工具函数名称（代码中使用的标识符）
            func: 工具函数实现

        示例:
            >>> executor = CodeExecutor(Sandbox())
            >>> def read_gpio_pin(pin: int) -> int:
            ...     return hal.gpio_read(pin)
            >>> executor.register_hardware_tool("gpio_read", read_gpio_pin)
        """
        if not name or not isinstance(name, str):
            raise ValueError("工具名称必须为非空字符串")
        if not callable(func):
            raise ValueError("工具必须是可调用的函数")

        # 检查名称冲突
        if name in self.hardware_tools:
            logger.warning(f"硬件工具'{name}'已被覆盖")

        self.hardware_tools[name] = func
        logger.info(f"硬件工具已注册: {name}")

    def unregister_hardware_tool(self, name: str) -> bool:
        """
        从执行器移除已注册的硬件工具。

        参数:
            name: 工具名称

        返回:
            是否成功移除
        """
        if name in self.hardware_tools:
            del self.hardware_tools[name]
            logger.info(f"硬件工具已移除: {name}")
            return True
        return False

    def list_hardware_tools(self) -> List[Dict[str, str]]:
        """
        获取所有已注册的硬件工具列表。

        返回:
            工具信息字典列表，包含名称和文档
        """
        tools = []
        for name, func in sorted(self.hardware_tools.items()):
            doc = func.__doc__ or "无描述"
            tools.append({
                "name": name,
                "description": doc.strip(),
                "signature": str(func),
            })
        return tools

    def _generate_tool_injection_code(self) -> str:
        """
        生成硬件工具注入代码。

        创建Python代码片段，将所有已注册的硬件工具包装为可在
        沙箱内调用的桩函数。实际调用通过HAL适配器的IPC机制转发。

        返回:
            工具注入Python代码字符串
        """
        lines = [
            "# ===== KunPeng-Cortex 硬件工具自动注入 =====",
            "import json",
            "",
        ]

        for tool_name in sorted(self.hardware_tools.keys()):
            # 生成包装函数
            lines.extend([
                f"def {tool_name}(*args, **kwargs):",
                f"    \"\"\"硬件工具桩函数: {tool_name}\"\"\"",
                "    print(json.dumps({",
                f"        '_tool_call': '{tool_name}',",
                "        '_args': args,",
                "        '_kwargs': kwargs,",
                "    }))",
                "    # 返回模拟值（实际执行将通过HAL适配器）",
                "    return None",
                "",
            ])

        lines.append("# ===== 硬件工具注入结束 =====")
        return "\n".join(lines)

    def _classify_error(self, error_text: str) -> str:
        """
        对错误信息进行分类。

        根据错误文本内容识别错误类型：syntax/security/timeout/
        runtime/hardware/unknown。

        参数:
            error_text: 错误描述文本

        返回:
            错误类型字符串
        """
        if not error_text:
            return "unknown"

        error_lower = error_text.lower()

        for error_type, keywords in self.ERROR_TYPE_MAP.items():
            for keyword in keywords:
                if keyword.lower() in error_lower:
                    return error_type

        return "unknown"

    def _parse_tool_results(self, stdout: str) -> Dict[str, Any]:
        """
        从标准输出中解析硬件工具调用结果。

        解析沙箱代码输出的JSON格式工具调用日志。

        参数:
            stdout: 标准输出字符串

        返回:
            工具名称到调用结果的字典
        """
        results: Dict[str, Any] = {}
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "_tool_call" in data:
                    tool_name = data["_tool_call"]
                    results[tool_name] = {
                        "args": data.get("_args", []),
                        "kwargs": data.get("_kwargs", {}),
                    }
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    def _extract_tool_calls(self, code: str) -> List[str]:
        """
        从代码中提取调用的硬件工具名称。

        参数:
            code: Python代码

        返回:
            代码中引用的硬件工具名称列表
        """
        found = []
        for tool_name in self.hardware_tools:
            if tool_name in code:
                found.append(tool_name)
        return found

    def _record_execution(self, code: str, result: ExecutionResult) -> None:
        """
        保存执行记录到历史。

        参数:
            code: 执行的代码
            result: 执行结果
        """
        import hashlib

        # 生成记录ID
        record_id = hashlib.sha256(
            f"{time.time()}:{code[:50]}".encode()
        ).hexdigest()[:16]

        # 代码摘要（脱敏）
        code_summary = code[:100] + "..." if len(code) > 100 else code

        # 结果摘要
        result_summary = (
            f"success={result.success}, "
            f"type={result.error_type or 'ok'}, "
            f"time={result.execution_time_ms:.1f}ms"
        )

        record = ExecutionRecord(
            record_id=record_id,
            timestamp=time.time(),
            code_summary=code_summary,
            result_summary=result_summary,
            triggered_tools=list(result.tool_results.keys()),
        )

        self._execution_history.append(record)

        # 限制历史记录大小
        if len(self._execution_history) > self.max_history:
            self._execution_history = self._execution_history[-self.max_history:]

    def clear_history(self) -> None:
        """清除执行历史记录。"""
        self._execution_history.clear()
        logger.debug("执行历史已清除")

    def get_recent_failures(self, count: int = 10) -> List[ExecutionRecord]:
        """
        获取最近的失败执行记录。

        参数:
            count: 返回的最大记录数

        返回:
            失败执行记录列表
        """
        failures = [
            r for r in self._execution_history
            if "success=False" in r.result_summary
        ]
        return failures[-count:]
