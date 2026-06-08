#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py — KunPeng-Cortex Claude Code能力引擎入口

ClaudeCodeEngine是融合Agent中Claude Code能力的核心入口，负责将自然语言指令
转换为可在RK3588嵌入式平台上安全执行的硬件控制代码。

核心职责：
    - 自然语言意图理解与解析
    - 代码模板匹配与生成
    - 安全沙箱中的代码执行
    - 硬件工具注册与管理
    - 执行结果反馈与错误处理
    - 与融合调度器(Orchestrator)的集成接口

流水线架构：
    NL输入 → 意图解析 → 模板匹配 → 代码生成 → 安全验证 → 沙箱执行 → 结果反馈
      ↑___________________________________________________________________________|
                                    (结果反馈循环)

安全设计：
    - 七层安全体系（命名空间隔离、Capability限制、Seccomp-BPF、硬件白名单、
      超时保护、文件系统只读、资源限制）
    - 物理层安全（独立MCU紧急停止）
    - 所有代码执行前必须通过安全审查
    - 高危险操作需额外确认

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

# 导入同模块组件
from .code_generator import CodeGenerator, GeneratedCode
from .code_executor import CodeExecutor, ExecutionResult
from .sandbox import Sandbox, SandboxResult
from .tool_definitions import (
    HARDWARE_TOOLS,
    get_tool_danger_level,
    get_tool_schema,
    get_tool_timeout,
    list_all_tools,
    validate_parameters_against_whitelist,
    validate_tool_call,
)

# 配置模块日志记录器
logger = logging.getLogger("kunpeng_cortex.claude_code.engine")

# =============================================================================
# 数据模型定义
# =============================================================================


@dataclasses.dataclass
class CodeResult:
    """
    Claude Code引擎处理结果数据类。

    属性:
        success: 处理是否成功（完整流水线完成）
        intent: 解析后的自然语言意图
        generated_code: 生成的Python代码
        template_name: 使用的代码模板名称
        execution_output: 代码执行的标准输出
        execution_error: 代码执行的标准错误
        tool_calls: 执行过程中调用的硬件工具列表
        execution_time_ms: 端到端处理时间(毫秒)
        stage_times: 各阶段耗时字典
        error_message: 错误描述信息
        needs_confirmation: 是否需要用户确认（高危险操作）
        confidence: 意图识别置信度
        metadata: 额外元数据
    """
    success: bool
    intent: str = ""
    generated_code: str = ""
    template_name: str = ""
    execution_output: str = ""
    execution_error: str = ""
    tool_calls: List[str] = dataclasses.field(default_factory=list)
    execution_time_ms: float = 0.0
    stage_times: Dict[str, float] = dataclasses.field(default_factory=dict)
    error_message: str = ""
    needs_confirmation: bool = False
    confidence: float = 0.0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class EngineStatus:
    """
    引擎状态数据类。

    属性:
        initialized: 是否已初始化
        template_count: 可用模板数量
        registered_tools: 已注册硬件工具数量
        total_requests: 总请求数
        total_success: 成功处理数
        total_failures: 失败数
        avg_execution_time_ms: 平均执行时间(毫秒)
        cache_hit_rate: 缓存命中率
        sandbox_stats: 沙箱统计信息
    """
    initialized: bool = False
    template_count: int = 0
    registered_tools: int = 0
    total_requests: int = 0
    total_success: int = 0
    total_failures: int = 0
    avg_execution_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    sandbox_stats: Dict[str, Any] = dataclasses.field(default_factory=dict)


# =============================================================================
# ClaudeCodeEngine 主类
# =============================================================================


class ClaudeCodeEngine:
    """
    Claude Code能力引擎 —— 自然语言到硬件控制的核心转换引擎。

    整合代码生成器(CodeGenerator)、代码执行器(CodeExecutor)和沙箱管理器(Sandbox)，
    实现从自然语言到安全代码执行的完整流水线。支持与硬件抽象层(HAL)的深度集成。

    属性:
        config: 引擎配置字典
        code_generator: 代码生成器实例
        code_executor: 代码执行器实例
        sandbox: 沙箱管理器实例
        hal_adapter: 硬件抽象层适配器（可选）
        _tool_registry: 硬件工具注册表
        _initialized: 初始化状态标记
        _request_count: 请求计数
        _success_count: 成功计数
        _total_execution_time: 累计执行时间(毫秒)

    示例:
        >>> config = {"sandbox_memory_mb": 256, "timeout": 5.0}
        >>> engine = ClaudeCodeEngine(config)
        >>> engine.register_hardware_tools(hal_adapter)
        >>> result = await engine.process("读取GPIO引脚15的电平")
        >>> print(result.execution_output)
    """

    def __init__(self, config: dict, hal_adapter: Any = None) -> None:
        """
        初始化Claude Code能力引擎。

        参数:
            config: 引擎配置字典，支持以下配置项：
                - sandbox_memory_mb: 沙箱内存限制(MB)，默认256
                - sandbox_cpu_percent: CPU限制百分比，默认50
                - sandbox_allowed_paths: 沙箱只读路径列表
                - sandbox_network: 是否允许网络，默认False
                - timeout: 默认执行超时(秒)，默认5.0
                - template_dir: 模板文件目录
                - max_history: 最大执行历史记录数
                - enable_cache: 是否启用代码缓存，默认True
                - require_confirmation_for_high_danger: 高危险操作是否需要确认，默认True
            hal_adapter: 硬件抽象层适配器实例（可选），提供HAL接口调用能力
        """
        self.config: Dict[str, Any] = dict(config)
        self.hal_adapter: Any = hal_adapter

        # 初始化沙箱
        self.sandbox: Sandbox = Sandbox(
            allowed_paths=self.config.get("sandbox_allowed_paths"),
            network=self.config.get("sandbox_network", False),
            max_memory_mb=self.config.get("sandbox_memory_mb", 256),
            max_cpu_percent=self.config.get("sandbox_cpu_percent", 50),
        )

        # 初始化代码执行器
        self.code_executor: CodeExecutor = CodeExecutor(
            sandbox=self.sandbox,
            timeout=self.config.get("timeout", 5.0),
            max_history=self.config.get("max_history", 1000),
        )

        # 初始化代码生成器
        self.code_generator: CodeGenerator = CodeGenerator(
            template_dir=self.config.get("template_dir", "templates")
        )

        # 工具注册表
        self._tool_registry: Dict[str, Dict[str, Any]] = {}
        self._initialized: bool = False

        # 统计信息
        self._request_count: int = 0
        self._success_count: int = 0
        self._total_execution_time: float = 0.0
        self._confirmation_count: int = 0

        logger.info(
            f"ClaudeCodeEngine已创建: memory={self.config.get('sandbox_memory_mb', 256)}MB, "
            f"timeout={self.config.get('timeout', 5.0)}s"
        )

    @property
    def status(self) -> EngineStatus:
        """
        获取引擎当前状态。

        返回:
            EngineStatus数据类，包含引擎运行状态信息
        """
        total = self._request_count
        success = self._success_count
        avg_time = self._total_execution_time / max(total, 1)

        return EngineStatus(
            initialized=self._initialized,
            template_count=len(self.code_generator.templates),
            registered_tools=len(self._tool_registry),
            total_requests=total,
            total_success=success,
            total_failures=total - success,
            avg_execution_time_ms=avg_time,
            cache_hit_rate=0.0,  # TODO: 从generator获取
            sandbox_stats=self.sandbox.execution_stats,
        )

    async def initialize(self) -> bool:
        """
        异步初始化引擎。

        完成以下初始化工作：
            1. 检查沙箱环境（bubblewrap可用性）
            2. 加载代码模板
            3. 注册HAL适配器工具（如提供）
            4. 预编译高频模板

        返回:
            是否初始化成功
        """
        try:
            logger.info("开始初始化ClaudeCodeEngine...")

            # 步骤1: 检查沙箱环境
            bwrap_ok = self.sandbox._bwrap_available
            if not bwrap_ok:
                logger.warning(
                    "bubblewrap不可用，将使用subprocess隔离模式（安全性降低）"
                )

            # 步骤2: 注册HAL工具（如果提供了适配器）
            if self.hal_adapter:
                self.register_hardware_tools(self.hal_adapter)

            # 步骤3: 验证工具schema完整性
            all_tools = list_all_tools()
            logger.info(f"已加载{len(all_tools)}个工具Schema定义")

            self._initialized = True
            logger.info("ClaudeCodeEngine初始化完成")
            return True

        except Exception as e:
            logger.error(f"引擎初始化失败: {type(e).__name__}: {e}")
            self._initialized = False
            return False

    async def process(
        self,
        natural_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CodeResult:
        """
        处理自然语言指令，执行完整的NL→代码→执行流水线。

        流水线阶段：
            1. 意图解析与参数提取（< 5ms）
            2. 代码模板匹配（< 2ms）
            3. 代码生成与参数填充（< 3ms）
            4. 安全验证（< 50ms）
            5. 沙箱执行（< 5s）
            6. 结果解析与反馈（< 5ms）

        参数:
            natural_language: 用户自然语言输入，如"打开LED灯"
            context: 额外上下文字典，可能包含：
                - timeout: 本次执行超时(秒)
                - confirm_high_danger: 是否自动确认高危险操作
                - inject_globals: 额外注入的全局变量
                - skip_execution: 仅生成代码不执行（调试模式）

        返回:
            CodeResult数据类，包含完整处理结果

        示例:
            >>> engine = ClaudeCodeEngine({})
            >>> await engine.initialize()
            >>> result = await engine.process("读取GPIO引脚15的电平")
            >>> print(result.success)
            True
        """
        if not self._initialized:
            await self.initialize()

        self._request_count += 1
        start_time = time.monotonic()
        stage_times: Dict[str, float] = {}
        context = context or {}

        # 阶段1: 意图解析
        stage_start = time.monotonic()
        intent, params = self._parse_intent(natural_language)
        stage_times["intent_parse"] = (time.monotonic() - stage_start) * 1000

        logger.debug(f"意图解析: '{natural_language[:50]}' -> 意图={intent}, 参数={params}")

        # 阶段2: 代码生成（模板匹配+填充）
        stage_start = time.monotonic()
        generated: GeneratedCode = self.code_generator.generate(intent, params)
        stage_times["code_generation"] = (time.monotonic() - stage_start) * 1000

        # 阶段3: 高危险操作确认
        needs_confirmation = False
        if generated.safety_level in ("high", "critical"):
            needs_confirmation = self.config.get(
                "require_confirmation_for_high_danger", True
            )
            if needs_confirmation and not context.get("confirm_high_danger", False):
                self._confirmation_count += 1
                logger.info(f"高危险操作需要确认: {generated.template_name}")
                total_time = (time.monotonic() - start_time) * 1000
                return CodeResult(
                    success=False,
                    intent=intent,
                    generated_code=generated.code,
                    template_name=generated.template_name,
                    needs_confirmation=True,
                    confidence=generated.confidence,
                    execution_time_ms=total_time,
                    stage_times=stage_times,
                    error_message=f"操作'{intent}'为{generated.safety_level}级别，需要用户确认",
                )

        # 阶段4: 安全验证
        stage_start = time.monotonic()
        safe, reason = self.code_generator.validate_generated_code(generated.code)
        stage_times["security_validation"] = (time.monotonic() - stage_start) * 1000

        if not safe:
            total_time = (time.monotonic() - start_time) * 1000
            logger.warning(f"生成代码未通过安全验证: {reason}")
            return CodeResult(
                success=False,
                intent=intent,
                generated_code=generated.code,
                template_name=generated.template_name,
                needs_confirmation=needs_confirmation,
                confidence=generated.confidence,
                execution_time_ms=total_time,
                stage_times=stage_times,
                error_message=f"安全验证失败: {reason}",
            )

        # 阶段5: 沙箱执行
        execution_output = ""
        execution_error = ""
        exec_success = False

        if not context.get("skip_execution", False):
            stage_start = time.monotonic()
            exec_result: ExecutionResult = await self.code_executor.execute(
                code=generated.code,
                globals_dict=context.get("inject_globals"),
                timeout=context.get("timeout"),
                inject_tools=True,
            )
            stage_times["sandbox_execution"] = (time.monotonic() - stage_start) * 1000

            execution_output = exec_result.output
            execution_error = exec_result.error
            exec_success = exec_result.success

            if not exec_success:
                logger.warning(
                    f"代码执行失败: type={exec_result.error_type}, "
                    f"msg={exec_result.error_message}"
                )
        else:
            stage_times["sandbox_execution"] = 0.0
            exec_success = True
            logger.debug("跳过执行（调试模式）")

        # 阶段6: 结果汇总
        total_time = (time.monotonic() - start_time) * 1000
        self._total_execution_time += total_time

        if exec_success:
            self._success_count += 1

        result = CodeResult(
            success=exec_success,
            intent=intent,
            generated_code=generated.code,
            template_name=generated.template_name,
            execution_output=execution_output,
            execution_error=execution_error,
            tool_calls=generated.required_tools,
            execution_time_ms=total_time,
            stage_times=stage_times,
            error_message=execution_error if not exec_success else "",
            needs_confirmation=needs_confirmation,
            confidence=generated.confidence,
            metadata={
                "safety_level": generated.safety_level,
                "estimated_time_ms": generated.estimated_time_ms,
                "parameter_values": generated.parameter_values,
                "natural_language": natural_language,
                "hal_adapter_present": self.hal_adapter is not None,
            },
        )

        logger.info(
            f"处理完成: intent='{intent[:30]}', success={result.success}, "
            f"time={total_time:.1f}ms, template={generated.template_name}"
        )
        return result

    def register_hardware_tools(self, hal_adapter: Any) -> None:
        """
        从HAL适配器注册所有硬件控制工具。

        扫描HAL适配器的方法，自动注册符合工具命名规范的硬件操作函数。
        这是引擎与硬件抽象层集成的关键接口。

        参数:
            hal_adapter: 硬件抽象层适配器实例，需实现以下方法：
                - gpio_read(pin), gpio_write(pin, value)
                - pwm_set(channel, freq, duty)
                - i2c_read(bus, addr, len), i2c_write(bus, addr, data)
                - motor_control(motor_id, speed)
                - sensor_read(sensor_id)
                - camera_capture(resolution, format)
                - arm_move(joint_angles, speed)
                - arm_gripper(position, force)
                - emergency_stop(scope)

        示例:
            >>> class HALAdapter:
            ...     def gpio_read(self, pin: int) -> int:
            ...         return 1
            >>> engine = ClaudeCodeEngine({})
            >>> engine.register_hardware_tools(HALAdapter())
        """
        if hal_adapter is None:
            logger.warning("HAL适配器为空，跳过工具注册")
            return

        self.hal_adapter = hal_adapter
        tool_count = 0

        # 定义工具名称到HAL方法的映射
        tool_method_map: Dict[str, Optional[str]] = {
            "gpio_read": "gpio_read",
            "gpio_write": "gpio_write",
            "pwm_set": "pwm_set",
            "servo_set": "set_servo",  # 可能的别名
            "i2c_read": "i2c_read",
            "i2c_write": "i2c_write",
            "uart_send": "uart_write",
            "motor_control": "motor_set_speed",
            "sensor_read": "sensor_read",
            "camera_capture": "camera_capture",
            "arm_move": "arm_move_joint",
            "arm_gripper": "arm_gripper_set",
            "emergency_stop": "hal_estop_activate",
        }

        for tool_name, method_name in tool_method_map.items():
            method = None

            # 尝试通过方法名获取
            if method_name and hasattr(hal_adapter, method_name):
                method = getattr(hal_adapter, method_name)

            # 回退：尝试直接通过工具名获取
            if method is None and hasattr(hal_adapter, tool_name):
                method = getattr(hal_adapter, tool_name)

            if method and callable(method):
                # 注册到代码执行器
                self.code_executor.register_hardware_tool(tool_name, method)
                self._tool_registry[tool_name] = {
                    "name": tool_name,
                    "handler": method,
                    "source": "hal_adapter",
                }
                tool_count += 1
                logger.debug(f"硬件工具已注册: {tool_name} -> {method.__qualname__}")

        logger.info(f"从HAL适配器注册了{tool_count}个硬件工具")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有可用的硬件工具列表。

        返回包含工具名称、描述、Schema定义和危险级别的完整工具信息列表。

        返回:
            工具信息字典列表，每个字典包含：
                - name: 工具名称
                - description: 工具描述
                - category: 工具类别
                - danger_level: 危险级别
                - timeout_ms: 超时时间(毫秒)
                - registered: 是否已注册handler
                - schema: 工具JSON Schema定义

        示例:
            >>> engine = ClaudeCodeEngine({})
            >>> tools = engine.get_available_tools()
            >>> print(tools[0]["name"])
            gpio_read
        """
        tools_info: List[Dict[str, Any]] = []
        all_schemas = list_all_tools()

        for schema in all_schemas:
            name = schema["name"]
            info: Dict[str, Any] = {
                "name": name,
                "description": schema.get("description", ""),
                "category": schema.get("category", "unknown"),
                "danger_level": schema.get("danger_level", "low"),
                "timeout_ms": schema.get("timeout_ms", 5000),
                "registered": name in self._tool_registry,
                "schema": schema,
            }
            tools_info.append(info)

        # 按类别排序
        tools_info.sort(key=lambda t: (t["category"], t["name"]))
        return tools_info

    def register_custom_tool(
        self,
        name: str,
        handler: Callable,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        注册自定义硬件工具。

        允许运行时注册自定义硬件操作，用于支持非标准外设或实验性功能。

        参数:
            name: 工具名称
            handler: 工具处理函数
            schema: 工具的JSON Schema定义（可选）

        返回:
            (是否成功, 状态信息) 元组
        """
        if not name or not isinstance(name, str):
            return False, "工具名称必须为非空字符串"
        if not callable(handler):
            return False, "handler必须是可调用的函数"

        try:
            # 注册到执行器
            self.code_executor.register_hardware_tool(name, handler)

            # 注册到本地注册表
            self._tool_registry[name] = {
                "name": name,
                "handler": handler,
                "schema": schema,
                "source": "custom",
            }

            # 添加到代码生成器的模板匹配（如果提供了关键词）
            if schema and "keywords" in schema:
                keywords = schema["keywords"]
                template_code = schema.get("template", f"# 自定义工具: {name}\nresult = {name}()\nprint(result)")
                self.code_generator.add_template(name, template_code, keywords)

            logger.info(f"自定义工具已注册: {name}")
            return True, f"工具'{name}'注册成功"

        except Exception as e:
            logger.error(f"注册自定义工具失败: {type(e).__name__}: {e}")
            return False, f"注册失败: {e}"

    def _parse_intent(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        从自然语言文本中提取意图和参数。

        基于启发式规则进行关键词提取和参数解析，将中文/英文自然语言
        转换为结构化的意图描述和参数字典。

        参数:
            text: 自然语言输入文本

        返回:
            (意图字符串, 参数字典) 元组
        """
        text_lower = text.lower().strip()
        params: Dict[str, Any] = {}

        # 提取引脚号（GPIO相关）
        pin_patterns = [
            r"引脚\s*(\d+)",
            r"pin\s*(\d+)",
            r"gpio\s*(\d+)",
            r"#(\d+)",
        ]
        for pattern in pin_patterns:
            match = __import__("re").search(pattern, text_lower)
            if match:
                params["pin_number"] = int(match.group(1))
                break

        # 提取数值参数
        value_match = __import__("re").search(r"(?:值为|设置为|设置成|to)\s*(\d+)", text_lower)
        if value_match:
            params["value"] = int(value_match.group(1))

        # 提取通道号
        channel_match = __import__("re").search(r"通道\s*(\d+)", text_lower)
        if channel_match:
            params["channel"] = int(channel_match.group(1))

        # 提取角度
        angle_match = __import__("re").search(r"(\d+)\s*度", text_lower)
        if angle_match:
            params["angle"] = int(angle_match.group(1))

        # 提取速度
        speed_keywords = ["速度", "转速", "speed"]
        for kw in speed_keywords:
            if kw in text_lower:
                speed_match = __import__("re").search(rf"{kw}\s*(\d+)", text_lower)
                if speed_match:
                    params["speed"] = int(speed_match.group(1))
                break

        # 提取传感器ID
        sensor_patterns = [
            r"传感器\s*([a-zA-Z0-9_]+)",
            r"sensor\s*([a-zA-Z0-9_]+)",
        ]
        for pattern in sensor_patterns:
            match = __import__("re").search(pattern, text_lower)
            if match:
                params["sensor_id"] = match.group(1)
                break

        # 提取电机ID
        motor_match = __import__("re").search(r"电机\s*(\d+)", text_lower)
        if motor_match:
            params["motor_id"] = int(motor_match.group(1))

        # 提取I2C地址和总线
        i2c_match = __import__("re").search(r"i2c\s*[-#]?\s*(\d+)", text_lower)
        if i2c_match:
            params["bus"] = int(i2c_match.group(1))
        addr_match = __import__("re").search(r"地址\s*(0x[0-9a-fA-F]+|\d+)", text_lower)
        if addr_match:
            addr_str = addr_match.group(1)
            params["address"] = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str)

        # 提取占空比
        duty_match = __import__("re").search(r"占空比\s*(\d+(?:\.\d+)?)", text_lower)
        if duty_match:
            params["duty_cycle"] = float(duty_match.group(1))

        # 提取频率
        freq_match = __import__("re").search(r"频率\s*(\d+(?:\.\d+)?)\s*[hH][zZ]?", text_lower)
        if freq_match:
            params["frequency"] = int(float(freq_match.group(1)))

        # 构建意图字符串（简化版本）
        intent = text.strip()

        return intent, params

    async def emergency_stop(self, scope: str = "all") -> CodeResult:
        """
        触发紧急停止操作。

        这是最高优先级的安全操作，立即停止所有硬件执行器。
        直接调用emergency_stop工具，绕过常规流水线。

        参数:
            scope: 停止范围，"all"=全部, "motors"=仅电机, "arm"=仅机械臂

        返回:
            CodeResult处理结果
        """
        logger.critical(f"触发紧急停止! scope={scope}")

        emergency_code = f"emergency_stop(scope='{scope}')"
        start_time = time.monotonic()

        # 直接通过执行器调用，不走模板匹配
        exec_result = await self.code_executor.execute(
            code=emergency_code,
            timeout=1.0,  # 紧急操作1秒超时
        )

        total_time = (time.monotonic() - start_time) * 1000

        return CodeResult(
            success=exec_result.success,
            intent="emergency_stop",
            generated_code=emergency_code,
            template_name="emergency_stop",
            execution_output=exec_result.output,
            execution_error=exec_result.error,
            tool_calls=["emergency_stop"],
            execution_time_ms=total_time,
            stage_times={"emergency": total_time},
            confidence=1.0,
            metadata={"emergency": True, "scope": scope},
        )

    def get_stats_report(self) -> str:
        """
        生成格式化的统计报告。

        返回:
            可读的统计报告字符串
        """
        s = self.status
        lines = [
            "=" * 50,
            "ClaudeCodeEngine 统计报告",
            "=" * 50,
            f"引擎状态: {'已初始化' if s.initialized else '未初始化'}",
            f"模板数量: {s.template_count}",
            f"已注册工具: {s.registered_tools}",
            "-" * 50,
            f"总请求数: {s.total_requests}",
            f"成功: {s.total_success}",
            f"失败: {s.total_failures}",
            f"成功率: {s.total_success / max(s.total_requests, 1) * 100:.1f}%",
            f"平均执行时间: {s.avg_execution_time_ms:.1f}ms",
            "-" * 50,
            "沙箱统计:",
            f"  总执行: {s.sandbox_stats.get('execution_count', 0)}",
            f"  阻止: {s.sandbox_stats.get('blocked_count', 0)}",
            f"  超时: {s.sandbox_stats.get('timeout_count', 0)}",
            "=" * 50,
        ]
        return "\n".join(lines)

    async def health_check(self) -> Tuple[bool, str]:
        """
        引擎健康检查。

        验证引擎各组件是否正常可用。

        返回:
            (是否健康, 状态信息) 元组
        """
        checks: List[Tuple[str, bool]] = []

        # 检查初始化状态
        checks.append(("initialized", self._initialized))

        # 检查沙箱
        checks.append(("sandbox", self.sandbox is not None))

        # 检查代码生成器
        checks.append(("generator", len(self.code_generator.templates) > 0))

        # 检查执行器
        checks.append(("executor", self.code_executor is not None))

        # 检查模板
        checks.append(("templates", self.status.template_count > 0))

        all_ok = all(ok for _, ok in checks)
        report = ", ".join(f"{name}={'OK' if ok else 'FAIL'}" for name, ok in checks)

        return all_ok, report
