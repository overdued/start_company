#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KunPeng-Cortex Claude Code能力引擎包

提供自然语言到硬件控制代码的完整转换能力：
    - engine: ClaudeCodeEngine主引擎入口
    - code_generator: 代码生成器（NL→Python）
    - code_executor: 代码执行器（沙箱执行）
    - sandbox: 沙箱管理（bubblewrap隔离）
    - tool_definitions: 硬件工具Schema定义

使用示例:
    from src.engines.claude_code import ClaudeCodeEngine
    
    engine = ClaudeCodeEngine(config={"timeout": 5.0})
    await engine.initialize()
    result = await engine.process("读取GPIO引脚15的电平")
    print(result.execution_output)
"""

from .engine import ClaudeCodeEngine, CodeResult, EngineStatus
from .code_generator import CodeGenerator, GeneratedCode
from .code_executor import CodeExecutor, ExecutionResult
from .sandbox import Sandbox, SandboxResult
from .tool_definitions import (
    HARDWARE_TOOLS,
    validate_tool_call,
    get_tool_schema,
    list_all_tools,
    register_custom_tool,
)

__version__ = "1.0.0"
__all__ = [
    "ClaudeCodeEngine",
    "CodeResult",
    "EngineStatus",
    "CodeGenerator",
    "GeneratedCode",
    "CodeExecutor",
    "ExecutionResult",
    "Sandbox",
    "SandboxResult",
    "HARDWARE_TOOLS",
    "validate_tool_call",
    "get_tool_schema",
    "list_all_tools",
    "register_custom_tool",
]
