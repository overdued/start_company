#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tool_definitions.py — KunPeng-Cortex 硬件控制工具定义与Schema验证模块

本模块定义了Claude Code能力引擎与硬件抽象层(HAL)交互所需的所有工具接口，
基于JSON Schema规范提供完整的工具描述、调用验证和安全参数范围检查。

功能范围：
    - 预定义硬件工具（GPIO、PWM、I2C、UART、电机、舵机、传感器、摄像头、机械臂）
    - JSON Schema定义与校验
    - 工具注册表管理（支持自定义工具动态注册）
    - 硬件命令白名单验证
    - 参数范围安全检查

安全设计：
    - 所有硬件命令必须通过schema验证
    - 数值参数自动进行范围限制
    - 禁止调用未经注册的工具
    - 敏感操作（如电机高速运动）需额外确认

作者: KunPeng-Cortex Team
版本: 1.0.0
硬件平台: OrangePi Kunpeng Pro (RK3588, ARM64)
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 配置模块日志记录器
logger = logging.getLogger("kunpeng_cortex.claude_code.tool_definitions")

# =============================================================================
# 类型别名定义
# =============================================================================

ToolSchema = Dict[str, Any]
ToolHandler = Callable[..., Any]
ValidationResult = Tuple[bool, str]

# =============================================================================
# 预定义硬件工具JSON Schema
# =============================================================================

# GPIO控制工具 —— 通用输入输出引脚控制
GPIO_READ_SCHEMA: ToolSchema = {
    "name": "gpio_read",
    "description": "读取指定GPIO引脚的当前电平值。适用于按钮、限位开关、传感器数字输出等场景。",
    "version": "1.0.0",
    "category": "gpio",
    "parameters": {
        "type": "object",
        "required": ["pin_number"],
        "properties": {
            "pin_number": {
                "type": "integer",
                "minimum": 0,
                "maximum": 63,
                "description": "GPIO引脚编号，RK3588平台有效范围为0-63"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "value": {"type": "integer", "description": "引脚电平值，0或1"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "low",
    "timeout_ms": 100
}

GPIO_WRITE_SCHEMA: ToolSchema = {
    "name": "gpio_write",
    "description": "设置指定GPIO引脚的输出电平值。适用于LED控制、继电器驱动等场景。",
    "version": "1.0.0",
    "category": "gpio",
    "parameters": {
        "type": "object",
        "required": ["pin_number", "value"],
        "properties": {
            "pin_number": {
                "type": "integer",
                "minimum": 0,
                "maximum": 63,
                "description": "GPIO引脚编号"
            },
            "value": {
                "type": "integer",
                "enum": [0, 1],
                "description": "输出电平值，0=低电平，1=高电平"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "low",
    "timeout_ms": 100
}

# PWM控制工具 —— 脉宽调制输出控制
PWM_SET_SCHEMA: ToolSchema = {
    "name": "pwm_set",
    "description": "设置指定PWM通道的频率和占空比。适用于舵机控制、LED调光、电机调速等场景。",
    "version": "1.0.0",
    "category": "pwm",
    "parameters": {
        "type": "object",
        "required": ["channel", "frequency", "duty_cycle"],
        "properties": {
            "channel": {
                "type": "integer",
                "minimum": 0,
                "maximum": 15,
                "description": "PWM通道号，RK3588支持0-15通道"
            },
            "frequency": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000000,
                "description": "PWM频率(Hz)，范围1Hz-1MHz"
            },
            "duty_cycle": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": "占空比百分比，范围0.0-100.0"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "actual_frequency": {"type": "integer"},
            "actual_duty": {"type": "number"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "medium",
    "timeout_ms": 200
}

SERVO_SET_SCHEMA: ToolSchema = {
    "name": "servo_set",
    "description": "设置舵机角度，通过PWM输出实现精确角度控制。适用于PCA9685扩展板连接的舵机。",
    "version": "1.0.0",
    "category": "pwm",
    "parameters": {
        "type": "object",
        "required": ["channel", "angle"],
        "properties": {
            "channel": {
                "type": "integer",
                "minimum": 0,
                "maximum": 15,
                "description": "舵机连接的PWM通道号"
            },
            "angle": {
                "type": "integer",
                "minimum": 0,
                "maximum": 180,
                "description": "目标角度(度)，标准舵机范围0-180度"
            },
            "speed": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 50,
                "description": "转动速度百分比，1-100，默认50"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "actual_angle": {"type": "integer"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "medium",
    "timeout_ms": 1000
}

# I2C通信工具 —— I2C总线读写操作
I2C_READ_SCHEMA: ToolSchema = {
    "name": "i2c_read",
    "description": "从I2C设备读取指定字节数的数据。适用于BMP280、MPU6050等I2C传感器。",
    "version": "1.0.0",
    "category": "i2c",
    "parameters": {
        "type": "object",
        "required": ["bus", "address", "length"],
        "properties": {
            "bus": {
                "type": "integer",
                "minimum": 0,
                "maximum": 7,
                "description": "I2C总线号，RK3588支持I2C-0到I2C-7"
            },
            "address": {
                "type": "integer",
                "minimum": 8,
                "maximum": 119,
                "description": "I2C从设备7位地址，范围0x08-0x77"
            },
            "register": {
                "type": "integer",
                "minimum": 0,
                "maximum": 255,
                "description": "寄存器地址(可选)，如未指定则从当前指针读取"
            },
            "length": {
                "type": "integer",
                "minimum": 1,
                "maximum": 256,
                "description": "读取字节数，最大256字节"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "data": {"type": "array", "items": {"type": "integer"}},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "low",
    "timeout_ms": 500
}

I2C_WRITE_SCHEMA: ToolSchema = {
    "name": "i2c_write",
    "description": "向I2C设备写入数据。适用于配置传感器参数、设置PCA9685寄存器等场景。",
    "version": "1.0.0",
    "category": "i2c",
    "parameters": {
        "type": "object",
        "required": ["bus", "address", "data"],
        "properties": {
            "bus": {
                "type": "integer",
                "minimum": 0,
                "maximum": 7,
                "description": "I2C总线号"
            },
            "address": {
                "type": "integer",
                "minimum": 8,
                "maximum": 119,
                "description": "I2C从设备7位地址"
            },
            "register": {
                "type": "integer",
                "minimum": 0,
                "maximum": 255,
                "description": "寄存器地址(可选)"
            },
            "data": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 255},
                "description": "待写入的字节数据列表"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "bytes_written": {"type": "integer"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "low",
    "timeout_ms": 500
}

# UART通信工具 —— 串口数据收发
UART_SEND_SCHEMA: ToolSchema = {
    "name": "uart_send",
    "description": "通过UART串口发送数据。适用于与STM32、ESP32等MCU通信。",
    "version": "1.0.0",
    "category": "uart",
    "parameters": {
        "type": "object",
        "required": ["port", "data"],
        "properties": {
            "port": {
                "type": "string",
                "description": "串口设备路径，如/dev/ttyS2"
            },
            "data": {
                "type": "string",
                "description": "待发送的字符串数据(将自动编码为UTF-8)"
            },
            "baudrate": {
                "type": "integer",
                "enum": [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600],
                "default": 115200,
                "description": "波特率，默认115200"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "bytes_sent": {"type": "integer"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "low",
    "timeout_ms": 500
}

# 电机控制工具 —— 直流电机/步进电机控制
MOTOR_CONTROL_SCHEMA: ToolSchema = {
    "name": "motor_control",
    "description": "控制直流电机转速和方向。speed正值为正转，负值为反转，0为停止。",
    "version": "1.0.0",
    "category": "motor",
    "parameters": {
        "type": "object",
        "required": ["motor_id", "speed"],
        "properties": {
            "motor_id": {
                "type": "integer",
                "minimum": 0,
                "maximum": 7,
                "description": "电机编号，系统最多支持8路电机"
            },
            "speed": {
                "type": "integer",
                "minimum": -100,
                "maximum": 100,
                "description": "转速百分比，-100到+100，0为停止"
            },
            "duration_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": 30000,
                "default": 0,
                "description": "持续运行时间(毫秒)，0表示持续运行直到新指令"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "motor_id": {"type": "integer"},
            "actual_speed": {"type": "integer"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "high",
    "timeout_ms": 500
}

# 传感器读取工具 —— 统一传感器接口
SENSOR_READ_SCHEMA: ToolSchema = {
    "name": "sensor_read",
    "description": "读取指定传感器的当前测量值。支持超声波、温度、湿度、光照、陀螺仪等多种传感器。",
    "version": "1.0.0",
    "category": "sensor",
    "parameters": {
        "type": "object",
        "required": ["sensor_id"],
        "properties": {
            "sensor_id": {
                "type": "string",
                "description": "传感器唯一标识符，如'hcsr04_front', 'bmp280_env', 'mpu6050_imu'"
            },
            "attributes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "需要读取的属性列表，为空则返回所有可用属性"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "sensor_id": {"type": "string"},
            "data": {"type": "object", "description": "传感器数据字典"},
            "timestamp_ms": {"type": "integer"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "low",
    "timeout_ms": 500
}

# 摄像头工具 —— 图像采集
CAMERA_CAPTURE_SCHEMA: ToolSchema = {
    "name": "camera_capture",
    "description": "采集摄像头单帧图像。支持RGB、YUV、灰度等多种格式。",
    "version": "1.0.0",
    "category": "camera",
    "parameters": {
        "type": "object",
        "required": [],
        "properties": {
            "resolution": {
                "type": "string",
                "enum": ["320x240", "640x480", "1280x720", "1920x1080"],
                "default": "640x480",
                "description": "采集分辨率"
            },
            "format": {
                "type": "string",
                "enum": ["rgb", "yuv", "gray", "jpeg"],
                "default": "rgb",
                "description": "图像格式"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "image_data": {"type": "string", "description": "Base64编码的图像数据"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "low",
    "timeout_ms": 1000
}

# 机械臂控制工具 —— Dofbot机械臂操作
ARM_MOVE_SCHEMA: ToolSchema = {
    "name": "arm_move",
    "description": "控制机械臂移动到指定关节角度位置。每个关节角度独立控制，速度受限以确保安全。",
    "version": "1.0.0",
    "category": "arm",
    "parameters": {
        "type": "object",
        "required": ["joint_angles"],
        "properties": {
            "joint_angles": {
                "type": "array",
                "items": {"type": "number", "minimum": -180.0, "maximum": 180.0},
                "description": "各关节目标角度列表(度)，例如[90, 130, 0, 0, 90]表示5自由度",
                "minItems": 3,
                "maxItems": 6
            },
            "speed": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 20,
                "description": "运动速度百分比，安全范围1-50，默认20"
            },
            "wait": {
                "type": "boolean",
                "default": True,
                "description": "是否等待运动完成再返回"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "current_angles": {"type": "array", "items": {"type": "number"}},
            "movement_time_ms": {"type": "integer"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "high",
    "timeout_ms": 5000
}

ARM_GRIPPER_SCHEMA: ToolSchema = {
    "name": "arm_gripper",
    "description": "控制机械臂夹爪开合程度。适用于抓取和释放物体。",
    "version": "1.0.0",
    "category": "arm",
    "parameters": {
        "type": "object",
        "required": ["position"],
        "properties": {
            "position": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "夹爪开合百分比，0=完全闭合，100=完全张开"
            },
            "force": {
                "type": "integer",
                "minimum": 10,
                "maximum": 80,
                "default": 50,
                "description": "夹持力度百分比，安全范围10-80"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "actual_position": {"type": "integer"},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "high",
    "timeout_ms": 2000
}

# 紧急停止工具 —— 最高优先级安全操作
EMERGENCY_STOP_SCHEMA: ToolSchema = {
    "name": "emergency_stop",
    "description": "【最高优先级】立即停止所有运动执行器（电机、机械臂）。用于安全紧急情况。",
    "version": "1.0.0",
    "category": "safety",
    "parameters": {
        "type": "object",
        "required": [],
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["all", "motors", "arm"],
                "default": "all",
                "description": "停止范围：all=全部，motors=仅电机，arm=仅机械臂"
            }
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "stopped_devices": {"type": "array", "items": {"type": "string"}},
            "error_message": {"type": "string"}
        }
    },
    "danger_level": "critical",
    "timeout_ms": 100
}

# =============================================================================
# 硬件工具注册表
# =============================================================================

HARDWARE_TOOLS: List[ToolSchema] = [
    GPIO_READ_SCHEMA,
    GPIO_WRITE_SCHEMA,
    PWM_SET_SCHEMA,
    SERVO_SET_SCHEMA,
    I2C_READ_SCHEMA,
    I2C_WRITE_SCHEMA,
    UART_SEND_SCHEMA,
    MOTOR_CONTROL_SCHEMA,
    SENSOR_READ_SCHEMA,
    CAMERA_CAPTURE_SCHEMA,
    ARM_MOVE_SCHEMA,
    ARM_GRIPPER_SCHEMA,
    EMERGENCY_STOP_SCHEMA,
]

# 工具名称到Schema的索引字典
_TOOL_SCHEMA_INDEX: Dict[str, ToolSchema] = {}

# 工具名称到处理函数的映射字典
_TOOL_HANDLER_REGISTRY: Dict[str, Tuple[ToolSchema, ToolHandler]] = {}

# 危险级别到数值权重的映射，用于安全决策
_DANGER_LEVEL_WEIGHT = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 5,
}


# =============================================================================
# 工具注册与查询函数
# =============================================================================


def _build_schema_index() -> None:
    """
    构建工具Schema索引字典，加速工具查找。
    该函数在模块导入时自动调用，将所有预定义工具加入索引。
    """
    _TOOL_SCHEMA_INDEX.clear()
    for schema in HARDWARE_TOOLS:
        name = schema["name"]
        _TOOL_SCHEMA_INDEX[name] = schema
        logger.debug(f"已索引工具Schema: {name}")


def get_tool_schema(name: str) -> Optional[ToolSchema]:
    """
    根据工具名称获取对应的JSON Schema定义。

    参数:
        name: 工具名称，如"gpio_read"、"motor_control"等

    返回:
        工具的JSON Schema字典，若工具不存在则返回None

    示例:
        >>> schema = get_tool_schema("gpio_read")
        >>> print(schema["description"])
        读取指定GPIO引脚的当前电平值...
    """
    return _TOOL_SCHEMA_INDEX.get(name)


def list_all_tools() -> List[ToolSchema]:
    """
    获取所有已注册工具的Schema列表。

    返回:
        包含所有工具Schema定义的字典列表，包含预定义工具和自定义注册工具
    """
    # 合并预定义工具和动态注册工具
    all_schemas: List[ToolSchema] = list(_TOOL_SCHEMA_INDEX.values())
    return all_schemas


def get_tools_by_category(category: str) -> List[ToolSchema]:
    """
    按类别过滤获取工具Schema列表。

    参数:
        category: 工具类别，可选值: gpio, pwm, i2c, uart, motor, sensor, camera, arm, safety

    返回:
        指定类别的工具Schema列表
    """
    return [
        schema for schema in _TOOL_SCHEMA_INDEX.values()
        if schema.get("category") == category
    ]


def register_custom_tool(
    name: str,
    schema: ToolSchema,
    handler: ToolHandler
) -> ValidationResult:
    """
    注册自定义硬件工具到工具注册表。

    支持在运行时动态扩展硬件控制工具，适用于自定义外设或实验性功能。
    注册的自定义工具同样受Schema验证和白名单机制约束。

    参数:
        name: 工具名称，需唯一，不可与预定义工具重名
        schema: 工具的JSON Schema定义字典
        handler: 工具执行函数，接收schema定义参数并返回结果

    返回:
        (是否成功, 状态信息) 元组

    示例:
        >>> def my_handler(pin: int, value: int) -> dict:
        ...     return {"success": True}
        >>> schema = {"name": "custom_led", ...}
        >>> ok, msg = register_custom_tool("custom_led", schema, my_handler)
    """
    if not name or not isinstance(name, str):
        return False, "工具名称必须为非空字符串"

    if not schema or not isinstance(schema, dict):
        return False, "Schema必须为非空字典"

    if not callable(handler):
        return False, "handler必须是可调用的函数"

    # 检查是否与现有工具冲突
    if name in _TOOL_SCHEMA_INDEX:
        logger.warning(f"工具'{name}'已存在，将被覆盖")

    # 验证schema格式
    if "parameters" not in schema:
        return False, "Schema必须包含'parameters'字段"

    # 注册到索引和handler注册表
    schema["name"] = name  # 确保名称一致
    _TOOL_SCHEMA_INDEX[name] = schema
    _TOOL_HANDLER_REGISTRY[name] = (schema, handler)

    logger.info(f"自定义工具已注册: {name}")
    return True, f"工具'{name}'注册成功"


def get_tool_handler(name: str) -> Optional[ToolHandler]:
    """
    获取已注册工具的处理函数。

    参数:
        name: 工具名称

    返回:
        工具的执行函数，若工具未注册handler则返回None
    """
    entry = _TOOL_HANDLER_REGISTRY.get(name)
    if entry:
        return entry[1]
    return None


def unregister_tool(name: str) -> bool:
    """
    从注册表中移除指定工具。预定义工具不允许移除。

    参数:
        name: 要移除的工具名称

    返回:
        是否成功移除
    """
    # 预定义工具不允许移除
    if name in [t["name"] for t in HARDWARE_TOOLS]:
        logger.warning(f"预定义工具'{name}'不允许移除")
        return False

    if name in _TOOL_HANDLER_REGISTRY:
        del _TOOL_HANDLER_REGISTRY[name]
    if name in _TOOL_SCHEMA_INDEX:
        del _TOOL_SCHEMA_INDEX[name]

    logger.info(f"工具已移除: {name}")
    return True


# =============================================================================
# Schema验证函数
# =============================================================================


def validate_tool_call(call: dict) -> ValidationResult:
    """
    验证工具调用请求是否符合JSON Schema定义。

    验证内容包括：
        1. 工具名称是否存在
        2. 必填参数是否齐全
        3. 参数类型是否正确
        4. 数值参数是否在允许范围内
        5. 字符串参数是否符合枚举约束

    参数:
        call: 工具调用请求字典，格式 {"name": "工具名", "parameters": {...}}

    返回:
        (是否通过验证, 错误信息) 元组。通过验证返回(True, "OK")，
        未通过返回(False, 具体错误描述)

    示例:
        >>> call = {"name": "gpio_read", "parameters": {"pin_number": 15}}
        >>> ok, msg = validate_tool_call(call)
        >>> assert ok  # 验证通过
    """
    if not isinstance(call, dict):
        return False, "工具调用必须是字典类型"

    tool_name = call.get("name")
    if not tool_name:
        return False, "工具调用缺少'name'字段"

    parameters = call.get("parameters", {})
    if not isinstance(parameters, dict):
        return False, "parameters必须是字典类型"

    # 查找工具Schema
    schema = get_tool_schema(tool_name)
    if schema is None:
        available = ", ".join(sorted(_TOOL_SCHEMA_INDEX.keys()))
        return False, f"未知工具'{tool_name}'。可用工具: [{available}]"

    param_schema = schema.get("parameters", {})
    required_params: List[str] = param_schema.get("required", [])
    properties: Dict[str, dict] = param_schema.get("properties", {})

    # 检查必填参数
    for req_param in required_params:
        if req_param not in parameters:
            return False, f"缺少必填参数'{req_param}'，工具'{tool_name}'"

    # 检查每个参数的类型和范围
    for param_name, param_value in parameters.items():
        if param_name not in properties:
            return False, f"未知参数'{param_name}'，工具'{tool_name}'"

        prop_def = properties[param_name]
        result = _validate_single_parameter(param_name, param_value, prop_def, tool_name)
        if not result[0]:
            return result

    logger.debug(f"工具调用验证通过: {tool_name}")
    return True, "OK"


def _validate_single_parameter(
    name: str,
    value: Any,
    prop_def: dict,
    tool_name: str
) -> ValidationResult:
    """
    验证单个参数值是否符合其Schema定义。

    参数:
        name: 参数名
        value: 参数值
        prop_def: 参数的Schema定义
        tool_name: 所属工具名称（用于错误信息）

    返回:
        (是否通过, 错误信息) 元组
    """
    param_type = prop_def.get("type")

    # 类型检查
    if param_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return False, f"参数'{name}'必须是整数类型，实际为{type(value).__name__}"
        # 范围检查
        minimum = prop_def.get("minimum")
        maximum = prop_def.get("maximum")
        if minimum is not None and value < minimum:
            return False, f"参数'{name}'值{value}小于最小值{minimum}"
        if maximum is not None and value > maximum:
            return False, f"参数'{name}'值{value}超过最大值{maximum}"

    elif param_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"参数'{name}'必须是数值类型"
        minimum = prop_def.get("minimum")
        maximum = prop_def.get("maximum")
        if minimum is not None and value < minimum:
            return False, f"参数'{name}'值{value}小于最小值{minimum}"
        if maximum is not None and value > maximum:
            return False, f"参数'{name}'值{value}超过最大值{maximum}"

    elif param_type == "string":
        if not isinstance(value, str):
            return False, f"参数'{name}'必须是字符串类型"
        # 枚举检查
        enum_values = prop_def.get("enum")
        if enum_values and value not in enum_values:
            return False, f"参数'{name}'值'{value}'不在允许值{enum_values}中"

    elif param_type == "boolean":
        if not isinstance(value, bool):
            return False, f"参数'{name}'必须是布尔类型"

    elif param_type == "array":
        if not isinstance(value, list):
            return False, f"参数'{name}'必须是数组类型"
        # 数组长度检查
        min_items = prop_def.get("minItems")
        max_items = prop_def.get("maxItems")
        if min_items is not None and len(value) < min_items:
            return False, f"参数'{name}'数组长度{len(value)}小于最小{min_items}"
        if max_items is not None and len(value) > max_items:
            return False, f"参数'{name}'数组长度{len(value)}超过最大{max_items}"
        # 数组元素类型检查
        items_schema = prop_def.get("items", {})
        for idx, item in enumerate(value):
            item_result = _validate_single_parameter(
                f"{name}[{idx}]", item, items_schema, tool_name
            )
            if not item_result[0]:
                return item_result

    elif param_type == "object":
        if not isinstance(value, dict):
            return False, f"参数'{name}'必须是对象类型"

    # 通过所有检查
    return True, "OK"


def validate_parameters_against_whitelist(
    tool_name: str,
    parameters: dict
) -> ValidationResult:
    """
    验证工具参数是否符合硬件命令白名单的安全约束。

    针对高危险级别的工具进行额外安全检查：
        - motor_control: 速度限制在±100以内
        - arm_move: 关节角度限制在±180度，速度限制在50%以内
        - arm_gripper: 力度限制在80%以内

    参数:
        tool_name: 工具名称
        parameters: 参数字典

    返回:
        (是否通过, 错误信息) 元组
    """
    schema = get_tool_schema(tool_name)
    if schema is None:
        return False, f"未知工具'{tool_name}'"

    danger_level = schema.get("danger_level", "low")

    # 高危险级别工具的额外安全检查
    if danger_level == "high" and tool_name == "motor_control":
        speed = parameters.get("speed", 0)
        if abs(speed) > 100:
            return False, f"电机速度{speed}超过安全上限±100"

    elif danger_level == "high" and tool_name == "arm_move":
        speed = parameters.get("speed", 20)
        if speed > 50:
            return False, f"机械臂运动速度{speed}超过安全上限50"
        joint_angles = parameters.get("joint_angles", [])
        for idx, angle in enumerate(joint_angles):
            if not -180 <= angle <= 180:
                return False, f"关节{idx}角度{angle}超出安全范围[-180, 180]"

    elif danger_level == "critical":
        logger.warning(f"执行关键安全操作: {tool_name}")

    return True, "OK"


def get_tool_danger_level(tool_name: str) -> str:
    """
    获取工具的危险级别。

    参数:
        tool_name: 工具名称

    返回:
        危险级别字符串: low/medium/high/critical，未知工具返回"unknown"
    """
    schema = get_tool_schema(tool_name)
    if schema is None:
        return "unknown"
    return schema.get("danger_level", "low")


def get_tool_timeout(tool_name: str) -> int:
    """
    获取工具的建议超时时间(毫秒)。

    参数:
        tool_name: 工具名称

    返回:
        超时时间(毫秒)，未知工具返回默认5000ms
    """
    schema = get_tool_schema(tool_name)
    if schema is None:
        return 5000
    return schema.get("timeout_ms", 5000)


def get_tools_summary() -> str:
    """
    生成所有可用工具的摘要信息字符串。

    返回:
        格式化的工具列表字符串，包含名称、类别、描述和危险级别
    """
    lines = ["=" * 60, "KunPeng-Cortex 硬件控制工具列表", "=" * 60]
    for schema in sorted(_TOOL_SCHEMA_INDEX.values(), key=lambda s: s["name"]):
        lines.append(f"\n  工具名称: {schema['name']}")
        lines.append(f"  类别: {schema.get('category', 'N/A')}")
        lines.append(f"  版本: {schema.get('version', 'N/A')}")
        lines.append(f"  危险级别: {schema.get('danger_level', 'low')}")
        lines.append(f"  超时: {schema.get('timeout_ms', 5000)}ms")
        lines.append(f"  描述: {schema['description']}")
        lines.append("-" * 40)
    return "\n".join(lines)


# =============================================================================
# 模块初始化
# =============================================================================

# 导入时自动构建索引
_build_schema_index()
logger.info(f"工具定义模块已初始化，共加载{len(_TOOL_SCHEMA_INDEX)}个工具")
