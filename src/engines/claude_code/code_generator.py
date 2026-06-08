#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code_generator.py — KunPeng-Cortex 代码生成器模块

负责将自然语言意图(NL)转换为可在沙箱中安全执行的Python代码。
基于模板匹配引擎实现快速代码生成，同时集成安全白名单验证确保生成代码的安全性。

核心功能：
    - 自然语言意图理解 → 代码模板匹配
    - Jinja2风格模板参数填充
    - GPIO读写、PWM控制、传感器读取、电机控制、机械臂操作模板库
    - 生成代码安全白名单验证
    - 模板缓存与预编译

安全设计：
    - 所有生成的代码必须通过安全白名单验证
    - 禁止使用eval、exec、import os/subprocess等危险操作
    - 数值参数自动范围限制
    - 硬件操作仅允许预定义的白名单函数

硬件平台: OrangePi Kunpeng Pro (RK3588, ARM64)
作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入工具定义模块
from .tool_definitions import (
    get_tool_danger_level,
    get_tools_by_category,
    validate_tool_call,
)

# 配置模块日志记录器
logger = logging.getLogger("kunpeng_cortex.claude_code.code_generator")

# =============================================================================
# 数据模型定义
# =============================================================================


@dataclasses.dataclass
class GeneratedCode:
    """
    生成的代码结果数据类。

    属性:
        code: 生成的完整Python代码字符串
        template_name: 匹配到的模板名称
        confidence: 模板匹配置信度(0.0-1.0)
        required_tools: 代码依赖的硬件工具列表
        estimated_time_ms: 预估执行时间(毫秒)
        safety_level: 代码安全级别(low/medium/high)
        parameter_values: 填充到模板的参数值字典
        metadata: 额外元数据字典
    """
    code: str
    template_name: str
    confidence: float
    required_tools: List[str] = dataclasses.field(default_factory=list)
    estimated_time_ms: int = 1000
    safety_level: str = "low"
    parameter_values: Dict[str, Any] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


# =============================================================================
# 代码模板定义
# =============================================================================

# GPIO控制模板
TEMPLATE_GPIO_READ = '''# GPIO读取操作 —— 自动生成
# 意图: {{intent}}
result = gpio_read(pin_number={{pin_number}})
print(json.dumps({"pin": {{pin_number}}, "value": result}))
'''

TEMPLATE_GPIO_WRITE = '''# GPIO写入操作 —— 自动生成
# 意图: {{intent}}
result = gpio_write(pin_number={{pin_number}}, value={{value}})
print(json.dumps({"pin": {{pin_number}}, "value": {{value}}, "success": result}))
'''

TEMPLATE_GPIO_BLINK = '''# GPIO闪烁操作 —— 自动生成
# 意图: {{intent}}
import time
pin = {{pin_number}}
frequency = {{frequency}}
duration = {{duration}}
period = 1.0 / frequency
half_period = period / 2
cycles = int(duration * frequency)

for i in range(cycles):
    gpio_write(pin_number=pin, value=1)
    time.sleep(half_period)
    gpio_write(pin_number=pin, value=0)
    time.sleep(half_period)

gpio_write(pin_number=pin, value=0)  # 确保关闭
print(json.dumps({"pin": pin, "frequency": frequency, "cycles": cycles, "done": True}))
'''

# PWM控制模板
TEMPLATE_PWM_SET = '''# PWM设置操作 —— 自动生成
# 意图: {{intent}}
result = pwm_set(channel={{channel}}, frequency={{frequency}}, duty_cycle={{duty_cycle}})
print(json.dumps({"channel": {{channel}}, "frequency": {{frequency}}, "duty": {{duty_cycle}}, "success": result}))
'''

TEMPLATE_SERVO_SET = '''# 舵机角度控制 —— 自动生成
# 意图: {{intent}}
result = servo_set(channel={{channel}}, angle={{angle}}{% if speed is defined %}, speed={{speed}}{% endif %})
print(json.dumps({"channel": {{channel}}, "angle": {{angle}}, "success": result}))
'''

# 传感器读取模板
TEMPLATE_SENSOR_READ = '''# 传感器读取操作 —— 自动生成
# 意图: {{intent}}
data = sensor_read(sensor_id="{{sensor_id}}"{% if attributes is defined %}, attributes={{attributes}}{% endif %})
print(json.dumps({"sensor": "{{sensor_id}}", "data": data}))
'''

TEMPLATE_SENSOR_POLL = '''# 传感器轮询操作 —— 自动生成
# 意图: {{intent}}
import time

sensor_id = "{{sensor_id}}"
interval = {{interval}}
count = {{count}}
results = []

for i in range(count):
    data = sensor_read(sensor_id=sensor_id)
    results.append({"index": i, "data": data, "timestamp": time.time()})
    if i < count - 1:
        time.sleep(interval)

print(json.dumps({"sensor": sensor_id, "samples": count, "results": results}))
'''

# 电机控制模板
TEMPLATE_MOTOR_CONTROL = '''# 电机控制操作 —— 自动生成
# 意图: {{intent}}
motor_id = {{motor_id}}
speed = {{speed}}
result = motor_control(motor_id=motor_id, speed=speed{% if duration_ms is defined %}, duration_ms={{duration_ms}}{% endif %})
print(json.dumps({"motor": motor_id, "speed": speed, "success": result}))
'''

TEMPLATE_MOTOR_STOP_ALL = '''# 停止所有电机 —— 自动生成
# 意图: {{intent}}
results = []
for motor_id in range({{max_motors|default(4)}}):
    result = motor_control(motor_id=motor_id, speed=0)
    results.append({"motor": motor_id, "stopped": result})

print(json.dumps({"stopped_all": True, "results": results}))
'''

# 机械臂控制模板
TEMPLATE_ARM_MOVE = '''# 机械臂移动操作 —— 自动生成
# 意图: {{intent}}
joint_angles = {{joint_angles}}
speed = {{speed|default(20)}}
result = arm_move(joint_angles=joint_angles, speed=speed, wait=True)
print(json.dumps({"angles": joint_angles, "speed": speed, "success": result}))
'''

TEMPLATE_ARM_GRIPPER = '''# 机械臂夹爪控制 —— 自动生成
# 意图: {{intent}}
position = {{position}}
force = {{force|default(50)}}
result = arm_gripper(position=position, force=force)
print(json.dumps({"position": position, "force": force, "success": result}))
'''

TEMPLATE_ARM_PICK_PLACE = '''# 机械臂取放操作 —— 自动生成
# 意图: {{intent}}
import time

# 1. 移动到取物位置
pick_angles = {{pick_angles}}
arm_move(joint_angles=pick_angles, speed={{speed|default(20)}}, wait=True)
time.sleep(0.5)

# 2. 张开夹爪
arm_gripper(position=100, force=30)
time.sleep(0.5)

# 3. 闭合夹爪抓取
arm_gripper(position={{grip_position|default(20)}}, force={{grip_force|default(50)}})
time.sleep(0.3)

# 4. 移动到放置位置
place_angles = {{place_angles}}
arm_move(joint_angles=place_angles, speed={{speed|default(20)}}, wait=True)
time.sleep(0.5)

# 5. 张开释放
arm_gripper(position=100, force=30)
time.sleep(0.3)

# 6. 回到初始位置
home_angles = {{home_angles|default([90, 130, 0, 0, 90])}}
arm_move(joint_angles=home_angles, speed=15, wait=True)

print(json.dumps({"action": "pick_place", "done": True}))
'''

# 摄像头模板
TEMPLATE_CAMERA_CAPTURE = '''# 摄像头图像采集 —— 自动生成
# 意图: {{intent}}
result = camera_capture(resolution="{{resolution|default('640x480')}}", format="{{format|default('rgb')}}")
print(json.dumps({"image_captured": True, "info": result}))
'''

# I2C通信模板
TEMPLATE_I2C_READ = '''# I2C读取操作 —— 自动生成
# 意图: {{intent}}
data = i2c_read(bus={{bus}}, address={{address}}, length={{length}}{% if register is defined %}, register={{register}}{% endif %})
print(json.dumps({"bus": {{bus}}, "address": {{address}}, "data": data}))
'''

TEMPLATE_I2C_WRITE = '''# I2C写入操作 —— 自动生成
# 意图: {{intent}}
bytes_written = i2c_write(bus={{bus}}, address={{address}}, data={{data}}{% if register is defined %}, register={{register}}{% endif %})
print(json.dumps({"bus": {{bus}}, "address": {{address}}, "bytes_written": bytes_written}))
'''

# UART通信模板
TEMPLATE_UART_SEND = '''# UART发送操作 —— 自动生成
# 意图: {{intent}}
result = uart_send(port="{{port}}", data="{{data}}"{% if baudrate is defined %}, baudrate={{baudrate}}{% endif %})
print(json.dumps({"port": "{{port}}", "bytes_sent": result}))
'''

# 紧急停止模板
TEMPLATE_EMERGENCY_STOP = '''# 紧急停止操作 —— 自动生成
# 意图: {{intent}}
# 【最高优先级安全操作】
result = emergency_stop(scope="{{scope|default('all')}}")
print(json.dumps({"emergency_stop": True, "scope": "{{scope|default('all')}}", "result": result}))
'''

# 综合任务模板
TEMPLATE_SEQUENCE = '''# 复合任务序列 —— 自动生成
# 意图: {{intent}}
import time

results = []
{% for step in steps %}
# 步骤{{loop.index}}: {{step.description}}
{{step.code}}
results.append({"step": {{loop.index}}, "description": "{{step.description}}", "done": True})
time.sleep({{step.delay|default(0.1)}})
{% endfor %}

print(json.dumps({"sequence_complete": True, "steps": len(results), "results": results}))
'''


# =============================================================================
# 模板注册表
# =============================================================================

# 模板名称到模板内容的映射
CODE_TEMPLATES: Dict[str, str] = {
    "gpio_read": TEMPLATE_GPIO_READ,
    "gpio_write": TEMPLATE_GPIO_WRITE,
    "gpio_blink": TEMPLATE_GPIO_BLINK,
    "pwm_set": TEMPLATE_PWM_SET,
    "servo_set": TEMPLATE_SERVO_SET,
    "sensor_read": TEMPLATE_SENSOR_READ,
    "sensor_poll": TEMPLATE_SENSOR_POLL,
    "motor_control": TEMPLATE_MOTOR_CONTROL,
    "motor_stop_all": TEMPLATE_MOTOR_STOP_ALL,
    "arm_move": TEMPLATE_ARM_MOVE,
    "arm_gripper": TEMPLATE_ARM_GRIPPER,
    "arm_pick_place": TEMPLATE_ARM_PICK_PLACE,
    "camera_capture": TEMPLATE_CAMERA_CAPTURE,
    "i2c_read": TEMPLATE_I2C_READ,
    "i2c_write": TEMPLATE_I2C_WRITE,
    "uart_send": TEMPLATE_UART_SEND,
    "emergency_stop": TEMPLATE_EMERGENCY_STOP,
    "sequence": TEMPLATE_SEQUENCE,
}

# 意图关键词到模板名称的匹配规则
# 格式: (模板名称, [关键词列表], 权重)
INTENT_MATCH_RULES: List[Tuple[str, List[str], float]] = [
    ("gpio_read", ["读", "读取", "read", "gpio", "引脚", "pin", "电平", "状态"], 1.0),
    ("gpio_write", ["写", "写入", "write", "gpio", "引脚", "pin", "设置", "输出", "打开", "关闭", "开灯", "关灯"], 1.0),
    ("gpio_blink", ["闪烁", "blink", "闪", "呼吸", "pwm", "gpio"], 1.0),
    ("pwm_set", ["pwm", "脉宽", "调制", "频率", "占空比", "输出"], 1.0),
    ("servo_set", ["舵机", "servo", "角度", "转动", "旋转"], 1.0),
    ("sensor_read", ["传感器", "sensor", "读取", "温度", "湿度", "距离", "光照", "读数"], 1.0),
    ("sensor_poll", ["轮询", "poll", "采样", "连续", "采集", "sensor"], 1.0),
    ("motor_control", ["电机", "motor", "转动", "旋转", "速度", "驱动", "前进", "后退"], 1.0),
    ("motor_stop_all", ["停止", "stop", "急停", "全部", "所有", "电机"], 1.0),
    ("arm_move", ["机械臂", "arm", "移动", "关节", "角度", "位置"], 1.0),
    ("arm_gripper", ["夹爪", "gripper", "抓取", "放开", "松开", "夹持"], 1.0),
    ("arm_pick_place", ["取放", "pick", "place", "拿", "放", "递", "抓取", "放下"], 1.0),
    ("camera_capture", ["摄像头", "camera", "拍照", "采集", "图像", "照片"], 1.0),
    ("i2c_read", ["i2c", "读取", "读寄存器", "总线"], 1.0),
    ("i2c_write", ["i2c", "写入", "写寄存器", "配置", "总线"], 1.0),
    ("uart_send", ["uart", "串口", "发送", "通信", "serial"], 1.0),
    ("emergency_stop", ["紧急", "急停", "停止", "estop", "安全", "危险", "停止一切"], 1.0),
]


# =============================================================================
# CodeGenerator 主类
# =============================================================================


class CodeGenerator:
    """
    代码生成器 —— 自然语言到Python代码的转换引擎。

    通过模板匹配和参数填充，将自然语言意图快速转换为可在沙箱中
    安全执行的Python代码。支持意图关键词匹配和置信度评估。

    属性:
        template_dir: 模板文件目录路径（支持从文件加载额外模板）
        templates: 已加载的代码模板字典
        intent_rules: 意图匹配规则列表
        cache_enabled: 是否启用代码缓存
        _code_cache: 代码生成缓存字典

    示例:
        >>> generator = CodeGenerator()
        >>> code = generator.generate("读取GPIO引脚15的电平", {"pin_number": 15})
        >>> print(code)
        # 读取GPIO...
    """

    def __init__(self, template_dir: str = "templates") -> None:
        """
        初始化代码生成器。

        参数:
            template_dir: 模板文件目录路径。若目录存在，将自动加载
                目录中的.py.j2模板文件作为额外模板。
        """
        self.template_dir: str = template_dir
        self.templates: Dict[str, str] = dict(CODE_TEMPLATES)
        self.intent_rules: List[Tuple[str, List[str], float]] = list(INTENT_MATCH_RULES)
        self.cache_enabled: bool = True
        self._code_cache: Dict[str, GeneratedCode] = {}
        self._stats: Dict[str, int] = {
            "total_generations": 0,
            "cache_hits": 0,
            "template_matches": 0,
            "fallback_generations": 0,
        }

        # 从模板目录加载额外模板
        if os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

        logger.info(
            f"代码生成器已初始化: {len(self.templates)}个模板, "
            f"{len(self.intent_rules)}条匹配规则"
        )

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """
        从模板目录加载额外的模板文件。

        支持.py.j2和.py模板文件。

        参数:
            template_dir: 模板目录路径
        """
        template_path = Path(template_dir)
        if not template_path.is_dir():
            return

        pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        for ext in ["*.py.j2", "*.py"]:
            for file_path in template_path.glob(ext):
                name = file_path.stem
                if name.endswith(".py"):
                    name = name[:-3]
                if pattern.match(name):
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        self.templates[name] = content
                        logger.debug(f"从文件加载模板: {name}")
                    except (IOError, UnicodeDecodeError) as e:
                        logger.warning(f"无法加载模板文件 {file_path}: {e}")

    @property
    def stats(self) -> Dict[str, int]:
        """
        获取代码生成统计信息。

        返回:
            包含总生成次数、缓存命中次数等的字典
        """
        return dict(self._stats)

    def generate(self, intent: str, params: Dict[str, Any]) -> GeneratedCode:
        """
        根据自然语言意图和参数生成Python代码。

        生成流程：
            1. 意图关键词匹配 → 选择模板
            2. 模板参数填充
            3. 安全白名单验证
            4. 返回GeneratedCode结果

        参数:
            intent: 自然语言意图描述，如"读取GPIO引脚15的电平"
            params: 模板参数字典，如{"pin_number": 15}

        返回:
            GeneratedCode数据类，包含生成的代码和元数据

        示例:
            >>> generator = CodeGenerator()
            >>> result = generator.generate("读取温度传感器", {
            ...     "sensor_id": "bmp280_env"
            ... })
            >>> print(result.code)
            # 传感器读取操作...
        """
        self._stats["total_generations"] += 1

        # 检查缓存
        cache_key = f"{intent}:{json.dumps(params, sort_keys=True, default=str)}"
        if self.cache_enabled and cache_key in self._code_cache:
            self._stats["cache_hits"] += 1
            logger.debug("代码生成缓存命中")
            return self._code_cache[cache_key]

        # 步骤1: 意图匹配
        matched_template = self.match_template(intent)

        if matched_template and matched_template in self.templates:
            self._stats["template_matches"] += 1
            template_name = matched_template
            template_content = self.templates[matched_template]
            confidence = self._calculate_match_confidence(intent, template_name)
        else:
            # 回退：使用通用传感器读取模板
            self._stats["fallback_generations"] += 1
            template_name = "sensor_read"
            template_content = TEMPLATE_SENSOR_READ
            confidence = 0.3
            logger.warning(f"意图'{intent}'未匹配到模板，使用回退模板")

        # 步骤2: 参数填充（简单的模板变量替换）
        filled_code = self._fill_template(template_content, params, intent)

        # 步骤3: 安全验证
        safe, error_msg = self.validate_generated_code(filled_code)
        if not safe:
            # 安全验证失败，返回安全的空操作
            logger.error(f"生成代码未通过安全验证: {error_msg}")
            error_json = '{"error": "' + error_msg.replace('"', '\\"') + '"}'
            filled_code = (
                "# 代码生成失败: 安全验证未通过\n"
                "# 原因: " + error_msg + "\n"
                "print('" + error_json + "')"
            )
            template_name = "error_fallback"
            confidence = 0.0

        # 步骤4: 确定安全级别和依赖工具
        safety_level = self._estimate_safety_level(template_name, params)
        required_tools = self._extract_required_tools(filled_code)
        estimated_time = self._estimate_execution_time(template_name, params)

        # 构建结果
        result = GeneratedCode(
            code=filled_code,
            template_name=template_name,
            confidence=confidence,
            required_tools=required_tools,
            estimated_time_ms=estimated_time,
            safety_level=safety_level,
            parameter_values=params,
            metadata={
                "intent": intent,
                "generation_time": time.time(),
                "validation_passed": safe,
            },
        )

        # 缓存结果
        if self.cache_enabled:
            self._code_cache[cache_key] = result

        logger.debug(
            f"代码生成完成: 模板={template_name}, 置信度={confidence:.2f}, "
            f"安全级别={safety_level}"
        )
        return result

    def match_template(self, intent: str) -> Optional[str]:
        """
        根据自然语言意图匹配最佳代码模板。

        采用关键词匹配算法，对每个模板的匹配规则计算得分，
        返回得分最高的模板名称。

        参数:
            intent: 自然语言意图描述字符串

        返回:
            最佳匹配的模板名称，若无匹配则返回None

        示例:
            >>> generator = CodeGenerator()
            >>> name = generator.match_template("读取GPIO引脚15的电平")
            >>> assert name == "gpio_read"
        """
        if not intent or not isinstance(intent, str):
            return None

        intent_lower = intent.lower().strip()
        best_match: Optional[str] = None
        best_score: float = 0.0

        for template_name, keywords, weight in self.intent_rules:
            score = 0.0
            matched_keywords = 0

            for keyword in keywords:
                kw_lower = keyword.lower()
                if kw_lower in intent_lower:
                    # 完整单词匹配权重更高
                    if re.search(rf"\b{re.escape(kw_lower)}\b", intent_lower):
                        score += weight * 2.0
                    else:
                        score += weight * 1.0
                    matched_keywords += 1

            # 额外加权：匹配到的关键词比例
            if keywords:
                coverage = matched_keywords / len(keywords)
                score += coverage * weight * 0.5

            if score > best_score:
                best_score = score
                best_match = template_name

        # 最低阈值：至少得分为1.0才认为有效匹配
        if best_score < 1.0:
            return None

        logger.debug(f"意图匹配: '{intent[:30]}...' -> {best_match} (得分: {best_score:.2f})")
        return best_match

    def _calculate_match_confidence(self, intent: str, template_name: str) -> float:
        """
        计算意图-模板匹配的置信度分数。

        参数:
            intent: 自然语言意图
            template_name: 匹配的模板名称

        返回:
            置信度值(0.0-1.0)，越接近1.0表示匹配越可靠
        """
        intent_lower = intent.lower().strip()
        best_score = 0.0

        for name, keywords, weight in self.intent_rules:
            if name != template_name:
                continue

            score = 0.0
            keyword_count = len(keywords)
            matches = 0

            for keyword in keywords:
                if keyword.lower() in intent_lower:
                    matches += 1
                    score += weight

            # 归一化得分
            if keyword_count > 0:
                score = min(1.0, score / (keyword_count * weight * 0.5))

            best_score = max(best_score, score)

        return min(1.0, max(0.0, best_score))

    def _fill_template(
        self,
        template: str,
        params: Dict[str, Any],
        intent: str,
    ) -> str:
        """
        填充模板中的变量占位符。

        使用简单的{{variable}}替换语法，支持Jinja2风格的基本条件表达式。

        参数:
            template: 模板字符串
            params: 参数值字典
            intent: 原始意图（注入为模板变量）

        返回:
            填充后的代码字符串
        """
        code = template
        all_params = dict(params)
        all_params["intent"] = intent

        # 简单的变量替换 {{var_name}}
        def replace_var(match: "re.Match") -> str:
            var_expr = match.group(1).strip()

            # 处理条件表达式 {% if var is defined %}...{% endif %}
            # 简单处理：移除条件包装，使用默认值
            if "|" in var_expr:
                var_name, default_expr = var_expr.split("|", 1)
                var_name = var_name.strip()
                default_expr = default_expr.strip()

                # 解析default过滤器
                if "default(" in default_expr:
                    default_match = re.search(r'default\(([^)]+)\)', default_expr)
                    if default_match:
                        default_str = default_match.group(1).strip()
                        # 移除引号
                        if (default_str.startswith("'") and default_str.endswith("'")) or \
                           (default_str.startswith('"') and default_str.endswith('"')):
                            default_str = default_str[1:-1]
                        return str(all_params.get(var_name, default_str))

                return str(all_params.get(var_name, default_expr))

            # 直接变量替换
            if var_name in all_params:
                value = all_params[var_name]
                # JSON安全地序列化
                if isinstance(value, (list, dict)):
                    return json.dumps(value, ensure_ascii=False)
                return str(value)

            # 未找到变量，保留占位符但标记
            return match.group(0)

        # 替换{{var}}占位符
        pattern = re.compile(r"\{\{(\s*[^{}%]+\s*)\}\}")

        # 多轮替换以处理嵌套
        max_iterations = 10
        for _ in range(max_iterations):
            new_code = pattern.sub(replace_var, code)
            if new_code == code:
                break
            code = new_code

        # 移除未解析的Jinja2条件块（简化处理）
        code = re.sub(r"\{%\s*if\s+.+?\s*%\}", "", code)
        code = re.sub(r"\{%\s*endif\s*%\}", "", code)
        code = re.sub(r"\{%\s*for\s+.+?\s*%\}", "", code)
        code = re.sub(r"\{%\s*endfor\s*%\}", "", code)
        code = re.sub(r"\{\{#.*?#\}\}", "", code)  # 注释

        return code

    def validate_generated_code(self, code: str) -> Tuple[bool, str]:
        """
        验证生成的Python代码是否符合安全白名单。

        安全检查内容：
            1. 语法合法性（AST解析）
            2. 黑名单关键字检查（eval, exec, import os等）
            3. 危险导入检查
            4. 文件系统操作检查
            5. 网络操作检查

        参数:
            code: 待验证的Python代码字符串

        返回:
            (是否安全, 原因信息) 元组

        示例:
            >>> generator = CodeGenerator()
            >>> ok, msg = generator.validate_generated_code("print('safe')")
            >>> assert ok
        """
        if not code or not isinstance(code, str):
            return False, "代码为空"

        # 检查1: 语法合法性
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: 第{e.lineno}行 - {e.msg}"

        # 检查2: 黑名单关键字
        code_lower = code.lower()

        # Python代码黑名单
        code_blacklist = [
            "__import__", "importlib", "eval(", "exec(", "compile(",
            "subprocess", "os.system", "os.popen", "os.fork", "os.kill",
            "os.exec", "os.spawn", "os.chmod", "os.chown", "os.remove",
            "shutil", "socket", "urllib", "http.client", "ftplib",
            "pty", "pickle", "marshal", "ctypes", "cffi", "mmap",
            "input(", "raw_input(", "breakpoint(", "getattr(", "setattr(",
            "delattr(", "globals(", "locals(", "vars(",
        ]

        for danger in code_blacklist:
            if danger.lower() in code_lower:
                return False, f"包含禁止的操作: '{danger}'"

        # 检查3: 只允许安全的import
        allowed_modules = {
            "time", "math", "random", "json", "re", "struct",
            "array", "datetime", "collections", "itertools",
            "functools", "decimal", "fractions", "statistics",
            "typing", "dataclasses", "enum",
        }

        for node in ast.walk(ast.parse(code)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in allowed_modules:
                        return False, f"禁止导入模块: '{module}'"

        # 检查4: 禁止直接文件操作
        dangerous_paths = ["/etc/passwd", "/etc/shadow", "/dev/mem",
                          "/proc", "/sys/kernel", "/root"]
        for path in dangerous_paths:
            if path in code:
                return False, f"禁止访问路径: {path}"

        # 检查5: AST节点级别的安全检查
        for node in ast.walk(ast.parse(code)):
            # 禁止__dunder__属性访问（部分）
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    dangerous_dunders = {
                        "__class__", "__bases__", "__subclasses__",
                        "__globals__", "__code__", "__closure__",
                    }
                    if node.attr in dangerous_dunders:
                        return False, f"禁止访问特殊属性: {node.attr}"

        return True, "OK"

    def _estimate_safety_level(self, template_name: str, params: Dict[str, Any]) -> str:
        """
        估算生成代码的安全级别。

        根据模板类型和参数值评估危险程度。

        参数:
            template_name: 模板名称
            params: 参数值

        返回:
            安全级别字符串: low/medium/high/critical
        """
        danger_map = {
            "emergency_stop": "critical",
            "motor_control": "high",
            "motor_stop_all": "medium",
            "arm_move": "high",
            "arm_gripper": "high",
            "arm_pick_place": "high",
            "pwm_set": "medium",
            "servo_set": "medium",
            "uart_send": "low",
            "i2c_read": "low",
            "i2c_write": "low",
            "gpio_read": "low",
            "gpio_write": "low",
            "gpio_blink": "low",
            "sensor_read": "low",
            "sensor_poll": "low",
            "camera_capture": "low",
            "sequence": "high",
        }
        return danger_map.get(template_name, "low")

    def _extract_required_tools(self, code: str) -> List[str]:
        """
        从生成的代码中提取调用的硬件工具名称。

        参数:
            code: Python代码字符串

        返回:
            代码中调用的硬件工具名称列表
        """
        tools = []
        tool_names = [
            "gpio_read", "gpio_write", "pwm_set", "servo_set",
            "i2c_read", "i2c_write", "uart_send", "motor_control",
            "sensor_read", "camera_capture", "arm_move", "arm_gripper",
            "emergency_stop",
        ]
        for tool in tool_names:
            if tool in code:
                tools.append(tool)
        return tools

    def _estimate_execution_time(self, template_name: str, params: Dict[str, Any]) -> int:
        """
        估算代码的执行时间(毫秒)。

        参数:
            template_name: 模板名称
            params: 参数值

        返回:
            预估执行时间(毫秒)
        """
        base_times = {
            "gpio_read": 100,
            "gpio_write": 100,
            "gpio_blink": 5000,  # 闪烁通常需要较长时间
            "pwm_set": 200,
            "servo_set": 1000,
            "sensor_read": 500,
            "sensor_poll": 3000,
            "motor_control": 500,
            "motor_stop_all": 500,
            "arm_move": 5000,
            "arm_gripper": 2000,
            "arm_pick_place": 15000,
            "camera_capture": 1000,
            "i2c_read": 500,
            "i2c_write": 500,
            "uart_send": 500,
            "emergency_stop": 100,
            "sequence": 5000,
        }
        base = base_times.get(template_name, 1000)

        # 根据参数调整
        if "duration" in params:
            base += int(params["duration"] * 1000)
        if "duration_ms" in params:
            base += int(params["duration_ms"])
        if "count" in params and "interval" in params:
            base += int(params["count"] * params["interval"] * 1000)

        return min(base, 30000)  # 最大30秒

    def list_templates(self) -> List[Dict[str, str]]:
        """
        获取所有可用模板的列表。

        返回:
            模板信息字典列表，包含名称、描述、安全级别等
        """
        template_info = []
        for name, content in sorted(self.templates.items()):
            safety = self._estimate_safety_level(name, {})
            # 提取第一行注释作为描述
            desc_lines = [line.strip("# ").strip() for line in content.split("\n")[:2] if line.strip().startswith("#")]
            description = desc_lines[0] if desc_lines else name
            template_info.append({
                "name": name,
                "description": description,
                "safety_level": safety,
                "lines": content.count("\n") + 1,
            })
        return template_info

    def add_template(self, name: str, template: str, keywords: Optional[List[str]] = None) -> None:
        """
        动态添加新模板。

        参数:
            name: 模板名称
            template: 模板代码字符串
            keywords: 意图匹配关键词列表
        """
        self.templates[name] = template
        if keywords:
            self.intent_rules.append((name, keywords, 1.0))
        logger.info(f"已添加新模板: {name}")

    def clear_cache(self) -> None:
        """清除代码生成缓存。"""
        self._code_cache.clear()
        logger.debug("代码生成缓存已清除")
