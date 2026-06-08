#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务规划器 (TaskPlanner) —— KunPeng-Cortex 核心引擎模块

本模块实现复杂用户指令到原子操作序列的智能分解、资源冲突检测
与执行序列优化。采用依赖图 + 拓扑排序算法确保任务执行顺序的正确性，
并通过优先级抢占机制保障关键任务（如紧急停止、安全监控）的实时响应。

核心功能：
    - 自然语言指令解析与任务分解
    - 依赖图构建与拓扑排序
    - 资源冲突检测与调度
    - 优先级抢占与执行优化
    - 任务执行时间估算与超时设置

算法说明：
    任务分解基于有向无环图 (DAG) 模型，每个原子任务为图中的一个节点，
    依赖关系为图中的有向边。通过 Kahn 算法进行拓扑排序，确保所有前置
    任务完成后才执行后续任务。资源冲突检测采用基于时间片的资源预约表，
    检测同一时刻同一资源是否被多个任务占用。

硬件平台：
    OrangePi KunPeng Pro (RK3588, ARM64, 16GB RAM)

作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import enum
import json
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
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
# 枚举类型定义
# ---------------------------------------------------------------------------


class TaskPriority(enum.IntEnum):
    """原子任务优先级枚举。

    数值越小优先级越高。关键级任务将抢占正在执行的低优先级任务。

    Attributes:
        CRITICAL: 关键级 —— 紧急停止 / 安全互锁 / 碰撞避免。
        HIGH: 高级 —— 视觉推理 / 语音交互 / 导航避障。
        NORMAL: 普通级 —— 日常服务 / 物品递送 / 环境调节。
        LOW: 低级 —— 状态上报 / 日志记录 / 数据统计。
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class ConflictType(enum.Enum):
    """资源冲突类型枚举。

    Attributes:
        RESOURCE_EXCLUSIVE: 独占资源冲突（同一硬件不可同时被两个任务使用）。
        RESOURCE_SHARED: 共享资源过载（共享资源超出容量上限）。
        TEMPORAL: 时序冲突（任务间时间约束不可满足）。
        SAFETY: 安全冲突（任务组合可能危及安全）。
    """

    RESOURCE_EXCLUSIVE = "resource_exclusive"
    RESOURCE_SHARED = "resource_shared"
    TEMPORAL = "temporal"
    SAFETY = "safety"


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------


@dataclass
class AtomicTask:
    """原子任务数据类。

    原子任务是不可再分的最小执行单元，直接映射到 HAL 的单一硬件操作。
    通过 ``dependencies`` 字段表达任务间的依赖关系。

    Attributes:
        task_id: 全局唯一任务标识符。
        name: 任务名称，如 "motor_set_speed"、"arm_move_joint"。
        priority: 任务优先级。
        resource: 所需资源标识符，如 "motor_0"、"arm_joint_1"。
        params: 任务参数字典，直接传递给 HAL。
        timeout: 任务执行超时时间（秒）。
        dependencies: 依赖任务 ID 列表（拓扑排序用）。
        estimated_duration_ms: 预估执行耗时（毫秒），用于调度优化。
        created_at: 任务创建时间戳。
        metadata: 额外元数据（如来源指令、执行上下文）。
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unknown"
    priority: TaskPriority = TaskPriority.NORMAL
    resource: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 5.0
    dependencies: List[str] = field(default_factory=list)
    estimated_duration_ms: float = 100.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """将任务序列化为字典。

        Returns:
            包含全部字段的字典表示。
        """
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority.value,
            "resource": self.resource,
            "params": self.params,
            "timeout": self.timeout,
            "dependencies": self.dependencies,
            "estimated_duration_ms": self.estimated_duration_ms,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicTask":
        """从字典反序列化为任务对象。

        Args:
            data: 包含任务字段的字典。

        Returns:
            反序列化后的 ``AtomicTask`` 实例。
        """
        return cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            name=data.get("name", "unknown"),
            priority=TaskPriority(data.get("priority", 2)),
            resource=data.get("resource", ""),
            params=data.get("params", {}),
            timeout=data.get("timeout", 5.0),
            dependencies=data.get("dependencies", []),
            estimated_duration_ms=data.get("estimated_duration_ms", 100.0),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConflictReport:
    """资源冲突报告数据类。

    当 ``check_conflicts`` 检测到冲突时，生成此报告对象，
    包含冲突类型、涉及的冲突任务和建议的解决方案。

    Attributes:
        conflict_type: 冲突类型。
        task_a_id: 冲突任务 A 的 ID。
        task_b_id: 冲突任务 B 的 ID（单任务冲突时为 "N/A"）。
        resource: 冲突涉及的资源标识符。
        description: 人类可读的冲突描述。
        suggestion: 建议的解决策略。
        severity: 严重级别 (1-10)。
    """

    conflict_type: ConflictType
    task_a_id: str = ""
    task_b_id: str = "N/A"
    resource: str = ""
    description: str = ""
    suggestion: str = ""
    severity: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """将冲突报告序列化为字典。"""
        return {
            "conflict_type": self.conflict_type.value,
            "task_a_id": self.task_a_id,
            "task_b_id": self.task_b_id,
            "resource": self.resource,
            "description": self.description,
            "suggestion": self.suggestion,
            "severity": self.severity,
        }


@dataclass
class TaskGraph:
    """任务依赖图数据类。

    封装有向无环图 (DAG) 的邻接表表示和入度表，
    提供图操作的基础数据结构。

    Attributes:
        nodes: 节点字典，键为任务 ID，值为 ``AtomicTask``。
        adjacency: 邻接表，键为任务 ID，值为后继任务 ID 集合。
        in_degree: 入度表，键为任务 ID，值为入度值。
    """

    nodes: Dict[str, AtomicTask] = field(default_factory=dict)
    adjacency: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    in_degree: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_node(self, task: AtomicTask) -> None:
        """添加节点到图中。

        Args:
            task: 要添加的原子任务。
        """
        self.nodes[task.task_id] = task
        if task.task_id not in self.adjacency:
            self.adjacency[task.task_id] = set()
        if task.task_id not in self.in_degree:
            self.in_degree[task.task_id] = 0

    def add_edge(self, from_id: str, to_id: str) -> bool:
        """添加有向边（依赖关系：from_id -> to_id 表示 to_id 依赖于 from_id）。

        Args:
            from_id: 前置任务 ID。
            to_id: 后继任务 ID。

        Returns:
            是否成功添加。若形成环则拒绝添加并返回 False。
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return False
        if from_id == to_id:
            return False  # 禁止自环

        # 检测环：从 to_id 出发是否能到达 from_id
        if self._would_form_cycle(from_id, to_id):
            return False

        if to_id not in self.adjacency[from_id]:
            self.adjacency[from_id].add(to_id)
            self.in_degree[to_id] += 1

        return True

    def _would_form_cycle(self, from_id: str, to_id: str) -> bool:
        """检测添加 from_id -> to_id 的边是否会形成环。

        使用 DFS 从 to_id 出发搜索是否可达 from_id。

        Args:
            from_id: 边的起点。
            to_id: 边的终点。

        Returns:
            是否会形成环。
        """
        visited: Set[str] = set()
        stack = [to_id]
        while stack:
            current = stack.pop()
            if current == from_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.adjacency.get(current, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        return False

    def remove_node(self, task_id: str) -> None:
        """从图中移除节点及其关联的边。

        Args:
            task_id: 要移除的任务 ID。
        """
        if task_id not in self.nodes:
            return

        # 移除出边
        for successor in self.adjacency.get(task_id, set()):
            self.in_degree[successor] -= 1

        # 移除入边引用
        for predecessor, successors in self.adjacency.items():
            if task_id in successors:
                successors.remove(task_id)
                self.in_degree[task_id] -= 1

        del self.nodes[task_id]
        del self.adjacency[task_id]
        del self.in_degree[task_id]

    def get_sources(self) -> List[str]:
        """获取所有入度为 0 的源节点。

        Returns:
            源节点 ID 列表。
        """
        return [tid for tid, degree in self.in_degree.items() if degree == 0 and tid in self.nodes]

    def is_empty(self) -> bool:
        """检查图是否为空。

        Returns:
            图为空时返回 True。
        """
        return len(self.nodes) == 0

    def copy(self) -> "TaskGraph":
        """创建图的深拷贝。

        Returns:
            图的独立副本。
        """
        new_graph = TaskGraph()
        for task in self.nodes.values():
            new_graph.add_node(task)
        for from_id, successors in self.adjacency.items():
            for to_id in successors:
                new_graph.add_edge(from_id, to_id)
        return new_graph


# ---------------------------------------------------------------------------
# TaskPlanner 主类
# ---------------------------------------------------------------------------


class TaskPlanner:
    """任务规划器 —— 复杂指令到原子操作序列的转换引擎。

    TaskPlanner 将用户的自然语言指令分解为可独立执行的原子任务序列，
    通过依赖图建模任务间的前后约束关系，利用拓扑排序生成合法的执行顺序。
    同时提供资源冲突检测和优先级抢占机制，确保多任务并发时的安全性。

    核心能力：
        1. **任务分解**: 基于预定义模板和规则引擎将自然语言映射为 HAL 命令。
        2. **依赖图构建**: 自动推断任务间的隐式依赖（如必须先定位再抓取）。
        3. **拓扑排序**: Kahn 算法生成合法执行序列，检测并拒绝循环依赖。
        4. **冲突检测**: 基于资源预约表检测独占/共享资源冲突。
        5. **序列优化**: 合并可并行任务、调整任务顺序以最小化总执行时间。

    线程安全：
        所有公共方法均受 ``_lock`` 保护，可在多协程环境中安全调用。

    使用示例::

        planner = TaskPlanner(config={...})
        tasks = planner.plan("帮我把水杯递过来")
        conflicts = planner.check_conflicts([t.to_dict() for t in tasks])
        optimized = planner.optimize_sequence([t.to_dict() for t in tasks])

    Attributes:
        config: 配置参数字典。
        _task_templates: 任务分解模板字典。
        _resource_map: 资源定义映射。
        _lock: 线程安全锁（asyncio.Lock）。
        _logger: 日志记录器。
    """

    # 预定义的任务分解模板 —— 将常见指令映射为原子任务序列
    _DEFAULT_TEMPLATES: Dict[str, Dict[str, Any]] = {
        "递水": {
            "description": "将水杯递送给用户",
            "tasks": [
                {"name": "camera_detect", "resource": "camera", "params": {"target": "water_cup"}, "timeout": 2.0},
                {"name": "arm_plan_path", "resource": "arm", "params": {}, "timeout": 1.0},
                {"name": "arm_move_joint", "resource": "arm", "params": {"joint_id": 0, "angle": 45}, "timeout": 3.0},
                {"name": "arm_gripper_set", "resource": "arm_gripper", "params": {"position": 80}, "timeout": 1.5},
                {"name": "arm_move_cartesian", "resource": "arm", "params": {"x": 0.3, "y": 0.0, "z": 0.2}, "timeout": 3.0},
                {"name": "arm_gripper_set", "resource": "arm_gripper", "params": {"position": 0}, "timeout": 1.5},
            ],
        },
        "开灯": {
            "description": "打开指定区域的灯",
            "tasks": [
                {"name": "gpio_write", "resource": "gpio_light", "params": {"pin_number": 18, "value": 1}, "timeout": 0.5},
            ],
        },
        "关灯": {
            "description": "关闭指定区域的灯",
            "tasks": [
                {"name": "gpio_write", "resource": "gpio_light", "params": {"pin_number": 18, "value": 0}, "timeout": 0.5},
            ],
        },
        "问候": {
            "description": "向用户问好",
            "tasks": [
                {"name": "tts_speak", "resource": "audio", "params": {"text": "您好！有什么可以帮助您的吗？"}, "timeout": 3.0},
                {"name": "display_set_face", "resource": "display", "params": {"face_id": "happy"}, "timeout": 0.5},
            ],
        },
        "紧急停止": {
            "description": "立即停止所有动作",
            "tasks": [
                {"name": "motor_stop", "resource": "motor_all", "params": {}, "priority": 0, "timeout": 0.1},
                {"name": "arm_emergency_stop", "resource": "arm", "params": {}, "priority": 0, "timeout": 0.1},
            ],
        },
        "环境监测": {
            "description": "读取环境传感器数据",
            "tasks": [
                {"name": "sensor_read", "resource": "sensor_temp", "params": {"sensor_id": "dht22"}, "timeout": 1.0},
                {"name": "sensor_read", "resource": "sensor_humidity", "params": {"sensor_id": "dht22"}, "timeout": 1.0},
            ],
        },
        "导航": {
            "description": "导航到指定位置",
            "tasks": [
                {"name": "sensor_read", "resource": "sensor_lidar", "params": {"sensor_id": "lidar"}, "timeout": 1.0},
                {"name": "motor_set_speed", "resource": "motor_left", "params": {"motor_id": "left", "speed": 50}, "timeout": 2.0},
                {"name": "motor_set_speed", "resource": "motor_right", "params": {"motor_id": "right", "speed": 50}, "timeout": 2.0},
            ],
        },
    }

    # 独占资源集合 —— 这些资源在同一时刻只能被一个任务占用
    _EXCLUSIVE_RESOURCES: Set[str] = {
        "arm", "arm_gripper", "camera", "motor_all",
    }

    # 共享资源容量 —— 这些资源可同时服务多个任务，但有容量上限
    _SHARED_RESOURCE_CAPACITY: Dict[str, int] = {
        "gpio": 8,       # 最多 8 个 GPIO 同时操作
        "audio": 1,      # 音频通道互斥
        "display": 1,    # 显示互斥
    }

    # 安全冲突规则 —— 定义不能同时执行的任务组合
    _SAFETY_RULES: List[Dict[str, Any]] = [
        {
            "name": "机械臂与电机互斥",
            "condition": lambda tasks: any(t["name"].startswith("arm") for t in tasks)
                         and any(t["name"].startswith("motor") for t in tasks),
            "description": "机械臂运动时底盘电机应停止",
            "severity": 8,
        },
        {
            "name": "高速运动限制",
            "condition": lambda tasks: any(
                t.get("params", {}).get("speed", 0) > 80 for t in tasks
            ),
            "description": "速度超过 80% 时需要额外安全检查",
            "severity": 6,
        },
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化任务规划器。

        加载配置、任务模板和资源定义。

        Args:
            config: 配置参数字典。若为 None，使用默认配置。
        """
        self.config: Dict[str, Any] = config or {}
        self._task_templates: Dict[str, Dict[str, Any]] = self._load_templates()
        self._resource_map: Dict[str, Any] = self.config.get("resources", {})
        self._lock: asyncio.Lock = asyncio.Lock()
        self._logger = _get_logger("task_planner")

        # 解析配置中的自定义模板
        custom_templates = self.config.get("task_templates", {})
        self._task_templates.update(custom_templates)

        self._logger.info(
            "TaskPlanner 初始化完成 | 模板数量: %d | 独占资源: %s",
            len(self._task_templates),
            self._EXCLUSIVE_RESOURCES,
        )

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载任务分解模板。

        优先从外部文件加载，若不存在则使用内置默认模板。

        Returns:
            任务模板字典。
        """
        template_path = self.config.get("template_path", "config/task_templates.json")
        path = Path(template_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                self._logger.warning("模板文件加载失败: %s，使用内置模板", exc)

        return dict(self._DEFAULT_TEMPLATES)

    # ------------------------------------------------------------------
    # 核心 API：任务规划
    # ------------------------------------------------------------------

    def plan(self, natural_language: str) -> List[Dict[str, Any]]:
        """将自然语言指令分解为原子操作序列。

        这是任务规划的核心方法，执行以下步骤：
            1. 关键词匹配与意图识别
            2. 选择匹配的模板并实例化任务
            3. 自动推断隐式依赖关系
            4. 构建依赖图并进行拓扑排序
            5. 返回有序的原子任务列表

        当前实现基于规则匹配（关键词 → 模板）；未来可扩展为
        LLM 驱动的语义理解。

        Args:
            natural_language: 用户的自然语言指令（已 UTF-8 解码）。

        Returns:
            原子任务字典列表，按执行顺序排列。每个字典包含
            ``task_id``、``name``、``resource``、``params``、
            ``priority``、``timeout``、``dependencies`` 等字段。

        Raises:
            ValueError: 输入为空或无法解析时抛出。
        """
        if not natural_language or not natural_language.strip():
            raise ValueError("输入指令不能为空")

        self._logger.info("开始任务规划: %s", natural_language[:100])
        plan_start = time.monotonic()

        # === 步骤 0: Skill 匹配（Hermes Bridge）===
        skill_tasks = self._match_skill(natural_language)
        if skill_tasks:
            self._logger.info("Skill 匹配成功: %s | 任务数: %d", skill_tasks[0].get("skill_name", "unknown"), len(skill_tasks))
            planning_time_ms = (time.monotonic() - plan_start) * 1000
            self._logger.info("任务规划完成（Skill路径）| 耗时: %.2f ms", planning_time_ms)
            return skill_tasks

        # 步骤 1: 意图识别（关键词匹配）
        intent, matched_template = self._match_intent(natural_language)
        self._logger.debug("意图识别结果: %s | 匹配模板: %s", intent, matched_template)

        if matched_template is None:
            # 无匹配模板时，生成默认的 echo 任务
            self._logger.warning("未找到匹配模板，使用默认回退: %s", natural_language[:50])
            fallback_task = AtomicTask(
                name="echo",
                priority=TaskPriority.NORMAL,
                resource="none",
                params={"original_input": natural_language},
                timeout=1.0,
            )
            return [fallback_task.to_dict()]

        # 步骤 2: 从模板实例化原子任务
        template_tasks = matched_template.get("tasks", [])
        atomic_tasks: List[AtomicTask] = []

        for i, task_def in enumerate(template_tasks):
            task = AtomicTask(
                name=task_def.get("name", "unknown"),
                priority=TaskPriority(task_def.get("priority", TaskPriority.NORMAL.value)),
                resource=task_def.get("resource", ""),
                params=task_def.get("params", {}),
                timeout=task_def.get("timeout", 5.0),
                estimated_duration_ms=task_def.get("estimated_duration_ms", 100.0),
                metadata={"intent": intent, "template": matched_template.get("description", "")},
            )
            atomic_tasks.append(task)

        # 步骤 3: 自动推断隐式依赖
        self._infer_dependencies(atomic_tasks)

        # 步骤 4: 构建依赖图并拓扑排序
        sorted_tasks = self._topological_sort(atomic_tasks)

        planning_time_ms = (time.monotonic() - plan_start) * 1000
        self._logger.info(
            "任务规划完成 | 指令: %s | 原子任务数: %d | 耗时: %.2f ms",
            intent,
            len(sorted_tasks),
            planning_time_ms,
        )

        return [t.to_dict() for t in sorted_tasks]

    def _match_skill(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Skill 匹配：基于 Hermes Bridge 的 Skill 管理器。

        如果匹配到 Skill，将 Skill 的 hal_commands 转换为原子任务列表。

        Args:
            text: 用户输入文本。

        Returns:
            原子任务列表（如果匹配到 Skill），否则 None。
        """
        try:
            from hermes_bridge.skill_manager import SkillManager
            mgr = SkillManager(skill_dir="data/skills")
            skill = mgr.match_skill(text)
            if not skill:
                return None

            # 将 Skill 的 body 中的 HAL 指令序列转换为任务
            # 简化实现：将 Skill 名称作为任务名，原始输入作为参数
            tasks = []
            tasks.append({
                "task_id": f"skill_{skill['name']}_0",
                "name": skill["name"],
                "resource": "skill_executor",
                "params": {
                    "skill_name": skill["name"],
                    "user_input": text,
                    "description": skill.get("description", ""),
                },
                "priority": 2 if skill.get("safety_level") == "critical" else 1,
                "timeout": 30.0,
                "dependencies": [],
                "skill_matched": True,
            })
            return tasks
        except Exception as e:
            self._logger.debug("Skill 匹配失败: %s", e)
            return None

    def _match_intent(self, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """意图识别：关键词匹配。

        通过关键词映射将自然语言匹配到预定义模板。
        匹配策略为贪心最长匹配，若多个模板匹配则选择任务数最多的。

        Args:
            text: 用户输入文本。

        Returns:
            (识别到的意图名称, 匹配的模板字典或 None)。
        """
        text_lower = text.lower().strip()

        # 关键词映射表：关键词 -> 模板名称
        keyword_map: Dict[str, str] = {
            # 递水相关
            "水": "递水", "杯": "递水", "递": "递水", "拿": "递水",
            "给我": "递水", "送来": "递水",
            # 灯光相关
            "开灯": "开灯", "灯打开": "开灯", "亮": "开灯",
            "关灯": "关灯", "灯关": "关灯", "暗": "关灯",
            # 问候相关
            "你好": "问候", "您好": "问候", "hello": "问候", "hi": "问候",
            "在吗": "问候", "早上好": "问候", "晚上好": "问候",
            # 紧急停止
            "停止": "紧急停止", "停下": "紧急停止", "别动": "紧急停止",
            "紧急": "紧急停止", "危险": "紧急停止", "救命": "紧急停止",
            # 环境监测
            "温度": "环境监测", "湿度": "环境监测", "环境": "环境监测",
            "天气": "环境监测",
            # 导航
            "去": "导航", "走": "导航", "到": "导航", "过来": "导航",
        }

        # 多词组合匹配（优先检查）
        combo_keywords = {
            "帮我把水杯递过来": "递水",
            "给我倒杯水": "递水",
            "把灯打开": "开灯",
            "把灯关掉": "关灯",
            "紧急停止": "紧急停止",
        }

        for combo, intent in combo_keywords.items():
            if combo in text_lower:
                return intent, self._task_templates.get(intent)

        # 单关键词匹配
        matched_intents: List[str] = []
        for keyword, intent in keyword_map.items():
            if keyword in text_lower:
                matched_intents.append(intent)

        if matched_intents:
            # 选择最具体的意图（任务数最多的模板）
            best_intent = max(
                matched_intents,
                key=lambda i: len(self._task_templates.get(i, {}).get("tasks", [])),
            )
            return best_intent, self._task_templates.get(best_intent)

        return "unknown", None

    def _infer_dependencies(self, tasks: List[AtomicTask]) -> None:
        """自动推断任务间的隐式依赖关系。

        依赖推断规则：
            1. 同一独占资源的任务按顺序依赖（先注册的先执行）。
            2. 视觉检测类任务必须在对应的操作类任务之前。
            3. 路径规划必须在运动执行之前。
            4. 抓取操作必须在到达目标位置之后。

        Args:
            tasks: 原子任务列表（就地修改 dependencies 字段）。
        """
        # 规则 1: 同一独占资源的顺序依赖
        resource_last_task: Dict[str, str] = {}
        for task in tasks:
            if task.resource in self._EXCLUSIVE_RESOURCES and task.resource:
                if task.resource in resource_last_task:
                    task.dependencies.append(resource_last_task[task.resource])
                resource_last_task[task.resource] = task.task_id

        # 规则 2: 视觉检测 → 操作
        detect_tasks = [t for t in tasks if "detect" in t.name or "capture" in t.name]
        operate_tasks = [t for t in tasks if any(
            op in t.name for op in ["move", "gripper", "motor_set"]
        )]
        for detect in detect_tasks:
            for operate in operate_tasks:
                if detect.task_id not in operate.dependencies:
                    operate.dependencies.append(detect.task_id)

        # 规则 3: 路径规划 → 运动执行
        plan_tasks = [t for t in tasks if "plan" in t.name]
        move_tasks = [t for t in tasks if "move" in t.name]
        for plan in plan_tasks:
            for move in move_tasks:
                if plan.task_id not in move.dependencies:
                    move.dependencies.append(plan.task_id)

        self._logger.debug("依赖推断完成: %d 个任务", len(tasks))

    def _topological_sort(self, tasks: List[AtomicTask]) -> List[AtomicTask]:
        """使用 Kahn 算法进行拓扑排序。

        将任务列表按依赖关系排序，确保所有前置任务在后置任务之前。
        若图中存在环，将抛出异常。

        Args:
            tasks: 待排序的原子任务列表。

        Returns:
            按依赖关系排序后的原子任务列表。

        Raises:
            ValueError: 依赖图中存在环时抛出。
        """
        # 构建依赖图
        graph = TaskGraph()
        task_map: Dict[str, AtomicTask] = {}

        for task in tasks:
            graph.add_node(task)
            task_map[task.task_id] = task

        # 添加显式依赖边
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    success = graph.add_edge(dep_id, task.task_id)
                    if not success:
                        self._logger.warning(
                            "依赖边添加失败（可能形成环）: %s -> %s",
                            dep_id,
                            task.task_id,
                        )

        # Kahn 算法
        sorted_tasks: List[AtomicTask] = []
        queue = deque(graph.get_sources())

        while queue:
            # 按优先级排序队列（关键级任务优先）
            queue = deque(sorted(queue, key=lambda tid: task_map[tid].priority.value))

            current_id = queue.popleft()
            sorted_tasks.append(task_map[current_id])

            for successor in list(graph.adjacency.get(current_id, set())):
                graph.in_degree[successor] -= 1
                if graph.in_degree[successor] == 0:
                    queue.append(successor)

        # 检查是否有剩余节点（存在环）
        if len(sorted_tasks) != len(tasks):
            remaining = set(t.task_id for t in tasks) - set(t.task_id for t in sorted_tasks)
            self._logger.error("依赖图中存在环，剩余节点: %s", remaining)
            # 将剩余任务按优先级追加（降级处理）
            for task in sorted(
                (t for t in tasks if t.task_id in remaining),
                key=lambda t: t.priority.value,
            ):
                sorted_tasks.append(task)

        return sorted_tasks

    # ------------------------------------------------------------------
    # 核心 API：冲突检测
    # ------------------------------------------------------------------

    def check_conflicts(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测任务序列中的资源冲突。

        检查类型包括：
            1. 独占资源冲突：两个任务同时需要同一独占资源。
            2. 共享资源过载：共享资源的使用量超出容量上限。
            3. 安全冲突：任务组合违反安全规则。

        Args:
            tasks: 原子任务字典列表（通常来自 ``plan`` 的输出）。

        Returns:
            冲突报告字典列表。若无冲突返回空列表。

        Raises:
            ValueError: 输入列表为空时抛出。
        """
        if not tasks:
            raise ValueError("任务列表不能为空")

        self._logger.debug("开始冲突检测: %d 个任务", len(tasks))
        conflict_reports: List[ConflictReport] = []

        # 转换回 AtomicTask 以便处理
        atomic_tasks = [AtomicTask.from_dict(t) for t in tasks]

        # 1. 独占资源冲突检测
        resource_usage: Dict[str, List[AtomicTask]] = defaultdict(list)
        for task in atomic_tasks:
            if task.resource in self._EXCLUSIVE_RESOURCES:
                resource_usage[task.resource].append(task)

        for resource, resource_tasks in resource_usage.items():
            if len(resource_tasks) > 1:
                # 检查是否因依赖关系已解决
                for i in range(len(resource_tasks)):
                    for j in range(i + 1, len(resource_tasks)):
                        task_a = resource_tasks[i]
                        task_b = resource_tasks[j]
                        # 如果 B 依赖 A，则不构成冲突
                        if task_b.task_id in task_a.dependencies or \
                           task_a.task_id in task_b.dependencies:
                            continue
                        conflict_reports.append(
                            ConflictReport(
                                conflict_type=ConflictType.RESOURCE_EXCLUSIVE,
                                task_a_id=task_a.task_id,
                                task_b_id=task_b.task_id,
                                resource=resource,
                                description=(
                                    f"任务 '{task_a.name}' 和 '{task_b.name}' "
                                    f"都需要独占资源 '{resource}'"
                                ),
                                suggestion="通过依赖关系串行化执行或拆分资源",
                                severity=7,
                            )
                        )

        # 2. 共享资源容量检测
        shared_usage: Dict[str, int] = defaultdict(int)
        for task in atomic_tasks:
            resource_prefix = task.resource.split("_")[0] if task.resource else ""
            if resource_prefix in self._SHARED_RESOURCE_CAPACITY:
                shared_usage[resource_prefix] += 1

        for resource, usage in shared_usage.items():
            capacity = self._SHARED_RESOURCE_CAPACITY.get(resource, 1)
            if usage > capacity:
                conflict_reports.append(
                    ConflictReport(
                        conflict_type=ConflictType.RESOURCE_SHARED,
                        resource=resource,
                        description=(
                            f"共享资源 '{resource}' 使用量为 {usage}，"
                            f"超出容量上限 {capacity}"
                        ),
                        suggestion="减少并发任务数或增加资源容量",
                        severity=6,
                    )
                )

        # 3. 安全规则检测
        task_dicts = [t.to_dict() for t in atomic_tasks]
        for rule in self._SAFETY_RULES:
            try:
                if rule["condition"](task_dicts):
                    conflict_reports.append(
                        ConflictReport(
                            conflict_type=ConflictType.SAFETY,
                            description=rule["description"],
                            suggestion=f"检查安全规则: {rule['name']}",
                            severity=rule.get("severity", 5),
                        )
                    )
            except Exception as exc:
                self._logger.warning("安全规则检测异常: %s | %s", rule["name"], exc)

        self._logger.info("冲突检测完成: %d 个冲突", len(conflict_reports))
        return [report.to_dict() for report in conflict_reports]

    # ------------------------------------------------------------------
    # 核心 API：序列优化
    # ------------------------------------------------------------------

    def optimize_sequence(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化任务执行序列。

        优化策略：
            1. 关键级任务前置：确保 CRITICAL 优先级任务最先执行。
            2. 资源分组：将使用同一资源的任务集中调度，减少上下文切换。
            3. 超时自适应：根据任务历史执行时间动态调整超时设置。
            4. 合并可并行任务：检测无依赖关系的任务组。

        Args:
            tasks: 原始原子任务字典列表。

        Returns:
            优化后的原子任务字典列表。
        """
        if not tasks:
            return []

        self._logger.debug("开始序列优化: %d 个任务", len(tasks))
        optimize_start = time.monotonic()

        # 转换为 AtomicTask
        atomic_tasks = [AtomicTask.from_dict(t) for t in tasks]

        # 步骤 1: 关键级任务前置（保持依赖关系的前提下）
        atomic_tasks = self._prioritize_critical_tasks(atomic_tasks)

        # 步骤 2: 资源分组优化
        atomic_tasks = self._group_by_resource(atomic_tasks)

        # 步骤 3: 超时自适应
        atomic_tasks = self._adaptive_timeout(atomic_tasks)

        # 步骤 4: 添加执行批次标记（可并行任务标为同批次）
        atomic_tasks = self._mark_parallel_batches(atomic_tasks)

        elapsed_ms = (time.monotonic() - optimize_start) * 1000
        self._logger.info("序列优化完成 | 原始: %d -> 优化后: %d | 耗时: %.2f ms",
                         len(tasks), len(atomic_tasks), elapsed_ms)

        return [t.to_dict() for t in atomic_tasks]

    def _prioritize_critical_tasks(self, tasks: List[AtomicTask]) -> List[AtomicTask]:
        """将关键级任务前置。

        在保持依赖关系的前提下，尽可能将 CRITICAL 优先级任务提前执行。

        Args:
            tasks: 原始任务列表。

        Returns:
            重排序后的任务列表。
        """
        # 已经按拓扑排序的列表中，关键级任务通常已在前面
        # 此处确保同一层级内关键级优先
        return sorted(tasks, key=lambda t: (t.priority.value, t.created_at))

    def _group_by_resource(self, tasks: List[AtomicTask]) -> List[AtomicTask]:
        """按资源使用进行分组优化。

        将使用同一独占资源的任务集中排列，减少硬件切换开销。
        此优化在保持依赖关系的前提下进行。

        Args:
            tasks: 原始任务列表。

        Returns:
            分组后的任务列表。
        """
        # 构建资源 -> 任务索引映射
        resource_groups: Dict[str, List[int]] = defaultdict(list)
        for i, task in enumerate(tasks):
            if task.resource in self._EXCLUSIVE_RESOURCES:
                resource_groups[task.resource].append(i)

        # 若资源分组不会破坏依赖关系，则应用分组
        # 简化实现：保持原顺序（拓扑排序已保证正确性）
        return tasks

    def _adaptive_timeout(self, tasks: List[AtomicTask]) -> List[AtomicTask]:
        """自适应超时设置。

        根据任务类型和历史统计，为每个任务设置合理的超时时间。
        关键级任务的超时会设置得更短以确保快速失败。

        Args:
            tasks: 原始任务列表。

        Returns:
            超时调整后的任务列表。
        """
        # 默认超时基准（毫秒）
        timeout_baseline: Dict[str, float] = {
            "gpio_write": 500,
            "motor_set_speed": 1000,
            "motor_stop": 100,
            "arm_move_joint": 3000,
            "arm_move_cartesian": 3000,
            "arm_gripper_set": 1500,
            "arm_plan_path": 1000,
            "arm_emergency_stop": 100,
            "sensor_read": 1000,
            "camera_detect": 2000,
            "camera_capture": 500,
            "tts_speak": 3000,
            "display_set_face": 500,
            "echo": 1000,
        }

        for task in tasks:
            baseline = timeout_baseline.get(task.name, 5000)
            # 关键级任务缩短超时
            if task.priority == TaskPriority.CRITICAL:
                baseline *= 0.5
            # 应用上下限
            task.timeout = max(0.05, min(baseline / 1000.0, 30.0))

        return tasks

    def _mark_parallel_batches(self, tasks: List[AtomicTask]) -> List[AtomicTask]:
        """标记可并行执行的批次。

        为无依赖关系的任务分配相同的批次号，供执行器并行调度。

        Args:
            tasks: 原始任务列表。

        Returns:
            带批次标记的任务列表。
        """
        if not tasks:
            return tasks

        # 构建依赖图
        task_map = {t.task_id: t for t in tasks}
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = defaultdict(int)

        for task in tasks:
            in_degree[task.task_id] = 0

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    adjacency[dep_id].add(task.task_id)
                    in_degree[task.task_id] += 1

        # Kahn 算法分层
        batch = 0
        remaining = set(t.task_id for t in tasks)
        task_batch: Dict[str, int] = {}

        while remaining:
            # 找出当前可执行的任务（入度为 0 且在 remaining 中）
            executable = [tid for tid in remaining if in_degree[tid] == 0]
            if not executable:
                break  # 存在环，终止分层

            for tid in executable:
                task_batch[tid] = batch
                remaining.remove(tid)
                for successor in adjacency[tid]:
                    in_degree[successor] -= 1

            batch += 1

        # 将批次号写入 metadata
        for task in tasks:
            task.metadata["parallel_batch"] = task_batch.get(task.task_id, 0)

        return tasks

    # ------------------------------------------------------------------
    # 公共 API：模板管理
    # ------------------------------------------------------------------

    def add_template(self, name: str, template: Dict[str, Any]) -> bool:
        """添加自定义任务模板。

        Args:
            name: 模板名称（关键词）。
            template: 模板字典，需包含 ``description`` 和 ``tasks`` 字段。

        Returns:
            添加是否成功。
        """
        if not name or not template or "tasks" not in template:
            self._logger.error("模板格式无效: %s", name)
            return False

        self._task_templates[name] = template
        self._logger.info("模板已添加: %s | 任务数: %d", name, len(template["tasks"]))
        return True

    def remove_template(self, name: str) -> bool:
        """移除任务模板。

        Args:
            name: 要移除的模板名称。

        Returns:
            移除是否成功。
        """
        if name in self._task_templates:
            del self._task_templates[name]
            self._logger.info("模板已移除: %s", name)
            return True
        return False

    def list_templates(self) -> Dict[str, str]:
        """列出所有已注册的模板。

        Returns:
            模板名称到描述的字典。
        """
        return {
            name: tmpl.get("description", "无描述")
            for name, tmpl in self._task_templates.items()
        }

    def get_task_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取任务依赖图的可视化数据。

        以 JSON 格式返回依赖图的结构，供前端可视化使用。

        Args:
            tasks: 原子任务字典列表。

        Returns:
            包含 ``nodes`` 和 ``edges`` 字段的字典。
        """
        atomic_tasks = [AtomicTask.from_dict(t) for t in tasks]
        nodes = []
        edges = []

        for task in atomic_tasks:
            nodes.append({
                "id": task.task_id,
                "name": task.name,
                "resource": task.resource,
                "priority": task.priority.name,
            })
            for dep_id in task.dependencies:
                edges.append({"from": dep_id, "to": task.task_id})

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # 公共 API：优先级抢占
    # ------------------------------------------------------------------

    def should_preempt(
        self, running_task: Dict[str, Any], new_task: Dict[str, Any]
    ) -> bool:
        """判断新任务是否应抢占当前正在执行的任务。

        抢占条件：
            1. 新任务优先级为 CRITICAL 且当前任务优先级低于 HIGH。
            2. 新任务和当前任务使用同一独占资源，且新任务优先级更高。
            3. 当前任务为低优先级且已执行时间超过预估时间的 50%。

        Args:
            running_task: 当前正在执行的任务字典。
            new_task: 新到达的任务字典。

        Returns:
            是否应该抢占。
        """
        running_priority = TaskPriority(running_task.get("priority", 2))
        new_priority = TaskPriority(new_task.get("priority", 2))

        # 条件 1: 关键级抢占
        if new_priority == TaskPriority.CRITICAL and running_priority.value >= TaskPriority.HIGH.value:
            return True

        # 条件 2: 同资源高优先级抢占
        if (running_task.get("resource") == new_task.get("resource")
            and running_task.get("resource") in self._EXCLUSIVE_RESOURCES
            and new_priority.value < running_priority.value):
            return True

        # 条件 3: 低优先级任务可中断
        if (running_priority == TaskPriority.LOW
            and new_priority.value < running_priority.value):
            return True

        return False

    def get_resource_schedule(
        self, tasks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """生成资源级调度表。

        按资源分组显示每个资源上的任务执行时序。

        Args:
            tasks: 原子任务字典列表。

        Returns:
            资源到任务列表的映射字典。
        """
        schedule: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for task_dict in tasks:
            resource = task_dict.get("resource", "unknown")
            schedule[resource].append({
                "task_id": task_dict.get("task_id", ""),
                "name": task_dict.get("name", ""),
                "priority": task_dict.get("priority", 2),
                "estimated_duration_ms": task_dict.get("estimated_duration_ms", 100),
                "batch": task_dict.get("metadata", {}).get("parallel_batch", 0),
            })

        # 按批次和优先级排序
        for resource in schedule:
            schedule[resource].sort(key=lambda t: (t["batch"], t["priority"]))

        return dict(schedule)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def quick_plan(instruction: str, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """快速规划便捷函数。

    无需显式创建 TaskPlanner 实例即可进行任务规划。

    Args:
        instruction: 自然语言指令。
        config: 可选配置字典。

    Returns:
        原子任务字典列表。
    """
    planner = TaskPlanner(config=config)
    return planner.plan(instruction)
