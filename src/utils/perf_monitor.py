"""
性能监控模块

提供CPU/内存/NPU使用率采集、延迟统计和性能告警功能。
支持Prometheus格式的数据导出。
适用于OrangePi Kunpeng Pro (RK3588)平台的KunPeng-Cortex项目。

功能特性:
    - CPU使用率采集(多核)
    - 内存使用率采集
    - NPU使用率采集(RK3588)
    - 延迟统计(P50/P95/P99)
    - 性能告警(阈值触发)
    - Prometheus格式导出
    - 历史数据保留
    - 异步监控循环

监控指标:
    kpcortex_cpu_percent        - CPU使用率
    kpcortex_memory_percent     - 内存使用率
    kpcortex_npu_percent        - NPU使用率
    kpcortex_task_latency_ms    - 任务延迟(分位数)
    kpcortex_frame_rate         - 帧率
    kpcortex_uptime_seconds     - 运行时间

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重级别枚举"""
    INFO = "info"           # 信息
    WARNING = "warning"     # 警告
    CRITICAL = "critical"   # 严重


class MetricType(Enum):
    """指标类型枚举"""
    GAUGE = "gauge"         # 瞬时值
    COUNTER = "counter"     # 累计值
    HISTOGRAM = "histogram" # 分布值
    SUMMARY = "summary"     # 汇总值


@dataclass
class MetricValue:
    """指标值数据结构

    属性:
        name: 指标名称
        value: 指标值
        metric_type: 指标类型
        labels: 标签字典
        timestamp: 时间戳
    """
    name: str = ""
    value: float = 0.0
    metric_type: MetricType = MetricType.GAUGE
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class LatencyStats:
    """延迟统计数据结构

    属性:
        count: 样本数量
        min_ms: 最小延迟(ms)
        max_ms: 最大延迟(ms)
        mean_ms: 平均延迟(ms)
        p50_ms: P50延迟(ms)
        p95_ms: P95延迟(ms)
        p99_ms: P99延迟(ms)
        stddev_ms: 标准差(ms)
    """
    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    stddev_ms: float = 0.0


@dataclass
class AlertRule:
    """告警规则

    属性:
        name: 规则名称
        metric: 监控指标名
        condition: 条件表达式
        threshold: 阈值
        severity: 严重级别
        cooldown: 冷却时间(秒)
        enabled: 是否启用
    """
    name: str = ""
    metric: str = ""
    condition: str = ">"        # >, <, >=, <=, ==
    threshold: float = 0.0
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown: float = 300.0     # 5分钟冷却
    enabled: bool = True
    _last_alert_time: float = 0.0


@dataclass
class PerformanceAlert:
    """性能告警

    属性:
        rule_name: 触发规则名
        severity: 严重级别
        message: 告警消息
        metric_value: 触发时的指标值
        threshold: 阈值
        timestamp: 告警时间
    """
    rule_name: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    timestamp: float = 0.0


@dataclass
class PerfMonitorConfig:
    """性能监控配置

    属性:
        collection_interval: 数据采集间隔(秒)
        latency_window_size: 延迟统计窗口大小
        history_size: 历史数据保留数量
        enable_prometheus: 是否启用Prometheus导出
        prometheus_port: Prometheus HTTP端口
        enable_alerts: 是否启用告警
    """
    collection_interval: float = 5.0
    latency_window_size: int = 1000
    history_size: int = 360       # 30分钟 @ 5秒间隔
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_alerts: bool = True


class PerfMonitor:
    """性能监控类

    提供系统性能指标采集、统计和告警功能。
    支持Prometheus格式导出。

    示例:
        >>> pm = PerfMonitor(PerfMonitorConfig())
        >>> await pm.initialize()
        >>> 
        >>> # 记录延迟
        >>> pm.record_latency("inference", 45.2)
        >>> pm.record_latency("control", 2.1)
        >>> 
        >>> # 获取统计
        >>> stats = pm.get_latency_stats("inference")
        >>> print(f"P95延迟: {stats.p95_ms:.1f}ms")
        >>> 
        >>> # 获取Prometheus格式
        >>> metrics = pm.to_prometheus()
        >>> 
        >>> await pm.shutdown()

    属性:
        config: 监控配置
        _metrics: 当前指标值
        _latency_windows: 延迟窗口字典
        _alert_rules: 告警规则列表
    """

    def __init__(self, config: PerfMonitorConfig | None = None) -> None:
        """初始化性能监控

        参数:
            config: 监控配置,None则使用默认配置
        """
        self.config: PerfMonitorConfig = config or PerfMonitorConfig()

        # 状态
        self._initialized: bool = False
        self._monitor_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._start_time: float = time.time()

        # 指标存储
        self._metrics: dict[str, MetricValue] = {}
        self._metrics_history: dict[str, deque[float]] = {}
        self._latency_windows: dict[str, deque[float]] = {}
        self._counters: dict[str, int] = {}

        # 告警
        self._alert_rules: list[AlertRule] = []
        self._alert_callbacks: list[Callable[[PerformanceAlert], None]] = []
        self._setup_default_rules()

        # 回调
        self._metrics_callbacks: list[Callable[[dict[str, MetricValue]], None]] = []

    def _setup_default_rules(self) -> None:
        """设置默认告警规则(内部方法)"""
        self._alert_rules = [
            AlertRule(
                name="cpu_high",
                metric="cpu_percent",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="cpu_critical",
                metric="cpu_percent",
                condition=">",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
            ),
            AlertRule(
                name="memory_high",
                metric="memory_percent",
                condition=">",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="memory_critical",
                metric="memory_percent",
                condition=">",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
            ),
            AlertRule(
                name="npu_high",
                metric="npu_percent",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="latency_p95_high",
                metric="latency_p95_ms",
                condition=">",
                threshold=100.0,
                severity=AlertSeverity.WARNING,
            ),
        ]

    async def initialize(self) -> bool:
        """初始化性能监控

        启动指标采集循环。

        返回:
            bool: 初始化成功返回True
        """
        if self._initialized:
            return True

        try:
            self._start_time = time.time()

            # 启动采集任务
            self._monitor_task = asyncio.create_task(
                self._collection_loop(), name="perf_collection"
            )

            self._initialized = True
            logger.info("性能监控初始化成功")
            return True

        except Exception as e:
            logger.error(f"性能监控初始化失败: {e}")
            return False

    async def _collection_loop(self) -> None:
        """指标采集循环(内部方法)

        定期采集系统性能指标。
        """
        logger.debug("性能采集循环已启动")

        while not self._stop_event.is_set():
            try:
                start_time = time.monotonic()

                # 采集指标
                await self._collect_metrics()

                # 检查告警
                if self.config.enable_alerts:
                    await self._check_alerts()

                # 通知回调
                metrics_copy = dict(self._metrics)
                for cb in self._metrics_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            asyncio.create_task(cb(metrics_copy))
                        else:
                            cb(metrics_copy)
                    except Exception as e:
                        logger.error(f"指标回调异常: {e}")

                # 间隔控制
                elapsed = time.monotonic() - start_time
                sleep_time = max(0, self.config.collection_interval - elapsed)

                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=sleep_time
                )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"性能采集异常: {e}")
                await asyncio.sleep(self.config.collection_interval)

        logger.debug("性能采集循环已退出")

    async def _collect_metrics(self) -> None:
        """采集系统指标(内部方法)

        采集CPU、内存、NPU等系统指标。
        """
        try:
            # CPU使用率
            cpu_percent = await self._get_cpu_percent()
            self._update_metric("cpu_percent", cpu_percent)

            # 内存使用率
            mem_percent = await self._get_memory_percent()
            self._update_metric("memory_percent", mem_percent)

            # NPU使用率(RK3588)
            npu_percent = await self._get_npu_percent()
            self._update_metric("npu_percent", npu_percent)

            # 运行时间
            uptime = time.time() - self._start_time
            self._update_metric("uptime_seconds", uptime)

            # 计算各延迟指标的统计值
            for name, window in self._latency_windows.items():
                if window:
                    stats = self._calc_latency_stats(list(window))
                    self._update_metric(f"latency_{name}_p50_ms", stats.p50_ms)
                    self._update_metric(f"latency_{name}_p95_ms", stats.p95_ms)
                    self._update_metric(f"latency_{name}_p99_ms", stats.p99_ms)

        except Exception as e:
            logger.error(f"指标采集异常: {e}")

    async def _get_cpu_percent(self) -> float:
        """获取CPU使用率(内部方法)

        返回:
            float: CPU使用百分比
        """
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
                fields = list(map(int, line.split()[1:]))

                idle = fields[3]
                total = sum(fields)

                # 计算使用率(需要前后两次采样)
                if hasattr(self, "_last_cpu"):
                    last_total, last_idle = self._last_cpu
                    total_diff = total - last_total
                    idle_diff = idle - last_idle

                    if total_diff > 0:
                        usage = (1.0 - idle_diff / total_diff) * 100.0
                        self._last_cpu = (total, idle)
                        return max(0.0, min(100.0, usage))

                self._last_cpu = (total, idle)
                return 0.0

        except Exception:
            # 回退到psutil
            try:
                import psutil
                return psutil.cpu_percent(interval=None)
            except ImportError:
                return 0.0

    async def _get_memory_percent(self) -> float:
        """获取内存使用率(内部方法)

        返回:
            float: 内存使用百分比
        """
        try:
            with open("/proc/meminfo", "r") as f:
                mem_total = 0
                mem_available = 0

                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_available = int(line.split()[1])

                if mem_total > 0:
                    used = mem_total - mem_available
                    return (used / mem_total) * 100.0

                return 0.0

        except Exception:
            try:
                import psutil
                return psutil.virtual_memory().percent
            except ImportError:
                return 0.0

    async def _get_npu_percent(self) -> float:
        """获取NPU使用率(内部方法)

        读取RK3588 NPU使用率。

        返回:
            float: NPU使用百分比
        """
        try:
            # 尝试读取RK3588 NPU状态
            npu_path = Path("/sys/class/misc/rockchip_npu/status")
            if npu_path.exists():
                content = npu_path.read_text()
                # 解析使用率
                for line in content.split("\n"):
                    if "usage" in line.lower():
                        value = line.split(":")[-1].strip()
                        return float(value)

            # 尝试通过NPU驱动读取
            npu_util_path = Path("/sys/kernel/debug/rknpu/load")
            if npu_util_path.exists():
                content = npu_util_path.read_text()
                # 格式: "NPU load: 45%%"
                for line in content.split("\n"):
                    if "load" in line.lower():
                        parts = line.split(":")
                        if len(parts) >= 2:
                            value = parts[1].replace("%%", "").strip()
                            return float(value)

            return 0.0

        except Exception:
            return 0.0

    def _update_metric(self, name: str, value: float) -> None:
        """更新指标值(内部方法)

        参数:
            name: 指标名称
            value: 指标值
        """
        now = time.time()

        self._metrics[name] = MetricValue(
            name=name,
            value=value,
            timestamp=now,
        )

        # 记录历史
        if name not in self._metrics_history:
            self._metrics_history[name] = deque(maxlen=self.config.history_size)
        self._metrics_history[name].append(value)

    def record_latency(self, name: str, latency_ms: float) -> None:
        """记录延迟样本

        将延迟样本加入滑动窗口,用于统计计算。

        参数:
            name: 延迟类别名称,如"inference"、"control"
            latency_ms: 延迟值(毫秒)

        示例:
            >>> start = time.time()
            >>> result = await do_something()
            >>> pm.record_latency("inference", (time.time() - start) * 1000)
        """
        if name not in self._latency_windows:
            self._latency_windows[name] = deque(
                maxlen=self.config.latency_window_size
            )

        self._latency_windows[name].append(latency_ms)

    def increment_counter(self, name: str, value: int = 1) -> None:
        """递增计数器

        用于记录事件次数,如帧数、请求数等。

        参数:
            name: 计数器名称
            value: 增量,默认1
        """
        if name not in self._counters:
            self._counters[name] = 0
        self._counters[name] += value

    def _calc_latency_stats(self, samples: list[float]) -> LatencyStats:
        """计算延迟统计(内部方法)

        参数:
            samples: 延迟样本列表

        返回:
            LatencyStats: 延迟统计结果
        """
        if not samples:
            return LatencyStats()

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        def percentile(p: float) -> float:
            idx = int(n * p / 100.0)
            return sorted_samples[min(idx, n - 1)]

        return LatencyStats(
            count=n,
            min_ms=min(samples),
            max_ms=max(samples),
            mean_ms=statistics.mean(samples),
            p50_ms=percentile(50),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            stddev_ms=statistics.stdev(samples) if n > 1 else 0.0,
        )

    def get_latency_stats(self, name: str) -> LatencyStats:
        """获取指定类别的延迟统计

        参数:
            name: 延迟类别名称

        返回:
            LatencyStats: 延迟统计结果
        """
        window = self._latency_windows.get(name, deque())
        return self._calc_latency_stats(list(window))

    def get_metric(self, name: str) -> float:
        """获取指定指标的当前值

        参数:
            name: 指标名称

        返回:
            float: 指标值,不存在返回0
        """
        metric = self._metrics.get(name)
        return metric.value if metric else 0.0

    def get_all_metrics(self) -> dict[str, MetricValue]:
        """获取所有当前指标

        返回:
            dict: 指标名称 -> MetricValue
        """
        return dict(self._metrics)

    def get_metric_history(self, name: str) -> list[float]:
        """获取指标历史数据

        参数:
            name: 指标名称

        返回:
            list: 历史值列表
        """
        return list(self._metrics_history.get(name, []))

    async def _check_alerts(self) -> None:
        """检查告警规则(内部方法)

        遍历所有规则,检查是否触发告警。
        """
        now = time.time()

        for rule in self._alert_rules:
            if not rule.enabled:
                continue

            # 冷却检查
            if now - rule._last_alert_time < rule.cooldown:
                continue

            # 获取指标值
            metric_value = self.get_metric(rule.metric)

            # 检查条件
            triggered = False
            if rule.condition == ">" and metric_value > rule.threshold:
                triggered = True
            elif rule.condition == ">=" and metric_value >= rule.threshold:
                triggered = True
            elif rule.condition == "<" and metric_value < rule.threshold:
                triggered = True
            elif rule.condition == "<=" and metric_value <= rule.threshold:
                triggered = True
            elif rule.condition == "==" and metric_value == rule.threshold:
                triggered = True

            if triggered:
                rule._last_alert_time = now

                alert = PerformanceAlert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=(
                        f"{rule.metric}={metric_value:.1f} "
                        f"{rule.condition} 阈值 {rule.threshold}"
                    ),
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    timestamp=now,
                )

                logger.warning(f"性能告警: {alert.message}")

                # 通知回调
                for cb in self._alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            asyncio.create_task(cb(alert))
                        else:
                            cb(alert)
                    except Exception as e:
                        logger.error(f"告警回调异常: {e}")

    def add_alert_rule(self, rule: AlertRule) -> None:
        """添加告警规则

        参数:
            rule: 告警规则
        """
        self._alert_rules.append(rule)

    def remove_alert_rule(self, name: str) -> bool:
        """移除告警规则

        参数:
            name: 规则名称

        返回:
            bool: 移除成功返回True
        """
        for i, rule in enumerate(self._alert_rules):
            if rule.name == name:
                self._alert_rules.pop(i)
                return True
        return False

    def register_alert_callback(
        self, callback: Callable[[PerformanceAlert], None]
    ) -> None:
        """注册告警回调

        参数:
            callback: 回调函数,接收PerformanceAlert
        """
        if callback not in self._alert_callbacks:
            self._alert_callbacks.append(callback)

    def register_metrics_callback(
        self, callback: Callable[[dict[str, MetricValue]], None]
    ) -> None:
        """注册指标更新回调

        每次采集完指标后调用。

        参数:
            callback: 回调函数,接收指标字典
        """
        if callback not in self._metrics_callbacks:
            self._metrics_callbacks.append(callback)

    def to_prometheus(self) -> str:
        """导出为Prometheus格式

        将当前指标导出为Prometheus文本格式,
        可用于Prometheus采集或调试查看。

        返回:
            str: Prometheus格式文本

        示例:
            >>> metrics = pm.to_prometheus()
            >>> print(metrics)
            # HELP kpcortex_cpu_percent CPU usage percent
            # TYPE kpcortex_cpu_percent gauge
            kpcortex_cpu_percent 45.2
            ...
        """
        lines: list[str] = []

        for name, metric in self._metrics.items():
            prom_name = f"kpcortex_{name}"
            metric_type = metric.metric_type.value

            lines.append(f"# HELP {prom_name} {name}")
            lines.append(f"# TYPE {prom_name} {metric_type}")

            # 标签
            labels = ", ".join(
                f'\"{k}\"=\"{v}\"'
                for k, v in metric.labels.items()
            )

            if labels:
                lines.append(f"{prom_name}{{{labels}}} {metric.value}")
            else:
                lines.append(f"{prom_name} {metric.value}")

        # 添加计数器
        for name, value in self._counters.items():
            prom_name = f"kpcortex_{name}_total"
            lines.append(f"# HELP {prom_name} Total {name}")
            lines.append(f"# TYPE {prom_name} counter")
            lines.append(f"{prom_name} {value}")

        return "\n".join(lines) + "\n"

    async def shutdown(self) -> None:
        """关闭性能监控"""
        self._stop_event.set()

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("性能监控已关闭")

    def __repr__(self) -> str:
        return (
            f"PerfMonitor(metrics={len(self._metrics)}, "
            f"alerts={len(self._alert_rules)}, "
            f"running={self._initialized})"
        )

    async def __aenter__(self) -> PerfMonitor:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
