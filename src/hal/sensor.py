"""
RK3588 传感器统一接口模块

本模块提供对所有传感器的统一抽象接口，包括：
    - 传感器基类（SensorBase）
    - 统一传感器管理器（UnifiedSensorManager）
    - 轮询和中断两种工作模式
    - 数据校准和滤波

支持的传感器类型：
    - 超声波距离传感器（HC-SR04）
    - 陀螺仪/加速度计（JY901/MPU6050）
    - 温度湿度传感器（DHT11/DHT22）
    - 红外传感器
    - 力/扭矩传感器

特性：
    - 抽象基类定义统一接口
    - 传感器自动注册和管理
    - 轮询和中断两种读取模式
    - 数据校准（零点/标度校准）
    - 数字滤波（滑动平均/中值/卡尔曼）
    - 超时保护（默认5秒）
    - 线程安全
    - 批量读取和事件回调

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

import os
import time
import logging
import threading
import statistics
from abc import ABC, abstractmethod
from typing import (
    Optional, Dict, List, Callable, Any, Tuple, Union, Set
)
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


class SensorMode(Enum):
    """传感器工作模式枚举"""
    POLLING = "polling"       # 轮询模式
    INTERRUPT = "interrupt"   # 中断模式
    HYBRID = "hybrid"         # 混合模式


class FilterType(Enum):
    """滤波器类型枚举"""
    NONE = "none"              # 无滤波
    MOVING_AVERAGE = "avg"     # 滑动平均
    MEDIAN = "median"          # 中值滤波
    KALMAN = "kalman"          # 卡尔曼滤波
    EXPONENTIAL = "exp"        # 指数平滑


@dataclass
class SensorData:
    """传感器数据容器

    Attributes:
        name: 传感器名称
        values: 传感器值字典
        timestamp: 时间戳（秒）
        unit: 单位
        raw: 原始值字典
        valid: 数据是否有效
    """
    name: str
    values: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)
    unit: str = ""
    raw: Dict[str, float] = field(default_factory=dict)
    valid: bool = True

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "name": self.name,
            "values": self.values,
            "timestamp": self.timestamp,
            "unit": self.unit,
            "raw": self.raw,
            "valid": self.valid,
        }


@dataclass
class CalibrationParams:
    """传感器校准参数

    Attributes:
        zero_offset: 零点偏移
        scale_factor: 标度因子
        linear_coef_a: 线性系数a（y = ax + b）
        linear_coef_b: 线性系数b
        calibration_date: 校准日期
    """
    zero_offset: float = 0.0
    scale_factor: float = 1.0
    linear_coef_a: float = 1.0
    linear_coef_b: float = 0.0
    calibration_date: str = ""


class KalmanFilter:
    """一维卡尔曼滤波器

    用于传感器数据的实时滤波，减少噪声影响。

    Args:
        process_variance: 过程噪声方差
        measurement_variance: 测量噪声方差
        estimated_error: 初始估计误差
        initial_value: 初始值
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
        estimated_error: float = 1.0,
        initial_value: float = 0.0,
    ) -> None:
        self._q: float = process_variance
        self._r: float = measurement_variance
        self._p: float = estimated_error
        self._x: float = initial_value
        self._k: float = 0.0

    def update(self, measurement: float) -> float:
        """更新滤波器

        Args:
            measurement: 新的测量值

        Returns:
            float: 滤波后的估计值
        """
        # 预测
        self._p = self._p + self._q

        # 更新
        self._k = self._p / (self._p + self._r)
        self._x = self._x + self._k * (measurement - self._x)
        self._p = (1 - self._k) * self._p

        return self._x

    def reset(self, value: float = 0.0) -> None:
        """重置滤波器

        Args:
            value: 重置后的初始值
        """
        self._x = value
        self._p = 1.0


class SensorFilter:
    """传感器数据滤波器

    支持多种滤波算法：滑动平均、中值、卡尔曼、指数平滑。

    Args:
        filter_type: 滤波器类型
        window_size: 滑动窗口大小
        alpha: 指数平滑系数
    """

    def __init__(
        self,
        filter_type: FilterType = FilterType.MOVING_AVERAGE,
        window_size: int = 5,
        alpha: float = 0.3,
    ) -> None:
        self._filter_type: FilterType = filter_type
        self._window_size: int = window_size
        self._alpha: float = alpha
        self._buffer: deque = deque(maxlen=window_size)
        self._kalman: Optional[KalmanFilter] = None
        self._exp_value: Optional[float] = None

        if filter_type == FilterType.KALMAN:
            self._kalman = KalmanFilter()

    def process(self, value: float) -> float:
        """处理新数据

        Args:
            value: 新的测量值

        Returns:
            float: 滤波后的值
        """
        if self._filter_type == FilterType.NONE:
            return value

        self._buffer.append(value)

        if self._filter_type == FilterType.MOVING_AVERAGE:
            return sum(self._buffer) / len(self._buffer)

        elif self._filter_type == FilterType.MEDIAN:
            return statistics.median(self._buffer)

        elif self._filter_type == FilterType.KALMAN:
            if self._kalman:
                return self._kalman.update(value)
            return value

        elif self._filter_type == FilterType.EXPONENTIAL:
            if self._exp_value is None:
                self._exp_value = value
            else:
                self._exp_value = self._alpha * value + (1 - self._alpha) * self._exp_value
            return self._exp_value

        return value

    def reset(self) -> None:
        """重置滤波器状态"""
        self._buffer.clear()
        self._exp_value = None
        if self._kalman:
            self._kalman.reset()


class SensorBase(ABC):
    """传感器抽象基类

    所有传感器的基类，定义统一接口。
    子类必须实现 read() 和 calibrate() 方法。

    Args:
        name: 传感器名称
        sensor_type: 传感器类型标识
        mode: 工作模式
        timeout: 操作超时时间（秒），默认5.0

    Example:
        >>> class MySensor(SensorBase):
        ...     def read(self) -> dict:
        ...         return {"temperature": 25.0}
        ...     def calibrate(self) -> bool:
        ...         return True
    """

    def __init__(
        self,
        name: str,
        sensor_type: str = "generic",
        mode: str = "polling",
        timeout: float = 5.0,
    ) -> None:
        self._name: str = name
        self._sensor_type: str = sensor_type
        self._mode: SensorMode = SensorMode(mode)
        self._timeout: float = timeout
        self._initialized: bool = False
        self._enabled: bool = True
        self._calibration: CalibrationParams = CalibrationParams()
        self._filters: Dict[str, SensorFilter] = {}
        self._instance_lock: threading.Lock = threading.Lock()
        self._last_read_time: float = 0.0
        self._last_data: Optional[SensorData] = None
        self._read_count: int = 0
        self._error_count: int = 0

    @abstractmethod
    def read(self) -> Dict[str, float]:
        """读取传感器数据

        子类必须实现此方法，返回传感器值字典。

        Returns:
            Dict[str, float]: 传感器值字典，如 {"temperature": 25.0, "humidity": 60.0}

        Raises:
            RuntimeError: 读取失败
        """
        pass

    @abstractmethod
    def calibrate(self) -> bool:
        """校准传感器

        子类必须实现此方法，执行校准操作。

        Returns:
            bool: True=校准成功, False=校准失败
        """
        pass

    def read_data(self) -> SensorData:
        """读取传感器数据（完整版）

        包含滤波、校准和错误处理。

        Returns:
            SensorData: 传感器数据对象
        """
        with self._instance_lock:
            if not self._enabled:
                return SensorData(
                    name=self._name,
                    values={},
                    valid=False,
                )

            try:
                raw_values = self.read()
                filtered_values = self._apply_filters(raw_values)
                calibrated_values = self._apply_calibration(filtered_values)

                data = SensorData(
                    name=self._name,
                    values=calibrated_values,
                    timestamp=time.monotonic(),
                    raw=raw_values,
                    valid=True,
                )

                self._last_data = data
                self._last_read_time = data.timestamp
                self._read_count += 1

                return data

            except Exception as e:
                self._error_count += 1
                logger.error(f"传感器 '{self._name}' 读取失败: {e}")
                return SensorData(
                    name=self._name,
                    values=self._last_data.values if self._last_data else {},
                    timestamp=time.monotonic(),
                    valid=False,
                )

    def _apply_filters(self, values: Dict[str, float]) -> Dict[str, float]:
        """应用滤波器

        Args:
            values: 原始值字典

        Returns:
            Dict[str, float]: 滤波后的值字典
        """
        filtered = {}
        for key, value in values.items():
            if key in self._filters:
                filtered[key] = self._filters[key].process(value)
            else:
                filtered[key] = value
        return filtered

    def _apply_calibration(self, values: Dict[str, float]) -> Dict[str, float]:
        """应用校准

        Args:
            values: 滤波后的值字典

        Returns:
            Dict[str, float]: 校准后的值字典
        """
        calibrated = {}
        for key, value in values.items():
            # 应用线性校准: y = ax + b
            value = value * self._calibration.linear_coef_a + self._calibration.linear_coef_b
            # 应用零点和标度
            value = (value - self._calibration.zero_offset) * self._calibration.scale_factor
            calibrated[key] = value
        return calibrated

    def add_filter(self, key: str, filter_type: FilterType, **kwargs) -> None:
        """为指定数据项添加滤波器

        Args:
            key: 数据项名称
            filter_type: 滤波器类型
            **kwargs: 滤波器参数
        """
        self._filters[key] = SensorFilter(filter_type=filter_type, **kwargs)
        logger.debug(f"传感器 '{self._name}' 数据项 '{key}' 添加 {filter_type.value} 滤波器")

    def remove_filter(self, key: str) -> None:
        """移除指定数据项的滤波器

        Args:
            key: 数据项名称
        """
        if key in self._filters:
            del self._filters[key]

    def set_calibration(self, params: CalibrationParams) -> None:
        """设置校准参数

        Args:
            params: 校准参数
        """
        self._calibration = params
        logger.info(f"传感器 '{self._name}' 校准参数已设置")

    def get_calibration(self) -> CalibrationParams:
        """获取当前校准参数

        Returns:
            CalibrationParams: 校准参数
        """
        return self._calibration

    def zero_calibration(self) -> None:
        """零点校准

        将当前读数设为零点偏移。
        """
        try:
            values = self.read()
            # 计算所有通道的平均值作为零点
            if values:
                avg = sum(values.values()) / len(values)
                self._calibration.zero_offset = avg
                self._calibration.calibration_date = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"传感器 '{self._name}' 零点校准完成: offset={avg:.4f}")
        except Exception as e:
            logger.error(f"传感器 '{self._name}' 零点校准失败: {e}")

    def reset_calibration(self) -> None:
        """重置校准参数"""
        self._calibration = CalibrationParams()
        for f in self._filters.values():
            f.reset()
        logger.info(f"传感器 '{self._name}' 校准参数已重置")

    def enable(self) -> None:
        """启用传感器"""
        self._enabled = True
        logger.debug(f"传感器 '{self._name}' 已启用")

    def disable(self) -> None:
        """禁用传感器"""
        self._enabled = False
        logger.debug(f"传感器 '{self._name}' 已禁用")

    @property
    def name(self) -> str:
        """获取传感器名称"""
        return self._name

    @property
    def sensor_type(self) -> str:
        """获取传感器类型"""
        return self._sensor_type

    @property
    def is_enabled(self) -> bool:
        """判断传感器是否启用"""
        return self._enabled

    @property
    def last_data(self) -> Optional[SensorData]:
        """获取最后一次读取的数据"""
        return self._last_data

    @property
    def read_count(self) -> int:
        """获取读取次数"""
        return self._read_count

    @property
    def error_count(self) -> int:
        """获取错误次数"""
        return self._error_count

    @property
    def error_rate(self) -> float:
        """获取错误率"""
        if self._read_count == 0:
            return 0.0
        return self._error_count / self._read_count

    @abstractmethod
    def close(self) -> None:
        """关闭传感器，释放资源

        子类应重写此方法进行资源清理。
        """
        self._enabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class UnifiedSensorManager:
    """统一传感器管理器

    管理多个传感器的注册、读取和事件处理。
    支持轮询和中断两种模式，提供批量读取和数据回调。

    Args:
        poll_interval: 轮询间隔（秒），默认0.1
        auto_start: 是否自动启动轮询，默认True

    Example:
        >>> manager = UnifiedSensorManager(poll_interval=0.1)
        >>> manager.register_sensor(my_temperature_sensor, "temp")
        >>> manager.register_sensor(my_distance_sensor, "distance")
        >>> data = manager.read_all()
        >>> temp = manager.read("temp")
    """

    def __init__(
        self,
        poll_interval: float = 0.1,
        auto_start: bool = True,
    ) -> None:
        self._sensors: Dict[str, SensorBase] = {}
        self._callbacks: Dict[str, List[Callable[[SensorData], None]]] = {}
        self._global_callbacks: List[Callable[[str, SensorData], None]] = []
        self._poll_interval: float = poll_interval
        self._auto_start: bool = auto_start
        self._manager_lock: threading.Lock = threading.Lock()
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_running: bool = False
        self._stop_event: threading.Event = threading.Event()
        self._sensor_data_cache: Dict[str, SensorData] = {}
        self._cache_lock: threading.Lock = threading.Lock()
        self._read_error_handlers: Dict[str, Callable[[Exception], None]] = {}
        self._total_reads: int = 0
        self._total_errors: int = 0

    def register_sensor(
        self,
        sensor: SensorBase,
        name: str,
        read_error_handler: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """注册传感器

        Args:
            sensor: 传感器实例
            name: 传感器注册名称
            read_error_handler: 读取错误处理回调

        Raises:
            ValueError: 名称已被注册
        """
        with self._manager_lock:
            if name in self._sensors:
                raise ValueError(f"传感器名称 '{name}' 已被注册")

            self._sensors[name] = sensor
            self._callbacks[name] = []
            self._sensor_data_cache[name] = SensorData(name=name, valid=False)

            if read_error_handler:
                self._read_error_handlers[name] = read_error_handler

        logger.info(f"传感器 '{name}' ({sensor.sensor_type}) 已注册")

        # 如果轮询在运行，新传感器会自动被读取
        if self._auto_start and not self._poll_running:
            self.start_polling()

    def unregister_sensor(self, name: str) -> None:
        """注销传感器

        Args:
            name: 传感器注册名称
        """
        with self._manager_lock:
            if name in self._sensors:
                sensor = self._sensors.pop(name)
                sensor.close()
                self._callbacks.pop(name, None)
                self._sensor_data_cache.pop(name, None)
                self._read_error_handlers.pop(name, None)
                logger.info(f"传感器 '{name}' 已注销")

    def read(self, name: str) -> SensorData:
        """读取指定传感器的数据

        Args:
            name: 传感器注册名称

        Returns:
            SensorData: 传感器数据

        Raises:
            KeyError: 传感器未注册
        """
        with self._manager_lock:
            if name not in self._sensors:
                raise KeyError(f"传感器 '{name}' 未注册")
            sensor = self._sensors[name]

        try:
            data = sensor.read_data()
            with self._cache_lock:
                self._sensor_data_cache[name] = data
            self._total_reads += 1

            # 触发回调
            self._trigger_callbacks(name, data)

            return data
        except Exception as e:
            self._total_errors += 1
            logger.error(f"读取传感器 '{name}' 失败: {e}")
            if name in self._read_error_handlers:
                self._read_error_handlers[name](e)
            raise

    def read_all(self) -> Dict[str, SensorData]:
        """读取所有传感器的数据

        Returns:
            Dict[str, SensorData]: 传感器名称=数据的字典
        """
        results = {}
        with self._manager_lock:
            sensor_names = list(self._sensors.keys())

        for name in sensor_names:
            try:
                results[name] = self.read(name)
            except Exception as e:
                logger.warning(f"批量读取中传感器 '{name}' 失败: {e}")
                results[name] = SensorData(name=name, valid=False)

        return results

    def read_all_dict(self) -> Dict[str, Dict[str, float]]:
        """读取所有传感器数据并返回纯数值字典

        Returns:
            Dict[str, Dict[str, float]]: {传感器名: {数据项: 值}}
        """
        all_data = self.read_all()
        return {
            name: data.values
            for name, data in all_data.items()
            if data.valid
        }

    def get_cached_data(self, name: str) -> SensorData:
        """获取缓存的传感器数据

        不触发实际读取，返回最近一次的数据。

        Args:
            name: 传感器注册名称

        Returns:
            SensorData: 缓存的传感器数据
        """
        with self._cache_lock:
            return self._sensor_data_cache.get(
                name, SensorData(name=name, valid=False)
            )

    def get_all_cached(self) -> Dict[str, SensorData]:
        """获取所有缓存的传感器数据

        Returns:
            Dict[str, SensorData]: 缓存数据字典
        """
        with self._cache_lock:
            return dict(self._sensor_data_cache)

    def register_callback(
        self,
        name: str,
        callback: Callable[[SensorData], None],
    ) -> None:
        """注册传感器数据回调

        当传感器数据更新时调用回调。

        Args:
            name: 传感器注册名称
            callback: 回调函数，参数为SensorData
        """
        with self._manager_lock:
            if name in self._callbacks:
                self._callbacks[name].append(callback)

    def register_global_callback(
        self,
        callback: Callable[[str, SensorData], None],
    ) -> None:
        """注册全局数据回调

        当任何传感器数据更新时调用回调。

        Args:
            callback: 回调函数，参数为(传感器名, SensorData)
        """
        self._global_callbacks.append(callback)

    def _trigger_callbacks(self, name: str, data: SensorData) -> None:
        """触发传感器数据回调

        Args:
            name: 传感器名称
            data: 传感器数据
        """
        # 触发特定回调
        if name in self._callbacks:
            for callback in self._callbacks[name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"传感器 '{name}' 回调执行失败: {e}")

        # 触发全局回调
        for callback in self._global_callbacks:
            try:
                callback(name, data)
            except Exception as e:
                logger.error(f"传感器 '{name}' 全局回调执行失败: {e}")

    def start_polling(self) -> None:
        """启动轮询线程"""
        if self._poll_running:
            return

        self._poll_running = True
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="SensorManager-Poll",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info("传感器轮询线程已启动")

    def _poll_loop(self) -> None:
        """轮询循环

        在独立线程中定期读取所有传感器数据。
        """
        while self._poll_running and not self._stop_event.is_set():
            try:
                start_time = time.monotonic()
                self.read_all()
                elapsed = time.monotonic() - start_time

                # 计算下次读取时间
                sleep_time = self._poll_interval - elapsed
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)
            except Exception as e:
                logger.error(f"传感器轮询异常: {e}")
                time.sleep(self._poll_interval)

    def stop_polling(self) -> None:
        """停止轮询线程"""
        self._poll_running = False
        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        logger.info("传感器轮询线程已停止")

    def get_sensor_names(self) -> List[str]:
        """获取所有已注册的传感器名称

        Returns:
            List[str]: 传感器名称列表
        """
        with self._manager_lock:
            return list(self._sensors.keys())

    def get_sensor_info(self) -> Dict[str, dict]:
        """获取所有传感器的信息

        Returns:
            Dict[str, dict]: 传感器信息字典
        """
        info = {}
        with self._manager_lock:
            for name, sensor in self._sensors.items():
                info[name] = {
                    "type": sensor.sensor_type,
                    "enabled": sensor.is_enabled,
                    "read_count": sensor.read_count,
                    "error_count": sensor.error_count,
                    "error_rate": sensor.error_rate,
                }
        return info

    def get_statistics(self) -> dict:
        """获取管理器统计信息

        Returns:
            dict: 统计信息字典
        """
        return {
            "registered_sensors": len(self._sensors),
            "total_reads": self._total_reads,
            "total_errors": self._total_errors,
            "polling_active": self._poll_running,
            "poll_interval": self._poll_interval,
        }

    def enable_sensor(self, name: str) -> None:
        """启用指定传感器

        Args:
            name: 传感器注册名称
        """
        with self._manager_lock:
            if name in self._sensors:
                self._sensors[name].enable()

    def disable_sensor(self, name: str) -> None:
        """禁用指定传感器

        Args:
            name: 传感器注册名称
        """
        with self._manager_lock:
            if name in self._sensors:
                self._sensors[name].disable()

    def close(self) -> None:
        """关闭管理器，释放所有资源"""
        self.stop_polling()

        with self._manager_lock:
            for name, sensor in list(self._sensors.items()):
                try:
                    sensor.close()
                except Exception as e:
                    logger.error(f"关闭传感器 '{name}' 失败: {e}")
            self._sensors.clear()
            self._callbacks.clear()
            self._global_callbacks.clear()

        logger.info("传感器管理器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DummySensor(SensorBase):
    """虚拟传感器（用于测试）

    生成模拟数据，用于测试和开发。

    Args:
        name: 传感器名称
        value_range: 数值范围 (min, max)
        noise: 噪声幅度
    """

    def __init__(
        self,
        name: str = "dummy",
        value_range: Tuple[float, float] = (0.0, 100.0),
        noise: float = 1.0,
    ) -> None:
        super().__init__(name, sensor_type="dummy", mode="polling")
        self._value_range = value_range
        self._noise = noise
        import random
        self._random = random.Random()
        self._base_value = (value_range[0] + value_range[1]) / 2.0

    def read(self) -> Dict[str, float]:
        """生成模拟数据"""
        import random
        value = self._base_value + random.gauss(0, self._noise)
        value = max(self._value_range[0], min(self._value_range[1], value))
        return {"value": value}

    def calibrate(self) -> bool:
        """虚拟校准"""
        return True

    def close(self) -> None:
        """关闭虚拟传感器"""
        super().close()
