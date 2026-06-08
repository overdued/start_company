"""
HC-SR04超声波测距传感器驱动模块

基于GPIO TRIG/ECHO引脚控制的超声波距离测量驱动。
支持温度补偿、中值滤波去噪和超时保护机制。
适用于OrangePi Kunpeng Pro (RK3588)平台的40pin GPIO接口。

功能特性:
    - 精确距离测量(2cm ~ 400cm)
    - 环境温度补偿(声速修正)
    - 中值滤波算法去噪
    - 超时保护(最大4m范围)
    - 连续测量模式(异步)
    - 异常检测与重试机制

硬件接线:
    VCC  -> 5V
    GND  -> GND
    TRIG -> GPIOx (输出)
    ECHO -> GPIOy (输入)

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


class FilterType(Enum):
    """滤波算法类型枚举"""
    NONE = "none"           # 无滤波(原始值)
    MEDIAN = "median"       # 中值滤波
    MEAN = "mean"           # 滑动平均
    KALMAN = "kalman"       # 卡尔曼滤波
    EXPONENTIAL = "exp"     # 指数平滑


@dataclass
class DistanceReading:
    """单次距离读数数据结构

    属性:
        distance_m: 测量距离(米)
        distance_cm: 测量距离(厘米)
        temperature_c: 测量时的环境温度(摄氏度)
        speed_of_sound: 实际使用的声速(m/s)
        timestamp: 测量时间戳(秒)
        raw_duration: 原始回波时间(秒)
        filtered: 是否经过滤波处理
        valid: 读数是否有效
    """
    distance_m: float = 0.0
    distance_cm: float = 0.0
    temperature_c: float = 20.0
    speed_of_sound: float = 343.0
    timestamp: float = 0.0
    raw_duration: float = 0.0
    filtered: bool = False
    valid: bool = False


@dataclass
class SensorConfig:
    """传感器配置参数

    属性:
        trig_pin: TRIG引脚编号(BCM编码)
        echo_pin: ECHO引脚编号(BCM编码)
        timeout_sec: 最大测量超时(秒),默认0.05(约8.6m范围)
        max_distance_m: 最大有效距离(米),默认4.0
        min_distance_m: 最小有效距离(米),默认0.02
        filter_type: 滤波类型
        filter_window: 滤波窗口大小
        temperature_compensation: 是否启用温度补偿
        ambient_temperature: 环境温度(摄氏度)
    """
    trig_pin: int = 17
    echo_pin: int = 27
    timeout_sec: float = 0.05
    max_distance_m: float = 4.0
    min_distance_m: float = 0.02
    filter_type: FilterType = FilterType.MEDIAN
    filter_window: int = 5
    temperature_compensation: bool = True
    ambient_temperature: float = 20.0


class KalmanFilter:
    """一维卡尔曼滤波器

    用于超声波距离测量的噪声抑制和状态估计。

    属性:
        q: 过程噪声协方差
        r: 测量噪声协方差
        x: 当前状态估计
        p: 估计误差协方差
        k: 卡尔曼增益
    """

    def __init__(
        self,
        initial_value: float = 0.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        """初始化卡尔曼滤波器

        参数:
            initial_value: 初始状态估计值
            process_noise: 过程噪声协方差Q
            measurement_noise: 测量噪声协方差R
        """
        self.q: float = process_noise
        self.r: float = measurement_noise
        self.x: float = initial_value
        self.p: float = 1.0
        self.k: float = 0.0

    def update(self, measurement: float) -> float:
        """执行一次卡尔曼滤波更新

        参数:
            measurement: 新的测量值

        返回:
            float: 滤波后的估计值
        """
        # 预测步骤
        self.p = self.p + self.q

        # 更新步骤
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p

        return self.x

    def reset(self, value: float = 0.0) -> None:
        """重置滤波器状态

        参数:
            value: 重置后的初始值
        """
        self.x = value
        self.p = 1.0
        self.k = 0.0


class HCSR04Ultrasonic:
    """HC-SR04超声波测距传感器驱动类

    提供精确的距离测量功能,支持温度补偿、多类型滤波和异步连续测量模式。
    内置超时保护和异常重试机制,确保硬件安全。

    示例:
        >>> sensor = HCSR04Ultrasonic(SensorConfig(trig_pin=17, echo_pin=27))
        >>> await sensor.initialize()
        >>> reading = await sensor.measure()
        >>> print(f"距离: {reading.distance_cm:.1f} cm")
        >>> await sensor.start_continuous(callback=on_distance)
        >>> await asyncio.sleep(10)
        >>> await sensor.shutdown()

    属性:
        config: 传感器配置参数
        _initialized: 初始化状态标志
        _continuous_task: 连续测量异步任务
        _history: 测量历史数据缓冲区
        _kalman: 卡尔曼滤波器实例
    """

    # 物理常量
    SPEED_OF_SOUND_20C: float = 343.0       # 20°C时声速(m/s)
    TEMP_COEFFICIENT: float = 0.606         # 声速温度系数(m/s/°C)
    MAX_PULSE_DURATION: float = 0.025       # 最大回波等待时间(25ms ~ 8.6m)
    TRIG_PULSE_DURATION: float = 0.00001    # TRIG触发脉宽(10μs)
    MEASUREMENT_INTERVAL: float = 0.06      # 最小测量间隔(60ms,避免串扰)
    DEFAULT_RETRIES: int = 3                # 默认重试次数

    def __init__(self, config: SensorConfig | None = None) -> None:
        """初始化HC-SR04传感器驱动

        参数:
            config: 传感器配置,None则使用默认配置
        """
        self.config: SensorConfig = config or SensorConfig()

        # 内部状态
        self._initialized: bool = False
        self._measuring: bool = False
        self._continuous_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._lock: asyncio.Lock = asyncio.Lock()

        # 数据缓冲区
        self._history: deque[float] = deque(
            maxlen=self.config.filter_window * 2
        )
        self._kalman: KalmanFilter = KalmanFilter()

        # 回调函数列表
        self._reading_callbacks: list[Callable[[DistanceReading], None]] = []

        # GPIO对象(延迟初始化)
        self._trig_gpio: Any = None
        self._echo_gpio: Any = None

        logger.debug(
            f"HC-SR04初始化: TRIG=GPIO{self.config.trig_pin}, "
            f"ECHO=GPIO{self.config.echo_pin}"
        )

    async def initialize(self) -> bool:
        """初始化GPIO引脚和传感器

        配置TRIG为输出模式,ECHO为输入模式,并进行初始测试测量。

        返回:
            bool: 初始化成功返回True

        异常:
            RuntimeError: GPIO初始化失败
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # 尝试使用OPi.GPIO库(OrangePi专用)
                try:
                    import OPi.GPIO as GPIO
                    self._gpio_lib = GPIO
                    self._gpio_lib.setmode(self._gpio_lib.BCM)
                except ImportError:
                    # 回退到标准RPi.GPIO风格接口
                    try:
                        import gpiod
                        self._gpio_lib = gpiod
                        self._gpio_chip = self._gpio_lib.Chip("0")
                    except ImportError:
                        logger.warning("GPIO库未安装,使用模拟模式")
                        self._gpio_lib = None

                if self._gpio_lib is not None:
                    # 配置TRIG引脚为输出
                    if hasattr(self._gpio_lib, 'setup'):
                        self._gpio_lib.setup(
                            self.config.trig_pin, self._gpio_lib.OUT
                        )
                        self._gpio_lib.setup(
                            self.config.echo_pin, self._gpio_lib.IN
                        )
                        # 初始状态置低
                        self._gpio_lib.output(
                            self.config.trig_pin, self._gpio_lib.LOW
                        )
                    elif hasattr(self, '_gpio_chip'):
                        # gpiod方式
                        self._trig_gpio = self._gpio_chip.get_line(
                            self.config.trig_pin
                        )
                        self._echo_gpio = self._gpio_chip.get_line(
                            self.config.echo_pin
                        )
                        self._trig_gpio.request(
                            consumer="hc-sr04-trig",
                            type=gpiod.LINE_REQ_DIR_OUT
                        )
                        self._echo_gpio.request(
                            consumer="hc-sr04-echo",
                            type=gpiod.LINE_REQ_DIR_IN
                        )
                        self._trig_gpio.set_value(0)

                self._initialized = True
                logger.info(
                    f"HC-SR04初始化成功: TRIG=GPIO{self.config.trig_pin}, "
                    f"ECHO=GPIO{self.config.echo_pin}"
                )

                # 执行一次测试测量验证传感器
                test_reading = await self._measure_single()
                if test_reading.valid:
                    logger.info(
                        f"传感器测试测量成功: {test_reading.distance_cm:.1f} cm"
                    )

                return True

            except Exception as e:
                logger.error(f"HC-SR04初始化失败: {e}")
                self._initialized = False
                return False

    def _calculate_speed_of_sound(self, temperature_c: float) -> float:
        """根据温度计算声速

        使用公式: v = 331.3 + 0.606 * T (m/s)
        其中T为摄氏温度。

        参数:
            temperature_c: 环境温度(摄氏度)

        返回:
            float: 声速(m/s)
        """
        speed = 331.3 + self.TEMP_COEFFICIENT * temperature_c
        return speed

    async def _measure_single(self) -> DistanceReading:
        """执行单次距离测量(内部方法)

        发送TRIG触发信号,测量ECHO回波持续时间,计算距离。
        包含完整的超时保护和异常处理。

        返回:
            DistanceReading: 包含距离、温度、时间戳等信息的读数对象
        """
        reading = DistanceReading(
            temperature_c=self.config.ambient_temperature,
            timestamp=time.time(),
        )

        try:
            # 计算声速(温度补偿)
            if self.config.temperature_compensation:
                sos = self._calculate_speed_of_sound(
                    self.config.ambient_temperature
                )
            else:
                sos = self.SPEED_OF_SOUND_20C

            reading.speed_of_sound = sos

            # 发送TRIG触发信号
            if self._gpio_lib is None:
                # 模拟模式:生成随机距离
                await asyncio.sleep(self.MEASUREMENT_INTERVAL)
                reading.distance_m = np.random.uniform(0.1, 2.0)
                reading.distance_cm = reading.distance_m * 100
                reading.raw_duration = reading.distance_m * 2 / sos
                reading.valid = True
                return reading

            # 10μs高电平触发
            if self._trig_gpio:
                if hasattr(self._trig_gpio, 'set_value'):
                    self._trig_gpio.set_value(1)
                    await asyncio.sleep(self.TRIG_PULSE_DURATION)
                    self._trig_gpio.set_value(0)
                else:
                    self._gpio_lib.output(
                        self.config.trig_pin, self._gpio_lib.HIGH
                    )
                    await asyncio.sleep(self.TRIG_PULSE_DURATION)
                    self._gpio_lib.output(
                        self.config.trig_pin, self._gpio_lib.LOW
                    )

            # 等待ECHO变高(超时保护)
            start_wait = time.monotonic()
            while True:
                if hasattr(self._echo_gpio, 'get_value'):
                    echo_val = self._echo_gpio.get_value()
                else:
                    echo_val = self._gpio_lib.input(self.config.echo_pin)

                if echo_val:
                    break

                if time.monotonic() - start_wait > self.config.timeout_sec:
                    logger.warning("等待ECHO上升沿超时")
                    return reading

                await asyncio.sleep(0.0001)  # 100μs轮询

            # 记录ECHO高电平开始时间
            pulse_start = time.monotonic()

            # 等待ECHO变低(超时保护)
            while True:
                if hasattr(self._echo_gpio, 'get_value'):
                    echo_val = self._echo_gpio.get_value()
                else:
                    echo_val = self._gpio_lib.input(self.config.echo_pin)

                if not echo_val:
                    break

                elapsed = time.monotonic() - pulse_start
                if elapsed > self.MAX_PULSE_DURATION:
                    logger.warning("ECHO高电平持续时间过长,可能无回波")
                    return reading

                await asyncio.sleep(0.0001)

            # 记录ECHO高电平结束时间
            pulse_end = time.monotonic()
            pulse_duration = pulse_end - pulse_start
            reading.raw_duration = pulse_duration

            # 计算距离: d = (v * t) / 2 (往返除以2)
            distance_m = (sos * pulse_duration) / 2.0

            # 有效性检查
            if distance_m < self.config.min_distance_m:
                logger.debug(f"距离过近: {distance_m*100:.1f}cm < 最小值")
                return reading

            if distance_m > self.config.max_distance_m:
                logger.debug(f"距离过远: {distance_m:.2f}m > 最大值")
                return reading

            reading.distance_m = distance_m
            reading.distance_cm = distance_m * 100.0
            reading.valid = True

        except Exception as e:
            logger.error(f"测量过程异常: {e}")

        return reading

    async def measure(
        self,
        retries: int = DEFAULT_RETRIES,
        apply_filter: bool = True,
    ) -> DistanceReading:
        """执行距离测量(带重试和滤波)

        多次测量并取有效结果的平均值,支持滤波算法去噪。

        参数:
            retries: 重试次数,默认3
            apply_filter: 是否应用滤波,默认True

        返回:
            DistanceReading: 滤波后的距离读数

        示例:
            >>> reading = await sensor.measure(retries=5)
            >>> if reading.valid:
            ...     print(f"距离: {reading.distance_cm:.1f} cm")
        """
        if not self._initialized:
            logger.error("传感器未初始化")
            return DistanceReading()

        async with self._lock:
            self._measuring = True
            measurements: list[float] = []

            for attempt in range(retries):
                try:
                    reading = await self._measure_single()
                    if reading.valid:
                        measurements.append(reading.distance_m)

                    # 测量间隔,避免串扰
                    if attempt < retries - 1:
                        await asyncio.sleep(self.MEASUREMENT_INTERVAL)

                except Exception as e:
                    logger.warning(f"第{attempt+1}次测量异常: {e}")

            self._measuring = False

            if not measurements:
                logger.warning("所有测量尝试均失败")
                return DistanceReading()

            # 应用滤波
            filtered_distance = self._apply_filter(measurements, apply_filter)

            result = DistanceReading(
                distance_m=filtered_distance,
                distance_cm=filtered_distance * 100.0,
                temperature_c=self.config.ambient_temperature,
                speed_of_sound=reading.speed_of_sound if "reading" in dir() else self.SPEED_OF_SOUND_20C,
                timestamp=time.time(),
                raw_duration=statistics.median(
                    [r.raw_duration for r in [reading]]
                ) if 'reading' in dir() else 0.0,
                filtered=apply_filter and len(measurements) > 1,
                valid=True,
            )

            # 更新历史记录
            self._history.append(filtered_distance)

            return result

    def _apply_filter(
        self, measurements: list[float], apply: bool = True
    ) -> float:
        """应用滤波算法(内部方法)

        根据配置的滤波类型对测量值进行处理。

        参数:
            measurements: 原始测量值列表
            apply: 是否应用滤波

        返回:
            float: 滤波后的距离值(米)
        """
        if not apply or len(measurements) < 2:
            return statistics.mean(measurements)

        filter_type = self.config.filter_type

        if filter_type == FilterType.MEDIAN:
            # 中值滤波:对异常值鲁棒
            return statistics.median(measurements)

        elif filter_type == FilterType.MEAN:
            # 滑动平均
            return statistics.mean(measurements)

        elif filter_type == FilterType.KALMAN:
            # 卡尔曼滤波
            for m in measurements:
                filtered = self._kalman.update(m)
            return filtered

        elif filter_type == FilterType.EXPONENTIAL:
            # 指数平滑(使用历史数据)
            alpha = 0.3  # 平滑因子
            if self._history:
                smoothed = alpha * statistics.mean(measurements) +                            (1 - alpha) * statistics.mean(self._history)
                return smoothed
            return statistics.mean(measurements)

        else:
            return statistics.mean(measurements)

    def register_callback(
        self, callback: Callable[[DistanceReading], None]
    ) -> None:
        """注册距离读数回调函数

        在连续测量模式下,每次测量完成后会调用所有回调。

        参数:
            callback: 回调函数,接收DistanceReading参数
        """
        if callback not in self._reading_callbacks:
            self._reading_callbacks.append(callback)

    def unregister_callback(
        self, callback: Callable[[DistanceReading], None]
    ) -> None:
        """注销距离读数回调函数

        参数:
            callback: 要移除的回调函数
        """
        if callback in self._reading_callbacks:
            self._reading_callbacks.remove(callback)

    async def start_continuous(
        self,
        interval: float = 0.1,
        callback: Callable[[DistanceReading], None] | None = None,
    ) -> bool:
        """启动连续测量模式

        以指定间隔持续进行距离测量,结果通过回调函数分发。

        参数:
            interval: 测量间隔(秒),默认0.1
            callback: 可选的回调函数

        返回:
            bool: 启动成功返回True
        """
        if callback:
            self.register_callback(callback)

        if self._continuous_task and not self._continuous_task.done():
            logger.warning("连续测量已在运行")
            return True

        self._stop_event.clear()
        self._continuous_task = asyncio.create_task(
            self._continuous_loop(interval), name="hcsr04_continuous"
        )
        logger.info(f"连续测量已启动,间隔: {interval}s")
        return True

    async def _continuous_loop(self, interval: float) -> None:
        """连续测量循环(内部方法)

        参数:
            interval: 测量间隔(秒)
        """
        logger.debug("连续测量循环已启动")

        while not self._stop_event.is_set():
            try:
                start_time = time.monotonic()

                reading = await self.measure(
                    retries=1, apply_filter=True
                )

                # 分发读数
                for cb in self._reading_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            asyncio.create_task(cb(reading))
                        else:
                            cb(reading)
                    except Exception as e:
                        logger.error(f"回调执行异常: {e}")

                # 间隔控制
                elapsed = time.monotonic() - start_time
                sleep_time = max(0, interval - elapsed)
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=sleep_time
                )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"连续测量异常: {e}")
                await asyncio.sleep(interval)

        logger.debug("连续测量循环已退出")

    async def stop_continuous(self) -> None:
        """停止连续测量模式"""
        self._stop_event.set()

        if self._continuous_task and not self._continuous_task.done():
            self._continuous_task.cancel()
            try:
                await self._continuous_task
            except asyncio.CancelledError:
                pass

        logger.info("连续测量已停止")

    def set_temperature(self, temperature_c: float) -> None:
        """设置环境温度用于声速补偿

        参数:
            temperature_c: 环境温度(摄氏度)
        """
        self.config.ambient_temperature = temperature_c
        logger.debug(f"环境温度已更新: {temperature_c}°C")

    def get_statistics(self) -> dict[str, float]:
        """获取测量统计信息

        返回:
            dict: 包含均值、中值、标准差、最小值、最大值的字典
        """
        if not self._history:
            return {}

        data = list(self._history)
        return {
            "mean_m": statistics.mean(data),
            "median_m": statistics.median(data),
            "stdev_m": statistics.stdev(data) if len(data) > 1 else 0.0,
            "min_m": min(data),
            "max_m": max(data),
            "count": len(data),
        }

    async def shutdown(self) -> None:
        """关闭传感器,释放GPIO资源

        停止连续测量,清理GPIO引脚配置。
        """
        await self.stop_continuous()

        async with self._lock:
            try:
                if self._gpio_lib and hasattr(self._gpio_lib, 'cleanup'):
                    self._gpio_lib.cleanup(self.config.trig_pin)
                    self._gpio_lib.cleanup(self.config.echo_pin)

                if self._trig_gpio and hasattr(self._trig_gpio, 'release'):
                    self._trig_gpio.release()
                if self._echo_gpio and hasattr(self._echo_gpio, 'release'):
                    self._echo_gpio.release()

                self._initialized = False
                logger.info("HC-SR04传感器已关闭")

            except Exception as e:
                logger.error(f"关闭传感器异常: {e}")

    @property
    def is_initialized(self) -> bool:
        """传感器初始化状态"""
        return self._initialized

    @property
    def is_measuring(self) -> bool:
        """是否正在测量中"""
        return self._measuring

    def __repr__(self) -> str:
        return (
            f"HCSR04Ultrasonic(trig={self.config.trig_pin}, "
            f"echo={self.config.echo_pin}, "
            f"initialized={self._initialized})"
        )

    async def __aenter__(self) -> HCSR04Ultrasonic:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
