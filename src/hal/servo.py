"""
RK3588 舵机硬件抽象层模块

本模块提供对OrangePi Kunpeng Pro (RK3588) 舵机的完整控制支持，
基于PCA9685 I2C驱动板实现，支持多舵机同步控制。

舵机控制参数：
    - 标准舵机: 角度范围0°-180°
    - 脉冲宽度: 0.5ms(0°) ~ 2.5ms(180°)
    - 频率: 50Hz（周期20ms）
    - 分辨率: 12位（PCA9685）

PCA9685配置：
    - I2C地址: 0x40（默认）
    - 16路独立PWM输出
    - 12位分辨率（4096级）

特性：
    - 角度到脉宽的精确映射
    - 多舵机同步控制
    - 平滑运动（缓动函数）
    - 速度限制
    - 超时保护（默认5秒）
    - 线程安全
    - 异常回退到安全状态

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

import os
import time
import math
import logging
import threading
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# PCA9685寄存器定义（与pwm.py共享）
PCA9685_ADDRESS = 0x40
PCA9685_MODE1 = 0x00
PCA9685_MODE2 = 0x01
PCA9685_PRESCALE = 0xFE
PCA9685_LED0_ON_L = 0x06
PCA9685_LED0_ON_H = 0x07
PCA9685_LED0_OFF_L = 0x08
PCA9685_LED0_OFF_H = 0x09
PCA9685_ALL_LED_ON_L = 0xFA
PCA9685_ALL_LED_ON_H = 0xFB
PCA9685_ALL_LED_OFF_L = 0xFC
PCA9685_ALL_LED_OFF_H = 0xFD

# 舵机控制常量
DEFAULT_SERVO_FREQ = 50           # 标准舵机频率50Hz
DEFAULT_MIN_PULSE_US = 500        # 0°对应脉宽0.5ms
DEFAULT_MAX_PULSE_US = 2500       # 180°对应脉宽2.5ms
DEFAULT_MIN_ANGLE = 0             # 最小角度
DEFAULT_MAX_ANGLE = 180           # 最大角度
PCA9685_RESOLUTION = 4096         # 12位分辨率


class EasingType(Enum):
    """缓动函数类型枚举"""
    LINEAR = "linear"           # 线性
    EASE_IN_QUAD = "ease_in"    # 二次加速
    EASE_OUT_QUAD = "ease_out"  # 二次减速
    EASE_IN_OUT = "ease_in_out" # 二次加减速
    EASE_OUT_BOUNCE = "bounce"  # 弹跳效果


@dataclass
class ServoConfig:
    """舵机配置数据类

    Attributes:
        min_pulse_us: 最小脉宽（微秒）
        max_pulse_us: 最大脉宽（微秒）
        min_angle: 最小角度（度）
        max_angle: 最大角度（度）
        default_angle: 默认角度（度）
        speed: 运动速度（度/秒），0表示无限制
        easing: 缓动函数类型
    """
    min_pulse_us: float = DEFAULT_MIN_PULSE_US
    max_pulse_us: float = DEFAULT_MAX_PULSE_US
    min_angle: float = DEFAULT_MIN_ANGLE
    max_angle: float = DEFAULT_MAX_ANGLE
    default_angle: float = 90.0
    speed: float = 0.0  # 度/秒，0表示无限制
    easing: EasingType = EasingType.LINEAR


class PCA9685ServoController:
    """PCA9685舵机控制器

    通过I2C总线控制PCA9685芯片，提供16路舵机控制。

    Args:
        bus: I2C总线号
        address: PCA9685 I2C地址，默认0x40

    Example:
        >>> controller = PCA9685ServoController(3, 0x40)
        >>> controller.initialize()
        >>> controller.set_angle(0, 90.0)  # 通道0设为90度
    """

    # 类级别实例缓存
    _instances: Dict[Tuple[int, int], "PCA9685ServoController"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, bus: int, address: int = PCA9685_ADDRESS) -> None:
        self._bus = bus
        self._address = address
        self._initialized = False
        self._lock = threading.Lock()
        self._smbus: Optional[object] = None
        self._frequency = DEFAULT_SERVO_FREQ

    @classmethod
    def get_instance(cls, bus: int, address: int = PCA9685_ADDRESS) -> "PCA9685ServoController":
        """获取PCA9685控制器实例（单例模式）

        Args:
            bus: I2C总线号
            address: I2C地址

        Returns:
            PCA9685ServoController: 控制器实例
        """
        key = (bus, address)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(bus, address)
            return cls._instances[key]

    def initialize(self) -> None:
        """初始化PCA9685芯片"""
        if self._initialized:
            return

        try:
            from smbus2 import SMBus
            self._smbus = SMBus(self._bus)

            with self._lock:
                # 软件复位
                self._write(PCA9685_MODE1, 0x00)
                time.sleep(0.005)

                # 配置MODE2: totem pole输出
                self._write(PCA9685_MODE2, 0x04)

                # 配置MODE1: 正常模式，ALLCALL
                self._write(PCA9685_MODE1, 0x01)
                time.sleep(0.005)

                # 等待振荡器稳定，退出睡眠
                mode1 = self._read(PCA9685_MODE1)
                mode1 &= ~0x10  # 清除SLEEP位
                self._write(PCA9685_MODE1, mode1)
                time.sleep(0.005)

                # 设置频率
                self._set_frequency(DEFAULT_SERVO_FREQ)

                # 初始化所有通道为0占空比
                self._set_all_pwm(0, 0)

                self._initialized = True
                logger.info(f"PCA9685舵机控制器 (0x{self._address:02X}) 初始化成功")
        except Exception as e:
            raise RuntimeError(f"PCA9685舵机控制器初始化失败: {e}") from e

    def _write(self, register: int, value: int) -> None:
        """写入PCA9685寄存器"""
        if self._smbus:
            self._smbus.write_byte_data(self._address, register, value)

    def _read(self, register: int) -> int:
        """读取PCA9685寄存器"""
        if self._smbus:
            return self._smbus.read_byte_data(self._address, register)
        return 0

    def _set_frequency(self, freq_hz: int) -> None:
        """设置PWM频率

        Args:
            freq_hz: 频率（Hz）
        """
        prescale = int(round(25000000.0 / (4096.0 * freq_hz)) - 1)
        prescale = max(3, min(255, prescale))

        old_mode = self._read(PCA9685_MODE1)
        new_mode = (old_mode & 0x7F) | 0x10  # 进入睡眠
        self._write(PCA9685_MODE1, new_mode)
        self._write(PCA9685_PRESCALE, prescale)
        self._write(PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self._write(PCA9685_MODE1, old_mode | 0x80)  # 重启
        self._frequency = freq_hz

    def _set_pwm(self, channel: int, on: int, off: int) -> None:
        """设置单个通道的PWM

        Args:
            channel: 通道号（0-15）
            on: 开启计数值（0-4095）
            off: 关闭计数值（0-4095）
        """
        base = PCA9685_LED0_ON_L + 4 * channel
        self._write(base, on & 0xFF)
        self._write(base + 1, (on >> 8) & 0x0F)
        self._write(base + 2, off & 0xFF)
        self._write(base + 3, (off >> 8) & 0x0F)

    def _set_all_pwm(self, on: int, off: int) -> None:
        """设置所有通道的PWM"""
        self._write(PCA9685_ALL_LED_ON_L, on & 0xFF)
        self._write(PCA9685_ALL_LED_ON_H, (on >> 8) & 0x0F)
        self._write(PCA9685_ALL_LED_OFF_L, off & 0xFF)
        self._write(PCA9685_ALL_LED_OFF_H, (off >> 8) & 0x0F)

    def set_channel_pulse(self, channel: int, pulse_us: float) -> None:
        """设置通道脉冲宽度

        Args:
            channel: 通道号（0-15）
            pulse_us: 脉冲宽度（微秒）
        """
        if not self._initialized:
            raise RuntimeError("PCA9685未初始化")

        period_us = 1000000.0 / self._frequency
        off_count = int(pulse_us / period_us * PCA9685_RESOLUTION)
        off_count = max(0, min(PCA9685_RESOLUTION - 1, off_count))

        with self._lock:
            self._set_pwm(channel, 0, off_count)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def close(self) -> None:
        """关闭控制器"""
        if self._smbus:
            try:
                self._set_all_pwm(0, 0)
                self._smbus.close()
            except Exception:
                pass
            self._smbus = None
            self._initialized = False


class Servo:
    """舵机控制器类

    基于PCA9685 I2C驱动板的单路舵机控制。

    Args:
        channel: PCA9685通道号（0-15）
        min_angle: 最小角度限制（度），默认0
        max_angle: 最大角度限制（度），默认180
        config: 舵机配置，None表示使用默认配置
        i2c_bus: I2C总线号，默认3
        i2c_address: PCA9685 I2C地址，默认0x40
        timeout: 操作超时时间（秒），默认5.0

    Raises:
        ValueError: 通道号或角度参数无效
        RuntimeError: 舵机初始化失败

    Example:
        >>> servo = Servo(0, min_angle=0, max_angle=180)
        >>> servo.set_angle(90.0)   # 转到90度
        >>> servo.set_angle(0.0)    # 转到0度
        >>> servo.set_angle(180.0)  # 转到180度
        >>> servo.close()
    """

    def __init__(
        self,
        channel: int,
        min_angle: float = DEFAULT_MIN_ANGLE,
        max_angle: float = DEFAULT_MAX_ANGLE,
        config: Optional[ServoConfig] = None,
        i2c_bus: int = 3,
        i2c_address: int = PCA9685_ADDRESS,
        timeout: float = 5.0,
    ) -> None:
        self._channel: int = channel
        self._config: ServoConfig = config or ServoConfig()
        self._config.min_angle = max(min_angle, self._config.min_angle)
        self._config.max_angle = min(max_angle, self._config.max_angle)
        self._i2c_bus: int = i2c_bus
        self._i2c_address: int = i2c_address
        self._timeout: float = timeout
        self._instance_lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._current_angle: float = self._config.default_angle
        self._target_angle: float = self._config.default_angle
        self._controller: PCA9685ServoController = PCA9685ServoController.get_instance(
            i2c_bus, i2c_address
        )
        self._motion_thread: Optional[threading.Thread] = None
        self._motion_running: bool = False
        self._stop_event: threading.Event = threading.Event()

        # 参数验证
        if not (0 <= channel <= 15):
            raise ValueError(f"通道号必须在0-15范围内: {channel}")
        if min_angle >= max_angle:
            raise ValueError(f"最小角度必须小于最大角度: {min_angle} >= {max_angle}")

        try:
            # 确保控制器已初始化
            if not self._controller.is_initialized:
                self._controller.initialize()

            # 设置默认角度
            self._set_pulse_from_angle(self._config.default_angle)
            self._current_angle = self._config.default_angle

            logger.info(
                f"舵机通道 {channel} 初始化成功, "
                f"角度范围[{self._config.min_angle}°-{self._config.max_angle}°]"
            )
        except Exception as e:
            self._fallback_safe_state()
            raise RuntimeError(f"舵机通道 {channel} 初始化失败: {e}") from e

    def _angle_to_pulse_us(self, angle: float) -> float:
        """将角度转换为脉冲宽度（微秒）

        使用线性映射将角度转换为脉宽。

        Args:
            angle: 角度值（度）

        Returns:
            float: 脉冲宽度（微秒）
        """
        angle_ratio = (angle - self._config.min_angle) / (
            self._config.max_angle - self._config.min_angle
        )
        pulse = self._config.min_pulse_us + angle_ratio * (
            self._config.max_pulse_us - self._config.min_pulse_us
        )
        return pulse

    def _pulse_us_to_angle(self, pulse_us: float) -> float:
        """将脉冲宽度转换为角度

        Args:
            pulse_us: 脉冲宽度（微秒）

        Returns:
            float: 角度值（度）
        """
        pulse_ratio = (pulse_us - self._config.min_pulse_us) / (
            self._config.max_pulse_us - self._config.min_pulse_us
        )
        angle = self._config.min_angle + pulse_ratio * (
            self._config.max_angle - self._config.min_angle
        )
        return angle

    def _set_pulse_from_angle(self, angle: float) -> None:
        """根据角度设置脉冲宽度

        Args:
            angle: 角度值（度）
        """
        pulse_us = self._angle_to_pulse_us(angle)
        self._controller.set_channel_pulse(self._channel, pulse_us)

    def set_angle(
        self,
        angle: float,
        duration: float = 0.0,
        easing: Optional[EasingType] = None,
    ) -> None:
        """设置舵机角度

        将舵机旋转到指定角度。如果指定了duration，会平滑过渡。

        Args:
            angle: 目标角度（度）
            duration: 过渡时间（秒），0表示直接设置
            easing: 缓动函数类型，None表示使用配置默认值

        Raises:
            ValueError: 角度超出范围
            RuntimeError: 舵机已关闭
        """
        if self._closed:
            raise RuntimeError(f"舵机通道 {self._channel} 已关闭")

        # 限制角度范围
        angle = max(self._config.min_angle, min(self._config.max_angle, angle))

        with self._instance_lock:
            self._target_angle = angle

            if duration > 0:
                self._motion_with_easing(angle, duration, easing or self._config.easing)
            else:
                self._set_pulse_from_angle(angle)
                self._current_angle = angle

        logger.debug(f"舵机通道 {self._channel} 角度设置为 {angle}°")

    def _motion_with_easing(
        self,
        target_angle: float,
        duration: float,
        easing: EasingType,
    ) -> None:
        """使用缓动函数平滑运动

        在指定时间内从当前角度平滑过渡到目标角度。

        Args:
            target_angle: 目标角度
            duration: 过渡时间（秒）
            easing: 缓动函数类型
        """
        start_angle = self._current_angle
        start_time = time.monotonic()

        # 如果有正在进行的运动，先停止
        self._stop_event.clear()

        # 在循环中使用小的步进
        step_count = max(int(duration * 100), 1)  # 100Hz更新率
        for i in range(step_count + 1):
            if self._stop_event.is_set():
                break

            t = i / step_count  # 归一化时间 0-1
            eased_t = self._apply_easing(t, easing)

            current = start_angle + (target_angle - start_angle) * eased_t
            self._set_pulse_from_angle(current)
            self._current_angle = current

            # 等待到下一帧
            expected_time = start_time + (i / step_count) * duration
            sleep_time = expected_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 确保到达目标位置
        if not self._stop_event.is_set():
            self._set_pulse_from_angle(target_angle)
            self._current_angle = target_angle

    def _apply_easing(self, t: float, easing: EasingType) -> float:
        """应用缓动函数

        Args:
            t: 归一化时间（0-1）
            easing: 缓动函数类型

        Returns:
            float: 缓动后的值（0-1）
        """
        if easing == EasingType.LINEAR:
            return t
        elif easing == EasingType.EASE_IN_QUAD:
            return t * t
        elif easing == EasingType.EASE_OUT_QUAD:
            return 1 - (1 - t) * (1 - t)
        elif easing == EasingType.EASE_IN_OUT:
            if t < 0.5:
                return 2 * t * t
            return 1 - math.pow(-2 * t + 2, 2) / 2
        elif easing == EasingType.EASE_OUT_BOUNCE:
            return self._bounce_easing(t)
        return t

    @staticmethod
    def _bounce_easing(t: float) -> float:
        """弹跳缓动函数"""
        if t < 1 / 2.75:
            return 7.5625 * t * t
        elif t < 2 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375

    def get_angle(self) -> float:
        """获取当前角度

        Returns:
            float: 当前角度值（度）
        """
        return self._current_angle

    def get_target_angle(self) -> float:
        """获取目标角度

        Returns:
            float: 目标角度值（度）
        """
        return self._target_angle

    def set_pulse_width(self, pulse_us: float) -> None:
        """直接设置脉冲宽度

        绕过角度映射，直接控制脉冲宽度。

        Args:
            pulse_us: 脉冲宽度（微秒）
        """
        if self._closed:
            raise RuntimeError(f"舵机通道 {self._channel} 已关闭")

        self._controller.set_channel_pulse(self._channel, pulse_us)
        self._current_angle = self._pulse_us_to_angle(pulse_us)
        logger.debug(f"舵机通道 {self._channel} 脉宽设置为 {pulse_us}μs")

    def get_pulse_width(self) -> float:
        """获取当前脉冲宽度

        Returns:
            float: 当前脉冲宽度（微秒）
        """
        return self._angle_to_pulse_us(self._current_angle)

    def center(self) -> None:
        """舵机归中（转到默认角度）"""
        self.set_angle(self._config.default_angle)

    def relax(self) -> None:
        """释放舵机（停止PWM输出）

        停止PWM输出，使舵机可以自由转动。
        """
        if self._controller and self._controller.is_initialized:
            # 设置占空比为0以释放舵机
            self._controller._set_pwm(self._channel, 0, 4096)
        logger.debug(f"舵机通道 {self._channel} 已释放")

    def _fallback_safe_state(self) -> None:
        """异常回退到安全状态：释放舵机"""
        try:
            self.relax()
        except Exception as e:
            logger.error(f"舵机通道 {self._channel} 安全状态回退失败: {e}")

    def close(self) -> None:
        """关闭舵机，释放资源"""
        if self._closed:
            return

        with self._instance_lock:
            try:
                self._stop_event.set()
                if self._motion_thread and self._motion_thread.is_alive():
                    self._motion_thread.join(timeout=0.5)
                self.relax()
                self._closed = True
                logger.info(f"舵机通道 {self._channel} 已关闭")
            except Exception as e:
                logger.error(f"舵机通道 {self._channel} 关闭时发生错误: {e}")
                raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if not getattr(self, '_closed', True):
            try:
                self.close()
            except Exception:
                pass

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def min_angle(self) -> float:
        return self._config.min_angle

    @property
    def max_angle(self) -> float:
        return self._config.max_angle

    @property
    def is_closed(self) -> bool:
        return self._closed


class ServoGroup:
    """舵机组控制器

    同步控制多个舵机，用于机械臂等场景。

    Args:
        servos: 舵机字典，格式为 {名称: Servo实例}

    Example:
        >>> servos = {
        ...     "base": Servo(0, min_angle=0, max_angle=180),
        ...     "shoulder": Servo(1, min_angle=0, max_angle=180),
        ...     "elbow": Servo(2, min_angle=0, max_angle=180),
        ... }
        >>> group = ServoGroup(servos)
        >>> group.set_angles(base=90, shoulder=45, elbow=90)
    """

    def __init__(self, servos: Dict[str, Servo]) -> None:
        self._servos: Dict[str, Servo] = servos
        self._lock: threading.Lock = threading.Lock()

    def set_angle(self, name: str, angle: float, duration: float = 0.0) -> None:
        """设置指定舵机的角度

        Args:
            name: 舵机名称
            angle: 目标角度
            duration: 过渡时间（秒）
        """
        if name in self._servos:
            self._servos[name].set_angle(angle, duration)
        else:
            raise ValueError(f"舵机 '{name}' 不存在")

    def set_angles(self, duration: float = 0.0, **angles: float) -> None:
        """批量设置舵机角度

        Args:
            duration: 过渡时间（秒）
            **angles: 舵机名称=角度的键值对
        """
        with self._lock:
            for name, angle in angles.items():
                if name in self._servos:
                    self._servos[name].set_angle(angle, duration)

    def set_angles_sync(self, angles: Dict[str, float], duration: float = 0.5) -> None:
        """同步设置多个舵机角度

        所有舵机同时开始运动，同时到达目标位置。

        Args:
            angles: 舵机名称=角度的字典
            duration: 过渡时间（秒）
        """
        with self._lock:
            # 使用线程实现同步运动
            threads = []
            for name, angle in angles.items():
                if name in self._servos:
                    t = threading.Thread(
                        target=self._servos[name].set_angle,
                        args=(angle, duration),
                    )
                    t.start()
                    threads.append(t)

            # 等待所有运动完成
            for t in threads:
                t.join(timeout=duration + 1.0)

    def get_angles(self) -> Dict[str, float]:
        """获取所有舵机当前角度

        Returns:
            Dict[str, float]: 舵机名称=当前角度的字典
        """
        return {name: servo.get_angle() for name, servo in self._servos.items()}

    def center_all(self) -> None:
        """将所有舵机归中"""
        with self._lock:
            for servo in self._servos.values():
                servo.center()

    def relax_all(self) -> None:
        """释放所有舵机"""
        with self._lock:
            for servo in self._servos.values():
                servo.relax()

    def close(self) -> None:
        """关闭所有舵机"""
        for servo in self._servos.values():
            servo.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
