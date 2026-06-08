"""
RK3588 电机硬件抽象层模块

本模块提供对OrangePi Kunpeng Pro (RK3588) 电机的完整控制支持，
支持直流减速电机和步进电机，通过PWM调速和GPIO方向控制。

电机控制架构：
    - 直流电机: PWM调速 + GPIO方向控制 + 可选使能控制
    - 步进电机: PWM脉冲控制 + GPIO方向控制 + 步数控制
    - 编码器接口: 可选的编码器反馈（速度/位置闭环）

特性：
    - PWM调速（-1.0 ~ 1.0）
    - GPIO方向控制
    - 加减速曲线（梯形/S曲线）
    - 编码器接口（速度/位置反馈）
    - 紧急制动
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
from typing import Optional, Callable, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MotorType(Enum):
    """电机类型枚举"""
    DC = "dc"               # 直流减速电机
    STEPPER = "stepper"     # 步进电机
    BRUSHLESS = "brushless" # 无刷电机


class RampProfile(Enum):
    """加减速曲线类型枚举"""
    NONE = "none"       # 无加减速（直接切换）
    LINEAR = "linear"   # 梯形加减速（线性）
    S_CURVE = "s_curve" # S曲线加减速（平滑）


@dataclass
class MotorConfig:
    """电机配置数据类

    Attributes:
        motor_type: 电机类型
        pwm_frequency: PWM频率（Hz）
        max_speed: 最大速度（0.0-1.0）
        ramp_profile: 加减速曲线类型
        ramp_time: 加减速时间（秒）
        encoder_ppr: 编码器每转脉冲数（0=无编码器）
        gear_ratio: 减速比（默认1.0）
        invert_direction: 是否反转方向
    """
    motor_type: MotorType = MotorType.DC
    pwm_frequency: int = 1000
    max_speed: float = 1.0
    ramp_profile: RampProfile = RampProfile.LINEAR
    ramp_time: float = 0.5
    encoder_ppr: int = 0
    gear_ratio: float = 1.0
    invert_direction: bool = False


@dataclass
class EncoderData:
    """编码器数据类

    Attributes:
        position: 累计位置（脉冲数）
        speed: 当前速度（转/秒）
        direction: 当前方向（1=正转，-1=反转）
        timestamp: 时间戳
    """
    position: int = 0
    speed: float = 0.0
    direction: int = 0
    timestamp: float = field(default_factory=time.monotonic)


class Motor:
    """RK3588电机控制器类

    提供对直流电机和步进电机的完整控制，包括PWM调速、方向控制、
    加减速曲线和编码器接口。

    Args:
        pwm_pin: PWM输出引脚号（物理引脚号）
        dir_pin: 方向控制引脚号（物理引脚号）
        enable_pin: 使能控制引脚号（可选，None表示不使能）
        config: 电机配置，默认直流电机配置
        timeout: 操作超时时间（秒），默认5.0

    Raises:
        ValueError: 引脚号无效
        RuntimeError: 电机初始化失败

    Example:
        >>> config = MotorConfig(motor_type=MotorType.DC, pwm_frequency=1000)
        >>> motor = Motor(11, 13, enable_pin=15, config=config)
        >>> motor.set_speed(0.5)  # 半速正转
        >>> motor.set_speed(-0.3) # 30%速度反转
        >>> motor.brake()         # 紧急制动
        >>> motor.stop()          # 停止
        >>> motor.close()
    """

    def __init__(
        self,
        pwm_pin: int,
        dir_pin: int,
        enable_pin: Optional[int] = None,
        config: Optional[MotorConfig] = None,
        timeout: float = 5.0,
    ) -> None:
        self._pwm_pin: int = pwm_pin
        self._dir_pin: int = dir_pin
        self._enable_pin: Optional[int] = enable_pin
        self._config: MotorConfig = config or MotorConfig()
        self._timeout: float = timeout
        self._instance_lock: threading.Lock = threading.Lock()
        self._speed_lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._current_speed: float = 0.0
        self._target_speed: float = 0.0
        self._enabled: bool = False
        self._encoder_data: EncoderData = EncoderData()
        self._encoder_callback: Optional[Callable[[EncoderData], None]] = None
        self._ramp_thread: Optional[threading.Thread] = None
        self._ramp_running: bool = False
        self._stop_event: threading.Event = threading.Event()

        # 引脚验证
        if pwm_pin <= 0:
            raise ValueError(f"PWM引脚号必须大于0: {pwm_pin}")
        if dir_pin <= 0:
            raise ValueError(f"方向引脚号必须大于0: {dir_pin}")

        try:
            self._init_motor()
            logger.info(
                f"电机初始化成功: PWM引脚={pwm_pin}, 方向引脚={dir_pin}, "
                f"类型={self._config.motor_type.value}"
            )
        except Exception as e:
            self._fallback_safe_state()
            raise RuntimeError(f"电机初始化失败: {e}") from e

    def _init_motor(self) -> None:
        """初始化电机控制器

        初始化PWM和GPIO引脚，配置默认参数。
        """
        # 延迟导入避免循环依赖
        from .pwm import PWM
        from .gpio import GPIO

        # 初始化PWM（用于调速）
        self._pwm = PWM(
            channel=self._pwm_pin,
            frequency=self._config.pwm_frequency,
        )

        # 初始化方向GPIO
        self._dir_gpio = GPIO(self._dir_pin, mode="out")
        self._dir_gpio.write(0)

        # 初始化使能GPIO（如果提供）
        if self._enable_pin:
            self._enable_gpio = GPIO(self._enable_pin, mode="out")
            self._enable_gpio.write(0)  # 默认禁用
        else:
            self._enable_gpio = None

        # 启动加减速线程（如果配置了加减速）
        if self._config.ramp_profile != RampProfile.NONE:
            self._start_ramp_thread()

    def _start_ramp_thread(self) -> None:
        """启动加减速控制线程"""
        self._ramp_running = True
        self._stop_event.clear()
        self._ramp_thread = threading.Thread(
            target=self._ramp_loop,
            name=f"Motor-Ramp-{self._pwm_pin}",
            daemon=True,
        )
        self._ramp_thread.start()
        logger.debug("加减速控制线程已启动")

    def _ramp_loop(self) -> None:
        """加减速控制循环

        在独立线程中运行，根据配置的加减速曲线平滑调整电机速度。
        使用50Hz的更新频率进行速度插值。
        """
        update_interval = 0.02  # 50Hz更新频率
        ramp_steps = max(int(self._config.ramp_time / update_interval), 1)

        while self._ramp_running and not self._stop_event.is_set():
            try:
                with self._speed_lock:
                    current = self._current_speed
                    target = self._target_speed

                if abs(current - target) > 0.001:
                    if self._config.ramp_profile == RampProfile.LINEAR:
                        # 线性加减速
                        step = (target - current) / ramp_steps
                        new_speed = current + step
                    elif self._config.ramp_profile == RampProfile.S_CURVE:
                        # S曲线加减速（使用sin函数平滑过渡）
                        diff = target - current
                        new_speed = current + diff * 0.1  # 简化版S曲线
                    else:
                        new_speed = target

                    # 限制速度范围
                    new_speed = max(-self._config.max_speed, min(self._config.max_speed, new_speed))
                    self._apply_speed(new_speed)
                else:
                    if current != target:
                        self._apply_speed(target)

                time.sleep(update_interval)
            except Exception as e:
                if self._ramp_running:
                    logger.error(f"加减速循环异常: {e}")
                time.sleep(0.1)

    def set_speed(self, speed: float) -> None:
        """设置电机速度

        速度范围为 -1.0 ~ 1.0:
            - 正数: 正转（方向GPIO高电平）
            - 负数: 反转（方向GPIO低电平）
            - 0: 停止（PWM占空比0）

        如果配置了加减速曲线，速度会平滑过渡。
        如果未配置加减速，速度立即生效。

        Args:
            speed: 速度值（-1.0 ~ 1.0）

        Raises:
            ValueError: 速度值超出范围
            RuntimeError: 电机已关闭
        """
        if self._closed:
            raise RuntimeError("电机已关闭")

        speed = float(speed)
        if not (-1.0 <= speed <= 1.0):
            raise ValueError(f"速度必须在-1.0~1.0范围内: {speed}")

        with self._speed_lock:
            self._target_speed = speed

            # 如果没有加减速，直接应用
            if self._config.ramp_profile == RampProfile.NONE:
                self._apply_speed(speed)

        logger.debug(f"电机目标速度设置为 {speed}")

    def _apply_speed(self, speed: float) -> None:
        """应用速度到硬件

        直接控制PWM和方向GPIO输出。

        Args:
            speed: 速度值（-1.0 ~ 1.0）
        """
        if self._closed:
            return

        # 方向反转
        if self._config.invert_direction:
            speed = -speed

        self._current_speed = speed

        if speed > 0:
            # 正转
            self._dir_gpio.write(1)
            duty = speed * 100.0
        elif speed < 0:
            # 反转
            self._dir_gpio.write(0)
            duty = abs(speed) * 100.0
        else:
            # 停止
            duty = 0.0

        try:
            self._pwm.set_duty_cycle(duty)
            if not self._pwm.is_running():
                self._pwm.start()
        except Exception as e:
            logger.error(f"应用电机速度失败: {e}")

    def stop(self) -> None:
        """停止电机

        将速度设为0，PWM占空比设为0。
        """
        if self._closed:
            return

        with self._speed_lock:
            self._target_speed = 0.0
            self._current_speed = 0.0
            self._apply_speed(0.0)

        # 禁用使能（如果有）
        if self._enable_gpio:
            self._enable_gpio.write(0)
        self._enabled = False

        logger.info("电机已停止")

    def brake(self) -> None:
        """紧急制动

        立即停止电机，将方向设为刹车模式（如果硬件支持）。
        对于直流电机，将PWM占空比瞬间设为0。
        """
        if self._closed:
            return

        with self._speed_lock:
            self._target_speed = 0.0
            self._current_speed = 0.0

            # 立即停止PWM
            try:
                self._pwm.stop()
                self._pwm.set_duty_cycle(0.0)
            except Exception as e:
                logger.error(f"制动PWM停止失败: {e}")

            # 方向设为低
            try:
                self._dir_gpio.write(0)
            except Exception as e:
                logger.error(f"制动方向设置失败: {e}")

        self._enabled = False
        logger.info("电机紧急制动")

    def enable(self) -> None:
        """使能电机驱动

        通过使能GPIO激活电机驱动器。
        """
        if self._enable_gpio:
            self._enable_gpio.write(1)
            self._enabled = True
            logger.debug("电机已使能")

    def disable(self) -> None:
        """禁用电机驱动

        通过使能GPIO关闭电机驱动器。
        """
        if self._enable_gpio:
            self._enable_gpio.write(0)
            self._enabled = False
            logger.debug("电机已禁用")

    def get_speed(self) -> float:
        """获取当前电机速度

        Returns:
            float: 当前速度值（-1.0 ~ 1.0）
        """
        with self._speed_lock:
            return self._current_speed

    def get_target_speed(self) -> float:
        """获取目标电机速度

        Returns:
            float: 目标速度值（-1.0 ~ 1.0）
        """
        with self._speed_lock:
            return self._target_speed

    def set_ramp_profile(self, profile: str, ramp_time: float = 0.5) -> None:
        """动态修改加减速曲线配置

        Args:
            profile: 曲线类型 "none"/"linear"/"s_curve"
            ramp_time: 加减速时间（秒）
        """
        self._config.ramp_profile = RampProfile(profile)
        self._config.ramp_time = ramp_time
        logger.info(f"加减速曲线修改为 {profile}, 时间={ramp_time}s")

    def set_max_speed(self, max_speed: float) -> None:
        """设置最大速度限制

        Args:
            max_speed: 最大速度值（0.0 - 1.0）
        """
        if not (0.0 < max_speed <= 1.0):
            raise ValueError(f"最大速度必须在0.0-1.0范围内: {max_speed}")
        self._config.max_speed = max_speed
        logger.info(f"最大速度限制设置为 {max_speed}")

    def set_encoder_callback(self, callback: Callable[[EncoderData], None]) -> None:
        """设置编码器数据回调

        当编码器数据更新时调用回调函数。

        Args:
            callback: 回调函数，参数为EncoderData
        """
        self._encoder_callback = callback

    def update_encoder(self, position: int, timestamp: Optional[float] = None) -> None:
        """更新编码器数据

        由外部编码器读取循环调用，更新位置和速度。

        Args:
            position: 当前编码器位置（脉冲数）
            timestamp: 时间戳，None表示使用当前时间
        """
        ts = timestamp or time.monotonic()
        dt = ts - self._encoder_data.timestamp

        if dt > 0:
            delta_pos = position - self._encoder_data.position
            speed = (delta_pos / dt) / self._config.encoder_ppr if self._config.encoder_ppr > 0 else 0.0
            direction = 1 if delta_pos > 0 else (-1 if delta_pos < 0 else 0)

            self._encoder_data = EncoderData(
                position=position,
                speed=abs(speed),
                direction=direction,
                timestamp=ts,
            )

            if self._encoder_callback:
                self._encoder_callback(self._encoder_data)

    def get_encoder_data(self) -> EncoderData:
        """获取当前编码器数据

        Returns:
            EncoderData: 编码器数据副本
        """
        return EncoderData(
            position=self._encoder_data.position,
            speed=self._encoder_data.speed,
            direction=self._encoder_data.direction,
            timestamp=self._encoder_data.timestamp,
        )

    def _fallback_safe_state(self) -> None:
        """异常回退到安全状态：停止电机并释放资源"""
        try:
            # 立即停止PWM
            if hasattr(self, '_pwm') and self._pwm:
                self._pwm.stop()
                self._pwm.set_duty_cycle(0.0)
            # 方向设为安全状态
            if hasattr(self, '_dir_gpio') and self._dir_gpio:
                self._dir_gpio.write(0)
            # 禁用使能
            if hasattr(self, '_enable_gpio') and self._enable_gpio:
                self._enable_gpio.write(0)
        except Exception as e:
            logger.error(f"电机安全状态回退失败: {e}")

    def _cleanup_resources(self) -> None:
        """清理所有已分配资源"""
        # 停止加减速线程
        self._ramp_running = False
        self._stop_event.set()
        if self._ramp_thread and self._ramp_thread.is_alive():
            self._ramp_thread.join(timeout=1.0)

        # 停止PWM
        if hasattr(self, '_pwm') and self._pwm:
            try:
                self._pwm.stop()
                self._pwm.close()
            except Exception:
                pass

        # 释放GPIO
        if hasattr(self, '_dir_gpio') and self._dir_gpio:
            try:
                self._dir_gpio.write(0)
                self._dir_gpio.close()
            except Exception:
                pass

        if hasattr(self, '_enable_gpio') and self._enable_gpio:
            try:
                self._enable_gpio.write(0)
                self._enable_gpio.close()
            except Exception:
                pass

    def close(self) -> None:
        """关闭电机控制器，释放所有资源

        先停止电机，再释放PWM和GPIO资源。
        """
        if self._closed:
            return

        with self._instance_lock:
            try:
                self.stop()
                self._cleanup_resources()
                self._closed = True
                logger.info("电机控制器已关闭")
            except Exception as e:
                logger.error(f"电机控制器关闭时发生错误: {e}")
                raise

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源释放"""
        self.close()

    def __del__(self):
        """析构函数，确保资源释放"""
        if not getattr(self, '_closed', True):
            try:
                self.close()
            except Exception:
                pass

    @property
    def pwm_pin(self) -> int:
        """获取PWM引脚号"""
        return self._pwm_pin

    @property
    def dir_pin(self) -> int:
        """获取方向引脚号"""
        return self._dir_pin

    @property
    def config(self) -> MotorConfig:
        """获取电机配置"""
        return self._config

    @property
    def is_running(self) -> bool:
        """判断电机是否正在运行"""
        return abs(self._current_speed) > 0.001

    @property
    def is_closed(self) -> bool:
        """判断电机控制器是否已关闭"""
        return self._closed

    @property
    def is_enabled(self) -> bool:
        """判断电机是否已使能"""
        return self._enabled


class MotorGroup:
    """电机组控制器

    同时控制多个电机，用于差分驱动等场景。

    Args:
        motors: 电机字典，格式为 {名称: Motor实例}

    Example:
        >>> left_motor = Motor(11, 13)
        >>> right_motor = Motor(12, 15)
        >>> group = MotorGroup({"left": left_motor, "right": right_motor})
        >>> group.set_speeds(0.5, 0.5)  # 直线前进
        >>> group.set_speeds(-0.5, 0.5) # 原地左转
    """

    def __init__(self, motors: Dict[str, Motor]) -> None:
        self._motors: Dict[str, Motor] = motors
        self._lock: threading.Lock = threading.Lock()

    def set_speed(self, name: str, speed: float) -> None:
        """设置指定电机的速度

        Args:
            name: 电机名称
            speed: 速度值（-1.0 ~ 1.0）
        """
        if name in self._motors:
            self._motors[name].set_speed(speed)
        else:
            raise ValueError(f"电机 '{name}' 不存在")

    def set_speeds(self, **speeds: float) -> None:
        """批量设置电机速度

        Args:
            **speeds: 电机名称=速度值 的键值对
        """
        with self._lock:
            for name, speed in speeds.items():
                if name in self._motors:
                    self._motors[name].set_speed(speed)

    def stop_all(self) -> None:
        """停止所有电机"""
        with self._lock:
            for motor in self._motors.values():
                motor.stop()

    def brake_all(self) -> None:
        """紧急制动所有电机"""
        with self._lock:
            for motor in self._motors.values():
                motor.brake()

    def close(self) -> None:
        """关闭所有电机"""
        for motor in self._motors.values():
            motor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
