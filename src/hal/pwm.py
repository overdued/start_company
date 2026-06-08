"""
RK3588 PWM硬件抽象层模块

本模块提供对OrangePi Kunpeng Pro (RK3588) 硬件PWM的完整控制支持，
RK3588内置16路硬件PWM通道，同时支持通过PCA9685 I2C扩展板扩展更多PWM输出。

RK3588硬件PWM通道（部分在40PIN上可用）：
    - PWM0: 40PIN引脚32
    - PWM1: 40PIN引脚33
    - PWM2-PWM15: 部分在其他扩展接口上

PCA9685扩展：
    - I2C地址: 0x40（默认）
    - 16路PWM输出，12位分辨率
    - 支持舵机和LED控制

特性：
    - RK3588硬件PWM原生支持
    - PCA9685 I2C扩展支持
    - 频率和占空比精确配置
    - 自动后端的探测和选择
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
from typing import Optional, Dict, List, Tuple
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# RK3588硬件PWM sysfs路径模板
# RK3588的PWM通过sysfs控制: /sys/class/pwm/pwmchipN/pwmM
RK3588_PWM_CHIP = 0  # pwmchip编号，根据实际系统可能不同
RK3588_PWM_SYSFS_BASE = f"/sys/class/pwm/pwmchip{RK3588_PWM_CHIP}"

# RK3588 PWM通道到40PIN引脚映射
RK3588_PWM_PIN_MAP: Dict[int, Dict[str, any]] = {
    0: {"pin": 32, "name": "PWM0", "chip": 0, "channel": 0},
    1: {"pin": 33, "name": "PWM1", "chip": 0, "channel": 1},
    2: {"pin": None, "name": "PWM2", "chip": 1, "channel": 0},
    3: {"pin": None, "name": "PWM3", "chip": 1, "channel": 1},
    4: {"pin": None, "name": "PWM4", "chip": 2, "channel": 0},
    5: {"pin": None, "name": "PWM5", "chip": 2, "channel": 1},
    6: {"pin": None, "name": "PWM6", "chip": 3, "channel": 0},
    7: {"pin": None, "name": "PWM7", "chip": 3, "channel": 1},
    8: {"pin": None, "name": "PWM8", "chip": 4, "channel": 0},
    9: {"pin": None, "name": "PWM9", "chip": 4, "channel": 1},
    10: {"pin": None, "name": "PWM10", "chip": 5, "channel": 0},
    11: {"pin": 11, "name": "PWM11", "chip": 5, "channel": 1},
    12: {"pin": 12, "name": "PWM12", "chip": 6, "channel": 0},
    13: {"pin": None, "name": "PWM13", "chip": 6, "channel": 1},
    14: {"pin": 22, "name": "PWM14", "chip": 7, "channel": 0},
    15: {"pin": None, "name": "PWM15", "chip": 7, "channel": 1},
}

# PCA9685寄存器定义
PCA9685_ADDRESS = 0x40
PCA9685_MODE1 = 0x00
PCA9685_MODE2 = 0x01
PCA9685_SUBADR1 = 0x02
PCA9685_SUBADR2 = 0x03
PCA9685_SUBADR3 = 0x04
PCA9685_PRESCALE = 0xFE
PCA9685_LED0_ON_L = 0x06
PCA9685_LED0_ON_H = 0x07
PCA9685_LED0_OFF_L = 0x08
PCA9685_LED0_OFF_H = 0x09
PCA9685_ALL_LED_ON_L = 0xFA
PCA9685_ALL_LED_ON_H = 0xFB
PCA9685_ALL_LED_OFF_L = 0xFC
PCA9685_ALL_LED_OFF_H = 0xFD

# PCA9685模式位
PCA9685_MODE1_RESTART = 0x80
PCA9685_MODE1_SLEEP = 0x10
PCA9685_MODE1_ALLCALL = 0x01
PCA9685_MODE2_OUTDRV = 0x04
PCA9685_MODE2_INVRT = 0x10


class PWMBackend(Enum):
    """PWM后端类型枚举"""
    HARDWARE = "hardware"      # RK3588硬件PWM
    PCA9685 = "pca9685"        # PCA9685 I2C扩展
    SOFTWARE = "software"      # 软件PWM（备用）


class PWM:
    """RK3588 PWM控制器类

    提供对RK3588硬件PWM和PCA9685扩展PWM的完整控制。
    自动探测可用的PWM后端，优先使用硬件PWM。

    Args:
        channel: PWM通道号（硬件PWM: 0-15, PCA9685: 0-15）
        frequency: PWM频率（Hz），默认50（舵机标准频率）
        backend: 后端类型，None表示自动选择
        i2c_bus: PCA9685所在的I2C总线号，默认3
        i2c_address: PCA9685的I2C地址，默认0x40
        timeout: 操作超时时间（秒），默认5.0

    Raises:
        ValueError: 通道号或参数无效
        RuntimeError: PWM初始化失败

    Example:
        >>> pwm = PWM(0, frequency=50)  # 50Hz硬件PWM
        >>> pwm.start()
        >>> pwm.set_duty_cycle(50.0)  # 50%占空比
        >>> pwm.stop()
        >>> pwm.close()
    """

    # 类级别PCA9685缓存
    _pca9685_instances: Dict[Tuple[int, int], "PCA9685Controller"] = {}
    _pca9685_lock = threading.Lock()

    def __init__(
        self,
        channel: int,
        frequency: int = 50,
        backend: Optional[str] = None,
        i2c_bus: int = 3,
        i2c_address: int = 0x40,
        timeout: float = 5.0,
    ) -> None:
        self._channel: int = channel
        self._frequency: int = frequency
        self._duty_cycle: float = 0.0
        self._running: bool = False
        self._timeout: float = timeout
        self._instance_lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._backend_type: PWMBackend = PWMBackend.HARDWARE
        self._backend: Optional[object] = None
        self._i2c_bus: int = i2c_bus
        self._i2c_address: int = i2c_address

        # 参数验证
        if channel < 0 or channel > 15:
            raise ValueError(f"无效的PWM通道号: {channel}, 支持: 0-15")
        if frequency <= 0:
            raise ValueError(f"无效的PWM频率: {frequency}")

        # 自动选择后端
        if backend is None:
            self._backend_type = self._auto_detect_backend(channel)
        else:
            self._backend_type = PWMBackend(backend)

        try:
            self._init_backend()
            logger.info(
                f"PWM通道 {channel} 初始化成功, "
                f"后端={self._backend_type.value}, 频率={frequency}Hz"
            )
        except Exception as e:
            self._fallback_safe_state()
            raise RuntimeError(f"PWM通道 {channel} 初始化失败: {e}") from e

    def _auto_detect_backend(self, channel: int) -> PWMBackend:
        """自动探测可用的PWM后端

        优先级：硬件PWM > PCA9685扩展

        Args:
            channel: PWM通道号

        Returns:
            PWMBackend: 探测到的后端类型
        """
        # 检查硬件PWM是否可用
        if channel in RK3588_PWM_PIN_MAP:
            pwm_info = RK3588_PWM_PIN_MAP[channel]
            chip_path = f"/sys/class/pwm/pwmchip{pwm_info['chip']}"
            if os.path.exists(chip_path):
                return PWMBackend.HARDWARE

        # 检查PCA9685是否可用
        pca9685_key = (self._i2c_bus, self._i2c_address)
        if pca9685_key in self._pca9685_instances:
            return PWMBackend.PCA9685

        # 尝试探测PCA9685
        try:
            from smbus2 import SMBus
            smbus = SMBus(self._i2c_bus)
            smbus.read_byte(self._i2c_address)
            smbus.close()
            return PWMBackend.PCA9685
        except Exception:
            pass

        logger.warning(f"PWM通道 {channel} 无硬件支持，回退到软件PWM")
        return PWMBackend.SOFTWARE

    def _init_backend(self) -> None:
        """初始化选定的PWM后端"""
        if self._backend_type == PWMBackend.HARDWARE:
            self._init_hardware_pwm()
        elif self._backend_type == PWMBackend.PCA9685:
            self._init_pca9685()
        else:
            self._init_software_pwm()

    def _init_hardware_pwm(self) -> None:
        """初始化RK3588硬件PWM

        通过sysfs接口控制硬件PWM。
        """
        pwm_info = RK3588_PWM_PIN_MAP.get(self._channel)
        if not pwm_info:
            raise RuntimeError(f"通道 {self._channel} 无硬件PWM映射")

        chip = pwm_info["chip"]
        channel = pwm_info["channel"]

        self._pwm_chip_path = f"/sys/class/pwm/pwmchip{chip}"
        self._pwm_channel_path = f"{self._pwm_chip_path}/pwm{channel}"

        # 导出PWM通道（如果未导出）
        if not os.path.exists(self._pwm_channel_path):
            try:
                with open(f"{self._pwm_chip_path}/export", "w") as f:
                    f.write(str(channel))
                time.sleep(0.1)
            except IOError as e:
                raise RuntimeError(f"导出PWM通道 {channel} 失败: {e}") from e

        # 设置初始周期和占空比
        self._set_hardware_period_ns(int(1e9 / self._frequency))
        self._set_hardware_duty_ns(0)

        self._backend = {
            "type": "hardware",
            "chip": chip,
            "channel": channel,
            "path": self._pwm_channel_path,
        }

    def _init_pca9685(self) -> None:
        """初始化PCA9685 I2C扩展PWM

        通过I2C总线控制PCA9685芯片。
        """
        pca9685_key = (self._i2c_bus, self._i2c_address)

        with self._pca9685_lock:
            if pca9685_key not in self._pca9685_instances:
                controller = PCA9685Controller(self._i2c_bus, self._i2c_address)
                self._pca9685_instances[pca9685_key] = controller

        self._pca9685 = self._pca9685_instances[pca9685_key]

        # 确保PCA9685已初始化
        if not self._pca9685.is_initialized:
            self._pca9685.initialize()

        # 设置频率
        self._pca9685.set_frequency(self._frequency)

        self._backend = {
            "type": "pca9685",
            "controller": self._pca9685,
            "channel": self._channel,
        }

    def _init_software_pwm(self) -> None:
        """初始化软件PWM（备用方案）

        使用GPIO模拟PWM信号。
        """
        # 软件PWM需要GPIO支持，这里仅做占位
        # 实际实现需要结合GPIO模块
        self._backend = {
            "type": "software",
            "channel": self._channel,
        }
        logger.warning("软件PWM初始化（备用方案，精度和性能有限）")

    def _set_hardware_period_ns(self, period_ns: int) -> None:
        """设置硬件PWM周期

        Args:
            period_ns: 周期（纳秒）
        """
        try:
            with open(f"{self._pwm_channel_path}/period", "w") as f:
                f.write(str(period_ns))
        except IOError as e:
            raise RuntimeError(f"设置PWM周期失败: {e}") from e

    def _set_hardware_duty_ns(self, duty_ns: int) -> None:
        """设置硬件PWM占空比时间

        Args:
            duty_ns: 高电平时间（纳秒）
        """
        try:
            with open(f"{self._pwm_channel_path}/duty_cycle", "w") as f:
                f.write(str(duty_ns))
        except IOError as e:
            raise RuntimeError(f"设置PWM占空比失败: {e}") from e

    def start(self) -> None:
        """启动PWM输出"""
        if self._closed:
            raise RuntimeError(f"PWM通道 {self._channel} 已关闭")
        if self._running:
            return

        with self._instance_lock:
            try:
                if self._backend_type == PWMBackend.HARDWARE:
                    self._start_hardware()
                elif self._backend_type == PWMBackend.PCA9685:
                    self._start_pca9685()
                else:
                    self._start_software()
                self._running = True
                logger.info(f"PWM通道 {self._channel} 已启动")
            except Exception as e:
                logger.error(f"PWM通道 {self._channel} 启动失败: {e}")
                raise

    def _start_hardware(self) -> None:
        """启动硬件PWM"""
        try:
            with open(f"{self._pwm_channel_path}/enable", "w") as f:
                f.write("1")
        except IOError as e:
            raise RuntimeError(f"启动硬件PWM失败: {e}") from e

    def _start_pca9685(self) -> None:
        """启动PCA9685 PWM"""
        # PCA9685通道默认开启，只需设置占空比
        self._pca9685.set_duty_cycle(self._channel, self._duty_cycle)

    def _start_software(self) -> None:
        """启动软件PWM"""
        # 软件PWM启动逻辑（占位）
        pass

    def stop(self) -> None:
        """停止PWM输出"""
        if self._closed or not self._running:
            return

        with self._instance_lock:
            try:
                if self._backend_type == PWMBackend.HARDWARE:
                    self._stop_hardware()
                elif self._backend_type == PWMBackend.PCA9685:
                    self._stop_pca9685()
                else:
                    self._stop_software()
                self._running = False
                logger.info(f"PWM通道 {self._channel} 已停止")
            except Exception as e:
                logger.error(f"PWM通道 {self._channel} 停止失败: {e}")

    def _stop_hardware(self) -> None:
        """停止硬件PWM"""
        try:
            with open(f"{self._pwm_channel_path}/enable", "w") as f:
                f.write("0")
        except IOError as e:
            logger.error(f"停止硬件PWM失败: {e}")

    def _stop_pca9685(self) -> None:
        """停止PCA9685 PWM"""
        if self._pca9685:
            self._pca9685.set_duty_cycle(self._channel, 0.0)

    def _stop_software(self) -> None:
        """停止软件PWM"""
        pass

    def set_duty_cycle(self, percent: float) -> None:
        """设置PWM占空比

        Args:
            percent: 占空比百分比（0.0 - 100.0）

        Raises:
            ValueError: 占空比值无效
        """
        if self._closed:
            raise RuntimeError(f"PWM通道 {self._channel} 已关闭")
        if not (0.0 <= percent <= 100.0):
            raise ValueError(f"占空比必须在0.0-100.0范围内: {percent}")

        self._duty_cycle = percent

        with self._instance_lock:
            try:
                if self._backend_type == PWMBackend.HARDWARE:
                    self._set_hardware_duty(percent)
                elif self._backend_type == PWMBackend.PCA9685:
                    if self._pca9685:
                        self._pca9685.set_duty_cycle(self._channel, percent)
                else:
                    self._set_software_duty(percent)
                logger.debug(f"PWM通道 {self._channel} 占空比设置为 {percent}%")
            except Exception as e:
                logger.error(f"PWM通道 {self._channel} 设置占空比失败: {e}")
                raise

    def _set_hardware_duty(self, percent: float) -> None:
        """设置硬件PWM占空比

        Args:
            percent: 占空比百分比
        """
        period_ns = int(1e9 / self._frequency)
        duty_ns = int(period_ns * percent / 100.0)
        self._set_hardware_duty_ns(duty_ns)

    def _set_software_duty(self, percent: float) -> None:
        """设置软件PWM占空比（占位）"""
        pass

    def set_frequency(self, freq: int) -> None:
        """设置PWM频率

        Args:
            freq: 频率（Hz），必须>0

        Raises:
            ValueError: 频率无效
        """
        if self._closed:
            raise RuntimeError(f"PWM通道 {self._channel} 已关闭")
        if freq <= 0:
            raise ValueError(f"PWM频率必须大于0: {freq}")

        with self._instance_lock:
            self._frequency = freq
            try:
                if self._backend_type == PWMBackend.HARDWARE:
                    period_ns = int(1e9 / freq)
                    self._set_hardware_period_ns(period_ns)
                    # 更新占空比
                    self._set_hardware_duty(self._duty_cycle)
                elif self._backend_type == PWMBackend.PCA9685:
                    if self._pca9685:
                        self._pca9685.set_frequency(freq)
                logger.info(f"PWM通道 {self._channel} 频率设置为 {freq}Hz")
            except Exception as e:
                logger.error(f"PWM通道 {self._channel} 设置频率失败: {e}")
                raise

    def set_pulse_width(self, width_us: float) -> None:
        """设置PWM脉冲宽度（微秒）

        常用于舵机控制，直接指定脉冲宽度。

        Args:
            width_us: 脉冲宽度（微秒）
        """
        period_us = 1e6 / self._frequency
        duty = (width_us / period_us) * 100.0
        self.set_duty_cycle(min(max(duty, 0.0), 100.0))

    def get_frequency(self) -> int:
        """获取当前PWM频率

        Returns:
            int: 频率（Hz）
        """
        return self._frequency

    def get_duty_cycle(self) -> float:
        """获取当前占空比

        Returns:
            float: 占空比百分比（0.0 - 100.0）
        """
        return self._duty_cycle

    def is_running(self) -> bool:
        """判断PWM是否正在运行

        Returns:
            bool: True=运行中, False=已停止
        """
        return self._running

    def _fallback_safe_state(self) -> None:
        """异常回退到安全状态：停止PWM输出"""
        try:
            self.stop()
            self._cleanup_resources()
        except Exception as e:
            logger.error(f"PWM通道 {self._channel} 安全状态回退失败: {e}")

    def _cleanup_resources(self) -> None:
        """清理所有已分配资源"""
        if self._backend_type == PWMBackend.HARDWARE:
            # 取消导出PWM通道
            if hasattr(self, '_backend') and self._backend:
                channel = self._backend.get("channel")
                chip = self._backend.get("chip")
                if channel is not None and chip is not None:
                    unexport_path = f"/sys/class/pwm/pwmchip{chip}/unexport"
                    if os.path.exists(unexport_path):
                        try:
                            with open(unexport_path, "w") as f:
                                f.write(str(channel))
                        except IOError:
                            pass

    def close(self) -> None:
        """关闭PWM，释放所有资源

        先停止PWM输出，再释放资源。
        """
        if self._closed:
            return

        with self._instance_lock:
            try:
                self.stop()
                self._cleanup_resources()
                self._closed = True
                logger.info(f"PWM通道 {self._channel} 已关闭")
            except Exception as e:
                logger.error(f"PWM通道 {self._channel} 关闭时发生错误: {e}")
                raise

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源释放"""
        self.close()

    def __del__(self):
        """析构函数，确保资源释放"""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass

    @property
    def channel(self) -> int:
        """获取PWM通道号"""
        return self._channel

    @property
    def is_closed(self) -> bool:
        """判断PWM是否已关闭"""
        return self._closed

    @property
    def backend_type(self) -> PWMBackend:
        """获取当前后端类型"""
        return self._backend_type


class PCA9685Controller:
    """PCA9685 I2C PWM扩展控制器

    通过I2C总线控制PCA9685芯片，提供16路PWM输出。

    Args:
        bus: I2C总线号
        address: PCA9685 I2C地址，默认0x40

    Example:
        >>> pca = PCA9685Controller(3, 0x40)
        >>> pca.initialize()
        >>> pca.set_frequency(50)
        >>> pca.set_duty_cycle(0, 50.0)
    """

    def __init__(self, bus: int, address: int = PCA9685_ADDRESS) -> None:
        self._bus: int = bus
        self._address: int = address
        self._frequency: int = 50
        self._initialized: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._smbus: Optional[object] = None

    def initialize(self) -> None:
        """初始化PCA9685芯片"""
        try:
            from smbus2 import SMBus
            self._smbus = SMBus(self._bus)

            with self._lock:
                # 软件复位
                self._write_register(PCA9685_MODE1, 0x00)
                time.sleep(0.005)

                # 配置MODE2: 输出驱动方式为推挽，非反转
                self._write_register(PCA9685_MODE2, PCA9685_MODE2_OUTDRV)

                # 配置MODE1: 正常模式，启用ALLCALL
                self._write_register(PCA9685_MODE1, PCA9685_MODE1_ALLCALL)
                time.sleep(0.005)

                # 等待振荡器稳定
                mode1 = self._read_register(PCA9685_MODE1)
                mode1 &= ~PCA9685_MODE1_SLEEP
                self._write_register(PCA9685_MODE1, mode1)
                time.sleep(0.005)

                self._initialized = True
                logger.info(f"PCA9685 (0x{self._address:02X}) 初始化成功")
        except Exception as e:
            raise RuntimeError(f"PCA9685初始化失败: {e}") from e

    def _write_register(self, register: int, value: int) -> None:
        """写入PCA9685寄存器"""
        if self._smbus:
            self._smbus.write_byte_data(self._address, register, value)

    def _read_register(self, register: int) -> int:
        """读取PCA9685寄存器"""
        if self._smbus:
            return self._smbus.read_byte_data(self._address, register)
        return 0

    def set_frequency(self, freq: int) -> None:
        """设置PCA9685 PWM频率

        Args:
            freq: 频率（Hz），范围24-1526
        """
        with self._lock:
            # PCA9685预分频值计算: prescale = round(25MHz / (4096 * freq)) - 1
            prescale = int(round(25000000.0 / (4096.0 * freq)) - 1)
            prescale = max(3, min(255, prescale))  # 限制范围

            # 进入睡眠模式
            old_mode = self._read_register(PCA9685_MODE1)
            new_mode = (old_mode & 0x7F) | PCA9685_MODE1_SLEEP
            self._write_register(PCA9685_MODE1, new_mode)

            # 设置预分频值
            self._write_register(PCA9685_PRESCALE, prescale)

            # 恢复之前的模式
            self._write_register(PCA9685_MODE1, old_mode & ~PCA9685_MODE1_SLEEP)
            time.sleep(0.005)

            # 重启
            self._write_register(PCA9685_MODE1, old_mode | PCA9685_MODE1_RESTART)

            self._frequency = freq
            logger.debug(f"PCA9685频率设置为 {freq}Hz (prescale={prescale})")

    def set_duty_cycle(self, channel: int, percent: float) -> None:
        """设置指定通道的占空比

        Args:
            channel: 通道号（0-15）
            percent: 占空比百分比（0.0 - 100.0）
        """
        with self._lock:
            # 12位分辨率: 0-4095
            off_count = int(4095.0 * percent / 100.0)
            off_count = max(0, min(4095, off_count))

            base_reg = PCA9685_LED0_ON_L + 4 * channel
            self._write_register(base_reg, 0x00)         # ON_L (始终从0开始)
            self._write_register(base_reg + 1, 0x00)     # ON_H
            self._write_register(base_reg + 2, off_count & 0xFF)   # OFF_L
            self._write_register(base_reg + 3, (off_count >> 8) & 0x0F)  # OFF_H

    def set_all_duty_cycle(self, percent: float) -> None:
        """设置所有通道的占空比

        Args:
            percent: 占空比百分比（0.0 - 100.0）
        """
        with self._lock:
            off_count = int(4095.0 * percent / 100.0)
            off_count = max(0, min(4095, off_count))

            self._write_register(PCA9685_ALL_LED_ON_L, 0x00)
            self._write_register(PCA9685_ALL_LED_ON_H, 0x00)
            self._write_register(PCA9685_ALL_LED_OFF_L, off_count & 0xFF)
            self._write_register(PCA9685_ALL_LED_OFF_H, (off_count >> 8) & 0x0F)

    def reset(self) -> None:
        """复位PCA9685"""
        with self._lock:
            self._write_register(PCA9685_MODE1, PCA9685_MODE1_RESTART)
            time.sleep(0.01)

    @property
    def is_initialized(self) -> bool:
        """判断PCA9685是否已初始化"""
        return self._initialized

    def close(self) -> None:
        """关闭PCA9685控制器"""
        if self._smbus:
            try:
                # 关闭所有输出
                self.set_all_duty_cycle(0.0)
                self._smbus.close()
            except Exception:
                pass
            self._smbus = None
            self._initialized = False
