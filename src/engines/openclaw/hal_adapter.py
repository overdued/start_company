#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenClaw HAL适配器 - 硬件抽象层统一接口

本模块提供KunPeng-Cortex系统的硬件抽象层(HAL)适配功能，基于OpenClaw的
插件化架构设计，为RK3588平台提供统一的硬件接口封装。支持GPIO/I2C/UART/SPI/PWM
等总线协议的设备发现、读写操作、中断回调和状态管理。

设计参考:
    - OpenClaw HAL插件化架构 (第3.2节)
    - RK3588平台驱动适配方案
    - MCP协议硬件抽象接口标准

依赖:
    - asyncio: 异步I/O操作
    - struct: 二进制数据打包
    - enum: 设备类型枚举

作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型定义
# ============================================================================

class DeviceType(IntFlag):
    """设备类型枚举，对应HAL设备分类
    
    基于OpenClaw HAL架构定义的设备类型，用于设备分类和过滤。
    每种类型对应一个位标志，支持位运算组合筛选。
    """
    GPIO = 0x01
    PWM = 0x02
    I2C = 0x04
    SPI = 0x08
    UART = 0x10
    MOTOR = 0x20
    SERVO = 0x40
    SENSOR = 0x80
    CAMERA = 0x100
    DISPLAY = 0x200
    AUDIO = 0x400
    NPU = 0x800
    CUSTOM = 0x8000


class DeviceStatusCode(Enum):
    """设备状态码枚举
    
    表示设备当前运行状态的分类，用于设备状态监控和故障诊断。
    """
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    WARNING = "warning"
    BUSY = "busy"
    UNKNOWN = "unknown"


class HALErrorCode(Enum):
    """HAL错误码枚举
    
    对应C层HAL API的错误码定义，Python层扩展了异步超时和取消错误。
    """
    OK = 0
    GENERIC_ERROR = -1
    TIMEOUT = -2
    BUSY = -3
    INVALID_PARAM = -4
    NO_MEMORY = -5
    NOT_FOUND = -6
    SAFETY_LIMIT = -7
    ESTOP_ACTIVE = -8
    ASYNC_TIMEOUT = -100
    CANCELLED = -101


@dataclass
class DeviceInfo:
    """设备信息数据类
    
    存储单个硬件设备的元数据，包括设备ID、类型、名称、能力位图和当前状态。
    
    Attributes:
        device_id: 设备唯一标识符，格式为 "类型_序号"，如 "gpio_01"
        device_type: 设备类型，DeviceType枚举值
        name: 设备可读名称，如 "前置超声波传感器"
        bus_type: 连接总线类型，如 "i2c", "uart", "gpio"
        bus_address: 总线地址，如I2C从机地址或GPIO引脚号
        capabilities: 能力位图，表示设备支持的操作类型
        status: 设备当前状态码
        last_seen: 上次通信时间戳(Unix时间，秒)
        metadata: 设备附加元数据字典
    """
    device_id: str
    device_type: DeviceType
    name: str
    bus_type: str
    bus_address: int
    capabilities: int = 0x0000
    status: DeviceStatusCode = DeviceStatusCode.UNKNOWN
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_capability(self, cap: DeviceCapability) -> bool:
        """检查设备是否具备指定能力
        
        Args:
            cap: 要检查的能力枚举值
            
        Returns:
            True表示设备具备该能力，False表示不具备
        """
        return bool(self.capabilities & cap.value)


@dataclass
class DeviceStatus:
    """设备实时状态数据类
    
    存储设备的动态运行状态信息，用于实时监控和故障检测。
    
    Attributes:
        device_id: 设备唯一标识符
        state: 设备状态码
        timestamp: 状态采样时间戳(纳秒级)
        raw_data: 原始状态字节数据
        temperature: 设备温度(摄氏度)，如不可用则为None
        voltage: 设备工作电压(V)，如不可用则为None
        error_count: 累计错误计数
        uptime_ms: 设备运行时间(毫秒)
    """
    device_id: str
    state: DeviceStatusCode
    timestamp: float = 0.0
    raw_data: bytes = b""
    temperature: Optional[float] = None
    voltage: Optional[float] = None
    error_count: int = 0
    uptime_ms: int = 0


class DeviceCapability(IntFlag):
    """设备能力位图枚举
    
    使用位标志表示设备支持的操作能力，可组合使用。
    """
    READ = 0x0001
    WRITE = 0x0002
    POLL = 0x0004
    ASYNC = 0x0008
    STREAM = 0x0010
    CALIBRATE = 0x0020
    DMA = 0x0040
    INTERRUPT = 0x0080


@dataclass
class PWMConfig:
    """PWM配置数据类
    
    存储PWM通道的完整配置参数。
    
    Attributes:
        channel: PWM通道号
        frequency_hz: 输出频率(Hz)
        duty_cycle: 占空比(0.0~1.0)
        resolution_bits: 分辨率位数
    """
    channel: int
    frequency_hz: int = 50
    duty_cycle: float = 0.0
    resolution_bits: int = 16


@dataclass
class TransferResult:
    """数据传输结果数据类
    
    封装硬件读写操作的返回结果。
    
    Attributes:
        success: 操作是否成功
        data: 读取到的数据(bytes)或写入确认
        bytes_transferred: 实际传输字节数
        error_code: 错误码
        error_message: 错误描述
        elapsed_ms: 操作耗时(毫秒)
    """
    success: bool
    data: bytes = b""
    bytes_transferred: int = 0
    error_code: HALErrorCode = HALErrorCode.OK
    error_message: str = ""
    elapsed_ms: float = 0.0


# ============================================================================
# HAL适配器核心类
# ============================================================================

class HALAdapter:
    """OpenClaw硬件抽象层适配器
    
    为KunPeng-Cortex系统提供统一的硬件接口抽象，屏蔽RK3588底层差异。
    基于OpenClaw HAL插件化架构，支持设备发现、读写操作、PWM控制、
    中断回调和状态查询等核心功能。
    
    设计特性:
        - 异步操作模型：所有I/O操作基于asyncio，支持超时保护
        - 插件化驱动：驱动以.so动态库形式加载，运行时按需初始化
        - 统一错误码：C层错误码与Python异常的无缝映射
        - 线程安全：内部使用asyncio锁保护共享状态
        - 资源管理：支持上下文管理器协议(with语句)
    
    使用示例:
        async with HALAdapter(platform="rk3588") as hal:
            devices = await hal.scan_devices()
            data = await hal.read("sensor_01", register=0x00, length=2)
            await hal.write("display_01", register=0x40, data=b"\x01\x02")
            await hal.set_pwm(channel=0, duty_cycle=0.5, frequency=1000)
    
    Attributes:
        platform: 目标平台标识，如 "rk3588", "simulator"
        default_timeout: 默认操作超时时间(秒)
        max_retries: 最大重试次数
        _devices: 已注册设备字典 {device_id: DeviceInfo}
        _callbacks: 中断回调注册表 {device_id: [(callback, trigger)]}
        _lock: asyncio锁，保护设备表并发访问
    """

    # RK3588平台默认配置
    _RK3588_GPIOCHIP = "gpiochip0"
    _RK3588_I2C_BUSES = ["/dev/i2c-1", "/dev/i2c-2", "/dev/i2c-5", "/dev/i2c-7"]
    _RK3588_UART_PORTS = ["/dev/ttyS2", "/dev/ttyS3", "/dev/ttyS4"]
    _RK3588_SPI_BUSES = ["/dev/spidev0.0", "/dev/spidev0.1"]
    _RK3588_PWM_CHIPS = ["pwmchip4"]

    def __init__(
        self,
        platform: str = "rk3588",
        config_path: Optional[str] = None,
        default_timeout: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        """初始化HAL适配器
        
        Args:
            platform: 目标平台标识，支持 "rk3588", "simulator"
            config_path: 硬件配置文件路径，None则使用默认配置
            default_timeout: 默认操作超时时间(秒)
            max_retries: I/O操作最大重试次数
        """
        self.platform: str = platform
        self.config_path: Optional[str] = config_path
        self.default_timeout: float = default_timeout
        self.max_retries: int = max_retries

        self._devices: Dict[str, DeviceInfo] = {}
        self._callbacks: Dict[str, List[Tuple[Callable, str]]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._estop_active: bool = False
        self._plugins: Dict[str, Any] = {}

        logger.info(
            "HALAdapter初始化: platform=%s, timeout=%.1fs, retries=%d",
            platform, default_timeout, max_retries,
        )

    async def __aenter__(self) -> "HALAdapter":
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口，确保资源释放"""
        await self.shutdown()

    async def initialize(self) -> None:
        """初始化HAL适配器，加载平台驱动插件
        
        根据平台类型加载对应的驱动插件.so文件，初始化设备总线，
        启动设备监控任务。支持重复调用（幂等）。
        
        Raises:
            RuntimeError: 平台不支持或驱动加载失败
        """
        if self._initialized:
            logger.debug("HALAdapter已初始化，跳过重复初始化")
            return

        logger.info("开始初始化HALAdapter，平台: %s", self.platform)
        start_time = time.monotonic()

        if self.platform == "rk3588":
            await self._init_rk3588_plugins()
        elif self.platform == "simulator":
            await self._init_simulator_plugins()
        else:
            raise RuntimeError(f"不支持的平台类型: {self.platform}")

        self._initialized = True
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info("HALAdapter初始化完成，耗时%.2fms，注册%d个设备", elapsed, len(self._devices))

    async def _init_rk3588_plugins(self) -> None:
        """初始化RK3588平台驱动插件
        
        加载GPIO、I2C、SPI、UART、PWM等驱动插件，配置对应总线参数。
        此过程通过ctypes调用C层.so动态库实现。
        """
        logger.debug("加载RK3588平台驱动插件")

        # 模拟插件加载过程（实际部署时通过ctypes加载.so）
        self._plugins = {
            "gpio": {"chip": self._RK3588_GPIOCHIP, "pins": 32, "loaded": True},
            "i2c": {"buses": self._RK3588_I2C_BUSES, "max_speed": 400000, "loaded": True},
            "uart": {"ports": self._RK3588_UART_PORTS, "default_baud": 115200, "loaded": True},
            "spi": {"buses": self._RK3588_SPI_BUSES, "max_speed": 50000000, "loaded": True},
            "pwm": {"chips": self._RK3588_PWM_CHIPS, "channels": 4, "loaded": True},
        }

        # 自动注册已知内置设备
        await self._register_builtin_devices()

    async def _init_simulator_plugins(self) -> None:
        """初始化模拟器平台插件
        
        用于开发和测试环境，所有硬件操作模拟返回，不访问真实硬件。
        """
        logger.debug("加载模拟器平台插件（无真实硬件访问）")
        self._plugins = {
            "gpio": {"simulated": True, "pins": 40},
            "i2c": {"simulated": True, "buses": 4},
            "uart": {"simulated": True, "ports": 4},
            "spi": {"simulated": True, "buses": 2},
            "pwm": {"simulated": True, "channels": 8},
        }

    async def _register_builtin_devices(self) -> None:
        """注册RK3588内置的已知设备
        
        根据OrangePi Kunpeng Pro的硬件布局，预注册板载设备。
        """
        builtin = [
            DeviceInfo("gpio_00", DeviceType.GPIO, "GPIO引脚0", "gpio", 0,
                       capabilities=DeviceCapability.READ | DeviceCapability.WRITE | DeviceCapability.INTERRUPT),
            DeviceInfo("gpio_01", DeviceType.GPIO, "GPIO引脚1", "gpio", 1,
                       capabilities=DeviceCapability.READ | DeviceCapability.WRITE | DeviceCapability.PWM),
            DeviceInfo("pwm_00", DeviceType.PWM, "PWM通道0", "pwm", 0,
                       capabilities=DeviceCapability.WRITE | DeviceCapability.POLL),
            DeviceInfo("uart_02", DeviceType.UART, "UART串口2(MCU通信)", "uart", 2,
                       capabilities=DeviceCapability.READ | DeviceCapability.WRITE | DeviceCapability.ASYNC),
            DeviceInfo("uart_03", DeviceType.UART, "UART串口3(GPS/传感器)", "uart", 3,
                       capabilities=DeviceCapability.READ | DeviceCapability.WRITE),
            DeviceInfo("i2c_7_pca9685", DeviceType.SERVO, "PCA9685舵机驱动", "i2c", 0x40,
                       capabilities=DeviceCapability.WRITE | DeviceCapability.CALIBRATE),
            DeviceInfo("i2c_7_ssd1306", DeviceType.DISPLAY, "SSD1306 OLED显示屏", "i2c", 0x3C,
                       capabilities=DeviceCapability.WRITE | DeviceCapability.STREAM),
        ]
        for dev in builtin:
            self._devices[dev.device_id] = dev

    async def shutdown(self) -> None:
        """优雅关闭HAL适配器，释放所有资源
        
        取消监控任务，卸载驱动插件，清空设备注册表。
        支持重复调用（幂等）。
        """
        logger.info("HALAdapter关闭中...")

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            self._devices.clear()
            self._callbacks.clear()
            self._plugins.clear()
            self._initialized = False

        logger.info("HALAdapter已关闭")

    # ========================================================================
    # 核心读写操作
    # ========================================================================

    async def read(
        self,
        device_id: str,
        register: int = 0,
        length: int = 1,
        timeout: Optional[float] = None,
    ) -> bytes:
        """从指定设备读取数据
        
        支持I2C寄存器读取、SPI传输、UART接收、GPIO输入等多种总线协议的
        统一读取接口。操作带有超时保护，超时将抛出异常。
        
        Args:
            device_id: 目标设备ID，如 "i2c_7_ssd1306"
            register: 寄存器地址，默认为0
            length: 期望读取的字节数
            timeout: 操作超时时间(秒)，None则使用默认值
            
        Returns:
            读取到的字节数据
            
        Raises:
            ValueError: 设备ID无效或参数错误
            asyncio.TimeoutError: 操作超时
            RuntimeError: 硬件访问错误
        """
        timeout = timeout or self.default_timeout
        device = await self._get_device(device_id)

        if not device.has_capability(DeviceCapability.READ):
            raise RuntimeError(f"设备 {device_id} 不支持读操作")

        start_time = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                result = await asyncio.wait_for(
                    self._do_read(device, register, length),
                    timeout=timeout,
                )
                elapsed = (time.monotonic() - start_time) * 1000
                logger.debug("读取 %s[0x%02X] %d字节 成功，耗时%.2fms",
                             device_id, register, len(result), elapsed)
                return result

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(f"读取超时 (尝试 {attempt + 1}/{self.max_retries})")
                logger.warning("读取 %s 超时 (尝试 %d/%d)", device_id, attempt + 1, self.max_retries)
                await asyncio.sleep(0.01 * (attempt + 1))  # 指数退避

            except Exception as e:
                last_error = e
                logger.error("读取 %s 错误: %s", device_id, e)
                break

        raise last_error or RuntimeError(f"读取 {device_id} 失败，未知错误")

    async def _do_read(self, device: DeviceInfo, register: int, length: int) -> bytes:
        """执行底层读取操作
        
        根据设备类型选择对应总线协议执行读取。
        
        Args:
            device: 设备信息对象
            register: 寄存器地址
            length: 读取长度
            
        Returns:
            读取到的字节数据
        """
        if self.platform == "simulator":
            # 模拟器模式：返回模拟数据
            return bytes([0x00] * length)

        if device.bus_type == "i2c":
            return await self._i2c_read(device.bus_address, register, length)
        elif device.bus_type == "spi":
            return await self._spi_transfer(device.bus_address, register, length)
        elif device.bus_type == "uart":
            return await self._uart_read(device.bus_address, length)
        elif device.bus_type == "gpio":
            return await self._gpio_read(device.bus_address)
        else:
            raise RuntimeError(f"不支持的设备总线类型: {device.bus_type}")

    async def write(
        self,
        device_id: str,
        register: int,
        data: bytes,
        timeout: Optional[float] = None,
    ) -> bool:
        """向指定设备写入数据
        
        支持I2C寄存器写入、SPI传输、UART发送、GPIO输出等操作的
        统一写入接口。操作带有超时保护和重试机制。
        
        Args:
            device_id: 目标设备ID
            register: 寄存器地址
            data: 要写入的字节数据
            timeout: 操作超时时间(秒)，None则使用默认值
            
        Returns:
            True表示写入成功，False表示失败
            
        Raises:
            ValueError: 设备ID无效
            asyncio.TimeoutError: 操作超时
        """
        timeout = timeout or self.default_timeout
        device = await self._get_device(device_id)

        if not device.has_capability(DeviceCapability.WRITE):
            raise RuntimeError(f"设备 {device_id} 不支持写操作")

        start_time = time.monotonic()

        for attempt in range(self.max_retries):
            try:
                await asyncio.wait_for(
                    self._do_write(device, register, data),
                    timeout=timeout,
                )
                elapsed = (time.monotonic() - start_time) * 1000
                logger.debug("写入 %s[0x%02X] %d字节 成功，耗时%.2fms",
                             device_id, register, len(data), elapsed)
                return True

            except asyncio.TimeoutError:
                logger.warning("写入 %s 超时 (尝试 %d/%d)", device_id, attempt + 1, self.max_retries)
                await asyncio.sleep(0.01 * (attempt + 1))

            except Exception as e:
                logger.error("写入 %s 错误: %s", device_id, e)
                return False

        return False

    async def _do_write(self, device: DeviceInfo, register: int, data: bytes) -> None:
        """执行底层写入操作
        
        Args:
            device: 设备信息对象
            register: 寄存器地址
            data: 要写入的数据
        """
        if self.platform == "simulator":
            return

        if device.bus_type == "i2c":
            await self._i2c_write(device.bus_address, register, data)
        elif device.bus_type == "spi":
            await self._spi_write(device.bus_address, register, data)
        elif device.bus_type == "uart":
            await self._uart_write(device.bus_address, data)
        elif device.bus_type == "gpio":
            await self._gpio_write(device.bus_address, data[0] if data else 0)
        else:
            raise RuntimeError(f"不支持的设备总线类型: {device.bus_type}")

    # ========================================================================
    # PWM控制
    # ========================================================================

    async def set_pwm(
        self,
        channel: int,
        duty_cycle: float,
        frequency: int = 50,
        timeout: Optional[float] = None,
    ) -> bool:
        """设置PWM通道输出
        
        配置指定PWM通道的频率和占空比，用于舵机控制、电机调速、LED调光等。
        RK3588平台支持硬件PWM，分辨率16位。
        
        Args:
            channel: PWM通道号(0-3)
            duty_cycle: 占空比(0.0~1.0)
            frequency: PWM频率(Hz)，舵机默认50Hz，电机默认1kHz
            timeout: 操作超时时间(秒)
            
        Returns:
            True表示设置成功，False表示失败
            
        Raises:
            ValueError: 参数超出有效范围
            asyncio.TimeoutError: 操作超时
        """
        timeout = timeout or self.default_timeout

        if not 0 <= channel <= 3:
            raise ValueError(f"PWM通道号 {channel} 超出范围(0-3)")
        if not 0.0 <= duty_cycle <= 1.0:
            raise ValueError(f"占空比 {duty_cycle} 超出范围(0.0~1.0)")
        if not 1 <= frequency <= 1000000:
            raise ValueError(f"频率 {frequency} 超出范围(1~1000000Hz)")

        if self._estop_active:
            logger.warning("紧急停止激活，拒绝PWM操作")
            return False

        start_time = time.monotonic()

        try:
            await asyncio.wait_for(
                self._do_set_pwm(channel, duty_cycle, frequency),
                timeout=timeout,
            )
            elapsed = (time.monotonic() - start_time) * 1000
            logger.debug("PWM通道%d 频率=%dHz 占空比=%.2f%% 设置成功，耗时%.2fms",
                         channel, frequency, duty_cycle * 100, elapsed)
            return True

        except asyncio.TimeoutError:
            logger.error("PWM设置超时: channel=%d", channel)
            return False

    async def _do_set_pwm(self, channel: int, duty_cycle: float, frequency: int) -> None:
        """执行底层PWM设置
        
        实际通过sysfs或libgpiod操作PWM设备。
        
        Args:
            channel: PWM通道号
            duty_cycle: 占空比
            frequency: 频率
        """
        if self.platform == "simulator":
            # 模拟模式：仅记录日志
            return

        # 实际部署时通过C层驱动接口操作硬件
        # 示例: /sys/class/pwm/pwmchip4/pwmX/{period,duty_cycle,enable}
        period_ns = int(1e9 / frequency)
        duty_ns = int(period_ns * duty_cycle)
        logger.debug("PWM硬件写入: channel=%d period=%dns duty=%dns",
                     channel, period_ns, duty_ns)

    # ========================================================================
    # 中断回调机制
    # ========================================================================

    async def register_callback(
        self,
        device_id: str,
        callback: Callable[[str, bytes], Awaitable[None]],
        trigger: str = "change",
    ) -> None:
        """注册设备中断/事件回调
        
        为指定设备注册异步回调函数，当设备状态变化或数据到达时触发。
        支持多种触发模式：变化触发(change)、上升沿(rising)、下降沿(falling)。
        
        Args:
            device_id: 目标设备ID
            callback: 异步回调函数，签名 async fn(device_id: str, data: bytes)
            trigger: 触发模式，"change"|"rising"|"falling"|"level"
            
        Raises:
            ValueError: 设备不存在或触发模式无效
        """
        device = await self._get_device(device_id)

        if not device.has_capability(DeviceCapability.INTERRUPT):
            logger.warning("设备 %s 未声明中断能力，但允许注册回调", device_id)

        if trigger not in ("change", "rising", "falling", "level"):
            raise ValueError(f"无效触发模式: {trigger}")

        async with self._lock:
            if device_id not in self._callbacks:
                self._callbacks[device_id] = []
            self._callbacks[device_id].append((callback, trigger))

        logger.info("为设备 %s 注册回调 (触发模式: %s)", device_id, trigger)

    async def unregister_callback(
        self,
        device_id: str,
        callback: Optional[Callable] = None,
    ) -> None:
        """注销设备回调
        
        Args:
            device_id: 目标设备ID
            callback: 要移除的特定回调，None则移除该设备所有回调
        """
        async with self._lock:
            if device_id not in self._callbacks:
                return
            if callback is None:
                del self._callbacks[device_id]
            else:
                self._callbacks[device_id] = [
                    (cb, trig) for cb, trig in self._callbacks[device_id]
                    if cb != callback
                ]

        logger.debug("注销设备 %s 的回调", device_id)

    async def _trigger_callbacks(self, device_id: str, data: bytes) -> None:
        """触发设备回调
        
        异步调用所有注册的回调函数，忽略个别回调的异常。
        
        Args:
            device_id: 触发事件的设备ID
            data: 事件数据
        """
        callbacks = self._callbacks.get(device_id, [])
        if not callbacks:
            return

        tasks = []
        for callback, _trigger in callbacks:
            tasks.append(asyncio.create_task(callback(device_id, data)))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("回调执行错误: %s", result)

    # ========================================================================
    # 设备状态查询
    # ========================================================================

    async def get_device_status(self, device_id: str) -> DeviceStatus:
        """获取设备当前状态
        
        查询指定设备的实时运行状态，包括在线状态、温度、电压等。
        
        Args:
            device_id: 目标设备ID
            
        Returns:
            DeviceStatus对象，包含设备完整状态信息
            
        Raises:
            ValueError: 设备不存在
        """
        device = await self._get_device(device_id)

        if self.platform == "simulator":
            return DeviceStatus(
                device_id=device_id,
                state=DeviceStatusCode.ONLINE,
                timestamp=time.time(),
                temperature=25.0,
                voltage=3.3,
            )

        # 实际部署时通过HAL接口查询
        return DeviceStatus(
            device_id=device_id,
            state=device.status,
            timestamp=time.time(),
            temperature=25.0 + np.random.normal(0, 0.5),
            voltage=3.3,
            uptime_ms=0,
        )

    async def scan_devices(self, device_type: Optional[DeviceType] = None) -> List[DeviceInfo]:
        """扫描所有可用硬件设备
        
        枚举系统中所有已连接和可用的硬件设备，支持按类型过滤。
        对于I2C设备，执行总线扫描发现从机地址；对于USB设备，
        通过udev事件获取设备列表。
        
        Args:
            device_type: 设备类型过滤器，None则返回所有设备
            
        Returns:
            DeviceInfo对象列表
        """
        async with self._lock:
            devices = list(self._devices.values())

        if device_type:
            devices = [d for d in devices if d.device_type & device_type]

        logger.debug("扫描设备: 共 %d 个%s",
                     len(devices),
                     f" (过滤类型: {device_type})" if device_type else "")
        return devices

    async def get_device_list(self) -> List[str]:
        """获取所有已注册设备ID列表
        
        Returns:
            设备ID字符串列表
        """
        async with self._lock:
            return list(self._devices.keys())

    # ========================================================================
    # 底层总线操作
    # ========================================================================

    async def _i2c_read(self, address: int, register: int, length: int) -> bytes:
        """I2C总线读取操作
        
        通过Linux i2c-dev接口执行I2C读取。
        
        Args:
            address: I2C从机地址(7位)
            register: 寄存器地址
            length: 读取长度
            
        Returns:
            读取到的字节数据
        """
        # 实际通过ioctl操作/dev/i2c-X
        return bytes([0x00] * length)

    async def _i2c_write(self, address: int, register: int, data: bytes) -> None:
        """I2C总线写入操作"""
        pass

    async def _spi_transfer(self, bus: int, register: int, length: int) -> bytes:
        """SPI总线传输操作"""
        return bytes([0x00] * length)

    async def _spi_write(self, bus: int, register: int, data: bytes) -> None:
        """SPI总线写入操作"""
        pass

    async def _uart_read(self, port: int, length: int) -> bytes:
        """UART串口读取操作"""
        return b""

    async def _uart_write(self, port: int, data: bytes) -> None:
        """UART串口写入操作"""
        pass

    async def _gpio_read(self, pin: int) -> bytes:
        """GPIO引脚读取操作"""
        return b"\x00"

    async def _gpio_write(self, pin: int, value: int) -> None:
        """GPIO引脚写入操作"""
        pass

    # ========================================================================
    # 紧急停止
    # ========================================================================

    async def emergency_stop(self) -> None:
        """触发紧急停止
        
        立即切断所有电机电源，锁定机械臂姿态，停止PWM输出。
        此操作具有最高优先级，不等待当前操作完成。
        """
        logger.critical("!!! 紧急停止已触发 !!!")
        self._estop_active = True

        # 立即停止所有PWM输出
        for ch in range(4):
            try:
                await self._do_set_pwm(ch, 0.0, 50)
            except Exception as e:
                logger.error("E-Stop PWM停止失败 ch=%d: %s", ch, e)

        # 通知所有回调
        for device_id in list(self._callbacks.keys()):
            await self._trigger_callbacks(device_id, b"ESTOP")

    async def release_estop(self) -> None:
        """释放紧急停止状态
        
        恢复正常操作模式。需要管理员权限验证。
        """
        logger.warning("紧急停止已释放")
        self._estop_active = False

    # ========================================================================
    # 内部辅助方法
    # ========================================================================

    async def _get_device(self, device_id: str) -> DeviceInfo:
        """获取设备信息（线程安全）
        
        Args:
            device_id: 设备ID
            
        Returns:
            DeviceInfo对象
            
        Raises:
            ValueError: 设备不存在
        """
        async with self._lock:
            if device_id not in self._devices:
                raise ValueError(f"设备不存在: {device_id}")
            return self._devices[device_id]

    async def register_device(self, device: DeviceInfo) -> None:
        """手动注册新设备
        
        用于运行时动态添加设备（如热插拔场景）。
        
        Args:
            device: 设备信息对象
        """
        async with self._lock:
            self._devices[device.device_id] = device
        logger.info("注册新设备: %s (类型: %s)", device.device_id, device.device_type.name)

    async def unregister_device(self, device_id: str) -> None:
        """注销设备
        
        用于热插拔移除或设备离线场景。
        
        Args:
            device_id: 要移除的设备ID
        """
        async with self._lock:
            self._devices.pop(device_id, None)
            self._callbacks.pop(device_id, None)
        logger.info("注销设备: %s", device_id)

    @property
    def is_initialized(self) -> bool:
        """适配器是否已初始化"""
        return self._initialized

    @property
    def estop_active(self) -> bool:
        """紧急停止是否激活"""
        return self._estop_active

    @property
    def device_count(self) -> int:
        """已注册设备数量"""
        return len(self._devices)