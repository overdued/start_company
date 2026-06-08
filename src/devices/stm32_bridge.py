"""
STM32桥接通信模块

基于UART通信协议的STM32微控制器桥接驱动。
实现命令封装/解析、心跳检测和固件版本查询功能。
适用于OrangePi Kunpeng Pro (RK3588)平台通过UART与STM32通信。

功能特性:
    - 标准化UART通信协议
    - 命令封装与CRC校验
    - 响应解析与错误处理
    - 心跳检测(50ms周期)
    - 固件版本查询
    - 看门狗喂狗
    - 自动重连机制

通信协议帧格式:
    [SOF:0xAA][CMD:1B][LEN:1B][DATA:N][CRC:1B][EOF:0x55]
    - SOF: 帧头 0xAA
    - CMD: 命令字节
    - LEN: 数据长度
    - DATA: 数据负载
    - CRC: 校验和(CRC-8)
    - EOF: 帧尾 0x55

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class STM32Command(Enum):
    """STM32命令码枚举

    定义主控发送到STM32的所有命令类型。
    """
    NOP = 0x00              # 空操作
    HEARTBEAT = 0x01        # 心跳包
    GET_VERSION = 0x02      # 查询固件版本
    GET_STATUS = 0x03       # 查询状态
    SET_GPIO = 0x10         # 设置GPIO
    READ_GPIO = 0x11        # 读取GPIO
    SET_PWM = 0x20          # 设置PWM输出
    READ_ADC = 0x30         # 读取ADC
    SET_DAC = 0x31          # 设置DAC输出
    MOTOR_CONTROL = 0x40    # 电机控制
    ESTOP = 0xFF            # 紧急停止


class STM32ResponseCode(Enum):
    """STM32响应码枚举"""
    OK = 0x00               # 执行成功
    ERR_INVALID_CMD = 0x01  # 无效命令
    ERR_INVALID_PARAM = 0x02 # 无效参数
    ERR_EXECUTION = 0x03    # 执行失败
    ERR_TIMEOUT = 0x04      # 执行超时
    ERR_CRC = 0x05          # CRC校验失败
    ERR_BUSY = 0x06         # 设备忙


@dataclass
class FirmwareVersion:
    """固件版本信息

    属性:
        major: 主版本号
        minor: 次版本号
        patch: 补丁版本号
        build_date: 编译日期字符串
        git_commit: Git提交哈希
    """
    major: int = 0
    minor: int = 0
    patch: int = 0
    build_date: str = ""
    git_commit: str = ""

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class STM32Status:
    """STM32状态信息

    属性:
        mcu_temp: MCU温度(°C)
        uptime_ms: 运行时间(毫秒)
        free_heap: 空闲堆内存(字节)
        cpu_usage: CPU使用率(%)
        estop_active: 紧急停止是否激活
        watchdog_fed: 看门狗是否已喂
        error_count: 累计错误计数
    """
    mcu_temp: float = 0.0
    uptime_ms: int = 0
    free_heap: int = 0
    cpu_usage: float = 0.0
    estop_active: bool = False
    watchdog_fed: bool = False
    error_count: int = 0
    timestamp: float = 0.0


@dataclass
class BridgeConfig:
    """桥接配置参数

    属性:
        port: UART设备路径
        baudrate: 波特率
        timeout: 通信超时(秒)
        heartbeat_interval: 心跳间隔(秒)
        heartbeat_timeout: 心跳超时(秒)
        max_retries: 最大重试次数
        auto_reconnect: 是否自动重连
    """
    port: str = "/dev/ttyS4"
    baudrate: int = 921600
    timeout: float = 1.0
    heartbeat_interval: float = 0.05   # 50ms
    heartbeat_timeout: float = 0.15    # 150ms (3个心跳周期)
    max_retries: int = 3
    auto_reconnect: bool = True


class STM32Bridge:
    """STM32桥接通信驱动类

    提供与STM32微控制器的可靠UART通信,包括命令发送、
    响应解析、心跳检测和自动重连功能。

    示例:
        >>> bridge = STM32Bridge(BridgeConfig(port="/dev/ttyS4"))
        >>> await bridge.initialize()
        >>> version = await bridge.get_firmware_version()
        >>> print(f"STM32固件版本: {version}")
        >>> status = await bridge.get_status()
        >>> await bridge.emergency_stop()
        >>> await bridge.shutdown()

    属性:
        config: 桥接配置
        _connected: 连接状态
        _heartbeat_task: 心跳任务
        _last_heartbeat_time: 上次收到心跳时间
    """

    # 协议常量
    FRAME_SOF: int = 0xAA
    FRAME_EOF: int = 0x55
    FRAME_MIN_LEN: int = 4  # SOF + CMD + LEN + CRC + EOF
    MAX_FRAME_SIZE: int = 256
    MAX_DATA_LEN: int = 250

    # CRC-8多项式: x^8 + x^2 + x + 1 (CRC-8-CCITT)
    CRC8_POLY: int = 0x07

    # 状态
    DEFAULT_TIMEOUT: float = 5.0
    RECONNECT_DELAY: float = 2.0

    def __init__(self, config: BridgeConfig | None = None) -> None:
        """初始化STM32桥接

        参数:
            config: 桥接配置,None则使用默认配置
        """
        self.config: BridgeConfig = config or BridgeConfig()

        # 状态
        self._initialized: bool = False
        self._connected: bool = False
        self._serial: Any = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._buffer: bytearray = bytearray()

        # 心跳
        self._heartbeat_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._last_heartbeat_time: float = 0.0
        self._heartbeat_missed: int = 0
        self._max_missed_heartbeats: int = 3

        # 状态回调
        self._status_callbacks: list[Callable[[STM32Status], None]] = []
        self._disconnect_callbacks: list[Callable[[], None]] = []

        # 最新状态缓存
        self._latest_status: STM32Status = STM32Status()
        self._firmware_version: FirmwareVersion | None = None

    async def initialize(self) -> bool:
        """初始化UART通信

        打开串口,配置波特率,启动心跳检测任务。

        返回:
            bool: 初始化成功返回True
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # 尝试导入pyserial
                try:
                    import serial
                    self._serial = serial.Serial(
                        port=self.config.port,
                        baudrate=self.config.baudrate,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=0.1,  # 非阻塞读取
                    )
                except ImportError:
                    logger.warning("pyserial未安装,使用模拟模式")
                    self._serial = None

                self._initialized = True
                self._connected = True
                self._last_heartbeat_time = time.monotonic()

                # 启动心跳任务
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(), name="stm32_heartbeat"
                )

                logger.info(
                    f"STM32桥接初始化成功: {self.config.port} "
                    f"@ {self.config.baudrate}bps"
                )
                return True

            except Exception as e:
                logger.error(f"STM32桥接初始化失败: {e}")
                return False

    def _crc8(self, data: bytes) -> int:
        """计算CRC-8校验值(内部方法)

        使用CRC-8-CCITT多项式。

        参数:
            data: 待校验数据

        返回:
            int: CRC-8校验值
        """
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ self.CRC8_POLY
                else:
                    crc <<= 1
            crc &= 0xFF
        return crc

    def _build_frame(self, cmd: STM32Command, data: bytes = b"") -> bytes:
        """构建通信帧(内部方法)

        按照协议格式封装命令和数据。

        参数:
            cmd: 命令码
            data: 数据负载

        返回:
            bytes: 完整帧数据
        """
        length = len(data)
        if length > self.MAX_DATA_LEN:
            raise ValueError(f"数据长度超过限制: {length} > {self.MAX_DATA_LEN}")

        frame = bytearray()
        frame.append(self.FRAME_SOF)
        frame.append(cmd.value)
        frame.append(length)
        frame.extend(data)

        # CRC校验(覆盖CMD+LEN+DATA)
        crc = self._crc8(bytes(frame[1:]))
        frame.append(crc)
        frame.append(self.FRAME_EOF)

        return bytes(frame)

    def _parse_frame(self, raw: bytes) -> tuple[STM32ResponseCode, bytes] | None:
        """解析响应帧(内部方法)

        从原始字节流中解析响应。

        参数:
            raw: 原始帧数据

        返回:
            tuple: (响应码, 数据负载) 或 None(解析失败)
        """
        if len(raw) < self.FRAME_MIN_LEN:
            return None

        # 验证帧头帧尾
        if raw[0] != self.FRAME_SOF or raw[-1] != self.FRAME_EOF:
            logger.debug("帧头/帧尾不匹配")
            return None

        # 验证长度
        data_len = raw[2]
        expected_len = 4 + data_len  # SOF+CMD+LEN+CRC+EOF + data
        if len(raw) != expected_len:
            logger.debug(f"帧长度不匹配: 期望={expected_len}, 实际={len(raw)}")
            return None

        # 验证CRC
        calc_crc = self._crc8(raw[1:-2])  # CMD+LEN+DATA
        if calc_crc != raw[-2]:
            logger.debug(f"CRC校验失败: 期望={raw[-2]}, 实际={calc_crc}")
            return None

        # 提取响应码和数据
        resp_code = STM32ResponseCode(raw[1])
        data = raw[3:-2] if data_len > 0 else b""

        return resp_code, data

    async def send_command(
        self,
        cmd: STM32Command,
        data: bytes = b"",
        timeout: float | None = None,
        retries: int | None = None,
    ) -> tuple[STM32ResponseCode, bytes] | None:
        """发送命令并等待响应

        封装命令帧,发送给STM32,并等待响应。
        支持超时保护和自动重试。

        参数:
            cmd: 命令码
            data: 命令数据负载
            timeout: 超时时间(秒)
            retries: 重试次数

        返回:
            tuple: (响应码, 数据) 或 None(通信失败)
        """
        if not self._initialized:
            logger.error("桥接未初始化")
            return None

        timeout = timeout or self.config.timeout
        retries = retries or self.config.max_retries

        frame = self._build_frame(cmd, data)

        for attempt in range(retries):
            try:
                if self._serial is None:
                    # 模拟模式:返回成功响应
                    await asyncio.sleep(0.001)
                    return STM32ResponseCode.OK, b"\x01\x02\x03"

                # 清空接收缓冲区
                self._serial.reset_input_buffer()

                # 发送帧
                self._serial.write(frame)
                self._serial.flush()

                # 等待响应
                response = await asyncio.wait_for(
                    self._read_response(), timeout=timeout
                )

                if response is not None:
                    return response

            except asyncio.TimeoutError:
                logger.warning(f"命令 {cmd.name} 超时 (尝试 {attempt+1})")
            except Exception as e:
                logger.error(f"发送命令异常: {e}")

            if attempt < retries - 1:
                await asyncio.sleep(0.01 * (attempt + 1))

        logger.error(f"命令 {cmd.name} 达到最大重试次数")
        return None

    async def _read_response(
        self,
    ) -> tuple[STM32ResponseCode, bytes] | None:
        """读取响应帧(内部方法)

        从串口读取并解析响应帧。

        返回:
            tuple: (响应码, 数据) 或 None
        """
        if self._serial is None:
            return None

        start_time = time.monotonic()

        while time.monotonic() - start_time < 1.0:
            # 读取可用数据
            available = self._serial.in_waiting
            if available > 0:
                self._buffer.extend(self._serial.read(available))

            # 查找完整帧
            while len(self._buffer) >= self.FRAME_MIN_LEN:
                # 查找SOF
                sof_idx = -1
                for i in range(len(self._buffer)):
                    if self._buffer[i] == self.FRAME_SOF:
                        sof_idx = i
                        break

                if sof_idx < 0:
                    self._buffer.clear()
                    break

                # 检查是否有足够长度
                if len(self._buffer) - sof_idx < 3:
                    break

                data_len = self._buffer[sof_idx + 2]
                frame_len = 4 + data_len

                if len(self._buffer) - sof_idx < frame_len:
                    break

                # 提取帧
                frame = bytes(self._buffer[sof_idx:sof_idx + frame_len])
                self._buffer = self._buffer[sof_idx + frame_len:]

                # 解析帧
                result = self._parse_frame(frame)
                if result is not None:
                    return result

            await asyncio.sleep(0.001)

        return None

    async def _heartbeat_loop(self) -> None:
        """心跳检测循环(内部方法)

        定期发送心跳包,检测STM32是否在线。
        若连续丢失心跳,则触发断开回调。
        """
        logger.debug("心跳检测循环已启动")

        while not self._stop_event.is_set():
            try:
                # 发送心跳
                result = await self.send_command(
                    STM32Command.HEARTBEAT,
                    timeout=self.config.heartbeat_timeout,
                    retries=1,
                )

                if result is not None and result[0] == STM32ResponseCode.OK:
                    self._last_heartbeat_time = time.monotonic()
                    self._heartbeat_missed = 0
                    self._connected = True
                else:
                    self._heartbeat_missed += 1
                    logger.warning(
                        f"心跳丢失: {self._heartbeat_missed}/"
                        f"{self._max_missed_heartbeats}"
                    )

                    if self._heartbeat_missed >= self._max_missed_heartbeats:
                        logger.error("STM32心跳超时,连接已断开")
                        self._connected = False

                        # 通知断开回调
                        for cb in self._disconnect_callbacks:
                            try:
                                cb()
                            except Exception as e:
                                logger.error(f"断开回调异常: {e}")

                        # 尝试重连
                        if self.config.auto_reconnect:
                            await self._attempt_reconnect()

                # 等待下次心跳
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.heartbeat_interval,
                )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳循环异常: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)

        logger.debug("心跳检测循环已退出")

    async def _attempt_reconnect(self) -> None:
        """尝试重新连接(内部方法)

        在连接断开后尝试重新建立通信。
        """
        logger.info("尝试重新连接STM32...")

        try:
            if self._serial and self._serial.is_open:
                self._serial.close()

            import serial
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )

            await asyncio.sleep(1.0)  # 等待STM32启动

            # 测试通信
            result = await self.send_command(STM32Command.GET_VERSION, retries=3)
            if result is not None:
                self._connected = True
                self._heartbeat_missed = 0
                logger.info("STM32重新连接成功")
            else:
                logger.error("STM32重新连接失败")

        except Exception as e:
            logger.error(f"重连异常: {e}")

    async def get_firmware_version(self) -> FirmwareVersion:
        """查询STM32固件版本

        发送版本查询命令,解析响应数据。

        返回:
            FirmwareVersion: 固件版本信息
        """
        result = await self.send_command(STM32Command.GET_VERSION)

        if result is None or result[0] != STM32ResponseCode.OK:
            logger.error("查询固件版本失败")
            return FirmwareVersion()

        data = result[1]

        if len(data) >= 3:
            version = FirmwareVersion(
                major=data[0],
                minor=data[1],
                patch=data[2],
            )
            self._firmware_version = version
            logger.info(f"STM32固件版本: {version}")
            return version

        return FirmwareVersion()

    async def get_status(self) -> STM32Status:
        """查询STM32当前状态

        获取MCU温度、运行时间、内存使用等状态信息。

        返回:
            STM32Status: 状态信息结构体
        """
        result = await self.send_command(STM32Command.GET_STATUS)

        if result is None or result[0] != STM32ResponseCode.OK:
            logger.error("查询状态失败")
            return self._latest_status

        data = result[1]

        try:
            if len(data) >= 12:
                status = STM32Status(
                    mcu_temp=data[0] + data[1] / 100.0,
                    uptime_ms=struct.unpack("<I", data[2:6])[0],
                    free_heap=struct.unpack("<I", data[6:10])[0],
                    cpu_usage=data[10],
                    estop_active=bool(data[11] & 0x01),
                    watchdog_fed=bool(data[11] & 0x02),
                    error_count=struct.unpack("<H", data[12:14])[0] if len(data) >= 14 else 0,
                    timestamp=time.time(),
                )
                self._latest_status = status

                # 通知状态回调
                for cb in self._status_callbacks:
                    try:
                        cb(status)
                    except Exception as e:
                        logger.error(f"状态回调异常: {e}")

                return status
        except Exception as e:
            logger.error(f"解析状态数据失败: {e}")

        return self._latest_status

    async def emergency_stop(self) -> bool:
        """触发紧急停止

        向STM32发送紧急停止命令,切断所有电机电源。

        返回:
            bool: 命令发送成功返回True
        """
        logger.critical("发送紧急停止命令到STM32")

        result = await self.send_command(
            STM32Command.ESTOP,
            timeout=0.5,
            retries=1,
        )

        return result is not None and result[0] == STM32ResponseCode.OK

    async def set_gpio(self, pin: int, value: bool) -> bool:
        """控制STM32 GPIO引脚

        参数:
            pin: 引脚编号
            value: 高低电平

        返回:
            bool: 设置成功返回True
        """
        data = bytes([pin, 1 if value else 0])
        result = await self.send_command(STM32Command.SET_GPIO, data)
        return result is not None and result[0] == STM32ResponseCode.OK

    async def read_adc(self, channel: int) -> int | None:
        """读取STM32 ADC通道

        参数:
            channel: ADC通道号

        返回:
            int: ADC原始值(12bit) 或 None(失败)
        """
        data = bytes([channel])
        result = await self.send_command(STM32Command.READ_ADC, data)

        if result and result[0] == STM32ResponseCode.OK and len(result[1]) >= 2:
            return struct.unpack("<H", result[1][:2])[0]
        return None

    async def set_pwm(self, channel: int, duty: float) -> bool:
        """设置STM32 PWM输出

        参数:
            channel: PWM通道
            duty: 占空比(0.0-1.0)

        返回:
            bool: 设置成功返回True
        """
        duty_int = int(max(0.0, min(1.0, duty)) * 1000)
        data = struct.pack("<BH", channel, duty_int)
        result = await self.send_command(STM32Command.SET_PWM, data)
        return result is not None and result[0] == STM32ResponseCode.OK

    def register_status_callback(
        self, callback: Callable[[STM32Status], None]
    ) -> None:
        """注册状态变更回调

        参数:
            callback: 状态回调函数
        """
        if callback not in self._status_callbacks:
            self._status_callbacks.append(callback)

    def register_disconnect_callback(self, callback: Callable[[], None]) -> None:
        """注册断开连接回调

        参数:
            callback: 断开回调函数
        """
        if callback not in self._disconnect_callbacks:
            self._disconnect_callbacks.append(callback)

    @property
    def is_connected(self) -> bool:
        """STM32连接状态"""
        return self._connected

    @property
    def latest_status(self) -> STM32Status:
        """最新状态缓存"""
        return self._latest_status

    @property
    def firmware_version(self) -> FirmwareVersion | None:
        """固件版本缓存"""
        return self._firmware_version

    async def shutdown(self) -> None:
        """关闭桥接,释放资源"""
        self._stop_event.set()

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            try:
                if self._serial and self._serial.is_open:
                    self._serial.close()

                self._initialized = False
                self._connected = False
                logger.info("STM32桥接已关闭")

            except Exception as e:
                logger.error(f"关闭桥接异常: {e}")

    def __repr__(self) -> str:
        return (
            f"STM32Bridge(port={self.config.port}, "
            f"baud={self.config.baudrate}, "
            f"connected={self._connected})"
        )

    async def __aenter__(self) -> STM32Bridge:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
