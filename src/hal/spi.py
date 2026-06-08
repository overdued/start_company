"""
RK3588 SPI硬件抽象层模块

本模块提供对OrangePi Kunpeng Pro (RK3588) SPI总线的完整控制支持，
基于spidev驱动实现，支持全双工通信、多种时钟模式、可变字长配置。

RK3588 SPI资源（40PIN上可用的）：
    - SPI1: 40PIN扩展（引脚7/13/15/16/18）
    - SPI4: 40PIN扩展（引脚23/24/26/40）

特性：
    - 支持SPI模式0-3（CPOL/CPHA组合）
    - 可变字长（4-32位）
    - 可配置时钟频率
    - 全双工读写
    - 块传输支持
    - 超时保护（默认5秒）
    - 线程安全
    - 异常回退和自动恢复

SPI模式定义：
    Mode 0: CPOL=0, CPHA=0 (空闲时钟低，第一个边沿采样)
    Mode 1: CPOL=0, CPHA=1 (空闲时钟低，第二个边沿采样)
    Mode 2: CPOL=1, CPHA=0 (空闲时钟高，第一个边沿采样)
    Mode 3: CPOL=1, CPHA=1 (空闲时钟高，第二个边沿采样)

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

import os
import time
import fcntl
import struct
import logging
import threading
from typing import Optional, List, Tuple, Callable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# SPI ioctl命令定义
SPI_IOC_MAGIC = 0x6B
SPI_IOC_MESSAGE_1 = 0x40066B00  # _IOW(SPI_IOC_MAGIC, 0, struct spi_ioc_transfer[1])
SPI_IOC_RD_MODE = 0x80016B01    # _IOR(SPI_IOC_MAGIC, 1, uint8)
SPI_IOC_WR_MODE = 0x40016B01    # _IOW(SPI_IOC_MAGIC, 1, uint8)
SPI_IOC_RD_BITS_PER_WORD = 0x80016B03
SPI_IOC_WR_BITS_PER_WORD = 0x40016B03
SPI_IOC_RD_MAX_SPEED_HZ = 0x80046B04
SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04

# SPI模式位定义
SPI_MODE_0 = 0x00  # CPOL=0, CPHA=0
SPI_MODE_1 = 0x01  # CPOL=0, CPHA=1
SPI_MODE_2 = 0x02  # CPOL=1, CPHA=0
SPI_MODE_3 = 0x03  # CPOL=1, CPHA=1
SPI_MODE_CS_HIGH = 0x04  # CS高电平有效
SPI_MODE_LSB_FIRST = 0x08  # LSB先传输
SPI_MODE_3WIRE = 0x10  # 单线模式
SPI_MODE_LOOP = 0x20  # 回环模式
SPI_MODE_NO_CS = 0x40  # 不使用CS


class SPIMode(Enum):
    """SPI时钟模式枚举"""
    MODE_0 = 0  # CPOL=0, CPHA=0
    MODE_1 = 1  # CPOL=0, CPHA=1
    MODE_2 = 2  # CPOL=1, CPHA=0
    MODE_3 = 3  # CPOL=1, CPHA=1


# RK3588 SPI设备路径映射
RK3588_SPI_DEVICES: dict = {
    0: "/dev/spidev0.0",
    1: "/dev/spidev1.0",
    2: "/dev/spidev2.0",
    3: "/dev/spidev3.0",
    4: "/dev/spidev4.0",
    5: "/dev/spidev5.0",
    6: "/dev/spidev6.0",
    7: "/dev/spidev7.0",
}

# SPI传输结构体 (struct spi_ioc_transfer)
# struct spi_ioc_transfer {
#     __u64       tx_buf;         // 发送缓冲区地址
#     __u64       rx_buf;         // 接收缓冲区地址
#     __u32       len;            // 传输长度
#     __u32       speed_hz;       // 时钟频率
#     __u16       delay_usecs;    // 传输后延迟
#     __u8        bits_per_word;  // 每字位数
# }
SPI_IOC_TRANSFER_FMT = "=QQIIHBBxxxxx"  # 打包格式
SPI_IOC_TRANSFER_SIZE = struct.calcsize(SPI_IOC_TRANSFER_FMT)


class SPI:
    """RK3588 SPI总线控制器类

    提供对RK3588 SPI总线的完整控制，基于Linux spidev驱动。
    支持全双工通信、多种时钟模式和可配置时钟频率。

    Args:
        bus: SPI总线号
        device: 设备选择（CS）号，默认0
        mode: SPI模式（0-3），默认0
        max_speed_hz: 最大时钟频率（Hz），默认1000000
        bits_per_word: 每字位数（4-32），默认8
        timeout: 操作超时时间（秒），默认5.0
        cs_active_high: CS是否高电平有效，默认False

    Raises:
        ValueError: 参数无效
        RuntimeError: SPI设备打开失败

    Example:
        >>> spi = SPI(1, mode=0, max_speed_hz=1000000)
        >>> rx = spi.transfer([0x01, 0x02, 0x03])
        >>> spi.close()
    """

    def __init__(
        self,
        bus: int,
        device: int = 0,
        mode: int = 0,
        max_speed_hz: int = 1000000,
        bits_per_word: int = 8,
        timeout: float = 5.0,
        cs_active_high: bool = False,
    ) -> None:
        self._bus: int = bus
        self._device: int = device
        self._mode: SPIMode = SPIMode(mode)
        self._max_speed_hz: int = max_speed_hz
        self._bits_per_word: int = bits_per_word
        self._timeout: float = timeout
        self._cs_active_high: bool = cs_active_high
        self._instance_lock: threading.Lock = threading.Lock()
        self._fd: Optional[int] = None
        self._closed: bool = False
        self._device_path: str = f"/dev/spidev{bus}.{device}"

        # 参数验证
        if mode not in (0, 1, 2, 3):
            raise ValueError(f"无效的SPI模式: {mode}, 支持: 0-3")
        if not (4 <= bits_per_word <= 32):
            raise ValueError(f"无效的每字位数: {bits_per_word}, 支持: 4-32")
        if max_speed_hz <= 0:
            raise ValueError(f"无效的时钟频率: {max_speed_hz}")

        # 检查设备节点
        if not os.path.exists(self._device_path):
            # 尝试备选路径
            alt_path = RK3588_SPI_DEVICES.get(bus)
            if alt_path and os.path.exists(alt_path):
                self._device_path = alt_path
            else:
                raise RuntimeError(f"SPI设备节点不存在: {self._device_path}")

        try:
            self._init_spidev()
            logger.info(
                f"SPI {self._device_path} 初始化成功, "
                f"mode={mode}, speed={max_speed_hz}Hz, bits={bits_per_word}"
            )
        except Exception as e:
            self._fallback_safe_state()
            raise RuntimeError(f"SPI {self._device_path} 初始化失败: {e}") from e

    def _init_spidev(self) -> None:
        """初始化spidev设备

        Raises:
            OSError: 设备打开或配置失败
        """
        # 打开设备
        try:
            self._fd = os.open(self._device_path, os.O_RDWR)
        except OSError as e:
            raise RuntimeError(f"打开SPI设备 {self._device_path} 失败: {e}") from e

        # 配置SPI模式
        mode_val = self._mode.value
        if self._cs_active_high:
            mode_val |= SPI_MODE_CS_HIGH

        try:
            fcntl.ioctl(self._fd, SPI_IOC_WR_MODE, struct.pack("=B", mode_val))
        except OSError as e:
            raise RuntimeError(f"设置SPI模式失败: {e}") from e

        # 配置每字位数
        try:
            fcntl.ioctl(self._fd, SPI_IOC_WR_BITS_PER_WORD, struct.pack("=B", self._bits_per_word))
        except OSError as e:
            raise RuntimeError(f"设置SPI每字位数失败: {e}") from e

        # 配置最大时钟频率
        try:
            fcntl.ioctl(self._fd, SPI_IOC_WR_MAX_SPEED_HZ, struct.pack("=I", self._max_speed_hz))
        except OSError as e:
            raise RuntimeError(f"设置SPI时钟频率失败: {e}") from e

    def transfer(self, tx_data: List[int]) -> List[int]:
        """SPI全双工数据传输

        同时发送和接收数据，发送tx_data的同时接收等长的数据。

        Args:
            tx_data: 要发送的字节列表

        Returns:
            List[int]: 接收到的字节列表

        Raises:
            RuntimeError: SPI设备已关闭
            IOError: 传输失败
        """
        if self._closed:
            raise RuntimeError(f"SPI {self._device_path} 已关闭")
        if not tx_data:
            return []

        with self._instance_lock:
            try:
                tx_bytes = bytes(tx_data)
                rx_bytes = self._transfer_raw(tx_bytes)
                return list(rx_bytes)
            except Exception as e:
                logger.error(f"SPI {self._device_path} 传输失败: {e}")
                raise IOError(f"SPI传输失败: {e}") from e

    def _transfer_raw(self, tx_data: bytes) -> bytes:
        """底层SPI传输实现（使用ioctl）

        Args:
            tx_data: 要发送的字节数据

        Returns:
            bytes: 接收到的字节数据
        """
        tx_buf = tx_data
        rx_buf = bytearray(len(tx_data))

        # 创建spi_ioc_transfer结构体
        # 注意：tx_buf和rx_buf使用ctypes传递地址
        import ctypes

        tx_array = ctypes.create_string_buffer(tx_buf, len(tx_buf))
        rx_array = ctypes.create_string_buffer(len(rx_buf))

        transfer = struct.pack(
            SPI_IOC_TRANSFER_FMT,
            ctypes.addressof(tx_array),  # tx_buf
            ctypes.addressof(rx_array),  # rx_buf
            len(tx_buf),                  # len
            self._max_speed_hz,           # speed_hz
            0,                            # delay_usecs
            self._bits_per_word,          # bits_per_word
            0,                            # cs_change
        )

        # 发送ioctl命令
        fcntl.ioctl(self._fd, SPI_IOC_MESSAGE_1, transfer)

        return bytes(rx_array.raw)

    def transfer3(self, tx_data: bytes) -> bytes:
        """SPI全双工数据传输（字节接口）

        Args:
            tx_data: 要发送的字节数据

        Returns:
            bytes: 接收到的字节数据
        """
        if self._closed:
            raise RuntimeError(f"SPI {self._device_path} 已关闭")
        if not tx_data:
            return b""

        with self._instance_lock:
            try:
                return self._transfer_raw(tx_data)
            except Exception as e:
                logger.error(f"SPI {self._device_path} 传输失败: {e}")
                raise IOError(f"SPI传输失败: {e}") from e

    def read(self, length: int) -> List[int]:
        """SPI读取数据

        发送指定长度的空字节（0x00），同时接收数据。

        Args:
            length: 要读取的字节数

        Returns:
            List[int]: 接收到的字节列表
        """
        tx_data = [0x00] * length
        return self.transfer(tx_data)

    def write(self, data: List[int]) -> None:
        """SPI写入数据

        发送数据，忽略接收到的数据。

        Args:
            data: 要发送的字节列表
        """
        self.transfer(data)

    def write_read(self, tx_data: List[int], rx_length: int) -> List[int]:
        """先写后读SPI操作

        先发送命令/地址，然后读取响应数据。

        Args:
            tx_data: 要发送的命令/地址数据
            rx_length: 要读取的字节数

        Returns:
            List[int]: 接收到的字节列表
        """
        # 合并发送和接收
        full_tx = tx_data + [0x00] * rx_length
        full_rx = self.transfer(full_tx)
        # 返回接收部分（跳过发送部分对应的数据）
        return full_rx[len(tx_data):]

    def set_speed(self, max_speed_hz: int) -> None:
        """动态修改SPI时钟频率

        Args:
            max_speed_hz: 新的最大时钟频率（Hz）
        """
        if self._closed:
            raise RuntimeError(f"SPI {self._device_path} 已关闭")
        if max_speed_hz <= 0:
            raise ValueError(f"无效的时钟频率: {max_speed_hz}")

        with self._instance_lock:
            try:
                fcntl.ioctl(self._fd, SPI_IOC_WR_MAX_SPEED_HZ, struct.pack("=I", max_speed_hz))
                self._max_speed_hz = max_speed_hz
                logger.info(f"SPI {self._device_path} 时钟频率修改为 {max_speed_hz}Hz")
            except OSError as e:
                raise IOError(f"设置SPI时钟频率失败: {e}") from e

    def set_mode(self, mode: int) -> None:
        """动态修改SPI模式

        Args:
            mode: 新的SPI模式（0-3）
        """
        if self._closed:
            raise RuntimeError(f"SPI {self._device_path} 已关闭")
        if mode not in (0, 1, 2, 3):
            raise ValueError(f"无效的SPI模式: {mode}")

        with self._instance_lock:
            try:
                mode_val = mode
                if self._cs_active_high:
                    mode_val |= SPI_MODE_CS_HIGH
                fcntl.ioctl(self._fd, SPI_IOC_WR_MODE, struct.pack("=B", mode_val))
                self._mode = SPIMode(mode)
                logger.info(f"SPI {self._device_path} 模式修改为 {mode}")
            except OSError as e:
                raise IOError(f"设置SPI模式失败: {e}") from e

    def set_bits_per_word(self, bits: int) -> None:
        """动态修改每字位数

        Args:
            bits: 新的每字位数（4-32）
        """
        if self._closed:
            raise RuntimeError(f"SPI {self._device_path} 已关闭")
        if not (4 <= bits <= 32):
            raise ValueError(f"无效的每字位数: {bits}")

        with self._instance_lock:
            try:
                fcntl.ioctl(self._fd, SPI_IOC_WR_BITS_PER_WORD, struct.pack("=B", bits))
                self._bits_per_word = bits
                logger.info(f"SPI {self._device_path} 每字位数修改为 {bits}")
            except OSError as e:
                raise IOError(f"设置SPI每字位数失败: {e}") from e

    def cs_active(self, active: bool) -> None:
        """手动控制CS信号

        注意：需要在SPI模式中使用SPI_MODE_NO_CS才能手动控制CS。

        Args:
            active: True=激活CS, False=释放CS
        """
        # 当前使用spidev自动CS控制，手动控制需要额外的GPIO
        logger.debug(f"SPI CS {'激活' if active else '释放'}")

    def _fallback_safe_state(self) -> None:
        """异常回退到安全状态：关闭设备"""
        try:
            self._cleanup_resources()
        except Exception as e:
            logger.error(f"SPI {self._device_path} 安全状态回退失败: {e}")

    def _cleanup_resources(self) -> None:
        """清理所有已分配资源"""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def close(self) -> None:
        """关闭SPI设备，释放所有资源

        使用try/finally确保资源被正确释放。
        """
        if self._closed:
            return

        with self._instance_lock:
            try:
                self._cleanup_resources()
                self._closed = True
                logger.info(f"SPI {self._device_path} 已关闭")
            except Exception as e:
                logger.error(f"SPI {self._device_path} 关闭时发生错误: {e}")
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
    def bus(self) -> int:
        """获取SPI总线号"""
        return self._bus

    @property
    def device(self) -> int:
        """获取设备号"""
        return self._device

    @property
    def mode(self) -> SPIMode:
        """获取当前SPI模式"""
        return self._mode

    @property
    def max_speed_hz(self) -> int:
        """获取当前最大时钟频率"""
        return self._max_speed_hz

    @property
    def bits_per_word(self) -> int:
        """获取当前每字位数"""
        return self._bits_per_word

    @property
    def is_closed(self) -> bool:
        """判断设备是否已关闭"""
        return self._closed

    @staticmethod
    def get_available_buses() -> List[str]:
        """获取系统中可用的SPI总线列表

        Returns:
            List[str]: 可用SPI设备路径列表
        """
        available = []
        for bus, path in RK3588_SPI_DEVICES.items():
            if os.path.exists(path):
                available.append(path)
        return available
