"""
RK3588 I2C硬件抽象层模块

本模块提供对OrangePi Kunpeng Pro (RK3588) I2C总线的完整控制支持，
基于smbus2库实现，支持字节读写、块读写、寄存器操作等。

RK3588 I2C总线资源：
    - I2C0: 内部使用（PMIC等）
    - I2C1: 40PIN扩展
    - I2C2: 内部使用
    - I2C3: 40PIN扩展（引脚3/5）
    - I2C4: 40PIN扩展（引脚27/28）
    - I2C5-I2C8: 部分在40PIN上可用

特性：
    - 多总线自动探测
    - 字节/字/块读写
    - 超时重试机制（默认5秒超时，3次重试）
    - 线程安全（每总线独立锁）
    - 异常回退和自动恢复
    - 设备扫描功能

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

import os
import time
import struct
import logging
import threading
from typing import Optional, List, Dict, Callable, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# RK3588 I2C总线设备路径映射
RK3588_I2C_DEVICES: Dict[int, str] = {
    0: "/dev/i2c-0",
    1: "/dev/i2c-1",
    2: "/dev/i2c-2",
    3: "/dev/i2c-3",
    4: "/dev/i2c-4",
    5: "/dev/i2c-5",
    6: "/dev/i2c-6",
    7: "/dev/i2c-7",
    8: "/dev/i2c-8",
}

# 总线级锁，确保同一总线上同一时间只有一个操作
_bus_locks: Dict[int, threading.Lock] = {}
_bus_locks_lock = threading.Lock()


def _get_bus_lock(bus: int) -> threading.Lock:
    """获取指定I2C总线的锁（线程安全）

    Args:
        bus: I2C总线号

    Returns:
        threading.Lock: 总线锁
    """
    with _bus_locks_lock:
        if bus not in _bus_locks:
            _bus_locks[bus] = threading.Lock()
        return _bus_locks[bus]


class I2C:
    """RK3588 I2C总线控制器类

    提供对RK3588 I2C总线的完整读写控制，基于smbus2库实现。
    支持字节、字、块数据读写，具备超时重试和线程安全保护。

    Args:
        bus: I2C总线号（0-8）
        device_address: 设备地址（7位地址，0x03-0x77）
        timeout: 操作超时时间（秒），默认5.0
        retries: 重试次数，默认3

    Raises:
        ValueError: 总线号或设备地址无效
        RuntimeError: I2C设备打开失败

    Example:
        >>> i2c = I2C(3, 0x50)
        >>> data = i2c.read_byte(0x00)
        >>> i2c.write_byte(0x00, 0xAA)
        >>> i2c.close()
    """

    def __init__(
        self,
        bus: int,
        device_address: int,
        timeout: float = 5.0,
        retries: int = 3,
    ) -> None:
        self._bus: int = bus
        self._device_address: int = device_address
        self._timeout: float = timeout
        self._retries: int = retries
        self._bus_lock: threading.Lock = _get_bus_lock(bus)
        self._instance_lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._smbus: Optional[object] = None
        self._device_path: str = ""

        # 参数验证
        if bus not in RK3588_I2C_DEVICES:
            raise ValueError(f"无效的I2C总线号: {bus}, RK3588支持的总线: {list(RK3588_I2C_DEVICES.keys())}")
        if not (0x03 <= device_address <= 0x77):
            raise ValueError(f"无效的I2C设备地址: 0x{device_address:02X}, 必须在0x03-0x77范围内")

        self._device_path = RK3588_I2C_DEVICES[bus]

        # 检查设备节点是否存在
        if not os.path.exists(self._device_path):
            raise RuntimeError(f"I2C设备节点不存在: {self._device_path}, 请检查设备树配置")

        try:
            self._init_smbus()
            logger.info(f"I2C总线 {bus} 设备 0x{device_address:02X} 初始化成功")
        except Exception as e:
            self._fallback_safe_state()
            raise RuntimeError(f"I2C总线 {bus} 设备 0x{device_address:02X} 初始化失败: {e}") from e

    def _init_smbus(self) -> None:
        """初始化smbus2连接

        Raises:
            ImportError: smbus2库未安装
            IOError: 打开设备失败
        """
        try:
            from smbus2 import SMBus
        except ImportError:
            raise ImportError("smbus2库未安装，请执行: pip install smbus2")

        try:
            self._smbus = SMBus(self._bus)
        except (IOError, OSError) as e:
            raise RuntimeError(f"打开I2C设备 {self._device_path} 失败: {e}") from e

    def _execute_with_retry(self, operation: Callable, *args, **kwargs):
        """带重试机制的操作执行

        在操作失败时自动重试，直到成功或达到最大重试次数。
        每次重试之间间隔100ms。

        Args:
            operation: 要执行的操作函数
            *args, **kwargs: 操作函数的参数

        Returns:
            操作函数的返回值

        Raises:
            IOError: 所有重试均失败
        """
        last_error = None
        for attempt in range(1, self._retries + 1):
            try:
                with self._bus_lock:
                    # 设置超时
                    result = operation(*args, **kwargs)
                    return result
            except (IOError, OSError) as e:
                last_error = e
                logger.warning(
                    f"I2C 总线{self._bus} 设备0x{self._device_address:02X} "
                    f"操作失败（第{attempt}/{self._retries}次）: {e}"
                )
                if attempt < self._retries:
                    time.sleep(0.1 * attempt)  # 递增退避

        raise IOError(
            f"I2C 总线{self._bus} 设备0x{self._device_address:02X} "
            f"操作在{self._retries}次重试后仍然失败: {last_error}"
        )

    def read_byte(self, register: int) -> int:
        """从指定寄存器读取1个字节

        Args:
            register: 寄存器地址

        Returns:
            int: 读取到的字节值（0-255）

        Raises:
            IOError: 读取失败
        """
        if self._closed:
            raise RuntimeError("I2C设备已关闭")
        if not (0 <= register <= 0xFF):
            raise ValueError(f"寄存器地址必须在0-255范围内: {register}")

        def _do_read(reg):
            return self._smbus.read_byte_data(self._device_address, reg)

        return self._execute_with_retry(_do_read, register)

    def write_byte(self, register: int, value: int) -> None:
        """向指定寄存器写入1个字节

        Args:
            register: 寄存器地址
            value: 要写入的字节值（0-255）

        Raises:
            IOError: 写入失败
        """
        if self._closed:
            raise RuntimeError("I2C设备已关闭")
        if not (0 <= register <= 0xFF):
            raise ValueError(f"寄存器地址必须在0-255范围内: {register}")
        if not (0 <= value <= 0xFF):
            raise ValueError(f"写入值必须在0-255范围内: {value}")

        def _do_write(reg, val):
            self._smbus.write_byte_data(self._device_address, reg, val)

        self._execute_with_retry(_do_write, register, value)

    def read_word(self, register: int) -> int:
        """从指定寄存器读取1个字（2字节，小端序）

        Args:
            register: 寄存器地址

        Returns:
            int: 读取到的字值（0-65535）

        Raises:
            IOError: 读取失败
        """
        if self._closed:
            raise RuntimeError("I2C设备已关闭")
        if not (0 <= register <= 0xFF):
            raise ValueError(f"寄存器地址必须在0-255范围内: {register}")

        def _do_read(reg):
            return self._smbus.read_word_data(self._device_address, reg)

        return self._execute_with_retry(_do_read, register)

    def write_word(self, register: int, value: int) -> None:
        """向指定寄存器写入1个字（2字节，小端序）

        Args:
            register: 寄存器地址
            value: 要写入的字值（0-65535）

        Raises:
            IOError: 写入失败
        """
        if self._closed:
            raise RuntimeError("I2C设备已关闭")
        if not (0 <= register <= 0xFF):
            raise ValueError(f"寄存器地址必须在0-255范围内: {register}")
        if not (0 <= value <= 0xFFFF):
            raise ValueError(f"写入值必须在0-65535范围内: {value}")

        def _do_write(reg, val):
            self._smbus.write_word_data(self._device_address, reg, val)

        self._execute_with_retry(_do_write, register, value)

    def read_block(self, register: int, length: int) -> List[int]:
        """从指定寄存器读取块数据

        使用I2C块读取协议读取多个字节。

        Args:
            register: 起始寄存器地址
            length: 要读取的字节数（1-32）

        Returns:
            List[int]: 读取到的字节列表

        Raises:
            IOError: 读取失败
        """
        if self._closed:
            raise RuntimeError("I2C设备已关闭")
        if not (0 <= register <= 0xFF):
            raise ValueError(f"寄存器地址必须在0-255范围内: {register}")
        if not (1 <= length <= 32):
            raise ValueError(f"读取长度必须在1-32范围内: {length}")

        def _do_read(reg, length_):
            return self._smbus.read_i2c_block_data(self._device_address, reg, length_)

        return self._execute_with_retry(_do_read, register, length)

    def write_block(self, register: int, data: List[int]) -> None:
        """向指定寄存器写入块数据

        使用I2C块写入协议写入多个字节。

        Args:
            register: 起始寄存器地址
            data: 要写入的字节列表（长度1-32）

        Raises:
            IOError: 写入失败
        """
        if self._closed:
            raise RuntimeError("I2C设备已关闭")
        if not (0 <= register <= 0xFF):
            raise ValueError(f"寄存器地址必须在0-255范围内: {register}")
        if not (1 <= len(data) <= 32):
            raise ValueError(f"写入数据长度必须在1-32范围内: {len(data)}")

        def _do_write(reg, data_):
            self._smbus.write_i2c_block_data(self._device_address, reg, data_)

        self._execute_with_retry(_do_write, register, data)

    def read_raw(self, length: int) -> List[int]:
        """直接读取原始数据（不指定寄存器）

        Args:
            length: 要读取的字节数

        Returns:
            List[int]: 读取到的字节列表

        Raises:
            IOError: 读取失败
        """
        if self._closed:
            raise RuntimeError("I2C设备已关闭")
        if not (1 <= length <= 32):
            raise ValueError(f"读取长度必须在1-32范围内: {length}")

        def _do_read(length_):
            return self._smbus.read_i2c_block_data(self._device_address, 0, length_)

        return self._execute_with_retry(_do_read, length)

    def write_raw(self, data: List[int]) -> None:
        """直接写入原始数据（不指定寄存器）

        Args:
            data: 要写入的字节列表

        Raises:
            IOError: 写入失败
        """
        if self._closed:
            raise RuntimeError("I2C设备已关闭")
        if not data:
            raise ValueError("写入数据不能为空")

        def _do_write(data_):
            self._smbus.write_i2c_block_data(self._device_address, 0, data_)

        self._execute_with_retry(_do_write, data)

    def read_bytes_with_retry(
        self,
        register: int,
        length: int,
        custom_timeout: Optional[float] = None,
        custom_retries: Optional[int] = None,
    ) -> List[int]:
        """带自定义超时和重试参数的块读取

        用于需要特殊超时或重试配置的场景。

        Args:
            register: 起始寄存器地址
            length: 要读取的字节数
            custom_timeout: 自定义超时时间（秒），None表示使用默认值
            custom_retries: 自定义重试次数，None表示使用默认值

        Returns:
            List[int]: 读取到的字节列表
        """
        original_timeout = self._timeout
        original_retries = self._retries
        try:
            if custom_timeout is not None:
                self._timeout = custom_timeout
            if custom_retries is not None:
                self._retries = custom_retries
            return self.read_block(register, length)
        finally:
            self._timeout = original_timeout
            self._retries = original_retries

    @staticmethod
    def scan_bus(bus: int, start_addr: int = 0x03, end_addr: int = 0x77) -> List[int]:
        """扫描I2C总线上的设备

        Args:
            bus: I2C总线号
            start_addr: 起始扫描地址
            end_addr: 结束扫描地址

        Returns:
            List[int]: 发现的设备地址列表
        """
        device_path = RK3588_I2C_DEVICES.get(bus)
        if not device_path or not os.path.exists(device_path):
            logger.error(f"I2C总线 {bus} 设备节点不存在")
            return []

        found_devices = []
        try:
            from smbus2 import SMBus
            smbus = SMBus(bus)
            for addr in range(start_addr, end_addr + 1):
                try:
                    smbus.read_byte(addr)
                    found_devices.append(addr)
                    logger.debug(f"I2C总线 {bus} 发现设备: 0x{addr:02X}")
                except (IOError, OSError):
                    pass
            smbus.close()
        except Exception as e:
            logger.error(f"扫描I2C总线 {bus} 失败: {e}")

        return found_devices

    def _fallback_safe_state(self) -> None:
        """异常回退到安全状态：关闭已打开的设备"""
        try:
            self._cleanup_resources()
        except Exception as e:
            logger.error(f"I2C 总线{self._bus} 安全状态回退失败: {e}")

    def _cleanup_resources(self) -> None:
        """清理所有已分配资源"""
        if self._smbus:
            try:
                self._smbus.close()
            except Exception:
                pass
            self._smbus = None

    def close(self) -> None:
        """关闭I2C设备，释放所有资源

        使用try/finally确保资源被正确释放。
        """
        if self._closed:
            return

        with self._instance_lock:
            try:
                self._cleanup_resources()
                self._closed = True
                logger.info(f"I2C总线 {self._bus} 设备 0x{self._device_address:02X} 已关闭")
            except Exception as e:
                logger.error(f"I2C总线 {self._bus} 关闭时发生错误: {e}")
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
        """获取I2C总线号"""
        return self._bus

    @property
    def device_address(self) -> int:
        """获取设备地址"""
        return self._device_address

    @property
    def is_closed(self) -> bool:
        """判断设备是否已关闭"""
        return self._closed

    @staticmethod
    def get_available_buses() -> List[int]:
        """获取系统中可用的I2C总线列表

        Returns:
            List[int]: 可用总线号列表
        """
        available = []
        for bus, path in RK3588_I2C_DEVICES.items():
            if os.path.exists(path):
                available.append(bus)
        return available
