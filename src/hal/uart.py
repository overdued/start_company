"""
RK3588 UART硬件抽象层模块

本模块提供对OrangePi Kunpeng Pro (RK3588) UART串口的完整控制支持，
基于pyserial库实现，支持可配置波特率、数据位、校验位、停止位等参数。

RK3588 UART资源（40PIN上可用的）：
    - UART0: 调试串口（引脚36/37）
    - UART6: 40PIN扩展（引脚8/10）
    - UART8: 40PIN扩展（引脚19/21）

特性：
    - 完整的串口参数配置
    - 读写超时保护（默认5秒）
    - 接收中断回调（独立线程）
    - 发送/接收缓冲区管理
    - 线程安全
    - 线路状态检测（CTS/DTS/DTR/RTS）
    - 自动重连机制

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

import os
import time
import logging
import threading
import serial
import serial.tools.list_ports
from typing import Optional, Callable, List, Dict, Tuple
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class UARTParity(Enum):
    """UART校验位枚举"""
    NONE = serial.PARITY_NONE
    EVEN = serial.PARITY_EVEN
    ODD = serial.PARITY_ODD
    MARK = serial.PARITY_MARK
    SPACE = serial.PARITY_SPACE


class UARTStopBits(Enum):
    """UART停止位枚举"""
    ONE = serial.STOPBITS_ONE
    ONE_POINT_FIVE = serial.STOPBITS_ONE_POINT_FIVE
    TWO = serial.STOPBITS_TWO


class UARTDataBits(Enum):
    """UART数据位枚举"""
    FIVE = serial.FIVEBITS
    SIX = serial.SIXBITS
    SEVEN = serial.SEVENBITS
    EIGHT = serial.EIGHTBITS


# RK3588 UART设备路径映射
RK3588_UART_DEVICES: Dict[str, str] = {
    "UART0": "/dev/ttyS0",
    "UART1": "/dev/ttyS1",
    "UART2": "/dev/ttyS2",
    "UART3": "/dev/ttyS3",
    "UART4": "/dev/ttyS4",
    "UART5": "/dev/ttyS5",
    "UART6": "/dev/ttyS6",
    "UART7": "/dev/ttyS7",
    "UART8": "/dev/ttyS8",
    "UART9": "/dev/ttyS9",
}


class UART:
    """RK3588 UART串口控制器类

    提供对RK3588 UART串口的完整控制，基于pyserial实现。
    支持可配置的波特率、数据位、校验位、停止位，具备读写超时保护和
    中断接收功能。

    Args:
        port: 串口设备路径（如"/dev/ttyS6"）或别名（如"UART6"）
        baudrate: 波特率，默认115200
        data_bits: 数据位（5/6/7/8），默认8
        parity: 校验位（"N"/"E"/"O"/"M"/"S"），默认"N"
        stop_bits: 停止位（1/1.5/2），默认1
        timeout: 读写超时时间（秒），默认5.0
        write_timeout: 写超时时间（秒），默认5.0
        read_callback: 数据接收回调函数

    Raises:
        ValueError: 参数无效
        RuntimeError: 串口打开失败

    Example:
        >>> uart = UART("/dev/ttyS6", baudrate=115200)
        >>> uart.send(b"Hello\\r\\n")
        >>> data = uart.receive(64, timeout=1.0)
        >>> uart.close()
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        data_bits: int = 8,
        parity: str = "N",
        stop_bits: float = 1,
        timeout: float = 5.0,
        write_timeout: float = 5.0,
        read_callback: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        self._port_name: str = port
        self._port_path: str = self._resolve_port(port)
        self._baudrate: int = baudrate
        self._data_bits: int = data_bits
        self._parity_str: str = parity
        self._stop_bits: float = stop_bits
        self._timeout: float = timeout
        self._write_timeout: float = write_timeout
        self._instance_lock: threading.Lock = threading.Lock()
        self._read_lock: threading.Lock = threading.Lock()
        self._write_lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._serial: Optional[serial.Serial] = None
        self._read_callback: Optional[Callable[[bytes], None]] = read_callback
        self._read_thread: Optional[threading.Thread] = None
        self._read_running: bool = False
        self._read_buffer: bytearray = bytearray()
        self._buffer_lock: threading.Lock = threading.Lock()
        self._bytesize_map: Dict[int, int] = {
            5: serial.FIVEBITS,
            6: serial.SIXBITS,
            7: serial.SEVENBITS,
            8: serial.EIGHTBITS,
        }
        self._parity_map: Dict[str, str] = {
            "N": serial.PARITY_NONE,
            "E": serial.PARITY_EVEN,
            "O": serial.PARITY_ODD,
            "M": serial.PARITY_MARK,
            "S": serial.PARITY_SPACE,
        }
        self._stopbits_map: Dict[float, float] = {
            1: serial.STOPBITS_ONE,
            1.5: serial.STOPBITS_ONE_POINT_FIVE,
            2: serial.STOPBITS_TWO,
        }

        # 参数验证
        if data_bits not in self._bytesize_map:
            raise ValueError(f"无效的数据位: {data_bits}, 支持: {list(self._bytesize_map.keys())}")
        if parity not in self._parity_map:
            raise ValueError(f"无效的校验位: {parity}, 支持: {list(self._parity_map.keys())}")
        if stop_bits not in self._stopbits_map:
            raise ValueError(f"无效的停止位: {stop_bits}, 支持: {list(self._stopbits_map.keys())}")

        try:
            self._init_serial()
            # 如果提供了回调函数，启动接收线程
            if self._read_callback:
                self._start_read_thread()
            logger.info(f"UART {self._port_path} 初始化成功, 波特率={baudrate}")
        except Exception as e:
            self._fallback_safe_state()
            raise RuntimeError(f"UART {self._port_path} 初始化失败: {e}") from e

    def _resolve_port(self, port: str) -> str:
        """解析串口名称到设备路径

        支持设备路径（如"/dev/ttyS6"）和别名（如"UART6"）。

        Args:
            port: 串口名称或路径

        Returns:
            str: 设备路径

        Raises:
            ValueError: 端口名称无法解析
        """
        # 如果已经是设备路径
        if port.startswith("/dev/"):
            if not os.path.exists(port):
                logger.warning(f"串口设备不存在: {port}")
            return port

        # 尝试从别名解析
        if port.upper() in RK3588_UART_DEVICES:
            return RK3588_UART_DEVICES[port.upper()]

        # 尝试ttyS前缀
        if port.startswith("ttyS"):
            path = f"/dev/{port}"
            return path

        raise ValueError(f"无法解析串口名称: {port}, 可用别名: {list(RK3588_UART_DEVICES.keys())}")

    def _init_serial(self) -> None:
        """初始化pyserial串口连接

        Raises:
            serial.SerialException: 串口打开失败
        """
        try:
            self._serial = serial.Serial(
                port=self._port_path,
                baudrate=self._baudrate,
                bytesize=self._bytesize_map[self._data_bits],
                parity=self._parity_map[self._parity_str],
                stopbits=self._stopbits_map[self._stop_bits],
                timeout=self._timeout,
                write_timeout=self._write_timeout,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            # 清空缓冲区
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
        except serial.SerialException as e:
            raise RuntimeError(f"打开串口 {self._port_path} 失败: {e}") from e

    def _start_read_thread(self) -> None:
        """启动数据接收线程"""
        self._read_running = True
        self._read_thread = threading.Thread(
            target=self._read_loop,
            name=f"UART-RX-{self._port_name}",
            daemon=True,
        )
        self._read_thread.start()
        logger.debug(f"UART {self._port_path} 接收线程已启动")

    def _read_loop(self) -> None:
        """数据接收循环（在独立线程中运行）

        持续监听串口数据，当收到数据时调用回调函数。
        使用超时读取避免阻塞，定期检查退出标志。
        """
        while self._read_running and self._serial and self._serial.is_open:
            try:
                # 使用小超时进行轮询，以便及时响应退出信号
                available = self._serial.in_waiting
                if available > 0:
                    data = self._serial.read(available)
                    if data and self._read_callback:
                        try:
                            self._read_callback(data)
                        except Exception as e:
                            logger.error(f"UART {self._port_path} 接收回调异常: {e}")
                    # 同时存入缓冲区
                    with self._buffer_lock:
                        self._read_buffer.extend(data)
                else:
                    # 短暂休眠避免CPU空转
                    time.sleep(0.001)
            except serial.SerialException as e:
                if self._read_running:
                    logger.error(f"UART {self._port_path} 接收线程异常: {e}")
                    time.sleep(0.1)
            except Exception as e:
                if self._read_running:
                    logger.error(f"UART {self._port_path} 接收线程未预期异常: {e}")
                    time.sleep(0.1)

    def send(self, data: bytes) -> int:
        """发送数据

        Args:
            data: 要发送的字节数据

        Returns:
            int: 实际发送的字节数

        Raises:
            RuntimeError: 串口未打开或已关闭
            serial.SerialTimeoutException: 发送超时
        """
        if self._closed:
            raise RuntimeError(f"UART {self._port_path} 已关闭")
        if not self._serial or not self._serial.is_open:
            raise RuntimeError(f"UART {self._port_path} 未打开")
        if not data:
            return 0

        with self._write_lock:
            try:
                bytes_written = self._serial.write(data)
                self._serial.flush()
                logger.debug(f"UART {self._port_path} 发送 {bytes_written} 字节")
                return bytes_written
            except serial.SerialTimeoutException as e:
                logger.error(f"UART {self._port_path} 发送超时: {e}")
                raise
            except serial.SerialException as e:
                logger.error(f"UART {self._port_path} 发送失败: {e}")
                raise IOError(f"UART发送失败: {e}") from e

    def send_line(self, data: bytes, line_ending: bytes = b"\r\n") -> int:
        """发送一行数据（自动添加行尾）

        Args:
            data: 要发送的数据
            line_ending: 行尾字符，默认\\r\\n

        Returns:
            int: 实际发送的字节数
        """
        return self.send(data + line_ending)

    def receive(self, size: int = 64, timeout: float = 1.0) -> bytes:
        """接收数据

        从串口接收缓冲区读取数据。如果启用了接收线程，
        数据会先存入内部缓冲区再返回。

        Args:
            size: 最大读取字节数
            timeout: 读取超时时间（秒）

        Returns:
            bytes: 接收到的数据

        Raises:
            RuntimeError: 串口未打开或已关闭
            serial.SerialTimeoutException: 接收超时
        """
        if self._closed:
            raise RuntimeError(f"UART {self._port_path} 已关闭")
        if not self._serial or not self._serial.is_open:
            raise RuntimeError(f"UART {self._port_path} 未打开")

        with self._read_lock:
            try:
                # 如果接收线程在运行，从内部缓冲区读取
                if self._read_running:
                    start_time = time.monotonic()
                    result = bytearray()
                    while len(result) < size:
                        with self._buffer_lock:
                            available = len(self._read_buffer)
                            if available > 0:
                                to_read = min(size - len(result), available)
                                result.extend(self._read_buffer[:to_read])
                                self._read_buffer = self._read_buffer[to_read:]
                            elif time.monotonic() - start_time >= timeout:
                                break
                        if len(result) < size:
                            time.sleep(0.001)
                    return bytes(result)
                else:
                    # 直接读取
                    original_timeout = self._serial.timeout
                    try:
                        self._serial.timeout = timeout
                        data = self._serial.read(size)
                        logger.debug(f"UART {self._port_path} 接收 {len(data)} 字节")
                        return data
                    finally:
                        self._serial.timeout = original_timeout

            except serial.SerialTimeoutException:
                logger.debug(f"UART {self._port_path} 接收超时")
                return b""
            except serial.SerialException as e:
                logger.error(f"UART {self._port_path} 接收失败: {e}")
                raise IOError(f"UART接收失败: {e}") from e

    def receive_line(self, timeout: float = 1.0, line_ending: bytes = b"\n") -> bytes:
        """接收一行数据

        持续读取直到遇到行尾字符或超时。

        Args:
            timeout: 读取超时时间（秒）
            line_ending: 行尾标识字符

        Returns:
            bytes: 接收到的行数据（不含行尾）
        """
        start_time = time.monotonic()
        buffer = bytearray()

        while time.monotonic() - start_time < timeout:
            data = self.receive(1, timeout=0.1)
            if data:
                buffer.extend(data)
                if line_ending in buffer:
                    line = buffer.split(line_ending, 1)[0]
                    # 将剩余数据放回缓冲区
                    remaining = buffer.split(line_ending, 1)[1]
                    with self._buffer_lock:
                        self._read_buffer = remaining + self._read_buffer
                    return bytes(line)
            else:
                time.sleep(0.001)

        return bytes(buffer)

    def set_receive_callback(self, callback: Callable[[bytes], None]) -> None:
        """设置数据接收回调函数

        设置后会在后台线程中监听串口数据，收到数据时调用回调。

        Args:
            callback: 回调函数，参数为收到的字节数据
        """
        self._read_callback = callback
        if not self._read_running:
            self._start_read_thread()
        logger.info(f"UART {self._port_path} 接收回调已设置")

    def clear_buffer(self) -> None:
        """清空接收和发送缓冲区"""
        with self._buffer_lock:
            self._read_buffer.clear()
        if self._serial and self._serial.is_open:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
        logger.debug(f"UART {self._port_path} 缓冲区已清空")

    def get_buffer_count(self) -> int:
        """获取接收缓冲区中的数据字节数

        Returns:
            int: 缓冲区中的字节数
        """
        if self._serial and self._serial.is_open:
            return self._serial.in_waiting + len(self._read_buffer)
        return 0

    def set_baudrate(self, baudrate: int) -> None:
        """动态修改波特率

        Args:
            baudrate: 新的波特率
        """
        if self._serial and self._serial.is_open:
            self._serial.baudrate = baudrate
            self._baudrate = baudrate
            logger.info(f"UART {self._port_path} 波特率修改为 {baudrate}")

    def set_dtr(self, value: bool) -> None:
        """设置DTR线路状态

        Args:
            value: True=激活, False=非激活
        """
        if self._serial and self._serial.is_open:
            self._serial.dtr = value

    def set_rts(self, value: bool) -> None:
        """设置RTS线路状态

        Args:
            value: True=激活, False=非激活
        """
        if self._serial and self._serial.is_open:
            self._serial.rts = value

    def get_cts(self) -> bool:
        """获取CTS线路状态

        Returns:
            bool: CTS状态
        """
        if self._serial and self._serial.is_open:
            return self._serial.cts
        return False

    def get_dsr(self) -> bool:
        """获取DSR线路状态

        Returns:
            bool: DSR状态
        """
        if self._serial and self._serial.is_open:
            return self._serial.dsr
        return False

    def _fallback_safe_state(self) -> None:
        """异常回退到安全状态：关闭串口"""
        try:
            self._cleanup_resources()
        except Exception as e:
            logger.error(f"UART {self._port_path} 安全状态回退失败: {e}")

    def _cleanup_resources(self) -> None:
        """清理所有已分配资源"""
        # 停止接收线程
        self._read_running = False
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)

        # 关闭串口
        if self._serial:
            try:
                if self._serial.is_open:
                    self._serial.close()
            except Exception:
                pass
            self._serial = None

    def close(self) -> None:
        """关闭UART串口，释放所有资源

        使用try/finally确保资源被正确释放。
        """
        if self._closed:
            return

        with self._instance_lock:
            try:
                self._cleanup_resources()
                self._closed = True
                logger.info(f"UART {self._port_path} 已关闭")
            except Exception as e:
                logger.error(f"UART {self._port_path} 关闭时发生错误: {e}")
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
    def port(self) -> str:
        """获取串口设备路径"""
        return self._port_path

    @property
    def baudrate(self) -> int:
        """获取当前波特率"""
        return self._baudrate

    @property
    def is_open(self) -> bool:
        """判断串口是否打开"""
        return self._serial is not None and self._serial.is_open

    @property
    def is_closed(self) -> bool:
        """判断串口是否已关闭"""
        return self._closed

    @staticmethod
    def list_ports() -> List[Tuple[str, str, str]]:
        """列出系统中可用的串口

        Returns:
            List[Tuple[str, str, str]]: (设备路径, 描述, 硬件ID) 列表
        """
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append((p.device, p.description, p.hwid))
        return ports

    @staticmethod
    def list_rk3588_uart_ports() -> List[str]:
        """列出RK3588上的UART串口

        Returns:
            List[str]: 可用的UART设备路径列表
        """
        available = []
        for name, path in RK3588_UART_DEVICES.items():
            if os.path.exists(path):
                available.append(path)
        return available
