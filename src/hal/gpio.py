"""
RK3588 GPIO硬件抽象层模块

本模块提供对OrangePi Kunpeng Pro (RK3588) 40PIN GPIO的完整控制支持，
基于sysfs和libgpiod两种后端实现，支持输入/输出/中断三种模式。

特性：
    - 支持sysfs和libgpiod双后端（自动探测优先使用libgpiod）
    - RK3588特定GPIO引脚映射表
    - 输入/输出/中断三种模式支持
    - 上下拉电阻配置
    - 硬件超时保护（默认5秒）
    - 异常回退到安全状态
    - 线程安全（锁保护）
    - 中断回调支持（边缘触发）

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

import os
import time
import threading
import logging
from enum import Enum
from typing import Optional, Callable, Dict, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class GPIOMode(Enum):
    """GPIO工作模式枚举"""
    INPUT = "in"
    OUTPUT = "out"
    INTERRUPT = "interrupt"


class GPIOEdge(Enum):
    """GPIO中断边沿触发模式枚举"""
    RISING = "rising"
    FALLING = "falling"
    BOTH = "both"
    NONE = "none"


class GPIOPull(Enum):
    """GPIO上下拉配置枚举"""
    NONE = None
    UP = "up"
    DOWN = "down"


# RK3588 GPIO银行映射表
# RK3588有5个GPIO银行: GPIO0-GPIO4, 每个银行有32个引脚
RK3588_GPIO_BANKS: Dict[str, Dict[str, any]] = {
    "GPIO0": {"base": 0, "pins": 32, "domain": 0},
    "GPIO1": {"base": 32, "pins": 32, "domain": 1},
    "GPIO2": {"base": 64, "pins": 32, "domain": 2},
    "GPIO3": {"base": 96, "pins": 32, "domain": 3},
    "GPIO4": {"base": 128, "pins": 32, "domain": 4},
}

# OrangePi Kunpeng Pro 40PIN引脚映射到RK3588 GPIO
# 格式: 物理引脚号 -> (银行名, 银行内引脚号, 功能列表)
ORANGEPI_40PIN_MAP: Dict[int, Dict[str, any]] = {
    # 第1排 (引脚1-20)
    3: {"bank": "GPIO1", "pin": 4, "func": ["GPIO", "I2C3_SDA"], "i2c_bus": 3},
    5: {"bank": "GPIO1", "pin": 5, "func": ["GPIO", "I2C3_SCL"], "i2c_bus": 3},
    7: {"bank": "GPIO1", "pin": 6, "func": ["GPIO", "SPI1_CLK"], "spi_bus": 1},
    8: {"bank": "GPIO4", "pin": 8, "func": ["GPIO", "UART6_TX"], "uart": "UART6"},
    10: {"bank": "GPIO4", "pin": 7, "func": ["GPIO", "UART6_RX"], "uart": "UART6"},
    11: {"bank": "GPIO4", "pin": 9, "func": ["GPIO", "PWM11"], "pwm": 11},
    12: {"bank": "GPIO4", "pin": 10, "func": ["GPIO", "PWM12"], "pwm": 12},
    13: {"bank": "GPIO1", "pin": 7, "func": ["GPIO", "SPI1_MOSI"], "spi_bus": 1},
    15: {"bank": "GPIO1", "pin": 8, "func": ["GPIO", "SPI1_MISO"], "spi_bus": 1},
    16: {"bank": "GPIO1", "pin": 9, "func": ["GPIO", "SPI1_CS0"], "spi_bus": 1},
    18: {"bank": "GPIO1", "pin": 10, "func": ["GPIO", "SPI1_CS1"]},
    19: {"bank": "GPIO3", "pin": 22, "func": ["GPIO", "UART8_TX"], "uart": "UART8"},
    21: {"bank": "GPIO3", "pin": 23, "func": ["GPIO", "UART8_RX"], "uart": "UART8"},
    22: {"bank": "GPIO3", "pin": 24, "func": ["GPIO", "PWM14"], "pwm": 14},
    23: {"bank": "GPIO3", "pin": 25, "func": ["GPIO", "SPI4_MOSI"], "spi_bus": 4},
    24: {"bank": "GPIO3", "pin": 26, "func": ["GPIO", "SPI4_MISO"], "spi_bus": 4},
    26: {"bank": "GPIO3", "pin": 27, "func": ["GPIO", "SPI4_CLK"], "spi_bus": 4},
    27: {"bank": "GPIO3", "pin": 28, "func": ["GPIO", "I2C4_SDA"], "i2c_bus": 4},
    28: {"bank": "GPIO3", "pin": 29, "func": ["GPIO", "I2C4_SCL"], "i2c_bus": 4},
    29: {"bank": "GPIO3", "pin": 30, "func": ["GPIO", "CAN1_RX"]},
    31: {"bank": "GPIO3", "pin": 31, "func": ["GPIO", "CAN1_TX"]},
    32: {"bank": "GPIO4", "pin": 0, "func": ["GPIO", "PWM0"], "pwm": 0},
    33: {"bank": "GPIO4", "pin": 1, "func": ["GPIO", "PWM1"], "pwm": 1},
    35: {"bank": "GPIO4", "pin": 2, "func": ["GPIO", "I2C8_SDA"], "i2c_bus": 8},
    36: {"bank": "GPIO4", "pin": 3, "func": ["GPIO", "UART0_TX"], "uart": "UART0"},
    37: {"bank": "GPIO4", "pin": 4, "func": ["GPIO", "UART0_RX"], "uart": "UART0"},
    38: {"bank": "GPIO4", "pin": 5, "func": ["GPIO", "I2C8_SCL"], "i2c_bus": 8},
    40: {"bank": "GPIO4", "pin": 6, "func": ["GPIO", "SPI4_CS0"], "spi_bus": 4},
}

# 电源和地线引脚（不可配置为GPIO）
POWER_PINS: set = {1, 2, 4, 6, 9, 14, 17, 20, 25, 30, 34, 39}


class GPIO:
    """RK3588 GPIO控制器类

    提供对RK3588 GPIO引脚的完整控制，支持sysfs和libgpiod两种后端。
    自动探测libgpiod可用性，优先使用高性能的libgpiod接口。

    Args:
        pin: GPIO引脚号（物理引脚号1-40）
        mode: 工作模式，可选 "in"/"out"/"interrupt"
        pull: 上下拉配置，可选 None/"up"/"down"
        timeout: 操作超时时间（秒），默认5.0
        use_libgpiod: 是否强制使用libgpiod，None表示自动探测

    Raises:
        ValueError: 引脚号无效或模式不支持
        RuntimeError: GPIO初始化失败

    Example:
        >>> gpio = GPIO(11, mode="out")  # 使用11号引脚作为输出
        >>> gpio.write(1)  # 输出高电平
        >>> gpio.close()   # 释放资源
    """

    # 类级别锁，保护GPIO全局操作
    _global_lock: threading.Lock = threading.Lock()
    _exported_pins: set = set()
    _libgpiod_available: Optional[bool] = None

    def __init__(
        self,
        pin: int,
        mode: str = "out",
        pull: Optional[str] = None,
        timeout: float = 5.0,
        use_libgpiod: Optional[bool] = None,
    ) -> None:
        self._pin: int = pin
        self._mode: GPIOMode = GPIOMode(mode)
        self._pull: GPIOPull = GPIOPull(pull) if pull else GPIOPull.NONE
        self._timeout: float = timeout
        self._instance_lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._value_fd: Optional[int] = None
        self._edge_callback: Optional[Callable[[int, int], None]] = None
        self._interrupt_thread: Optional[threading.Thread] = None
        self._interrupt_running: bool = False
        self._chip: Optional[object] = None
        self._line: Optional[object] = None
        self._using_libgpiod: bool = False

        # 验证引脚有效性
        if pin in POWER_PINS:
            raise ValueError(f"引脚 {pin} 是电源/地线引脚，不可配置为GPIO")
        if pin not in ORANGEPI_40PIN_MAP:
            raise ValueError(f"引脚 {pin} 不在OrangePi Kunpeng Pro 40PIN映射表中")

        # 自动探测libgpiod
        if use_libgpiod is None:
            use_libgpiod = self._detect_libgpiod()
        self._using_libgpiod = use_libgpiod

        try:
            if self._using_libgpiod:
                self._init_libgpiod()
            else:
                self._init_sysfs()
            logger.info(f"GPIO {pin} 初始化成功, 模式={mode}, 后端={'libgpiod' if self._using_libgpiod else 'sysfs'}")
        except Exception as e:
            self._fallback_safe_state()
            raise RuntimeError(f"GPIO {pin} 初始化失败: {e}") from e

    @classmethod
    def _detect_libgpiod(cls) -> bool:
        """探测libgpiod库是否可用

        Returns:
            bool: libgpiod是否可用
        """
        if cls._libgpiod_available is not None:
            return cls._libgpiod_available
        try:
            import gpiod
            cls._libgpiod_available = True
            return True
        except ImportError:
            cls._libgpiod_available = False
            logger.warning("libgpiod库未安装，回退到sysfs接口")
            return False

    def _get_linux_gpio_num(self) -> int:
        """将物理引脚号转换为Linux GPIO编号

        Returns:
            int: Linux GPIO编号
        """
        pin_info = ORANGEPI_40PIN_MAP[self._pin]
        bank = RK3588_GPIO_BANKS[pin_info["bank"]]
        return bank["base"] + pin_info["pin"]

    def _init_sysfs(self) -> None:
        """使用sysfs接口初始化GPIO

        Raises:
            RuntimeError: sysfs导出失败
        """
        gpio_num = self._get_linux_gpio_num()
        gpio_path = Path(f"/sys/class/gpio/gpio{gpio_num}")

        with self._global_lock:
            if gpio_num not in self._exported_pins:
                if not gpio_path.exists():
                    try:
                        with open("/sys/class/gpio/export", "w") as f:
                            f.write(str(gpio_num))
                        time.sleep(0.1)  # 等待内核创建设备节点
                    except IOError as e:
                        raise RuntimeError(f"导出GPIO {gpio_num} 失败: {e}") from e
                self._exported_pins.add(gpio_num)

        # 设置方向
        direction = "in" if self._mode in (GPIOMode.INPUT, GPIOMode.INTERRUPT) else "out"
        try:
            with open(gpio_path / "direction", "w") as f:
                f.write(direction)
        except IOError as e:
            raise RuntimeError(f"设置GPIO {gpio_num} 方向失败: {e}") from e

        # 设置上下拉（如果内核支持）
        if self._pull != GPIOPull.NONE:
            pull_path = gpio_path / "pull"
            if pull_path.exists():
                try:
                    with open(pull_path, "w") as f:
                        f.write(self._pull.value)
                except IOError as e:
                    logger.warning(f"设置GPIO {gpio_num} 上下拉失败: {e}")

        # 设置中断边沿（中断模式下）
        if self._mode == GPIOMode.INTERRUPT:
            try:
                with open(gpio_path / "edge", "w") as f:
                    f.write(GPIOEdge.BOTH.value)
            except IOError as e:
                raise RuntimeError(f"设置GPIO {gpio_num} 中断边沿失败: {e}") from e

        # 打开value文件描述符以提高性能
        try:
            self._value_fd = os.open(str(gpio_path / "value"), os.O_RDWR if direction == "out" else os.O_RDONLY)
        except OSError as e:
            raise RuntimeError(f"打开GPIO {gpio_num} value文件失败: {e}") from e

    def _init_libgpiod(self) -> None:
        """使用libgpiod接口初始化GPIO

        Raises:
            RuntimeError: libgpiod初始化失败
        """
        try:
            import gpiod
        except ImportError:
            raise RuntimeError("libgpiod库未安装")

        gpio_num = self._get_linux_gpio_num()
        bank_info = ORANGEPI_40PIN_MAP[self._pin]
        bank_name = bank_info["bank"]
        bank_idx = RK3588_GPIO_BANKS[bank_name]["domain"]
        pin_in_bank = bank_info["pin"]

        # 打开GPIO chip
        chip_path = f"/dev/gpiochip{bank_idx}"
        try:
            self._chip = gpiod.Chip(chip_path)
        except OSError as e:
            raise RuntimeError(f"打开GPIO chip {chip_path} 失败: {e}") from e

        # 请求GPIO line
        line_offset = pin_in_bank
        direction_map = {
            GPIOMode.INPUT: gpiod.LINE_REQ_DIR_IN,
            GPIOMode.OUTPUT: gpiod.LINE_REQ_DIR_OUT,
            GPIOMode.INTERRUPT: gpiod.LINE_REQ_EV_BOTH_EDGES,
        }

        # 上下拉配置
        pull_map = {
            GPIOPull.NONE: gpiod.LINE_REQ_FLAG_BIAS_DISABLE,
            GPIOPull.UP: gpiod.LINE_REQ_FLAG_BIAS_PULL_UP,
            GPIOPull.DOWN: gpiod.LINE_REQ_FLAG_BIAS_PULL_DOWN,
        }

        flags = pull_map.get(self._pull, gpiod.LINE_REQ_FLAG_BIAS_DISABLE)
        req_type = direction_map.get(self._mode, gpiod.LINE_REQ_DIR_OUT)

        try:
            self._line = self._chip.get_line(line_offset)
            consumer = "kunpeng-cortex"
            if self._mode == GPIOMode.INTERRUPT:
                self._line.request(
                    consumer=consumer,
                    type=req_type,
                    flags=flags,
                )
                # 启动中断监听线程
                self._start_interrupt_thread()
            elif self._mode == GPIOMode.OUTPUT:
                self._line.request(
                    consumer=consumer,
                    type=req_type,
                    flags=flags,
                    default_val=0,
                )
            else:
                self._line.request(
                    consumer=consumer,
                    type=req_type,
                    flags=flags,
                )
        except OSError as e:
            raise RuntimeError(f"请求GPIO line {line_offset} 失败: {e}") from e

    def _start_interrupt_thread(self) -> None:
        """启动中断监听线程（libgpiod模式下）"""
        self._interrupt_running = True
        self._interrupt_thread = threading.Thread(
            target=self._interrupt_loop,
            name=f"GPIO-IRQ-{self._pin}",
            daemon=True,
        )
        self._interrupt_thread.start()
        logger.debug(f"GPIO {self._pin} 中断监听线程已启动")

    def _interrupt_loop(self) -> None:
        """中断事件循环（在独立线程中运行）"""
        import select
        while self._interrupt_running:
            try:
                if self._line:
                    # 使用poll等待事件，带超时以便检查退出标志
                    fd = self._line.event_get_fd()
                    poll = select.poll()
                    poll.register(fd, select.POLLIN)
                    events = poll.poll(int(self._timeout * 1000))
                    if events:
                        event = self._line.event_read()
                        if event and self._edge_callback:
                            # event.type: 1=RISING_EDGE, 2=FALLING_EDGE
                            edge_type = 1 if event.type == 1 else 0
                            self._edge_callback(self._pin, edge_type)
            except Exception as e:
                if self._interrupt_running:
                    logger.error(f"GPIO {self._pin} 中断处理异常: {e}")
                time.sleep(0.01)

    def write(self, value: int) -> None:
        """向GPIO引脚写入电平值

        Args:
            value: 电平值，0=低电平，1=高电平

        Raises:
            RuntimeError: GPIO未初始化为输出模式
            ValueError: 写入值无效
            IOError: 写入操作失败
        """
        if self._closed:
            raise RuntimeError(f"GPIO {self._pin} 已关闭")
        if self._mode != GPIOMode.OUTPUT:
            raise RuntimeError(f"GPIO {self._pin} 不是输出模式")
        if value not in (0, 1):
            raise ValueError(f"GPIO写入值必须为0或1，收到: {value}")

        with self._instance_lock:
            try:
                if self._using_libgpiod and self._line:
                    self._line.set_value(value)
                elif self._value_fd is not None:
                    os.lseek(self._value_fd, 0, os.SEEK_SET)
                    os.write(self._value_fd, b"1" if value else b"0")
                else:
                    raise RuntimeError("GPIO未正确初始化")
            except Exception as e:
                logger.error(f"GPIO {self._pin} 写入 {value} 失败: {e}")
                raise IOError(f"GPIO {self._pin} 写入失败: {e}") from e

    def read(self) -> int:
        """读取GPIO引脚电平值

        Returns:
            int: 0=低电平，1=高电平

        Raises:
            RuntimeError: GPIO已关闭
            IOError: 读取操作失败
        """
        if self._closed:
            raise RuntimeError(f"GPIO {self._pin} 已关闭")

        with self._instance_lock:
            try:
                if self._using_libgpiod and self._line:
                    return self._line.get_value()
                elif self._value_fd is not None:
                    os.lseek(self._value_fd, 0, os.SEEK_SET)
                    return int(os.read(self._value_fd, 1).decode())
                else:
                    raise RuntimeError("GPIO未正确初始化")
            except Exception as e:
                logger.error(f"GPIO {self._pin} 读取失败: {e}")
                raise IOError(f"GPIO {self._pin} 读取失败: {e}") from e

    def set_interrupt(self, edge: str, callback: Callable[[int, int], None]) -> None:
        """配置GPIO中断回调

        Args:
            edge: 触发边沿，"rising"/"falling"/"both"
            callback: 回调函数，参数为(pin, value)

        Raises:
            RuntimeError: GPIO不是中断模式
            ValueError: 边沿参数无效
        """
        if self._closed:
            raise RuntimeError(f"GPIO {self._pin} 已关闭")
        if self._mode != GPIOMode.INTERRUPT:
            raise RuntimeError(f"GPIO {self._pin} 不是中断模式，无法设置中断")

        edge_enum = GPIOEdge(edge)
        self._edge_callback = callback

        # 如果是sysfs模式，需要更新edge文件并启动监听线程
        if not self._using_libgpiod:
            gpio_num = self._get_linux_gpio_num()
            try:
                with open(f"/sys/class/gpio/gpio{gpio_num}/edge", "w") as f:
                    f.write(edge_enum.value)
            except IOError as e:
                raise RuntimeError(f"设置GPIO {gpio_num} 中断边沿失败: {e}") from e

            # 启动sysfs中断监听线程
            self._interrupt_running = True
            self._interrupt_thread = threading.Thread(
                target=self._sysfs_interrupt_loop,
                name=f"GPIO-IRQ-sysfs-{self._pin}",
                daemon=True,
            )
            self._interrupt_thread.start()

        logger.info(f"GPIO {self._pin} 中断回调已配置, 边沿={edge}")

    def _sysfs_interrupt_loop(self) -> None:
        """sysfs模式下的中断事件循环"""
        import select
        gpio_num = self._get_linux_gpio_num()
        value_path = f"/sys/class/gpio/gpio{gpio_num}/value"

        try:
            fd = os.open(value_path, os.O_RDONLY)
        except OSError as e:
            logger.error(f"GPIO {self._pin} 中断循环打开value文件失败: {e}")
            return

        try:
            while self._interrupt_running:
                os.lseek(fd, 0, os.SEEK_SET)
                os.read(fd, 1)

                poll = select.poll()
                poll.register(fd, select.POLLPRI)
                events = poll.poll(int(self._timeout * 1000))

                if events:
                    os.lseek(fd, 0, os.SEEK_SET)
                    data = os.read(fd, 1).decode().strip()
                    if data and self._edge_callback:
                        self._edge_callback(self._pin, int(data))
        except Exception as e:
            if self._interrupt_running:
                logger.error(f"GPIO {self._pin} sysfs中断处理异常: {e}")
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def set_pull(self, pull: str) -> None:
        """动态设置上下拉电阻

        Args:
            pull: "up"/"down"/None

        Raises:
            RuntimeError: GPIO已关闭
        """
        if self._closed:
            raise RuntimeError(f"GPIO {self._pin} 已关闭")

        pull_enum = GPIOPull(pull) if pull else GPIOPull.NONE
        self._pull = pull_enum

        # sysfs模式下尝试设置
        if not self._using_libgpiod:
            gpio_num = self._get_linux_gpio_num()
            pull_path = Path(f"/sys/class/gpio/gpio{gpio_num}/pull")
            if pull_path.exists():
                try:
                    with open(pull_path, "w") as f:
                        f.write(pull_enum.value if pull_enum.value else "none")
                except IOError as e:
                    logger.warning(f"GPIO {self._pin} 设置上下拉失败: {e}")

        logger.debug(f"GPIO {self._pin} 上下拉设置为 {pull}")

    def _fallback_safe_state(self) -> None:
        """异常回退到安全状态：释放已分配资源"""
        try:
            self._cleanup_resources()
        except Exception as e:
            logger.error(f"GPIO {self._pin} 安全状态回退失败: {e}")

    def _cleanup_resources(self) -> None:
        """清理所有已分配资源"""
        # 停止中断线程
        self._interrupt_running = False
        if self._interrupt_thread and self._interrupt_thread.is_alive():
            self._interrupt_thread.join(timeout=1.0)

        # 关闭value文件描述符
        if self._value_fd is not None:
            try:
                os.close(self._value_fd)
            except OSError:
                pass
            self._value_fd = None

        # 释放libgpiod资源
        if self._line:
            try:
                self._line.release()
            except Exception:
                pass
            self._line = None

        if self._chip:
            try:
                self._chip.close()
            except Exception:
                pass
            self._chip = None

    def close(self) -> None:
        """关闭GPIO，释放所有资源

        使用try/finally确保资源被正确释放，即使发生异常。
        """
        if self._closed:
            return

        with self._instance_lock:
            try:
                self._cleanup_resources()

                # sysfs模式下取消导出
                if not self._using_libgpiod:
                    gpio_num = self._get_linux_gpio_num()
                    if gpio_num in self._exported_pins:
                        try:
                            with open("/sys/class/gpio/unexport", "w") as f:
                                f.write(str(gpio_num))
                            self._exported_pins.discard(gpio_num)
                        except IOError as e:
                            logger.warning(f"GPIO {self._pin} 取消导出失败: {e}")

                self._closed = True
                logger.info(f"GPIO {self._pin} 已关闭")
            except Exception as e:
                logger.error(f"GPIO {self._pin} 关闭时发生错误: {e}")
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
    def pin(self) -> int:
        """获取物理引脚号"""
        return self._pin

    @property
    def mode(self) -> GPIOMode:
        """获取当前工作模式"""
        return self._mode

    @property
    def is_closed(self) -> bool:
        """判断GPIO是否已关闭"""
        return self._closed

    @staticmethod
    def get_available_pins() -> List[int]:
        """获取可用的GPIO引脚列表

        Returns:
            List[int]: 可用引脚号列表
        """
        return sorted(ORANGEPI_40PIN_MAP.keys())

    @staticmethod
    def get_pin_info(pin: int) -> Optional[Dict[str, any]]:
        """获取引脚详细信息

        Args:
            pin: 物理引脚号

        Returns:
            Dict或None: 引脚信息字典
        """
        return ORANGEPI_40PIN_MAP.get(pin)
