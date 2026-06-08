#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenClaw平台检测器 - RK3588硬件平台自动检测

本模块实现KunPeng-Cortex系统的平台自动检测功能，针对OrangePi Kunpeng Pro
(RK3588)进行深度适配。能够自动检测运行平台、可用硬件资源、外设连接状态，
并生成优化后的硬件配置文件。

检测能力:
    1. CPU架构与性能: 核心数、频率、温度、负载
    2. NPU可用性: RKNN-Toolkit Lite运行时检测
    3. GPIO版本: libgpiod / sysfs接口可用性
    4. 内核版本: Linux内核版本与实时扩展检测
    5. I2C总线: 已连接从机设备地址扫描
    6. USB设备: 连接的外设枚举
    7. 串口: UART端口可用性检测
    8. 内存与存储: 可用容量检测
    9. 性能基准: 快速性能测试

设计参考:
    - RK3588硬件规格书
    - OpenClaw平台适配层设计
    - Linux系统检测最佳实践

依赖:
    - asyncio: 异步检测
    - subprocess: 系统命令调用
    - os/platform: 平台信息

作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform as sys_platform
import re
import struct
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型定义
# ============================================================================

class PlatformType(Enum):
    """平台类型枚举
    
    支持检测的目标平台类型。
    """
    RK3588 = "rk3588"                   # Rockchip RK3588 (OrangePi Kunpeng Pro)
    RK3568 = "rk3568"                   # Rockchip RK3568
    RK3399 = "rk3399"                   # Rockchip RK3399
    RPI4 = "rpi4"                       # Raspberry Pi 4
    RPI5 = "rpi5"                       # Raspberry Pi 5
    X86_64 = "x86_64"                   # x86_64通用平台
    AARCH64 = "aarch64"                 # 其他ARM64平台
    SIMULATOR = "simulator"             # 模拟器/开发环境
    UNKNOWN = "unknown"                 # 未知平台


class NPUStatus(Enum):
    """NPU状态枚举"""
    AVAILABLE = "available"             # 完全可用
    DRIVER_MISSING = "driver_missing"   # 驱动缺失
    LIBRARY_MISSING = "library_missing" # 运行时库缺失
    PERMISSION_DENIED = "permission_denied"  # 权限不足
    UNKNOWN = "unknown"                 # 状态未知
    NOT_SUPPORTED = "not_supported"     # 平台不支持NPU


@dataclass
class CPUInfo:
    """CPU信息数据类
    
    Attributes:
        architecture: CPU架构
        model_name: CPU型号名称
        core_count: 逻辑核心数
        physical_cores: 物理核心数
        max_freq_mhz: 最大频率(MHz)
        current_freq_mhz: 当前频率(MHz)
        temperature_c: CPU温度(摄氏度)
        load_percent: 当前负载百分比
        governor: 频率调节策略
        flags: CPU特性标志
    """
    architecture: str = ""
    model_name: str = ""
    core_count: int = 0
    physical_cores: int = 0
    max_freq_mhz: float = 0.0
    current_freq_mhz: float = 0.0
    temperature_c: float = 0.0
    load_percent: float = 0.0
    governor: str = ""
    flags: str = ""


@dataclass
class NPUInfo:
    """NPU信息数据类
    
    Attributes:
        available: 是否可用
        status: NPU状态
        tops: 理论算力(TOPS)
        driver_version: 驱动版本
        runtime_version: 运行时版本
        memory_mb: NPU专用内存(MB)
        supported_ops: 支持的算子列表
        error_message: 错误信息(不可用时)
    """
    available: bool = False
    status: NPUStatus = NPUStatus.UNKNOWN
    tops: float = 0.0
    driver_version: str = ""
    runtime_version: str = ""
    memory_mb: int = 0
    supported_ops: List[str] = field(default_factory=list)
    error_message: str = ""


@dataclass
class GPIOInfo:
    """GPIO信息数据类
    
    Attributes:
        available: 是否可用
        interface: 接口类型(libgpiod/sysfs)
        gpiochip: GPIO芯片名称
        pin_count: 可用引脚数
        kernel_version: gpio相关内核版本
        groups: GPIO分组信息
    """
    available: bool = False
    interface: str = ""
    gpiochip: str = ""
    pin_count: int = 0
    kernel_version: str = ""
    groups: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryInfo:
    """内存信息数据类
    
    Attributes:
        total_mb: 总内存(MB)
        available_mb: 可用内存(MB)
        used_mb: 已用内存(MB)
        swap_total_mb: 交换空间(MB)
        swap_used_mb: 已用交换(MB)
        hugepages_total: 大页内存数量
        hugepages_free: 空闲大页数量
    """
    total_mb: int = 0
    available_mb: int = 0
    used_mb: int = 0
    swap_total_mb: int = 0
    swap_used_mb: int = 0
    hugepages_total: int = 0
    hugepages_free: int = 0


@dataclass
class I2CBusInfo:
    """I2C总线信息数据类
    
    Attributes:
        bus_number: 总线编号
        device_path: 设备路径
        speed_hz: 当前速率(Hz)
        slave_addresses: 检测到的从机地址列表
        is_available: 总线是否可用
    """
    bus_number: int = 0
    device_path: str = ""
    speed_hz: int = 400000
    slave_addresses: List[int] = field(default_factory=list)
    is_available: bool = False


@dataclass
class USBDeviceInfo:
    """USB设备信息数据类
    
    Attributes:
        bus_number: 总线编号
        device_address: 设备地址
        vendor_id: 厂商ID
        product_id: 产品ID
        vendor_name: 厂商名称
        product_name: 产品名称
        device_class: 设备类别
    """
    bus_number: int = 0
    device_address: int = 0
    vendor_id: str = ""
    product_id: str = ""
    vendor_name: str = ""
    product_name: str = ""
    device_class: str = ""


@dataclass
class BenchmarkResult:
    """性能基准测试结果数据类
    
    Attributes:
        cpu_score: CPU综合评分(越高越好)
        memory_score: 内存带宽评分
        io_score: I/O评分
        npu_score: NPU推理评分(0表示无NPU)
        total_score: 综合评分
        matrix_multiply_ms: 矩阵乘法耗时(ms)
        memory_copy_ms: 内存拷贝耗时(ms)
        numpy_fft_ms: FFT运算耗时(ms)
        inference_ms: 模拟推理耗时(ms)
    """
    cpu_score: float = 0.0
    memory_score: float = 0.0
    io_score: float = 0.0
    npu_score: float = 0.0
    total_score: float = 0.0
    matrix_multiply_ms: float = 0.0
    memory_copy_ms: float = 0.0
    numpy_fft_ms: float = 0.0
    inference_ms: float = 0.0


@dataclass
class PlatformInfo:
    """平台完整信息数据类
    
    汇总所有检测结果的完整平台信息。
    
    Attributes:
        platform_type: 平台类型
        board_name: 开发板名称
        kernel_version: Linux内核版本
        distribution: 发行版信息
        hostname: 主机名
        cpu: CPU信息
        npu: NPU信息
        gpio: GPIO信息
        memory: 内存信息
        i2c_buses: I2C总线列表
        usb_devices: USB设备列表
        uart_ports: 可用串口列表
        spi_buses: 可用SPI总线列表
        benchmark: 性能基准结果
        detection_timestamp: 检测时间戳
        raw_data: 原始检测数据字典
    """
    platform_type: PlatformType = PlatformType.UNKNOWN
    board_name: str = ""
    kernel_version: str = ""
    distribution: str = ""
    hostname: str = ""
    cpu: CPUInfo = field(default_factory=CPUInfo)
    npu: NPUInfo = field(default_factory=NPUInfo)
    gpio: GPIOInfo = field(default_factory=GPIOInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    i2c_buses: List[I2CBusInfo] = field(default_factory=list)
    usb_devices: List[USBDeviceInfo] = field(default_factory=list)
    uart_ports: List[str] = field(default_factory=list)
    spi_buses: List[str] = field(default_factory=list)
    benchmark: BenchmarkResult = field(default_factory=BenchmarkResult)
    detection_timestamp: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 平台检测器核心类
# ============================================================================

class PlatformDetector:
    """平台检测器
    
    自动检测运行平台的硬件配置和可用资源，生成优化后的运行配置。
    专为OrangePi Kunpeng Pro (RK3588)深度适配。
    
    检测流程:
        1. CPU/平台识别: 通过/proc/cpuinfo和device tree
        2. NPU检测: 检查/dev/rknpu和rknn库
        3. GPIO检测: 检查libgpiod和sysfs接口
        4. 总线扫描: I2C/SPI/UART设备枚举
        5. USB扫描: lsusb枚举外设
        6. 性能基准: 快速CPU/内存/NPU测试
        7. 配置生成: 输出硬件配置文件
    
    使用示例:
        detector = PlatformDetector()
        info = await detector.detect()
        config = detector.generate_config()
        
        # 单项检测
        npu_ok, npu_msg = detector.check_npu()
        gpio_info = detector.check_gpio()
        i2c_addrs = await detector.scan_i2c(bus=7)
    
    Attributes:
        _cached_info: 缓存的检测结果
        _detection_done: 是否已完成检测
    """

    # RK3588已知I2C设备地址映射
    _RK3588_KNOWN_I2C_DEVICES: Dict[int, str] = {
        0x34: "AXP2101电源管理",
        0x40: "PCA9685舵机驱动",
        0x3C: "SSD1306 OLED显示屏",
        0x50: "EEPROM",
        0x68: "MPU6050/JY901陀螺仪",
        0x76: "BMP280气压传感器",
        0x77: "BMP180气压传感器",
    }

    # RK3588 UART端口映射
    _RK3588_UART_PORTS = ["/dev/ttyS2", "/dev/ttyS3", "/dev/ttyS4", "/dev/ttyS7"]

    # RK3588 SPI总线映射
    _RK3588_SPI_BUSES = ["/dev/spidev0.0", "/dev/spidev0.1", "/dev/spidev1.0"]

    def __init__(self) -> None:
        """初始化平台检测器"""
        self._cached_info: Optional[PlatformInfo] = None
        self._detection_done: bool = False
        logger.info("平台检测器初始化")

    # ========================================================================
    # 主检测流程
    # ========================================================================

    async def detect(self) -> PlatformInfo:
        """执行完整平台检测
        
        检测所有硬件组件并返回完整的平台信息。检测过程采用异步并行
        以提高效率。
        
        检测顺序:
            1. 基础平台信息(CPU、内核、发行版)
            2. NPU可用性
            3. GPIO接口
            4. I2C总线扫描
            5. USB设备枚举
            6. 串口检测
            7. SPI总线检测
            8. 内存信息
            9. 性能基准测试
        
        Returns:
            完整的PlatformInfo对象
        """
        logger.info("开始完整平台检测...")
        start_time = time.monotonic()

        info = PlatformInfo()
        info.detection_timestamp = time.time()

        # 1. 基础平台信息
        info.platform_type, info.board_name = self._detect_platform_type()
        info.kernel_version = self._detect_kernel_version()
        info.distribution = self._detect_distribution()
        info.hostname = self._detect_hostname()

        # 2. CPU信息
        info.cpu = await self._detect_cpu_info()

        # 3. 内存信息
        info.memory = self._detect_memory_info()

        # 4. NPU检测(并行)
        info.npu = await self._detect_npu()

        # 5. GPIO检测(并行)
        info.gpio = self._detect_gpio()

        # 6. 总线扫描(并行执行)
        i2c_task = asyncio.create_task(self._scan_all_i2c())
        usb_task = asyncio.create_task(self._scan_usb_async())
        uart_task = asyncio.create_task(self._detect_uart_ports_async())
        spi_task = asyncio.create_task(self._detect_spi_buses_async())

        info.i2c_buses = await i2c_task
        info.usb_devices = await usb_task
        info.uart_ports = await uart_task
        info.spi_buses = await spi_task

        # 7. 性能基准测试
        info.benchmark = await self.benchmark()

        # 汇总原始数据
        info.raw_data = {
            "platform_type": info.platform_type.value,
            "board_name": info.board_name,
            "kernel_version": info.kernel_version,
            "cpu_cores": info.cpu.core_count,
            "cpu_freq_mhz": info.cpu.current_freq_mhz,
            "cpu_temp_c": info.cpu.temperature_c,
            "npu_available": info.npu.available,
            "memory_total_mb": info.memory.total_mb,
            "memory_available_mb": info.memory.available_mb,
            "i2c_devices_found": sum(len(b.slave_addresses) for b in info.i2c_buses),
            "usb_devices_found": len(info.usb_devices),
            "uart_ports": len(info.uart_ports),
            "spi_buses": len(info.spi_buses),
            "benchmark_total": info.benchmark.total_score,
        }

        elapsed = (time.monotonic() - start_time) * 1000
        logger.info("平台检测完成，耗时%.1fms", elapsed)
        logger.info("平台: %s (%s), CPU: %s %d核 %.0fMHz, NPU: %s, 内存: %dMB",
                    info.platform_type.value,
                    info.board_name,
                    info.cpu.model_name,
                    info.cpu.core_count,
                    info.cpu.current_freq_mhz,
                    "可用" if info.npu.available else "不可用",
                    info.memory.total_mb)

        self._cached_info = info
        self._detection_done = True
        return info

    # ========================================================================
    # 单项检测方法
    # ========================================================================

    def check_npu(self) -> Tuple[bool, str]:
        """检查NPU可用性
        
        检测RK3588 NPU的可用状态，检查驱动、运行时库和权限。
        
        Returns:
            (是否可用, 状态描述)元组
        """
        # 检查NPU设备节点
        npu_dev_paths = ["/dev/rknpu", "/dev/rknpu0", "/dev/rknn"]
        npu_found = any(Path(p).exists() for p in npu_dev_paths)

        if not npu_found:
            # 检查是否在其他路径
            try:
                result = subprocess.run(
                    ["ls", "/dev/"],
                    capture_output=True, text=True, timeout=5
                )
                if "rknpu" in result.stdout:
                    npu_found = True
            except Exception:
                pass

        if not npu_found:
            return False, "NPU设备节点未找到，请检查内核驱动是否加载"

        # 检查RKNN库
        rknn_paths = [
            "/usr/local/lib/librknnrt.so",
            "/usr/lib/librknnrt.so",
            "/usr/lib/aarch64-linux-gnu/librknnrt.so",
        ]
        rknn_found = any(Path(p).exists() for p in rknn_paths)

        if not rknn_found:
            try:
                result = subprocess.run(
                    ["python3", "-c", "import rknnlite; print('OK')"],
                    capture_output=True, text=True, timeout=5
                )
                if "OK" in result.stdout:
                    rknn_found = True
            except Exception:
                pass

        if npu_found and rknn_found:
            return True, "NPU可用: RK3588 6-8 TOPS"
        elif npu_found and not rknn_found:
            return False, "NPU驱动已加载但RKNN运行时库未安装"
        else:
            return False, "NPU不可用"

    def check_gpio(self) -> Dict[str, Any]:
        """检查GPIO接口可用性
        
        检测libgpiod和sysfs两种GPIO接口的可用性。
        
        Returns:
            GPIO信息字典
        """
        result: Dict[str, Any] = {
            "available": False,
            "interface": "",
            "gpiochip": "",
            "pin_count": 0,
            "error": "",
        }

        # 检查libgpiod
        try:
            gpiodetect = subprocess.run(
                ["gpiodetect"],
                capture_output=True, text=True, timeout=5
            )
            if gpiodetect.returncode == 0:
                result["available"] = True
                result["interface"] = "libgpiod"
                # 解析gpiochip信息
                for line in gpiodetect.stdout.strip().split("\n"):
                    if "gpiochip" in line:
                        result["gpiochip"] = line.split()[0]
                        # 解析引脚数
                        match = re.search(r'(\d+)\s+lines', line)
                        if match:
                            result["pin_count"] = int(match.group(1))
                return result
        except FileNotFoundError:
            pass
        except Exception as e:
            result["error"] = f"libgpiod检测失败: {e}"

        # 检查sysfs GPIO
        sysfs_gpio = Path("/sys/class/gpio")
        if sysfs_gpio.exists():
            result["available"] = True
            result["interface"] = "sysfs"
            result["gpiochip"] = "sysfs"
            # 尝试导出引脚检查可用数量
            try:
                gpio_lines = list(sysfs_gpio.glob("gpio*"))
                result["pin_count"] = len(gpio_lines)
            except Exception:
                result["pin_count"] = 40  # 默认值
            return result

        return result

    async def scan_i2c(self, bus: int = 7) -> List[int]:
        """扫描I2C总线上的从机设备
        
        使用i2cdetect工具扫描指定I2C总线上的设备地址。
        RK3588标准I2C总线编号: 0, 1, 2, 3, 4, 5, 7
        
        Args:
            bus: I2C总线编号，RK3588常用7号总线
            
        Returns:
            检测到的从机地址列表(7位地址)
        """
        device_path = f"/dev/i2c-{bus}"
        if not Path(device_path).exists():
            logger.debug("I2C总线 %d 设备不存在: %s", bus, device_path)
            return []

        addresses: List[int] = []

        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "i2cdetect", "-y", str(bus),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=10.0
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=15.0)

            if result.returncode != 0:
                logger.warning("i2cdetect扫描失败: %s", stderr.decode().strip())
                return addresses

            # 解析i2cdetect输出
            for line in stdout.decode().split("\n"):
                # 跳过表头
                if not line or line[0] not in "0123456789abcdef":
                    continue
                # 解析每行 (格式: "00: -- -- -- ...")
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                row = int(parts[0], 16)
                cols = parts[1].strip().split()
                for col_idx, cell in enumerate(cols[:16]):
                    if cell not in ("--", "UU"):
                        addr = row + col_idx
                        addresses.append(addr)

        except asyncio.TimeoutError:
            logger.warning("I2C扫描总线 %d 超时", bus)
        except FileNotFoundError:
            logger.debug("i2cdetect工具未安装")
        except Exception as e:
            logger.warning("I2C扫描总线 %d 错误: %s", bus, e)

        if addresses:
            logger.info("I2C总线 %d 发现 %d 个设备: %s",
                        bus, len(addresses),
                        [f"0x{a:02X}({self._RK3588_KNOWN_I2C_DEVICES.get(a, '未知')})"
                         for a in addresses])
        return addresses

    async def scan_usb(self) -> List[Dict[str, str]]:
        """扫描USB设备
        
        使用lsusb命令枚举连接的USB设备。
        
        Returns:
            USB设备信息字典列表
        """
        devices: List[Dict[str, str]] = []

        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "lsusb",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=10.0
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10.0)

            if result.returncode != 0:
                return devices

            for line in stdout.decode().split("\n"):
                if not line.strip():
                    continue
                # 解析: Bus 001 Device 002: ID 1a2b:3c4d Vendor Product
                match = re.match(
                    r'Bus\s+(\d+)\s+Device\s+(\d+):\s+ID\s+([0-9a-f]{4}):([0-9a-f]{4})\s+(.*)',
                    line, re.IGNORECASE
                )
                if match:
                    vendor_id = match.group(3).upper()
                    product_id = match.group(4).upper()
                    name = match.group(5).strip()

                    # 识别已知设备
                    device_info = self._identify_usb_device(vendor_id, product_id, name)
                    devices.append(device_info)

        except asyncio.TimeoutError:
            logger.warning("USB扫描超时")
        except FileNotFoundError:
            logger.debug("lsusb工具未安装")
        except Exception as e:
            logger.warning("USB扫描错误: %s", e)

        return devices

    # ========================================================================
    # 性能基准测试
    # ========================================================================

    async def benchmark(self) -> BenchmarkResult:
        """执行快速性能基准测试
        
        通过CPU矩阵运算、内存拷贝、FFT变换等测试评估系统性能。
        测试时间控制在5秒以内，适合启动时快速评估。
        
        Returns:
            BenchmarkResult对象
        """
        logger.info("开始性能基准测试...")
        start_time = time.monotonic()
        result = BenchmarkResult()

        # 1. 矩阵乘法测试 (CPU性能)
        try:
            t0 = time.monotonic()
            a = np.random.randn(512, 512).astype(np.float32)
            b = np.random.randn(512, 512).astype(np.float32)
            c = np.matmul(a, b)
            result.matrix_multiply_ms = (time.monotonic() - t0) * 1000
            result.cpu_score = max(0, 1000 - result.matrix_multiply_ms)
        except Exception as e:
            logger.warning("矩阵乘法测试失败: %s", e)

        # 2. 内存拷贝测试
        try:
            t0 = time.monotonic()
            arr = np.random.randn(1000000).astype(np.float32)
            arr_copy = arr.copy()
            result.memory_copy_ms = (time.monotonic() - t0) * 1000
            result.memory_score = max(0, 1000 - result.memory_copy_ms * 10)
        except Exception as e:
            logger.warning("内存拷贝测试失败: %s", e)

        # 3. FFT运算测试
        try:
            t0 = time.monotonic()
            arr = np.random.randn(65536).astype(np.complex64)
            fft_result = np.fft.fft(arr)
            result.numpy_fft_ms = (time.monotonic() - t0) * 1000
        except Exception as e:
            logger.warning("FFT测试失败: %s", e)

        # 4. 模拟推理测试
        try:
            t0 = time.monotonic()
            # 模拟卷积运算
            input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)
            kernel = np.random.randn(3, 3, 3, 16).astype(np.float32)
            # 简化的卷积模拟
            output = np.tanh(np.dot(input_data.reshape(1, -1),
                                     kernel.reshape(-1, 16)))
            result.inference_ms = (time.monotonic() - t0) * 1000
        except Exception as e:
            logger.warning("推理模拟测试失败: %s", e)

        # 计算综合评分
        result.total_score = (result.cpu_score * 0.4 +
                              result.memory_score * 0.3 +
                              result.io_score * 0.1 +
                              result.npu_score * 0.2)

        elapsed = (time.monotonic() - start_time) * 1000
        logger.info("性能基准测试完成，耗时%.1fms，综合评分 %.1f",
                    elapsed, result.total_score)
        return result

    # ========================================================================
    # 配置生成
    # ========================================================================

    def generate_config(self) -> Dict[str, Any]:
        """生成硬件配置文件
        
        基于检测结果生成优化后的硬件配置字典，用于系统初始化。
        
        Returns:
            硬件配置字典
        """
        if not self._cached_info:
            logger.warning("尚未执行检测，先生成默认配置")
            return self._generate_default_config()

        info = self._cached_info

        config: Dict[str, Any] = {
            "platform": info.platform_type.value,
            "board_name": info.board_name,
            "version": "1.0.0",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),

            "cpu": {
                "architecture": info.cpu.architecture,
                "core_count": info.cpu.core_count,
                "max_freq_mhz": info.cpu.max_freq_mhz,
                "governor": info.cpu.governor,
            },

            "npu": {
                "available": info.npu.available,
                "tops": info.npu.tops if info.npu.available else 0,
                "memory_mb": info.npu.memory_mb,
            },

            "gpio": {
                "available": info.gpio.available,
                "interface": info.gpio.interface,
                "gpiochip": info.gpio.gpiochip,
                "pin_count": info.gpio.pin_count,
            },

            "i2c": {
                "available": len(info.i2c_buses) > 0,
                "buses": [
                    {
                        "bus_number": b.bus_number,
                        "device_path": b.device_path,
                        "slave_count": len(b.slave_addresses),
                        "slaves": [
                            {"address": f"0x{a:02X}",
                             "name": self._RK3588_KNOWN_I2C_DEVICES.get(a, "未知")}
                            for a in b.slave_addresses
                        ],
                    }
                    for b in info.i2c_buses
                ],
            },

            "uart": {
                "available": len(info.uart_ports) > 0,
                "ports": info.uart_ports,
                "default_baudrate": 115200,
            },

            "spi": {
                "available": len(info.spi_buses) > 0,
                "buses": info.spi_buses,
            },

            "pwm": {
                "available": info.gpio.available,  # PWM依赖GPIO子系统
                "channels": 4,
                "default_frequency": 50,
            },

            "memory": {
                "total_mb": info.memory.total_mb,
                "available_mb": info.memory.available_mb,
            },

            "benchmark": {
                "total_score": info.benchmark.total_score,
                "cpu_score": info.benchmark.cpu_score,
                "memory_score": info.benchmark.memory_score,
            },

            "hal_plugins": self._generate_hal_plugin_config(info),
        }

        return config

    def _generate_default_config(self) -> Dict[str, Any]:
        """生成默认配置(检测失败时使用)
        
        Returns:
            默认配置字典
        """
        return {
            "platform": "rk3588",
            "version": "1.0.0-default",
            "cpu": {"architecture": "aarch64", "core_count": 8, "max_freq_mhz": 2400},
            "npu": {"available": False, "tops": 0},
            "gpio": {"available": True, "interface": "sysfs", "pin_count": 40},
            "i2c": {"available": True, "buses": [1, 2, 5, 7]},
            "uart": {"available": True, "ports": ["/dev/ttyS2", "/dev/ttyS3"]},
            "spi": {"available": True, "buses": ["/dev/spidev0.0"]},
            "pwm": {"available": True, "channels": 4},
        }

    def _generate_hal_plugin_config(self, info: PlatformInfo) -> List[Dict[str, Any]]:
        """生成HAL插件配置
        
        根据检测的硬件配置生成对应的HAL插件配置。
        
        Args:
            info: 平台信息
            
        Returns:
            插件配置列表
        """
        plugins: List[Dict[str, Any]] = []

        # GPIO插件
        if info.gpio.available:
            plugins.append({
                "name": "hal_gpio_rk3588",
                "priority": 0,
                "auto_load": True,
                "config": {
                    "gpiochip": info.gpio.gpiochip or "gpiochip0",
                    "pin_count": info.gpio.pin_count or 32,
                },
            })

        # I2C插件
        if info.i2c_buses:
            plugins.append({
                "name": "hal_i2c_dev",
                "priority": 0,
                "auto_load": True,
                "config": {
                    "buses": [b.device_path for b in info.i2c_buses],
                    "max_speed_hz": 400000,
                },
            })

        # UART插件
        if info.uart_ports:
            plugins.append({
                "name": "hal_uart_linux",
                "priority": 0,
                "auto_load": True,
                "config": {
                    "ports": info.uart_ports,
                    "default_baudrate": 115200,
                },
            })

        # SPI插件
        if info.spi_buses:
            plugins.append({
                "name": "hal_spi_dev",
                "priority": 0,
                "auto_load": True,
                "config": {
                    "buses": info.spi_buses,
                    "max_speed_hz": 50000000,
                },
            })

        # PWM插件
        if info.gpio.available:
            plugins.append({
                "name": "hal_pwm_rk3588",
                "priority": 0,
                "auto_load": True,
                "config": {
                    "pwm_chip": "pwmchip4",
                    "num_channels": 4,
                    "max_frequency": 1000000,
                    "resolution_bits": 16,
                },
            })

        # NPU插件
        if info.npu.available:
            plugins.append({
                "name": "hal_npu_rk3588",
                "priority": 1,
                "auto_load": True,
                "config": {
                    "tops": info.npu.tops,
                    "memory_mb": info.npu.memory_mb,
                    "supported_formats": ["INT8", "FP16"],
                },
            })

        return plugins

    # ========================================================================
    # 私有检测方法
    # ========================================================================

    def _detect_platform_type(self) -> Tuple[PlatformType, str]:
        """检测平台类型
        
        通过/proc/device-tree/model和/proc/cpuinfo识别平台。
        
        Returns:
            (平台类型, 板子名称)元组
        """
        # 尝试读取device tree model
        model_paths = [
            "/proc/device-tree/model",
            "/sys/firmware/devicetree/base/model",
        ]
        for path in model_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    model = f.read().strip().strip("\x00")
                    if "rk3588" in model.lower() or "rockchip" in model.lower():
                        if "orangepi" in model.lower() or "orange pi" in model.lower():
                            return PlatformType.RK3588, "OrangePi Kunpeng Pro"
                        return PlatformType.RK3588, model
                    if "raspberry pi 4" in model.lower():
                        return PlatformType.RPI4, model
                    if "raspberry pi 5" in model.lower():
                        return PlatformType.RPI5, model
            except FileNotFoundError:
                continue
            except Exception:
                continue

        # 通过CPU信息检测
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
                if "rk3588" in cpuinfo:
                    return PlatformType.RK3588, "RK3588 (detected from cpuinfo)"
                if "rk3568" in cpuinfo:
                    return PlatformType.RK3568, "RK3568"
                if "rk3399" in cpuinfo:
                    return PlatformType.RK3399, "RK3399"
                if "cortex-a76" in cpuinfo and "cortex-a55" in cpuinfo:
                    # RK3588特征: 4xA76 + 4xA55
                    return PlatformType.RK3588, "RK3588 (detected from CPU cores)"
        except Exception:
            pass

        # 通过架构检测
        machine = sys_platform.machine().lower()
        if "aarch64" in machine or "arm64" in machine:
            return PlatformType.AARCH64, f"Generic ARM64 ({machine})"
        elif "x86_64" in machine:
            return PlatformType.X86_64, f"x86_64 ({machine})"

        return PlatformType.UNKNOWN, f"Unknown ({machine})"

    def _detect_kernel_version(self) -> str:
        """检测Linux内核版本
        
        Returns:
            内核版本字符串
        """
        try:
            result = subprocess.run(
                ["uname", "-r"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        try:
            with open("/proc/version", "r") as f:
                version = f.read()
                match = re.search(r'\d+\.\d+\.\d+', version)
                if match:
                    return match.group(0)
        except Exception:
            pass

        return "unknown"

    def _detect_distribution(self) -> str:
        """检测Linux发行版
        
        Returns:
            发行版信息字符串
        """
        try:
            with open("/etc/os-release", "r") as f:
                content = f.read()
                name_match = re.search(r'PRETTY_NAME="([^"]*)"', content)
                if name_match:
                    return name_match.group(1)
                name_match = re.search(r'NAME="([^"]*)"', content)
                if name_match:
                    return name_match.group(1)
        except Exception:
            pass
        return "unknown"

    def _detect_hostname(self) -> str:
        """检测主机名
        
        Returns:
            主机名字符串
        """
        try:
            result = subprocess.run(
                ["hostname"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"

    async def _detect_cpu_info(self) -> CPUInfo:
        """检测CPU详细信息
        
        Returns:
            CPUInfo对象
        """
        info = CPUInfo()
        info.architecture = sys_platform.machine()

        # 读取/proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()

            # 解析型号名
            model_match = re.search(r'model name\s*:\s*(.+)', content)
            if model_match:
                info.model_name = model_match.group(1).strip()
            else:
                # ARM平台可能使用Processor或CPU implementer
                proc_match = re.search(r'Processor\s*:\s*(.+)', content)
                if proc_match:
                    info.model_name = proc_match.group(1).strip()

            # 核心数
            cores = re.findall(r'^processor\s*:', content, re.MULTILINE)
            info.core_count = len(cores) if cores else os.cpu_count() or 1

            # 物理核心
            physical = set(re.findall(r'physical id\s*:\s*(\d+)', content))
            if physical:
                info.physical_cores = len(physical)
            else:
                info.physical_cores = info.core_count // 2 if info.core_count > 1 else 1

            # CPU flags
            flags_match = re.search(r'Features\s*:\s*(.+)', content)
            if flags_match:
                info.flags = flags_match.group(1).strip()

        except Exception as e:
            logger.warning("读取cpuinfo失败: %s", e)
            info.core_count = os.cpu_count() or 1

        # 频率信息
        try:
            freq_path = Path("/sys/devices/system/cpu/cpufreq/policy0")
            if (freq_path / "scaling_max_freq").exists():
                with open(freq_path / "scaling_max_freq", "r") as f:
                    info.max_freq_mhz = int(f.read().strip()) / 1000
            if (freq_path / "scaling_cur_freq").exists():
                with open(freq_path / "scaling_cur_freq", "r") as f:
                    info.current_freq_mhz = int(f.read().strip()) / 1000
            if (freq_path / "scaling_governor").exists():
                with open(freq_path / "scaling_governor", "r") as f:
                    info.governor = f.read().strip()
        except Exception:
            pass

        # CPU温度
        info.temperature_c = self._read_cpu_temperature()

        # CPU负载
        try:
            with open("/proc/loadavg", "r") as f:
                load = f.read().split()
                if load:
                    info.load_percent = float(load[0]) / info.core_count * 100
        except Exception:
            pass

        return info

    def _read_cpu_temperature(self) -> float:
        """读取CPU温度
        
        Returns:
            温度(摄氏度)
        """
        thermal_paths = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
            "/sys/class/hwmon/hwmon0/temp1_input",
        ]
        for path in thermal_paths:
            try:
                with open(path, "r") as f:
                    temp_raw = int(f.read().strip())
                    # 可能是毫摄氏度
                    if temp_raw > 1000:
                        return temp_raw / 1000.0
                    return float(temp_raw)
            except (FileNotFoundError, ValueError):
                continue
        return 0.0

    def _detect_memory_info(self) -> MemoryInfo:
        """检测内存信息
        
        Returns:
            MemoryInfo对象
        """
        info = MemoryInfo()

        try:
            with open("/proc/meminfo", "r") as f:
                content = f.read()

            # 总内存
            total_match = re.search(r'MemTotal:\s+(\d+)', content)
            if total_match:
                info.total_mb = int(total_match.group(1)) // 1024

            # 可用内存
            avail_match = re.search(r'MemAvailable:\s+(\d+)', content)
            if avail_match:
                info.available_mb = int(avail_match.group(1)) // 1024

            info.used_mb = info.total_mb - info.available_mb

            # 交换空间
            swap_total = re.search(r'SwapTotal:\s+(\d+)', content)
            swap_free = re.search(r'SwapFree:\s+(\d+)', content)
            if swap_total:
                info.swap_total_mb = int(swap_total.group(1)) // 1024
            if swap_free:
                swap_free_mb = int(swap_free.group(1)) // 1024
                info.swap_used_mb = info.swap_total_mb - swap_free_mb

        except Exception as e:
            logger.warning("读取内存信息失败: %s", e)

        # 大页内存
        try:
            with open("/proc/sys/vm/nr_hugepages", "r") as f:
                info.hugepages_total = int(f.read().strip())
            huge_path = Path("/sys/kernel/mm/hugepages")
            if huge_path.exists():
                free_pages = list(huge_path.glob("*/free_hugepages"))
                if free_pages:
                    with open(free_pages[0], "r") as f:
                        info.hugepages_free = int(f.read().strip())
        except Exception:
            pass

        return info

    async def _detect_npu(self) -> NPUInfo:
        """检测NPU信息
        
        Returns:
            NPUInfo对象
        """
        info = NPUInfo()
        available, message = self.check_npu()
        info.available = available

        if available:
            info.status = NPUStatus.AVAILABLE
            info.tops = 6.0  # RK3588理论算力
            info.memory_mb = 512  # NPU专用内存

            # 尝试获取驱动版本
            try:
                result = subprocess.run(
                    ["cat", "/sys/module/rknpu/version"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.driver_version = result.stdout.strip()
            except Exception:
                pass

            # 尝试获取RKNN版本
            try:
                result = subprocess.run(
                    ["python3", "-c",
                     "import rknnlite; print(rknnlite.__version__)"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.runtime_version = result.stdout.strip()
            except Exception:
                info.runtime_version = "unknown"

            info.supported_ops = ["CONV", "RELU", "POOL", "FC", "SOFTMAX",
                                   "BATCHNORM", "CONCAT", "ADD", "MUL"]
        else:
            if "驱动" in message:
                info.status = NPUStatus.DRIVER_MISSING
            elif "库" in message:
                info.status = NPUStatus.LIBRARY_MISSING
            else:
                info.status = NPUStatus.NOT_SUPPORTED
            info.error_message = message

        return info

    def _detect_gpio(self) -> GPIOInfo:
        """检测GPIO信息
        
        Returns:
            GPIOInfo对象
        """
        gpio_dict = self.check_gpio()
        info = GPIOInfo(
            available=gpio_dict.get("available", False),
            interface=gpio_dict.get("interface", ""),
            gpiochip=gpio_dict.get("gpiochip", ""),
            pin_count=gpio_dict.get("pin_count", 0),
        )

        # 内核版本
        info.kernel_version = self._detect_kernel_version()

        # GPIO分组(RK3588)
        if info.available:
            info.groups = {
                "GPIO0": {"base": 0, "pins": 32},
                "GPIO1": {"base": 32, "pins": 32},
                "GPIO2": {"base": 64, "pins": 32},
                "GPIO3": {"base": 96, "pins": 32},
                "GPIO4": {"base": 128, "pins": 32},
            }

        return info

    async def _scan_all_i2c(self) -> List[I2CBusInfo]:
        """扫描所有I2C总线
        
        Returns:
            I2CBusInfo列表
        """
        buses = []
        # RK3588常用I2C总线
        for bus_num in [0, 1, 2, 3, 4, 5, 7]:
            device_path = f"/dev/i2c-{bus_num}"
            if Path(device_path).exists():
                addresses = await self.scan_i2c(bus_num)
                buses.append(I2CBusInfo(
                    bus_number=bus_num,
                    device_path=device_path,
                    slave_addresses=addresses,
                    is_available=True,
                ))
        return buses

    async def scan_usb(self) -> List[dict]:
        """扫描USB设备 (同步包装)
        
        Returns:
            USB设备字典列表
        """
        devices = await self._scan_usb_async()
        return [d.__dict__ if hasattr(d, "__dict__") else d for d in devices]

    async def _scan_usb_async(self) -> List[USBDeviceInfo]:
        """异步扫描USB设备
        
        Returns:
            USBDeviceInfo列表
        """
        devices: List[USBDeviceInfo] = []
        usb_dicts = await asyncio.get_event_loop().run_in_executor(
            None, self._scan_usb_sync
        )
        for d in usb_dicts:
            devices.append(USBDeviceInfo(
                bus_number=d.get("bus_number", 0),
                device_address=d.get("device_address", 0),
                vendor_id=d.get("vendor_id", ""),
                product_id=d.get("product_id", ""),
                vendor_name=d.get("vendor_name", ""),
                product_name=d.get("product_name", ""),
                device_class=d.get("device_class", ""),
            ))
        return devices

    def _scan_usb_sync(self) -> List[Dict[str, str]]:
        """同步扫描USB设备
        
        Returns:
            USB设备字典列表
        """
        devices: List[Dict[str, str]] = []
        try:
            result = subprocess.run(
                ["lsusb"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return devices

            for line in result.stdout.split("\n"):
                if not line.strip():
                    continue
                match = re.match(
                    r'Bus\s+(\d+)\s+Device\s+(\d+):\s+ID\s+([0-9a-f]{4}):([0-9a-f]{4})\s+(.*)',
                    line, re.IGNORECASE
                )
                if match:
                    devices.append(self._identify_usb_device(
                        match.group(3).upper(),
                        match.group(4).upper(),
                        match.group(5).strip()
                    ))
        except Exception as e:
            logger.warning("USB同步扫描错误: %s", e)
        return devices

    def _identify_usb_device(self, vendor_id: str, product_id: str,
                             name: str) -> Dict[str, str]:
        """识别USB设备
        
        Args:
            vendor_id: 厂商ID
            product_id: 产品ID
            name: 设备名称
            
        Returns:
            设备信息字典
        """
        # 已知USB设备映射表
        known_devices: Dict[Tuple[str, str], Tuple[str, str]] = {
            ("1D27", "0601"): ("Orbbec", "Astra Pro深度相机"),
            ("046D", "0825"): ("Logitech", "C270摄像头"),
            ("045E", "0773"): ("Microsoft", "Kinect"),
            ("1A86", "7523"): ("QinHeng", "CH340串口"),
            ("10C4", "EA60"): ("Silicon Labs", "CP210x串口"),
            ("0403", "6001"): ("FTDI", "FT232串口"),
            ("0BDA", "C811"): ("Realtek", "RTL8811CU WiFi"),
            ("148F", "5370"): ("Ralink", "RT5370 WiFi"),
            ("0AC8", "C344"): ("Z-Star", "USB摄像头"),
        }

        vendor_name, product_name = known_devices.get(
            (vendor_id, product_id), ("", name)
        )
        if not vendor_name:
            vendor_name = name.split()[0] if name else "Unknown"
            product_name = name

        return {
            "bus_number": 0,
            "device_address": 0,
            "vendor_id": vendor_id,
            "product_id": product_id,
            "vendor_name": vendor_name,
            "product_name": product_name,
            "device_class": "",
        }

    async def _detect_uart_ports_async(self) -> List[str]:
        """异步检测可用串口
        
        Returns:
            可用串口路径列表
        """
        available = []
        for port in self._RK3588_UART_PORTS:
            if Path(port).exists():
                available.append(port)
        return available

    async def _detect_spi_buses_async(self) -> List[str]:
        """异步检测可用SPI总线
        
        Returns:
            可用SPI总线路径列表
        """
        available = []
        for bus in self._RK3588_SPI_BUSES:
            if Path(bus).exists():
                available.append(bus)
        return available

    @property
    def detection_done(self) -> bool:
        """是否已完成检测"""
        return self._detection_done

    @property
    def cached_info(self) -> Optional[PlatformInfo]:
        """获取缓存的检测结果"""
        return self._cached_info