"""
ESP32桥接通信模块

基于WiFi/串口双模通信的ESP32微控制器桥接驱动。
支持远程传感器读取、OTA更新触发和MQTT遥测。
适用于OrangePi Kunpeng Pro (RK3588)平台。

功能特性:
    - WiFi TCP/UDP通信 + UART串口双模
    - 远程传感器数据读取
    - OTA固件更新触发
    - MQTT状态上报
    - 自动重连与心跳检测
    - WiFi信号质量监控

通信协议:
    命令帧: {JSON-RPC 2.0格式}
    {
        "jsonrpc": "2.0",
        "method": "method_name",
        "params": {...},
        "id": 1
    }

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ESP32Command(Enum):
    """ESP32 JSON-RPC方法名枚举"""
    GET_STATUS = "getStatus"
    GET_SENSOR_DATA = "getSensorData"
    GET_WIFI_INFO = "getWiFiInfo"
    SET_GPIO = "setGPIO"
    READ_GPIO = "readGPIO"
    SET_PWM = "setPWM"
    TRIGGER_OTA = "triggerOTA"
    OTA_STATUS = "getOTAStatus"
    SET_LED = "setLED"
    RESTART = "restart"
    PING = "ping"


class ESP32Transport(Enum):
    """通信方式枚举"""
    WIFI_TCP = "wifi_tcp"
    WIFI_UDP = "wifi_udp"
    UART = "uart"
    AUTO = "auto"


@dataclass
class WiFiInfo:
    """WiFi连接信息

    属性:
        ssid: 连接的WiFi名称
        rssi: 信号强度(dBm)
        ip: 本机IP地址
        mac: MAC地址
        channel: WiFi信道
        connected: 是否已连接
    """
    ssid: str = ""
    rssi: int = 0
    ip: str = "0.0.0.0"
    mac: str = "00:00:00:00:00:00"
    channel: int = 0
    connected: bool = False


@dataclass
class OTAStatus:
    """OTA更新状态

    属性:
        in_progress: 是否正在进行OTA
        progress_percent: 下载进度(%)
        current_version: 当前版本
        target_version: 目标版本
        last_error: 上次错误信息
        reboot_required: 是否需要重启
    """
    in_progress: bool = False
    progress_percent: float = 0.0
    current_version: str = ""
    target_version: str = ""
    last_error: str = ""
    reboot_required: bool = False


@dataclass
class ESP32Config:
    """ESP32桥接配置

    属性:
        transport: 通信方式
        uart_port: UART设备路径
        uart_baudrate: UART波特率
        wifi_host: WiFi连接IP地址
        wifi_port: WiFi端口号
        heartbeat_interval: 心跳间隔(秒)
        timeout: 通信超时(秒)
        auto_reconnect: 是否自动重连
        max_retries: 最大重试次数
    """
    transport: ESP32Transport = ESP32Transport.UART
    uart_port: str = "/dev/ttyS3"
    uart_baudrate: int = 115200
    wifi_host: str = "192.168.4.1"
    wifi_port: int = 8080
    heartbeat_interval: float = 1.0
    timeout: float = 2.0
    auto_reconnect: bool = True
    max_retries: int = 3


class ESP32Bridge:
    """ESP32桥接通信驱动类

    提供通过WiFi或UART与ESP32通信的接口,支持远程传感器读取、
    GPIO控制和OTA固件更新。

    示例:
        >>> config = ESP32Config(transport=ESP32Transport.UART)
        >>> bridge = ESP32Bridge(config)
        >>> await bridge.initialize()
        >>> wifi = await bridge.get_wifi_info()
        >>> print(f"WiFi信号: {wifi.rssi} dBm")
        >>> await bridge.trigger_ota("http://server/firmware.bin")
        >>> await bridge.shutdown()

    属性:
        config: 桥接配置
        _connected: 连接状态
        _transport: 当前使用的传输层
    """

    # JSON-RPC常量
    JSONRPC_VERSION: str = "2.0"
    DEFAULT_TIMEOUT: float = 5.0
    RECONNECT_DELAY: float = 2.0

    def __init__(self, config: ESP32Config | None = None) -> None:
        """初始化ESP32桥接

        参数:
            config: 桥接配置,None则使用默认配置
        """
        self.config: ESP32Config = config or ESP32Config()

        # 状态
        self._initialized: bool = False
        self._connected: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        self._request_id: int = 0

        # 传输层对象(延迟初始化)
        self._serial: Any = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None

        # 心跳
        self._heartbeat_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._last_heartbeat: float = 0.0
        self._heartbeat_missed: int = 0

        # 状态缓存
        self._wifi_info: WiFiInfo = WiFiInfo()
        self._ota_status: OTAStatus = OTAStatus()

        # 回调
        self._sensor_callbacks: list[Callable[[dict], None]] = []
        self._disconnect_callbacks: list[Callable[[], None]] = []

    async def initialize(self) -> bool:
        """初始化ESP32通信

        根据配置的传输方式初始化UART或WiFi连接,
        启动心跳检测。

        返回:
            bool: 初始化成功返回True
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                if self.config.transport in (ESP32Transport.UART, ESP32Transport.AUTO):
                    success = await self._init_uart()
                    if success:
                        self.config.transport = ESP32Transport.UART
                    elif self.config.transport == ESP32Transport.AUTO:
                        success = await self._init_wifi()
                        if success:
                            self.config.transport = ESP32Transport.WIFI_TCP
                else:
                    success = await self._init_wifi()

                if not success:
                    logger.error("ESP32初始化失败: 所有传输方式均不可用")
                    return False

                self._initialized = True
                self._connected = True
                self._last_heartbeat = time.monotonic()

                # 启动心跳
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(), name="esp32_heartbeat"
                )

                logger.info(
                    f"ESP32桥接初始化成功: "
                    f"transport={self.config.transport.value}"
                )
                return True

            except Exception as e:
                logger.error(f"ESP32桥接初始化失败: {e}")
                return False

    async def _init_uart(self) -> bool:
        """初始化UART串口通信(内部方法)

        返回:
            bool: 初始化成功返回True
        """
        try:
            import serial
            self._serial = serial.Serial(
                port=self.config.uart_port,
                baudrate=self.config.uart_baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
            logger.debug(f"UART初始化: {self.config.uart_port}")
            return True
        except ImportError:
            logger.debug("pyserial未安装,UART不可用")
            return False
        except Exception as e:
            logger.debug(f"UART初始化失败: {e}")
            return False

    async def _init_wifi(self) -> bool:
        """初始化WiFi TCP通信(内部方法)

        返回:
            bool: 初始化成功返回True
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.config.wifi_host, self.config.wifi_port
                ),
                timeout=self.config.timeout,
            )
            logger.debug(f"WiFi TCP连接: {self.config.wifi_host}:{self.config.wifi_port}")
            return True
        except Exception as e:
            logger.debug(f"WiFi初始化失败: {e}")
            return False

    def _build_jsonrpc(
        self, method: str, params: dict | None = None
    ) -> dict[str, Any]:
        """构建JSON-RPC请求(内部方法)

        参数:
            method: 方法名
            params: 参数字典

        返回:
            dict: JSON-RPC请求字典
        """
        self._request_id += 1
        request: dict[str, Any] = {
            "jsonrpc": self.JSONRPC_VERSION,
            "method": method,
            "id": self._request_id,
        }
        if params is not None:
            request["params"] = params
        return request

    async def _send_request(
        self,
        request: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """发送JSON-RPC请求并等待响应(内部方法)

        参数:
            request: JSON-RPC请求字典
            timeout: 超时时间

        返回:
            dict: 响应字典或None
        """
        timeout = timeout or self.config.timeout
        request_json = json.dumps(request) + "\n"

        for attempt in range(self.config.max_retries):
            try:
                if self._serial is not None:
                    # UART模式
                    self._serial.reset_input_buffer()
                    self._serial.write(request_json.encode())
                    self._serial.flush()

                    # 读取响应
                    response = await asyncio.wait_for(
                        self._read_uart_response(), timeout=timeout
                    )

                elif self._writer is not None:
                    # WiFi TCP模式
                    self._writer.write(request_json.encode())
                    await self._writer.drain()

                    response = await asyncio.wait_for(
                        self._read_tcp_response(), timeout=timeout
                    )
                else:
                    # 模拟模式
                    await asyncio.sleep(0.01)
                    return self._mock_response(request)

                if response is not None:
                    return response

            except asyncio.TimeoutError:
                logger.warning(f"请求超时 (尝试 {attempt+1})")
            except Exception as e:
                logger.error(f"请求异常: {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(0.1 * (attempt + 1))

        return None

    async def _read_uart_response(self) -> dict[str, Any] | None:
        """读取UART响应(内部方法)

        返回:
            dict: 解析后的JSON响应或None
        """
        if self._serial is None:
            return None

        buffer = bytearray()
        start_time = time.monotonic()

        while time.monotonic() - start_time < 2.0:
            available = self._serial.in_waiting
            if available > 0:
                buffer.extend(self._serial.read(available))

                # 查找完整JSON
                try:
                    text = buffer.decode("utf-8")
                    for line in text.split("\n"):
                        line = line.strip()
                        if line:
                            response = json.loads(line)
                            return response
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            await asyncio.sleep(0.001)

        return None

    async def _read_tcp_response(self) -> dict[str, Any] | None:
        """读取TCP响应(内部方法)

        返回:
            dict: 解析后的JSON响应或None
        """
        if self._reader is None:
            return None

        try:
            line = await self._reader.readline()
            if line:
                return json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug(f"TCP响应解析失败: {e}")
        except Exception as e:
            logger.error(f"TCP读取异常: {e}")

        return None

    def _mock_response(self, request: dict[str, Any]) -> dict[str, Any]:
        """生成模拟响应(内部方法)

        在模拟模式下返回预定义的响应数据。

        参数:
            request: 请求字典

        返回:
            dict: 模拟响应
        """
        method = request.get("method", "")
        req_id = request.get("id", 0)

        mock_results: dict[str, Any] = {
            ESP32Command.GET_STATUS.value: {
                "uptime": 3600,
                "free_heap": 120000,
                "cpu_usage": 45,
                "temperature": 42.5,
            },
            ESP32Command.GET_WIFI_INFO.value: {
                "ssid": "KunPeng-Home",
                "rssi": -45,
                "ip": "192.168.1.100",
                "mac": "A0:B1:C2:D3:E4:F5",
                "channel": 6,
                "connected": True,
            },
            ESP32Command.GET_SENSOR_DATA.value: {
                "temperature": 24.5,
                "humidity": 55.0,
                "pressure": 1013.25,
                "ambient_light": 320,
                "motion_detected": False,
            },
            ESP32Command.OTA_STATUS.value: {
                "in_progress": False,
                "progress": 0,
                "current_version": "1.2.3",
                "target_version": "",
                "error": "",
                "reboot_required": False,
            },
        }

        result = mock_results.get(method, {"status": "ok"})

        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": req_id,
        }

    async def _heartbeat_loop(self) -> None:
        """心跳检测循环(内部方法)

        定期发送ping命令检测ESP32是否在线。
        """
        logger.debug("ESP32心跳循环已启动")

        while not self._stop_event.is_set():
            try:
                request = self._build_jsonrpc(ESP32Command.PING.value)
                response = await self._send_request(request, timeout=1.0)

                if response and "result" in response:
                    self._last_heartbeat = time.monotonic()
                    self._heartbeat_missed = 0
                    self._connected = True
                else:
                    self._heartbeat_missed += 1
                    logger.warning(f"ESP32心跳丢失: {self._heartbeat_missed}/3")

                    if self._heartbeat_missed >= 3:
                        logger.error("ESP32连接超时")
                        self._connected = False

                        for cb in self._disconnect_callbacks:
                            try:
                                cb()
                            except Exception as e:
                                logger.error(f"断开回调异常: {e}")

                        if self.config.auto_reconnect:
                            await self._attempt_reconnect()

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

        logger.debug("ESP32心跳循环已退出")

    async def _attempt_reconnect(self) -> None:
        """尝试重新连接(内部方法)"""
        logger.info("尝试重新连接ESP32...")

        try:
            if self.config.transport == ESP32Transport.UART:
                if self._serial:
                    self._serial.close()
                await self._init_uart()
            else:
                if self._writer:
                    self._writer.close()
                await self._init_wifi()

            # 验证连接
            request = self._build_jsonrpc(ESP32Command.PING.value)
            response = await self._send_request(request, timeout=2.0)

            if response:
                self._connected = True
                self._heartbeat_missed = 0
                logger.info("ESP32重新连接成功")
            else:
                logger.error("ESP32重新连接失败")

        except Exception as e:
            logger.error(f"ESP32重连异常: {e}")

    async def get_wifi_info(self) -> WiFiInfo:
        """获取WiFi连接信息

        返回:
            WiFiInfo: WiFi信息结构体
        """
        request = self._build_jsonrpc(ESP32Command.GET_WIFI_INFO.value)
        response = await self._send_request(request)

        if response and "result" in response:
            data = response["result"]
            self._wifi_info = WiFiInfo(
                ssid=data.get("ssid", ""),
                rssi=data.get("rssi", 0),
                ip=data.get("ip", ""),
                mac=data.get("mac", ""),
                channel=data.get("channel", 0),
                connected=data.get("connected", False),
            )
            return self._wifi_info

        return self._wifi_info

    async def get_sensor_data(self) -> dict[str, Any] | None:
        """获取远程传感器数据

        从ESP32连接的传感器读取数据(温湿度、光照、人体检测等)。

        返回:
            dict: 传感器数据字典或None
        """
        request = self._build_jsonrpc(ESP32Command.GET_SENSOR_DATA.value)
        response = await self._send_request(request)

        if response and "result" in response:
            data = response["result"]

            # 通知回调
            for cb in self._sensor_callbacks:
                try:
                    cb(data)
                except Exception as e:
                    logger.error(f"传感器回调异常: {e}")

            return data

        return None

    async def set_gpio(self, pin: int, value: bool) -> bool:
        """控制ESP32 GPIO引脚

        参数:
            pin: GPIO引脚号
            value: True=HIGH, False=LOW

        返回:
            bool: 设置成功返回True
        """
        request = self._build_jsonrpc(
            ESP32Command.SET_GPIO.value,
            {"pin": pin, "value": value},
        )
        response = await self._send_request(request)
        return response is not None and "result" in response

    async def set_led(self, r: int, g: int, b: int, brightness: float = 1.0) -> bool:
        """控制ESP32 RGB LED

        参数:
            r: 红色(0-255)
            g: 绿色(0-255)
            b: 蓝色(0-255)
            brightness: 亮度(0.0-1.0)

        返回:
            bool: 设置成功返回True
        """
        request = self._build_jsonrpc(
            ESP32Command.SET_LED.value,
            {
                "r": max(0, min(255, r)),
                "g": max(0, min(255, g)),
                "b": max(0, min(255, b)),
                "brightness": max(0.0, min(1.0, brightness)),
            },
        )
        response = await self._send_request(request)
        return response is not None and "result" in response

    async def trigger_ota(self, firmware_url: str, sha256_hash: str = "") -> bool:
        """触发OTA固件更新

        向ESP32发送OTA更新命令,开始从指定URL下载新固件。

        参数:
            firmware_url: 固件二进制文件URL
            sha256_hash: 固件SHA256校验值(可选)

        返回:
            bool: 触发成功返回True

        注意:
            调用后应轮询get_ota_status()监控更新进度。
        """
        logger.info(f"触发OTA更新: {firmware_url}")

        params: dict[str, Any] = {"url": firmware_url}
        if sha256_hash:
            params["sha256"] = sha256_hash

        request = self._build_jsonrpc(
            ESP32Command.TRIGGER_OTA.value, params
        )
        response = await self._send_request(request, timeout=10.0)

        if response and "result" in response:
            self._ota_status.in_progress = True
            logger.info("OTA更新已触发")
            return True

        logger.error("OTA更新触发失败")
        return False

    async def get_ota_status(self) -> OTAStatus:
        """查询OTA更新状态

        返回:
            OTAStatus: OTA状态信息
        """
        request = self._build_jsonrpc(ESP32Command.OTA_STATUS.value)
        response = await self._send_request(request)

        if response and "result" in response:
            data = response["result"]
            self._ota_status = OTAStatus(
                in_progress=data.get("in_progress", False),
                progress_percent=data.get("progress", 0.0),
                current_version=data.get("current_version", ""),
                target_version=data.get("target_version", ""),
                last_error=data.get("error", ""),
                reboot_required=data.get("reboot_required", False),
            )

        return self._ota_status

    async def restart(self) -> bool:
        """重启ESP32

        返回:
            bool: 命令发送成功返回True
        """
        request = self._build_jsonrpc(ESP32Command.RESTART.value)
        response = await self._send_request(request)
        return response is not None

    def register_sensor_callback(
        self, callback: Callable[[dict], None]
    ) -> None:
        """注册传感器数据回调

        参数:
            callback: 回调函数,接收传感器数据字典
        """
        if callback not in self._sensor_callbacks:
            self._sensor_callbacks.append(callback)

    def register_disconnect_callback(
        self, callback: Callable[[], None]
    ) -> None:
        """注册断开连接回调

        参数:
            callback: 断开回调函数
        """
        if callback not in self._disconnect_callbacks:
            self._disconnect_callbacks.append(callback)

    @property
    def is_connected(self) -> bool:
        """ESP32连接状态"""
        return self._connected

    @property
    def wifi_info(self) -> WiFiInfo:
        """WiFi信息缓存"""
        return self._wifi_info

    async def shutdown(self) -> None:
        """关闭ESP32桥接,释放资源"""
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

                if self._writer:
                    self._writer.close()
                    await self._writer.wait_closed()

                self._initialized = False
                self._connected = False
                logger.info("ESP32桥接已关闭")

            except Exception as e:
                logger.error(f"关闭ESP32桥接异常: {e}")

    def __repr__(self) -> str:
        return (
            f"ESP32Bridge(transport={self.config.transport.value}, "
            f"connected={self._connected})"
        )

    async def __aenter__(self) -> ESP32Bridge:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
