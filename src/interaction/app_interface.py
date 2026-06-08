"""
APP通信接口模块

基于WebSocket的JSON-RPC 2.0协议通信服务器。
支持多客户端连接、实时状态推送和指令接收。
适用于OrangePi Kunpeng Pro (RK3588)平台与移动端APP通信。

功能特性:
    - WebSocket服务器(基于websockets库)
    - JSON-RPC 2.0协议
    - 多客户端连接管理
    - 实时状态广播
    - 点对点消息发送
    - 心跳检测
    - 连接认证(可选)
    - 消息队列(背压保护)

协议规范:
    请求:  {"jsonrpc": "2.0", "method": "...", "params": {...}, "id": 1}
    响应: {"jsonrpc": "2.0", "result": {...}, "id": 1}
    通知: {"jsonrpc": "2.0", "method": "...", "params": {...}}

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ClientState(Enum):
    """客户端连接状态枚举"""
    CONNECTING = "connecting"   # 连接中
    CONNECTED = "connected"     # 已连接
    AUTHENTICATED = "auth"      # 已认证
    DISCONNECTING = "disconnecting" # 断开中
    DISCONNECTED = "disconnected"   # 已断开


class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"         # JSON-RPC请求
    RESPONSE = "response"       # JSON-RPC响应
    NOTIFICATION = "notification"   # JSON-RPC通知
    STATUS = "status"           # 状态推送
    HEARTBEAT = "heartbeat"     # 心跳
    ERROR = "error"             # 错误


@dataclass
class ClientInfo:
    """客户端信息

    属性:
        client_id: 客户端唯一ID
        remote_addr: 远程地址
        connected_at: 连接时间戳
        last_heartbeat: 上次心跳时间
        state: 连接状态
        metadata: 客户端元数据(设备类型、版本等)
    """
    client_id: str = ""
    remote_addr: str = ""
    connected_at: float = 0.0
    last_heartbeat: float = 0.0
    state: ClientState = ClientState.CONNECTING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RPCRequest:
    """JSON-RPC请求结构

    属性:
        method: 方法名
        params: 参数
        request_id: 请求ID
        client_id: 来源客户端ID
    """
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    request_id: int | None = None
    client_id: str = ""


@dataclass
class RPCResponse:
    """JSON-RPC响应结构

    属性:
        result: 结果数据
        error: 错误信息
        request_id: 对应请求ID
    """
    result: Any = None
    error: str | None = None
    request_id: int | None = None


@dataclass
class ServerConfig:
    """服务器配置参数

    属性:
        host: 监听地址
        port: 监听端口
        heartbeat_interval: 心跳间隔(秒)
        heartbeat_timeout: 心跳超时(秒)
        max_clients: 最大客户端数
        max_message_size: 最大消息大小(字节)
        enable_auth: 是否启用认证
        auth_token: 认证令牌
        status_push_interval: 状态推送间隔(秒)
    """
    host: str = "0.0.0.0"
    port: int = 8765
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 15.0
    max_clients: int = 10
    max_message_size: int = 64 * 1024  # 64KB
    enable_auth: bool = False
    auth_token: str = ""
    status_push_interval: float = 1.0


class AppInterface:
    """APP通信接口类

    提供WebSocket服务器,支持多客户端实时通信。
    实现JSON-RPC 2.0协议进行指令交互和状态推送。

    示例:
        >>> app = AppInterface(host="0.0.0.0", port=8765)
        >>> await app.start()
        >>> 
        >>> # 广播状态
        >>> await app.broadcast_status({"cpu": 45, "memory": 60})
        >>> 
        >>> # 发送给特定客户端
        >>> await app.send_to_client("client-001", {"type": "alert"})
        >>> 
        >>> await app.shutdown()

    属性:
        config: 服务器配置
        _clients: 客户端连接字典
        _running: 服务器运行状态
    """

    JSONRPC_VERSION: str = "2.0"
    DEFAULT_TIMEOUT: float = 10.0

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        config: ServerConfig | None = None,
    ) -> None:
        """初始化APP通信接口

        参数:
            host: 监听地址,默认0.0.0.0
            port: 监听端口,默认8765
            config: 服务器配置,None则使用默认配置
        """
        self.config: ServerConfig = config or ServerConfig()

        # 允许通过参数覆盖
        if host:
            self.config.host = host
        if port:
            self.config.port = port

        # 状态
        self._initialized: bool = False
        self._running: bool = False
        self._server: Any = None
        self._lock: asyncio.Lock = asyncio.Lock()

        # 客户端管理
        self._clients: dict[str, Any] = {}              # client_id -> websocket
        self._client_info: dict[str, ClientInfo] = {}    # client_id -> info

        # 请求处理
        self._request_counter: int = 0
        self._pending_requests: dict[int, asyncio.Future] = {}

        # 回调注册
        self._method_handlers: dict[str, Callable] = {}
        self._connect_callbacks: list[Callable[[str], None]] = []
        self._disconnect_callbacks: list[Callable[[str], None]] = []
        self._message_callbacks: list[Callable[[str, dict], None]] = []

        # 状态推送任务
        self._status_push_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

        # 消息队列(背压保护)
        self._message_queues: dict[str, asyncio.Queue] = {}
        self._max_queue_size: int = 100

    async def start(self) -> bool:
        """启动WebSocket服务器

        开始监听客户端连接,处理消息和心跳检测。

        返回:
            bool: 启动成功返回True

        示例:
            >>> await app.start()
            >>> # 服务器在后台运行
            >>> await asyncio.sleep(3600)  # 运行1小时
            >>> await app.shutdown()
        """
        if self._running:
            logger.warning("服务器已在运行")
            return True

        try:
            import websockets

            self._server = await websockets.serve(
                self._handle_client,
                self.config.host,
                self.config.port,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.heartbeat_timeout,
                max_size=self.config.max_message_size,
                max_queue=16,
            )

            self._running = True
            self._stop_event.clear()

            # 启动状态推送任务
            self._status_push_task = asyncio.create_task(
                self._status_push_loop(), name="status_push"
            )

            logger.info(
                f"WebSocket服务器已启动: "
                f"ws://{self.config.host}:{self.config.port}"
            )
            return True

        except ImportError:
            logger.error("websockets库未安装")
            return False
        except Exception as e:
            logger.error(f"启动服务器失败: {e}")
            return False

    async def _handle_client(self, websocket: Any, path: str = "") -> None:
        """处理客户端连接(内部方法)

        参数:
            websocket: WebSocket连接对象
            path: 连接路径
        """
        client_id = str(uuid.uuid4())[:8]
        remote = websocket.remote_address
        remote_str = f"{remote[0]}:{remote[1]}" if remote else "unknown"

        logger.info(f"客户端连接: {client_id} from {remote_str}")

        # 检查最大连接数
        if len(self._clients) >= self.config.max_clients:
            logger.warning(f"达到最大连接数,拒绝连接: {client_id}")
            await websocket.close(1013, "Server full")
            return

        # 注册客户端
        self._clients[client_id] = websocket
        self._client_info[client_id] = ClientInfo(
            client_id=client_id,
            remote_addr=remote_str,
            connected_at=time.time(),
            last_heartbeat=time.time(),
            state=ClientState.CONNECTED,
        )
        self._message_queues[client_id] = asyncio.Queue(
            maxsize=self._max_queue_size
        )

        # 通知连接回调
        for cb in self._connect_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(client_id))
                else:
                    cb(client_id)
            except Exception as e:
                logger.error(f"连接回调异常: {e}")

        try:
            async for message in websocket:
                try:
                    await self._process_message(client_id, message)
                except Exception as e:
                    logger.error(f"消息处理异常: {e}")

        except Exception as e:
            logger.debug(f"客户端连接异常: {client_id}: {e}")
        finally:
            # 清理客户端
            await self._remove_client(client_id)

    async def _process_message(self, client_id: str, message: str) -> None:
        """处理客户端消息(内部方法)

        解析JSON-RPC消息,分发到对应处理方法。

        参数:
            client_id: 客户端ID
            message: 原始消息字符串
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            await self._send_error(client_id, None, -32700, "Parse error")
            return

        # 更新心跳时间
        if client_id in self._client_info:
            self._client_info[client_id].last_heartbeat = time.time()

        # 处理JSON-RPC消息
        jsonrpc = data.get("jsonrpc")
        method = data.get("method")
        params = data.get("params", {})
        req_id = data.get("id")

        if jsonrpc != self.JSONRPC_VERSION:
            await self._send_error(client_id, req_id, -32600, "Invalid Request")
            return

        # 心跳处理
        if method == "heartbeat":
            await self._send_response(client_id, req_id, {"status": "ok"})
            return

        # 认证处理
        if method == "authenticate":
            token = params.get("token", "")
            if self._authenticate(token):
                self._client_info[client_id].state = ClientState.AUTHENTICATED
                await self._send_response(client_id, req_id, {"auth": True})
            else:
                await self._send_error(
                    client_id, req_id, -32001, "Authentication failed"
                )
            return

        # 检查认证
        if self.config.enable_auth:
            info = self._client_info.get(client_id)
            if not info or info.state != ClientState.AUTHENTICATED:
                await self._send_error(
                    client_id, req_id, -32001, "Not authenticated"
                )
                return

        # 路由到注册的方法处理器
        if method and method in self._method_handlers:
            handler = self._method_handlers[method]
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(params)
                else:
                    result = handler(params)

                if req_id is not None:
                    await self._send_response(client_id, req_id, result)

            except Exception as e:
                logger.error(f"方法处理异常: {method}: {e}")
                if req_id is not None:
                    await self._send_error(
                        client_id, req_id, -32603, str(e)
                    )
        else:
            # 通知类型的消息(无id)
            if req_id is not None:
                await self._send_error(
                    client_id, req_id, -32601, f"Method not found: {method}"
                )

        # 通知消息回调
        for cb in self._message_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(client_id, data))
                else:
                    cb(client_id, data)
            except Exception as e:
                logger.error(f"消息回调异常: {e}")

    def _authenticate(self, token: str) -> bool:
        """验证客户端令牌(内部方法)

        参数:
            token: 认证令牌

        返回:
            bool: 验证成功返回True
        """
        if not self.config.enable_auth:
            return True
        return token == self.config.auth_token

    async def _send_response(
        self, client_id: str, request_id: int | None, result: Any
    ) -> None:
        """发送JSON-RPC响应(内部方法)

        参数:
            client_id: 客户端ID
            request_id: 请求ID
            result: 结果数据
        """
        response = {
            "jsonrpc": self.JSONRPC_VERSION,
            "result": result,
            "id": request_id,
        }
        await self._send_raw(client_id, response)

    async def _send_error(
        self,
        client_id: str,
        request_id: int | None,
        code: int,
        message: str,
    ) -> None:
        """发送JSON-RPC错误响应(内部方法)

        参数:
            client_id: 客户端ID
            request_id: 请求ID
            code: 错误码
            message: 错误信息
        """
        response = {
            "jsonrpc": self.JSONRPC_VERSION,
            "error": {"code": code, "message": message},
            "id": request_id,
        }
        await self._send_raw(client_id, response)

    async def _send_raw(self, client_id: str, data: dict) -> bool:
        """发送原始JSON数据(内部方法)

        参数:
            client_id: 客户端ID
            data: 数据字典

        返回:
            bool: 发送成功返回True
        """
        websocket = self._clients.get(client_id)
        if websocket is None:
            return False

        try:
            message = json.dumps(data, ensure_ascii=False)
            await websocket.send(message)
            return True
        except Exception as e:
            logger.debug(f"发送消息失败: {client_id}: {e}")
            return False

    async def _remove_client(self, client_id: str) -> None:
        """移除客户端(内部方法)

        参数:
            client_id: 要移除的客户端ID
        """
        if client_id in self._clients:
            del self._clients[client_id]
        if client_id in self._client_info:
            del self._client_info[client_id]
        if client_id in self._message_queues:
            del self._message_queues[client_id]

        logger.info(f"客户端断开: {client_id}")

        # 通知断开回调
        for cb in self._disconnect_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(client_id))
                else:
                    cb(client_id)
            except Exception as e:
                logger.error(f"断开回调异常: {e}")

    async def _status_push_loop(self) -> None:
        """状态推送循环(内部方法)

        定期推送系统状态给所有客户端。
        """
        logger.debug("状态推送循环已启动")

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.status_push_interval,
                )
            except asyncio.TimeoutError:
                pass

        logger.debug("状态推送循环已退出")

    def register_method_handler(
        self, method: str, handler: Callable
    ) -> None:
        """注册JSON-RPC方法处理器

        注册处理特定方法名的回调函数。

        参数:
            method: 方法名
            handler: 处理函数,接收params字典,返回结果

        示例:
            >>> def handle_move(params):
            ...     x = params.get("x", 0)
            ...     y = params.get("y", 0)
            ...     return {"status": "ok", "position": [x, y]}
            >>> app.register_method_handler("arm.move", handle_move)
        """
        self._method_handlers[method] = handler
        logger.debug(f"已注册方法处理器: {method}")

    def register_connect_callback(
        self, callback: Callable[[str], None]
    ) -> None:
        """注册客户端连接回调

        参数:
            callback: 回调函数,接收client_id
        """
        if callback not in self._connect_callbacks:
            self._connect_callbacks.append(callback)

    def register_disconnect_callback(
        self, callback: Callable[[str], None]
    ) -> None:
        """注册客户端断开回调

        参数:
            callback: 回调函数,接收client_id
        """
        if callback not in self._disconnect_callbacks:
            self._disconnect_callbacks.append(callback)

    def register_message_callback(
        self, callback: Callable[[str, dict], None]
    ) -> None:
        """注册消息接收回调

        参数:
            callback: 回调函数,接收(client_id, message_dict)
        """
        if callback not in self._message_callbacks:
            self._message_callbacks.append(callback)

    async def broadcast_status(self, status: dict[str, Any]) -> int:
        """广播状态到所有客户端

        将系统状态推送给所有已连接的客户端。

        参数:
            status: 状态字典,例如:
                {"cpu_percent": 45, "memory_percent": 60,
                 "status": "running", "timestamp": 1234567890}

        返回:
            int: 成功发送的客户端数量

        示例:
            >>> await app.broadcast_status({
            ...     "cpu": 45.2,
            ...     "memory": 62.1,
            ...     "status": "idle",
            ...     "timestamp": time.time(),
            ... })
        """
        if not self._clients:
            return 0

        message = {
            "jsonrpc": self.JSONRPC_VERSION,
            "method": "status.update",
            "params": status,
        }

        sent = 0
        dead_clients: list[str] = []

        for client_id in list(self._clients.keys()):
            success = await self._send_raw(client_id, message)
            if success:
                sent += 1
            else:
                dead_clients.append(client_id)

        # 清理失效连接
        for cid in dead_clients:
            await self._remove_client(cid)

        return sent

    async def send_to_client(
        self, client_id: str, message: dict[str, Any]
    ) -> bool:
        """发送消息给指定客户端

        向特定客户端发送JSON-RPC通知消息。

        参数:
            client_id: 目标客户端ID
            message: 消息字典,会自动包装为JSON-RPC通知格式

        返回:
            bool: 发送成功返回True

        示例:
            >>> await app.send_to_client("client-001", {
            ...     "type": "alert",
            ...     "level": "warning",
            ...     "message": "低电量警告",
            ... })
        """
        if client_id not in self._clients:
            logger.warning(f"客户端不存在: {client_id}")
            return False

        rpc_message = {
            "jsonrpc": self.JSONRPC_VERSION,
            "method": "server.notification",
            "params": message,
        }

        return await self._send_raw(client_id, rpc_message)

    async def send_response(
        self, client_id: str, request_id: int, result: Any
    ) -> bool:
        """发送响应给指定客户端

        用于异步处理场景:先收到请求,处理后通过此方法发送响应。

        参数:
            client_id: 目标客户端ID
            request_id: 对应请求ID
            result: 结果数据

        返回:
            bool: 发送成功返回True
        """
        await self._send_response(client_id, request_id, result)
        return True

    def get_client_count(self) -> int:
        """获取当前连接数

        返回:
            int: 已连接客户端数量
        """
        return len(self._clients)

    def get_clients(self) -> dict[str, ClientInfo]:
        """获取所有客户端信息

        返回:
            dict: client_id -> ClientInfo
        """
        return dict(self._client_info)

    async def disconnect_client(self, client_id: str) -> bool:
        """主动断开指定客户端

        参数:
            client_id: 要断开的客户端ID

        返回:
            bool: 断开成功返回True
        """
        websocket = self._clients.get(client_id)
        if websocket:
            try:
                await websocket.close()
            except Exception:
                pass

        await self._remove_client(client_id)
        return True

    async def shutdown(self) -> None:
        """关闭WebSocket服务器

        断开所有客户端,停止服务器。
        """
        self._stop_event.set()
        self._running = False

        # 取消状态推送任务
        if self._status_push_task and not self._status_push_task.done():
            self._status_push_task.cancel()
            try:
                await self._status_push_task
            except asyncio.CancelledError:
                pass

        # 断开所有客户端
        for client_id in list(self._clients.keys()):
            await self.disconnect_client(client_id)

        # 关闭服务器
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket服务器已关闭")

    def __repr__(self) -> str:
        return (
            f"AppInterface(host={self.config.host}, "
            f"port={self.config.port}, "
            f"clients={len(self._clients)}, "
            f"running={self._running})"
        )

    async def __aenter__(self) -> AppInterface:
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
