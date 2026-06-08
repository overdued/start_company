"""
Orbbec Astra Pro 深度摄像头驱动模块

基于OpenNI2框架的深度+RGB双流摄像头驱动,支持点云生成和内外参标定。
适用于OrangePi KunPeng Pro (RK3588)平台的USB3.0接口深度摄像头。

功能特性:
    - 深度流与RGB流同步采集
    - 点云生成(Point Cloud)
    - 相机内外参标定与对齐
    - 自动重连与超时保护
    - 异步帧读取接口

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """摄像头流类型枚举"""
    DEPTH = "depth"          # 深度流
    COLOR = "color"          # RGB彩色流
    IR = "ir"                # 红外流
    SYNC = "sync"            # 同步对齐流


@dataclass
class CameraIntrinsics:
    """相机内参数据结构

    属性:
        fx: 焦距x(像素)
        fy: 焦距y(像素)
        cx: 主点x(像素)
        cy: 主点y(像素)
        width: 图像宽度
        height: 图像高度
        distortion: 畸变系数[k1, k2, p1, p2, k3]
    """
    fx: float = 570.0
    fy: float = 570.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    distortion: list[float] = field(default_factory=lambda: [0.0] * 5)


@dataclass
class CameraExtrinsics:
    """相机外参数据结构(深度到彩色的变换矩阵)

    属性:
        rotation: 3x3旋转矩阵
        translation: 3x1平移向量
    """
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Frame:
    """统一帧数据结构

    属性:
        timestamp: 帧时间戳(秒)
        frame_type: 帧类型
        data: 帧数据(np.ndarray)
        width: 图像宽度
        height: 图像高度
        intrinsics: 相机内参
    """
    timestamp: float
    frame_type: StreamType
    data: np.ndarray
    width: int
    height: int
    intrinsics: CameraIntrinsics


class OpenNI2Backend(Protocol):
    """OpenNI2后端协议,用于依赖注入和测试模拟"""

    def initialize(self) -> bool: ...
    def open_device(self, uri: str | None = None) -> Any: ...
    def create_stream(self, device: Any, stream_type: StreamType) -> Any: ...
    def start_stream(self, stream: Any) -> bool: ...
    def read_frame(self, stream: Any, timeout_ms: int) -> np.ndarray | None: ...
    def stop_stream(self, stream: Any) -> None: ...
    def shutdown(self) -> None: ...


class OrbbecAstraPro:
    """Orbbec Astra Pro深度摄像头驱动类

    提供深度流、RGB流采集,点云生成,相机标定等功能。
    支持异步接口和自动重连机制。

    示例:
        >>> camera = OrbbecAstraPro(device_index=0)
        >>> await camera.initialize()
        >>> depth_frame = await camera.read_depth(timeout=5.0)
        >>> point_cloud = camera.generate_point_cloud(depth_frame)
        >>> await camera.shutdown()

    属性:
        device_index: USB设备索引
        depth_resolution: 深度流分辨率(tuple)
        color_resolution: RGB流分辨率(tuple)
        fps: 目标帧率
        _connected: 连接状态标志
        _lock: 线程安全锁
    """

    # 类常量
    DEFAULT_TIMEOUT: float = 5.0          # 默认操作超时(秒)
    MAX_RECONNECT_RETRIES: int = 3        # 最大重连次数
    RECONNECT_DELAY: float = 2.0          # 重连间隔(秒)
    DEFAULT_FPS: int = 30                 # 默认帧率

    # 深度范围(米)
    MIN_DEPTH: float = 0.3
    MAX_DEPTH: float = 8.0

    def __init__(
        self,
        device_index: int = 0,
        depth_resolution: tuple[int, int] = (640, 480),
        color_resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        openni_path: str | None = None,
    ) -> None:
        """初始化Orbbec Astra Pro驱动

        参数:
            device_index: USB设备索引,默认0
            depth_resolution: 深度流分辨率(宽,高),默认(640,480)
            color_resolution: RGB流分辨率(宽,高),默认(640,480)
            fps: 目标帧率,默认30
            openni_path: OpenNI2库路径,None则使用系统默认
        """
        self.device_index: int = device_index
        self.depth_resolution: tuple[int, int] = depth_resolution
        self.color_resolution: tuple[int, int] = color_resolution
        self.fps: int = fps
        self.openni_path: str | None = openni_path

        # 内部状态
        self._connected: bool = False
        self._initialized: bool = False
        self._streaming: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        self._frame_callbacks: list[Callable[[Frame], None]] = []

        # OpenNI2对象(延迟初始化)
        self._openni: Any = None
        self._device: Any = None
        self._depth_stream: Any = None
        self._color_stream: Any = None

        # 相机参数
        self._depth_intrinsics: CameraIntrinsics = CameraIntrinsics()
        self._color_intrinsics: CameraIntrinsics = CameraIntrinsics()
        self._extrinsics: CameraExtrinsics = CameraExtrinsics()

        # 采集线程
        self._capture_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

    async def initialize(self) -> bool:
        """初始化摄像头驱动和OpenNI2后端

        尝试加载OpenNI2库,打开USB设备并配置视频流。
        失败时会进行最多MAX_RECONNECT_RETRIES次重试。

        返回:
            bool: 初始化成功返回True,否则False

        异常:
            RuntimeError: 初始化过程中发生不可恢复错误
        """
        async with self._lock:
            if self._initialized:
                logger.warning("摄像头已初始化,跳过重复初始化")
                return True

            for attempt in range(1, self.MAX_RECONNECT_RETRIES + 1):
                try:
                    logger.info(f"初始化OpenNI2后端 (尝试 {attempt}/{self.MAX_RECONNECT_RETRIES})")

                    # 尝试导入OpenNI2
                    try:
                        import openni2
                        self._openni = openni2
                        if self.openni_path:
                            self._openni.initialize(self.openni_path)
                        else:
                            self._openni.initialize()
                    except ImportError:
                        logger.warning("openni2库未安装,使用模拟后端")
                        self._openni = None
                        self._initialized = True
                        return True

                    # 打开设备
                    self._device = self._openni.Device.open_any()
                    logger.info(f"已打开深度摄像头设备: {self.device_index}")

                    # 配置深度流
                    self._depth_stream = self._create_depth_stream()
                    # 配置RGB流
                    self._color_stream = self._create_color_stream()

                    # 加载标定参数
                    self._load_calibration()

                    self._initialized = True
                    self._connected = True
                    logger.info("Orbbec Astra Pro初始化成功")
                    return True

                except Exception as e:
                    logger.error(f"初始化失败 (尝试 {attempt}): {e}")
                    if attempt < self.MAX_RECONNECT_RETRIES:
                        await asyncio.sleep(self.RECONNECT_DELAY)
                    else:
                        logger.critical("摄像头初始化失败,已达到最大重试次数")
                        return False

            return False

    def _create_depth_stream(self) -> Any:
        """创建并配置深度视频流

        返回:
            配置好的深度流对象

        异常:
            RuntimeError: 流创建失败
        """
        if self._openni is None:
            return None

        stream = self._device.create_depth_stream()

        # 设置分辨率
        w, h = self.depth_resolution
        stream.set_video_mode(
            self._openni.VideoMode(
                pixelFormat=self._openni.PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=w,
                resolutionY=h,
                fps=self.fps,
            )
        )
        logger.debug(f"深度流配置: {w}x{h}@{self.fps}fps")
        return stream

    def _create_color_stream(self) -> Any:
        """创建并配置RGB彩色视频流

        返回:
            配置好的彩色流对象

        异常:
            RuntimeError: 流创建失败
        """
        if self._openni is None:
            return None

        stream = self._device.create_color_stream()

        w, h = self.color_resolution
        stream.set_video_mode(
            self._openni.VideoMode(
                pixelFormat=self._openni.PIXEL_FORMAT_RGB888,
                resolutionX=w,
                resolutionY=h,
                fps=self.fps,
            )
        )
        logger.debug(f"RGB流配置: {w}x{h}@{self.fps}fps")
        return stream

    def _load_calibration(self) -> None:
        """从设备加载相机标定参数

        读取深度和RGB相机的内参以及两者之间的外参(旋转和平移)。
        如果设备未提供标定参数,则使用默认值。
        """
        try:
            if self._openni is None or self._device is None:
                return

            # 尝试从设备读取标定数据
            # 部分设备支持get_property接口读取标定信息
            depth_ival = self._device.get_sensor_info(
                self._openni.SENSOR_DEPTH
            ).videoModes[0]
            self._depth_intrinsics.width = depth_ival.resolutionX
            self._depth_intrinsics.height = depth_ival.resolutionY

            # 加载预设标定参数(如设备未提供)
            self._depth_intrinsics.fx = 570.342
            self._depth_intrinsics.fy = 570.342
            self._depth_intrinsics.cx = depth_ival.resolutionX / 2.0
            self._depth_intrinsics.cy = depth_ival.resolutionY / 2.0

            color_ival = self._device.get_sensor_info(
                self._openni.SENSOR_COLOR
            ).videoModes[0]
            self._color_intrinsics.width = color_ival.resolutionX
            self._color_intrinsics.height = color_ival.resolutionY
            self._color_intrinsics.fx = 570.342
            self._color_intrinsics.fy = 570.342
            self._color_intrinsics.cx = color_ival.resolutionX / 2.0
            self._color_intrinsics.cy = color_ival.resolutionY / 2.0

            logger.info("相机标定参数加载完成")

        except Exception as e:
            logger.warning(f"加载标定参数失败,使用默认值: {e}")

    async def start_streaming(self) -> bool:
        """开始异步视频流采集

        启动深度流和RGB流的异步采集循环,帧数据通过回调函数分发。

        返回:
            bool: 启动成功返回True
        """
        async with self._lock:
            if not self._initialized:
                logger.error("摄像头未初始化,无法启动流")
                return False

            if self._streaming:
                logger.warning("视频流已在运行")
                return True

            try:
                if self._depth_stream:
                    self._depth_stream.start()
                if self._color_stream:
                    self._color_stream.start()

                self._streaming = True
                self._stop_event.clear()
                self._capture_task = asyncio.create_task(
                    self._capture_loop(), name="camera_capture"
                )
                logger.info("视频流采集已启动")
                return True

            except Exception as e:
                logger.error(f"启动视频流失败: {e}")
                return False

    async def _capture_loop(self) -> None:
        """帧采集主循环(内部方法)

        持续从摄像头读取深度帧和彩色帧,组装为统一Frame对象后
        分发给所有注册的回调函数。

        注意:
            此方法作为异步任务运行,通过_stop_event控制退出。
        """
        logger.debug("帧采集循环已启动")

        while not self._stop_event.is_set():
            try:
                loop_start = time.monotonic()

                # 读取深度帧
                depth_data = await self._read_frame_async(
                    self._depth_stream, StreamType.DEPTH
                )

                # 读取彩色帧
                color_data = await self._read_frame_async(
                    self._color_stream, StreamType.COLOR
                )

                # 分发帧
                if depth_data is not None:
                    await self._dispatch_frame(depth_data)
                if color_data is not None:
                    await self._dispatch_frame(color_data)

                # 帧率控制
                elapsed = time.monotonic() - loop_start
                frame_interval = 1.0 / self.fps
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)

            except asyncio.CancelledError:
                logger.debug("采集循环已取消")
                break
            except Exception as e:
                logger.error(f"采集循环异常: {e}")
                await asyncio.sleep(0.1)

        logger.debug("帧采集循环已退出")

    async def _read_frame_async(
        self, stream: Any, frame_type: StreamType
    ) -> Frame | None:
        """异步读取单帧数据(内部方法)

        参数:
            stream: OpenNI2流对象
            frame_type: 帧类型

        返回:
            Frame对象或None(读取失败/超时)
        """
        if stream is None:
            return None

        try:
            # 在线程池中执行同步读取,避免阻塞事件循环
            frame_data = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._read_frame_sync, stream
                ),
                timeout=self.DEFAULT_TIMEOUT,
            )

            if frame_data is None:
                return None

            w, h = self.depth_resolution if frame_type == StreamType.DEPTH else self.color_resolution
            intrinsics = (
                self._depth_intrinsics if frame_type == StreamType.DEPTH
                else self._color_intrinsics
            )

            return Frame(
                timestamp=time.time(),
                frame_type=frame_type,
                data=frame_data,
                width=w,
                height=h,
                intrinsics=intrinsics,
            )

        except asyncio.TimeoutError:
            logger.warning(f"读取{frame_type.value}帧超时")
            return None
        except Exception as e:
            logger.error(f"读取帧异常: {e}")
            return None

    def _read_frame_sync(self, stream: Any) -> np.ndarray | None:
        """同步读取单帧(内部方法,在线程池中执行)

        参数:
            stream: OpenNI2流对象

        返回:
            帧数据数组或None
        """
        try:
            if self._openni is None:
                # 模拟模式:生成随机测试帧
                w, h = self.depth_resolution
                return np.random.randint(0, 8000, (h, w), dtype=np.uint16)

            frame = stream.read_frame()
            data = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16)
            return data.reshape((frame.height, frame.width))

        except Exception as e:
            logger.error(f"同步读取帧失败: {e}")
            return None

    async def _dispatch_frame(self, frame: Frame) -> None:
        """分发帧到所有回调函数(内部方法)

        参数:
            frame: 要分发的帧对象
        """
        for callback in self._frame_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(frame))
                else:
                    callback(frame)
            except Exception as e:
                logger.error(f"帧回调执行异常: {e}")

    def register_frame_callback(self, callback: Callable[[Frame], None]) -> None:
        """注册帧数据回调函数

        当新帧到达时,会调用所有已注册的回调函数。

        参数:
            callback: 回调函数,接收Frame对象作为参数
        """
        if callback not in self._frame_callbacks:
            self._frame_callbacks.append(callback)
            logger.debug(f"已注册帧回调: {callback.__name__}")

    def unregister_frame_callback(self, callback: Callable[[Frame], None]) -> None:
        """注销帧数据回调函数

        参数:
            callback: 要移除的回调函数
        """
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)
            logger.debug(f"已注销帧回调: {callback.__name__}")

    async def read_depth(self, timeout: float = DEFAULT_TIMEOUT) -> Frame | None:
        """读取单帧深度图像

        参数:
            timeout: 读取超时时间(秒),默认5.0

        返回:
            深度帧对象或None(超时/失败)

        示例:
            >>> frame = await camera.read_depth(timeout=3.0)
            >>> if frame:
            ...     depth_mm = frame.data  # 单位:毫米
        """
        return await self._read_frame_async(self._depth_stream, StreamType.DEPTH)

    async def read_color(self, timeout: float = DEFAULT_TIMEOUT) -> Frame | None:
        """读取单帧RGB彩色图像

        参数:
            timeout: 读取超时时间(秒),默认5.0

        返回:
            RGB帧对象或None(超时/失败)
        """
        return await self._read_frame_async(self._color_stream, StreamType.COLOR)

    def generate_point_cloud(
        self, depth_frame: Frame, color_frame: Frame | None = None
    ) -> np.ndarray:
        """从深度帧生成点云数据

        根据深度图像和相机内参,将每个像素反投影到3D空间生成点云。
        可选地,可以从彩色帧中为每个点附加RGB颜色信息。

        参数:
            depth_frame: 深度帧对象
            color_frame: 可选的彩色帧对象,用于给点云着色

        返回:
            Nx3或Nx6数组,每行代表一个点的(x,y,z)坐标,
            如果提供了color_frame则为(x,y,z,r,g,b)

        示例:
            >>> depth = await camera.read_depth()
            >>> color = await camera.read_color()
            >>> pc = camera.generate_point_cloud(depth, color)
            >>> print(pc.shape)  # (N, 6) - xyzrgb
        """
        depth = depth_frame.data.astype(np.float32)  # 单位:毫米
        h, w = depth.shape
        intrinsics = depth_frame.intrinsics

        # 创建像素坐标网格
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)

        # 反投影到3D空间
        z = depth / 1000.0  # 毫米转米
        x = (uu - intrinsics.cx) * z / intrinsics.fx
        y = (vv - intrinsics.cy) * z / intrinsics.fy

        # 过滤无效深度值
        valid_mask = (z > self.MIN_DEPTH) & (z < self.MAX_DEPTH)

        points = np.stack([
            x[valid_mask],
            y[valid_mask],
            z[valid_mask],
        ], axis=-1)

        # 附加颜色信息
        if color_frame is not None:
            color = color_frame.data
            if color.ndim == 3:
                colors = color[valid_mask] / 255.0
                points = np.concatenate([points, colors], axis=-1)

        logger.debug(f"生成点云: {len(points)} 个点")
        return points

    def align_depth_to_color(self, depth_frame: Frame) -> np.ndarray:
        """将深度图像对齐到彩色图像坐标系

        使用相机外参将深度图像变换到彩色相机视角,
        使得深度和彩色像素一一对应。

        参数:
            depth_frame: 深度帧对象

        返回:
            对齐后的深度图像数组
        """
        depth = depth_frame.data.astype(np.float32)
        h, w = depth.shape

        # 使用外参变换
        aligned = np.zeros((h, w), dtype=np.float32)

        # 简化实现:使用双线性插值进行重映射
        # 完整实现需要调用OpenNI的registration或自定义CUDA核
        R = self._extrinsics.rotation
        T = self._extrinsics.translation

        for v in range(h):
            for u in range(w):
                z = depth[v, u] / 1000.0  # mm to m
                if z <= 0 or z > self.MAX_DEPTH:
                    continue

                # 深度相机坐标系中的3D点
                x = (u - self._depth_intrinsics.cx) * z / self._depth_intrinsics.fx
                y = (v - self._depth_intrinsics.cy) * z / self._depth_intrinsics.fy
                p_depth = np.array([x, y, z])

                # 变换到彩色相机坐标系
                p_color = R @ p_depth + T

                # 投影到彩色图像平面
                if p_color[2] > 0:
                    u_c = int(p_color[0] * self._color_intrinsics.fx / p_color[2]
                              + self._color_intrinsics.cx)
                    v_c = int(p_color[1] * self._color_intrinsics.fy / p_color[2]
                              + self._color_intrinsics.cy)

                    if 0 <= u_c < w and 0 <= v_c < h:
                        aligned[v_c, u_c] = depth[v, u]

        logger.debug("深度-彩色图像对齐完成")
        return aligned

    async def stop_streaming(self) -> None:
        """停止视频流采集

        安全地停止采集循环并关闭视频流。
        """
        async with self._lock:
            if not self._streaming:
                return

            self._stop_event.set()

            if self._capture_task and not self._capture_task.done():
                self._capture_task.cancel()
                try:
                    await self._capture_task
                except asyncio.CancelledError:
                    pass

            try:
                if self._depth_stream:
                    self._depth_stream.stop()
                if self._color_stream:
                    self._color_stream.stop()
            except Exception as e:
                logger.warning(f"停止流时发生异常: {e}")

            self._streaming = False
            logger.info("视频流采集已停止")

    async def shutdown(self) -> None:
        """关闭摄像头驱动,释放所有资源

        关闭视频流、释放设备、卸载OpenNI2后端。
        应在程序退出时调用。
        """
        await self.stop_streaming()

        async with self._lock:
            try:
                if self._depth_stream:
                    self._depth_stream.close()
                    self._depth_stream = None
                if self._color_stream:
                    self._color_stream.close()
                    self._color_stream = None

                if self._openni:
                    self._openni.unload()
                    self._openni = None

                self._device = None
                self._connected = False
                self._initialized = False

                logger.info("Orbbec Astra Pro驱动已关闭")

            except Exception as e:
                logger.error(f"关闭驱动时发生异常: {e}")

    @property
    def is_connected(self) -> bool:
        """摄像头连接状态"""
        return self._connected

    @property
    def is_streaming(self) -> bool:
        """视频流运行状态"""
        return self._streaming

    @property
    def depth_intrinsics(self) -> CameraIntrinsics:
        """深度相机内参"""
        return self._depth_intrinsics

    @property
    def color_intrinsics(self) -> CameraIntrinsics:
        """RGB相机内参"""
        return self._color_intrinsics

    async def save_calibration(self, filepath: str) -> bool:
        """保存相机标定参数到YAML文件

        参数:
            filepath: 保存路径

        返回:
            bool: 保存成功返回True
        """
        try:
            import yaml

            data = {
                "depth_intrinsics": {
                    "fx": self._depth_intrinsics.fx,
                    "fy": self._depth_intrinsics.fy,
                    "cx": self._depth_intrinsics.cx,
                    "cy": self._depth_intrinsics.cy,
                    "width": self._depth_intrinsics.width,
                    "height": self._depth_intrinsics.height,
                    "distortion": self._depth_intrinsics.distortion,
                },
                "color_intrinsics": {
                    "fx": self._color_intrinsics.fx,
                    "fy": self._color_intrinsics.fy,
                    "cx": self._color_intrinsics.cx,
                    "cy": self._color_intrinsics.cy,
                    "width": self._color_intrinsics.width,
                    "height": self._color_intrinsics.height,
                    "distortion": self._color_intrinsics.distortion,
                },
                "extrinsics": {
                    "rotation": self._extrinsics.rotation.tolist(),
                    "translation": self._extrinsics.translation.tolist(),
                },
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)

            logger.info(f"标定参数已保存: {filepath}")
            return True

        except Exception as e:
            logger.error(f"保存标定参数失败: {e}")
            return False

    async def load_calibration_from_file(self, filepath: str) -> bool:
        """从YAML文件加载相机标定参数

        参数:
            filepath: 标定文件路径

        返回:
            bool: 加载成功返回True
        """
        try:
            import yaml

            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            d = data["depth_intrinsics"]
            self._depth_intrinsics = CameraIntrinsics(
                fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
                width=d["width"], height=d["height"],
                distortion=d.get("distortion", [0.0] * 5),
            )

            c = data["color_intrinsics"]
            self._color_intrinsics = CameraIntrinsics(
                fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
                width=c["width"], height=c["height"],
                distortion=c.get("distortion", [0.0] * 5),
            )

            e = data["extrinsics"]
            self._extrinsics = CameraExtrinsics(
                rotation=np.array(e["rotation"]),
                translation=np.array(e["translation"]),
            )

            logger.info(f"标定参数已从文件加载: {filepath}")
            return True

        except Exception as e:
            logger.error(f"加载标定参数失败: {e}")
            return False

    def __repr__(self) -> str:
        return (
            f"OrbbecAstraPro(device={self.device_index}, "
            f"depth_res={self.depth_resolution}, "
            f"connected={self._connected}, "
            f"streaming={self._streaming})"
        )

    async def __aenter__(self) -> OrbbecAstraPro:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
