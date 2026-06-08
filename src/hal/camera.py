"""
RK3588 摄像头硬件抽象层模块

本模块提供对OrangePi Kunpeng Pro (RK3588) 摄像头的完整控制支持，
支持Orbbec Astra Pro深度摄像头（OpenNI2协议）和普通USB摄像头（V4L2协议）。

支持的摄像头：
    - Orbbec Astra Pro: RGB + 深度双路输出
    - 普通USB摄像头: RGB单路输出（V4L2）
    - MIPI CSI摄像头: 通过GStreamer支持

特性：
    - 自动摄像头探测
    - RGB流和深度流获取
    - 分辨率、帧率配置
    - 自动曝光、白平衡
    - 超时保护（默认5秒）
    - 线程安全
    - 异常回退和自动恢复

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

import os
import time
import logging
import threading
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable, Union
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CameraType(Enum):
    """摄像头类型枚举"""
    USB_V4L2 = "usb_v4l2"           # USB摄像头（V4L2）
    ORBBEC_ASTRA = "orbbec_astra"   # Orbbec Astra Pro（OpenNI2）
    MIPI_CSI = "mipi_csi"           # MIPI CSI摄像头
    AUTO = "auto"                    # 自动检测


@dataclass
class CameraIntrinsics:
    """摄像头内参数据结构

    Attributes:
        width: 图像宽度（像素）
        height: 图像高度（像素）
        fx: X轴焦距（像素）
        fy: Y轴焦距（像素）
        cx: X轴主点坐标
        cy: Y轴主点坐标
        k1: 径向畸变系数k1
        k2: 径向畸变系数k2
        p1: 切向畸变系数p1
        p2: 切向畸变系数p2
        k3: 径向畸变系数k3
    """
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "k3": self.k3,
        }


class Camera:
    """RK3588摄像头控制器类

    提供对RK3588平台上多种摄像头的统一控制接口。
    支持USB摄像头（V4L2）、Orbbec Astra Pro（OpenNI2）和MIPI CSI摄像头。

    Args:
        device: 摄像头设备标识，"auto"表示自动检测
        width: 图像宽度，默认640
        height: 图像高度，默认480
        fps: 帧率，默认30
        camera_type: 摄像头类型，None表示自动检测
        timeout: 操作超时时间（秒），默认5.0

    Raises:
        RuntimeError: 摄像头初始化失败
        ValueError: 参数无效

    Example:
        >>> cam = Camera(device="auto", width=640, height=480)
        >>> cam.start()
        >>> frame = cam.capture()  # 获取RGB帧
        >>> cam.stop()
        >>> cam.close()
    """

    def __init__(
        self,
        device: str = "auto",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        camera_type: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        self._device: str = device
        self._width: int = width
        self._height: int = height
        self._fps: int = fps
        self._timeout: float = timeout
        self._instance_lock: threading.Lock = threading.Lock()
        self._capture_lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._running: bool = False
        self._camera_type: CameraType = CameraType.AUTO
        self._backend: Optional[object] = None
        self._frame_buffer: Optional[np.ndarray] = None
        self._depth_buffer: Optional[np.ndarray] = None
        self._buffer_lock: threading.Lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running: bool = False

        # 摄像头内参
        self._intrinsics: CameraIntrinsics = CameraIntrinsics(
            width=width, height=height,
            fx=width * 0.8, fy=height * 0.8,
            cx=width / 2.0, cy=height / 2.0,
        )
        self._depth_intrinsics: Optional[CameraIntrinsics] = None

        # 自动白平衡和曝光
        self._auto_exposure: bool = True
        self._auto_white_balance: bool = True

        # 参数验证
        if width <= 0 or height <= 0:
            raise ValueError(f"分辨率必须为正: {width}x{height}")
        if fps <= 0:
            raise ValueError(f"帧率必须为正: {fps}")

        try:
            self._detect_and_init(camera_type)
            logger.info(
                f"摄像头初始化成功: 类型={self._camera_type.value}, "
                f"分辨率={width}x{height}, 帧率={fps}fps"
            )
        except Exception as e:
            self._fallback_safe_state()
            raise RuntimeError(f"摄像头初始化失败: {e}") from e

    def _detect_and_init(self, camera_type: Optional[str]) -> None:
        """自动检测并初始化摄像头

        Args:
            camera_type: 指定的摄像头类型，None表示自动检测
        """
        if camera_type:
            self._camera_type = CameraType(camera_type)
        else:
            self._camera_type = self._auto_detect_camera()

        if self._camera_type == CameraType.ORBBEC_ASTRA:
            self._init_orbbec_astra()
        elif self._camera_type == CameraType.MIPI_CSI:
            self._init_mipi_csi()
        else:
            self._init_usb_v4l2()

    def _auto_detect_camera(self) -> CameraType:
        """自动检测摄像头类型

        Returns:
            CameraType: 检测到的摄像头类型
        """
        # 检查OpenNI2是否可用（Orbbec Astra Pro）
        try:
            import openni
            openni.Device.open_any()
            logger.info("检测到Orbbec Astra Pro摄像头")
            return CameraType.ORBBEC_ASTRA
        except ImportError:
            logger.debug("OpenNI2库未安装，跳过Orbbec检测")
        except Exception:
            logger.debug("未检测到Orbbec Astra Pro")

        # 检查V4L2设备
        v4l2_devices = self._list_v4l2_devices()
        if v4l2_devices:
            logger.info(f"检测到USB摄像头: {v4l2_devices}")
            return CameraType.USB_V4L2

        # 检查MIPI CSI
        if os.path.exists("/dev/video10") or os.path.exists("/dev/rkisp0"):
            logger.info("检测到MIPI CSI摄像头")
            return CameraType.MIPI_CSI

        logger.warning("未检测到摄像头，回退到USB V4L2")
        return CameraType.USB_V4L2

    def _list_v4l2_devices(self) -> List[str]:
        """列出系统中的V4L2设备

        Returns:
            List[str]: V4L2设备路径列表
        """
        devices = []
        video_dir = Path("/dev")
        for dev in sorted(video_dir.glob("video*")):
            devices.append(str(dev))
        return devices

    def _init_orbbec_astra(self) -> None:
        """初始化Orbbec Astra Pro摄像头（OpenNI2）

        Raises:
            ImportError: OpenNI2库未安装
            RuntimeError: 设备连接失败
        """
        try:
            import openni
        except ImportError:
            raise ImportError(
                "OpenNI2库未安装，Orbbec Astra Pro需要openni2-python. "
                "请执行: pip install openni2-python"
            )

        try:
            openni.initialize()
            self._backend = {
                "type": "openni2",
                "device": openni.Device.open_any(),
            }

            # 配置RGB流
            color_stream = self._backend["device"].create_color_stream()
            color_stream.set_video_mode(
                openni.VideoMode(
                    pixel_format=openni.PIXEL_FORMAT_RGB888,
                    resolution_x=self._width,
                    resolution_y=self._height,
                    fps=self._fps,
                )
            )
            self._backend["color_stream"] = color_stream

            # 配置深度流
            depth_stream = self._backend["device"].create_depth_stream()
            depth_stream.set_video_mode(
                openni.VideoMode(
                    pixel_format=openni.PIXEL_FORMAT_DEPTH_1_MM,
                    resolution_x=self._width,
                    resolution_y=self._height,
                    fps=self._fps,
                )
            )
            self._backend["depth_stream"] = depth_stream

            # 设置深度摄像头内参
            self._depth_intrinsics = CameraIntrinsics(
                width=self._width, height=self._height,
                fx=self._width * 0.7, fy=self._height * 0.7,
                cx=self._width / 2.0, cy=self._height / 2.0,
            )

        except Exception as e:
            raise RuntimeError(f"Orbbec Astra Pro初始化失败: {e}") from e

    def _init_usb_v4l2(self) -> None:
        """初始化USB摄像头（V4L2）

        使用OpenCV的VideoCapture作为后端。

        Raises:
            ImportError: OpenCV未安装
            RuntimeError: 设备打开失败
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV未安装，USB摄像头需要opencv-python. "
                "请执行: pip install opencv-python-headless"
            )

        device_path = self._device if self._device != "auto" else 0
        if isinstance(device_path, str) and device_path.startswith("/dev/video"):
            device_path = int(device_path.replace("/dev/video", ""))

        try:
            cap = cv2.VideoCapture(device_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头设备: {device_path}")

            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._fps)

            # 自动曝光和白平衡
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if self._auto_exposure else 0)

            # 获取实际分辨率
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._backend = {
                "type": "v4l2",
                "capture": cap,
                "device": device_path,
                "actual_width": actual_width,
                "actual_height": actual_height,
            }

            # 更新内参
            self._intrinsics.width = actual_width
            self._intrinsics.height = actual_height
            self._intrinsics.cx = actual_width / 2.0
            self._intrinsics.cy = actual_height / 2.0

        except Exception as e:
            raise RuntimeError(f"USB摄像头初始化失败: {e}") from e

    def _init_mipi_csi(self) -> None:
        """初始化MIPI CSI摄像头

        使用GStreamer管道捕获MIPI CSI摄像头图像。
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV未安装，MIPI CSI需要opencv-python")

        # RK3588 MIPI CSI GStreamer管道
        gst_pipeline = (
            f"v4l2src device=/dev/video0 ! "
            f"video/x-raw, width={self._width}, height={self._height}, framerate={self._fps}/1 ! "
            f"videoconvert ! "
            f"appsink"
        )

        try:
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                # 回退到默认V4L2
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("无法打开MIPI CSI摄像头")

            self._backend = {
                "type": "mipi_csi",
                "capture": cap,
                "pipeline": gst_pipeline,
            }
        except Exception as e:
            raise RuntimeError(f"MIPI CSI摄像头初始化失败: {e}") from e

    def start(self) -> None:
        """启动摄像头采集"""
        if self._closed:
            raise RuntimeError("摄像头已关闭")
        if self._running:
            return

        with self._instance_lock:
            try:
                if self._camera_type == CameraType.ORBBEC_ASTRA:
                    self._start_orbbec()
                else:
                    self._start_v4l2()
                self._running = True
                logger.info("摄像头采集已启动")
            except Exception as e:
                logger.error(f"摄像头启动失败: {e}")
                raise

    def _start_orbbec(self) -> None:
        """启动Orbbec Astra Pro采集"""
        if self._backend and "color_stream" in self._backend:
            self._backend["color_stream"].start()
        if self._backend and "depth_stream" in self._backend:
            self._backend["depth_stream"].start()

    def _start_v4l2(self) -> None:
        """启动V4L2采集"""
        # OpenCV VideoCapture自动启动
        pass

    def capture(self) -> np.ndarray:
        """捕获一帧RGB图像

        Returns:
            np.ndarray: BGR格式的图像数组 (H, W, 3), dtype=uint8

        Raises:
            RuntimeError: 摄像头未启动或已关闭
            TimeoutError: 捕获超时
        """
        if self._closed:
            raise RuntimeError("摄像头已关闭")
        if not self._running:
            raise RuntimeError("摄像头未启动，请先调用start()")

        with self._capture_lock:
            try:
                if self._camera_type == CameraType.ORBBEC_ASTRA:
                    return self._capture_orbbec_rgb()
                else:
                    return self._capture_v4l2()
            except Exception as e:
                logger.error(f"RGB图像捕获失败: {e}")
                raise

    def _capture_orbbec_rgb(self) -> np.ndarray:
        """从Orbbec Astra Pro捕获RGB帧"""
        import openni
        color_stream = self._backend.get("color_stream")
        if not color_stream:
            raise RuntimeError("RGB流未初始化")

        frame = color_stream.read_frame()
        frame_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8)
        frame_data = frame_data.reshape((self._height, self._width, 3))
        # OpenNI返回RGB，转换为BGR以兼容OpenCV
        frame_data = frame_data[:, :, ::-1]
        return frame_data

    def _capture_v4l2(self) -> np.ndarray:
        """从V4L2设备捕获帧"""
        cap = self._backend.get("capture")
        if not cap:
            raise RuntimeError("V4L2捕获器未初始化")

        start_time = time.monotonic()
        while time.monotonic() - start_time < self._timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
            time.sleep(0.001)

        raise TimeoutError(f"V4L2图像捕获超时 ({self._timeout}s)")

    def capture_depth(self) -> np.ndarray:
        """捕获一帧深度图像

        仅Orbbec Astra Pro支持深度流。

        Returns:
            np.ndarray: 深度图像数组 (H, W), dtype=uint16, 单位毫米

        Raises:
            RuntimeError: 摄像头不支持深度流
            TimeoutError: 捕获超时
        """
        if self._closed:
            raise RuntimeError("摄像头已关闭")
        if not self._running:
            raise RuntimeError("摄像头未启动")
        if self._camera_type != CameraType.ORBBEC_ASTRA:
            raise RuntimeError(f"摄像头类型 {self._camera_type.value} 不支持深度流")

        with self._capture_lock:
            try:
                depth_stream = self._backend.get("depth_stream")
                if not depth_stream:
                    raise RuntimeError("深度流未初始化")

                frame = depth_stream.read_frame()
                frame_data = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16)
                frame_data = frame_data.reshape((self._height, self._width))
                return frame_data
            except Exception as e:
                logger.error(f"深度图像捕获失败: {e}")
                raise

    def capture_aligned(self) -> Tuple[np.ndarray, np.ndarray]:
        """同时捕获RGB和深度图像

        Returns:
            Tuple[np.ndarray, np.ndarray]: (RGB图像, 深度图像)

        Raises:
            RuntimeError: 摄像头不支持深度流
        """
        rgb = self.capture()
        depth = self.capture_depth()
        return rgb, depth

    def stop(self) -> None:
        """停止摄像头采集"""
        if self._closed or not self._running:
            return

        with self._instance_lock:
            try:
                if self._camera_type == CameraType.ORBBEC_ASTRA:
                    if self._backend and "color_stream" in self._backend:
                        self._backend["color_stream"].stop()
                    if self._backend and "depth_stream" in self._backend:
                        self._backend["depth_stream"].stop()
                self._running = False
                logger.info("摄像头采集已停止")
            except Exception as e:
                logger.error(f"摄像头停止失败: {e}")

    def set_resolution(self, width: int, height: int) -> None:
        """动态修改分辨率

        Args:
            width: 新的宽度
            height: 新的高度
        """
        if self._backend and self._backend.get("type") in ("v4l2", "mipi_csi"):
            import cv2
            cap = self._backend.get("capture")
            if cap:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"分辨率修改为 {self._width}x{self._height}")

    def set_fps(self, fps: int) -> None:
        """动态修改帧率

        Args:
            fps: 新的帧率
        """
        if self._backend and self._backend.get("type") in ("v4l2", "mipi_csi"):
            import cv2
            cap = self._backend.get("capture")
            if cap:
                cap.set(cv2.CAP_PROP_FPS, fps)
                self._fps = int(cap.get(cv2.CAP_PROP_FPS))
                logger.info(f"帧率修改为 {self._fps}fps")

    def set_auto_exposure(self, enable: bool) -> None:
        """设置自动曝光

        Args:
            enable: True=开启, False=关闭
        """
        self._auto_exposure = enable
        if self._backend and self._backend.get("type") in ("v4l2", "mipi_csi"):
            import cv2
            cap = self._backend.get("capture")
            if cap:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if enable else 0)

    def set_auto_white_balance(self, enable: bool) -> None:
        """设置自动白平衡

        Args:
            enable: True=开启, False=关闭
        """
        self._auto_white_balance = enable

    def get_intrinsics(self) -> dict:
        """获取RGB摄像头内参

        Returns:
            dict: 摄像头内参字典
        """
        return self._intrinsics.to_dict()

    def get_depth_intrinsics(self) -> Optional[dict]:
        """获取深度摄像头内参

        Returns:
            dict或None: 深度摄像头内参字典
        """
        if self._depth_intrinsics:
            return self._depth_intrinsics.to_dict()
        return None

    def _fallback_safe_state(self) -> None:
        """异常回退到安全状态：停止采集并释放资源"""
        try:
            self.stop()
            self._cleanup_resources()
        except Exception as e:
            logger.error(f"摄像头安全状态回退失败: {e}")

    def _cleanup_resources(self) -> None:
        """清理所有已分配资源"""
        if self._backend:
            try:
                backend_type = self._backend.get("type")
                if backend_type == "openni2":
                    import openni
                    if "color_stream" in self._backend:
                        self._backend["color_stream"].stop()
                    if "depth_stream" in self._backend:
                        self._backend["depth_stream"].stop()
                    if "device" in self._backend:
                        self._backend["device"].close()
                    openni.unload()
                elif backend_type in ("v4l2", "mipi_csi"):
                    cap = self._backend.get("capture")
                    if cap:
                        cap.release()
            except Exception as e:
                logger.error(f"摄像头资源清理失败: {e}")
            self._backend = None

    def close(self) -> None:
        """关闭摄像头，释放所有资源"""
        if self._closed:
            return

        with self._instance_lock:
            try:
                self._cleanup_resources()
                self._closed = True
                logger.info("摄像头已关闭")
            except Exception as e:
                logger.error(f"摄像头关闭时发生错误: {e}")
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
    def width(self) -> int:
        """获取图像宽度"""
        return self._width

    @property
    def height(self) -> int:
        """获取图像高度"""
        return self._height

    @property
    def fps(self) -> int:
        """获取帧率"""
        return self._fps

    @property
    def camera_type(self) -> CameraType:
        """获取摄像头类型"""
        return self._camera_type

    @property
    def is_running(self) -> bool:
        """判断摄像头是否正在采集"""
        return self._running

    @property
    def is_closed(self) -> bool:
        """判断摄像头是否已关闭"""
        return self._closed

    @staticmethod
    def list_cameras() -> List[Dict[str, str]]:
        """列出系统中可用的摄像头

        Returns:
            List[Dict[str, str]]: 摄像头信息列表
        """
        cameras = []
        video_dir = Path("/dev")
        for dev in sorted(video_dir.glob("video*")):
            cameras.append({
                "device": str(dev),
                "name": f"Video Device {dev.name}",
            })
        return cameras
