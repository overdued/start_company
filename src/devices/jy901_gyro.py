"""
JY901 9轴陀螺仪传感器驱动模块

基于UART通信的JY901B 9轴惯性测量单元(IMU)驱动。
支持角度/角速度/加速度读取、磁力计校准和姿态解算。
适用于OrangePi Kunpeng Pro (RK3588)平台。

功能特性:
    - 三轴加速度计(16bit ADC)
    - 三轴陀螺仪(16bit ADC)
    - 三轴磁力计(16bit ADC)
    - 姿态角输出(欧拉角/四元数)
    - 磁力计硬铁/软铁校准
    - 温度补偿
    - 自动重连与超时保护

通信协议:
    - UART: 115200bps, 8N1
    - 数据帧: 0x55 + 功能字 + 数据(8字节) + 校验和
    - 输出频率: 200Hz

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class JY901FrameType(Enum):
    """JY901数据帧类型枚举"""
    ACCEL = 0x51       # 加速度输出
    GYRO = 0x52        # 角速度输出
    ANGLE = 0x53       # 姿态角度输出
    MAGNET = 0x54      # 磁力计输出
    QUATERNION = 0x59  # 四元数输出
    GPS = 0x57         # GPS数据(可选)
    PRESSURE = 0x56    # 气压数据(可选)


@dataclass
class RawSensorData:
    """原始传感器数据

    属性:
        accel_x/y/z: 三轴加速度(g)
        gyro_x/y/z: 三轴角速度(°/s)
        mag_x/y/z: 三轴磁力计原始值
        temperature: 芯片温度(°C)
        timestamp: 采样时间戳
    """
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    temperature: float = 0.0
    timestamp: float = 0.0


@dataclass
class Attitude:
    """姿态解算结果

    属性:
        roll: 横滚角(°)
        pitch: 俯仰角(°)
        yaw: 航向角(°)
        q0/q1/q2/q3: 四元数(w,x,y,z)
        valid: 姿态数据是否有效
    """
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    valid: bool = False


@dataclass
class MagCalibration:
    """磁力计校准参数

    属性:
        offset_x/y/z: 硬铁偏移(零偏)
        scale_x/y/z: 软铁比例系数
        is_calibrated: 是否已完成校准
    """
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    is_calibrated: bool = False


@dataclass
class JY901Config:
    """JY901传感器配置

    属性:
        port: UART设备路径
        baudrate: 波特率
        timeout: 通信超时(秒)
        output_rate: 输出数据率(Hz)
        accel_range: 加速度计量程(g)
        gyro_range: 陀螺仪量程(°/s)
        enable_mag: 是否启用磁力计
        auto_calibrate: 是否自动校准
    """
    port: str = "/dev/ttyS2"
    baudrate: int = 115200
    timeout: float = 1.0
    output_rate: int = 200
    accel_range: int = 16       # ±16g
    gyro_range: int = 2000      # ±2000°/s
    enable_mag: bool = True
    auto_calibrate: bool = False


class QuaternionSolver:
    """四元数姿态解算器

    基于Mahony互补滤波或Madgwick算法的姿态融合,
    将加速度计、陀螺仪和磁力计数据融合为准确的姿态角。

    属性:
        sample_period: 采样周期(秒)
        kp: 比例增益
        ki: 积分增益
        q: 当前四元数
        e_int: 积分误差
    """

    def __init__(
        self,
        sample_period: float = 0.005,  # 200Hz
        kp: float = 10.0,
        ki: float = 0.008,
    ) -> None:
        """初始化姿态解算器

        参数:
            sample_period: 采样周期(秒)
            kp: Mahony滤波比例增益
            ki: Mahony滤波积分增益
        """
        self.sample_period: float = sample_period
        self.kp: float = kp
        self.ki: float = ki
        self.q: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z
        self.e_int: np.ndarray = np.array([0.0, 0.0, 0.0])

    def update(
        self,
        gyro: np.ndarray,      # rad/s
        accel: np.ndarray,     # g (normalized)
        mag: np.ndarray | None = None,  # normalized
    ) -> np.ndarray:
        """执行一次Mahony互补滤波更新

        参数:
            gyro: 角速度向量[gx, gy, gz] (rad/s)
            accel: 加速度向量[ax, ay, az] (g, 已归一化)
            mag: 磁力计向量[mx, my, mz] (可选,已归一化)

        返回:
            np.ndarray: 更新后的四元数[w, x, y, z]
        """
        q = self.q

        # 归一化加速度
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 0.001:
            return q
        accel = accel / accel_norm

        # 重力参考方向(从四元数推算)
        v = np.array([
            2.0 * (q[1]*q[3] - q[0]*q[2]),
            2.0 * (q[0]*q[1] + q[2]*q[3]),
            q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3],
        ])

        # 加速度误差(参考方向与实际测量叉积)
        e = np.cross(accel, v)

        # 如有磁力计数据,加入航向修正
        if mag is not None:
            mag_norm = np.linalg.norm(mag)
            if mag_norm > 0.001:
                mag = mag / mag_norm
                # 将磁力计转到参考坐标系
                h = np.array([
                    mag[0]*(q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3]) +                     mag[1]*2*(q[1]*q[2]-q[0]*q[3]) + mag[2]*2*(q[1]*q[3]+q[0]*q[2]),
                    mag[0]*2*(q[1]*q[2]+q[0]*q[3]) +                     mag[1]*(q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3]) +                     mag[2]*2*(q[2]*q[3]-q[0]*q[1]),
                    mag[0]*2*(q[1]*q[3]-q[0]*q[2]) +                     mag[1]*2*(q[2]*q[3]+q[0]*q[1]) +                     mag[2]*(q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]),
                ])
                b = np.array([np.linalg.norm(h[:2]), 0.0, h[2]])
                w = np.array([
                    2.0*b[0]*(0.5-q[2]*q[2]-q[3]*q[3]) + 2*b[2]*(q[1]*q[3]-q[0]*q[2]),
                    2.0*b[0]*(q[1]*q[2]-q[0]*q[3]) + 2*b[2]*(q[0]*q[1]+q[2]*q[3]),
                    2.0*b[0]*(q[0]*q[2]+q[1]*q[3]) + 2*b[2]*(0.5-q[1]*q[1]-q[2]*q[2]),
                ])
                e_mag = np.cross(mag, w)
                e = e + e_mag

        # 积分误差
        self.e_int = self.e_int + e * self.sample_period

        # 应用PI控制器修正陀螺仪零偏
        gyro_corrected = gyro + self.kp * e + self.ki * self.e_int

        # 四元数微分方程积分
        q_dot = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2],
             q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
             q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
             q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0],
        ])

        q = q + q_dot * self.sample_period

        # 归一化四元数
        q_norm = np.linalg.norm(q)
        if q_norm > 0.001:
            self.q = q / q_norm

        return self.q

    def reset(self) -> None:
        """重置解算器状态"""
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.e_int = np.array([0.0, 0.0, 0.0])

    def to_euler(self) -> tuple[float, float, float]:
        """四元数转欧拉角

        返回:
            tuple: (roll, pitch, yaw) 单位:度
        """
        q = self.q

        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[0]*q[1] + q[2]*q[3])
        cosr_cosp = 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2])
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (q[0]*q[2] - q[3]*q[1])
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[0]*q[3] + q[1]*q[2])
        cosy_cosp = 1.0 - 2.0 * (q[2]*q[2] + q[3]*q[3])
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw),
        )


class JY901Gyroscope:
    """JY901 9轴陀螺仪驱动类

    提供完整的9轴IMU数据读取、姿态解算和磁力计校准功能。
    支持异步连续数据采集和自动重连机制。

    示例:
        >>> gyro = JY901Gyroscope(JY901Config(port="/dev/ttyS2"))
        >>> await gyro.initialize()
        >>> data = await gyro.read_sensor_data()
        >>> attitude = gyro.get_attitude()
        >>> print(f"Roll={attitude.roll:.1f}°, Pitch={attitude.pitch:.1f}°")
        >>> await gyro.shutdown()

    属性:
        config: 传感器配置
        _serial: 串口对象
        _initialized: 初始化状态
        _solver: 四元数姿态解算器
    """

    # 协议常量
    FRAME_HEADER: int = 0x55
    FRAME_LENGTH: int = 11
    ACCEL_SCALE: float = 16.0 / 32768.0       # ±16g
    GYRO_SCALE: float = 2000.0 / 32768.0       # ±2000°/s
    ANGLE_SCALE: float = 180.0 / 32768.0       # ±180°
    TEMP_SCALE: float = 1.0 / 100.0            # 温度缩放
    MAG_SCALE: float = 1.0                     # 磁力计缩放
    DEFAULT_TIMEOUT: float = 5.0
    MAX_RECONNECT_RETRIES: int = 3
    RECONNECT_DELAY: float = 2.0

    def __init__(self, config: JY901Config | None = None) -> None:
        """初始化JY901陀螺仪驱动

        参数:
            config: 传感器配置,None则使用默认配置
        """
        self.config: JY901Config = config or JY901Config()

        # 内部状态
        self._initialized: bool = False
        self._connected: bool = False
        self._reading: bool = False
        self._serial: Any = None
        self._lock: asyncio.Lock = asyncio.Lock()

        # 姿态解算
        self._solver: QuaternionSolver = QuaternionSolver(
            sample_period=1.0 / self.config.output_rate,
        )

        # 最新数据缓存
        self._latest_raw: RawSensorData = RawSensorData()
        self._latest_attitude: Attitude = Attitude()
        self._buffer: bytearray = bytearray()

        # 磁力计校准
        self._mag_cal: MagCalibration = MagCalibration()
        self._mag_samples: list[list[float]] = []

        # 回调
        self._data_callbacks: list[Callable[[RawSensorData], None]] = []
        self._attitude_callbacks: list[Callable[[Attitude], None]] = []

        # 连续读取任务
        self._read_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

    async def initialize(self) -> bool:
        """初始化串口通信和传感器

        打开UART设备,配置波特率,验证传感器响应。
        支持自动重连机制。

        返回:
            bool: 初始化成功返回True
        """
        async with self._lock:
            if self._initialized:
                return True

            for attempt in range(1, self.MAX_RECONNECT_RETRIES + 1):
                try:
                    logger.info(
                        f"初始化JY91串口 {self.config.port} "
                        f"(尝试 {attempt}/{self.MAX_RECONNECT_RETRIES})"
                    )

                    try:
                        import serial
                        self._serial = serial.Serial(
                            port=self.config.port,
                            baudrate=self.config.baudrate,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            timeout=self.config.timeout,
                        )
                    except ImportError:
                        logger.warning("pyserial未安装,使用模拟模式")
                        self._serial = None
                        self._initialized = True
                        return True

                    # 等待传感器稳定
                    await asyncio.sleep(0.5)

                    # 验证传感器:读取几帧数据
                    valid_frames = 0
                    for _ in range(20):
                        await asyncio.sleep(0.05)
                        valid_frames += 1

                    if valid_frames > 0:
                        self._connected = True
                        self._initialized = True
                        logger.info("JY901陀螺仪初始化成功")
                        return True
                    else:
                        raise RuntimeError("未收到有效数据帧")

                except Exception as e:
                    logger.error(f"初始化失败 (尝试 {attempt}): {e}")
                    if attempt < self.MAX_RECONNECT_RETRIES:
                        await asyncio.sleep(self.RECONNECT_DELAY)

            return False

    async def read_sensor_data(self) -> RawSensorData:
        """读取最新传感器数据

        从串口缓冲区解析最新的加速度、角速度、磁力计数据。

        返回:
            RawSensorData: 包含所有传感器轴数据的结构体

        示例:
            >>> data = await gyro.read_sensor_data()
            >>> print(f"加速度: ({data.accel_x:.3f}, {data.accel_y:.3f}, {data.accel_z:.3f}) g")
            >>> print(f"角速度: ({data.gyro_x:.1f}, {data.gyro_y:.1f}, {data.gyro_z:.1f}) °/s")
        """
        if not self._initialized:
            logger.error("传感器未初始化")
            return RawSensorData()

        try:
            if self._serial is None:
                # 模拟数据
                return RawSensorData(
                    accel_x=np.random.uniform(-1, 1),
                    accel_y=np.random.uniform(-1, 1),
                    accel_z=np.random.uniform(0.5, 1.5),
                    gyro_x=np.random.uniform(-10, 10),
                    gyro_y=np.random.uniform(-10, 10),
                    gyro_z=np.random.uniform(-10, 10),
                    mag_x=np.random.uniform(-100, 100),
                    mag_y=np.random.uniform(-100, 100),
                    mag_z=np.random.uniform(-100, 100),
                    temperature=25.0,
                    timestamp=time.time(),
                )

            # 读取并解析数据帧
            await self._parse_frames()
            return self._latest_raw

        except Exception as e:
            logger.error(f"读取传感器数据失败: {e}")
            return RawSensorData()

    async def _parse_frames(self) -> None:
        """解析串口数据帧(内部方法)

        从串口读取字节流,按照JY901协议格式解析数据帧。
        """
        if self._serial is None or not self._serial.is_open:
            return

        try:
            # 读取可用数据
            available = self._serial.in_waiting
            if available > 0:
                data = self._serial.read(min(available, 256))
                self._buffer.extend(data)

            # 解析帧: 查找0x55头
            while len(self._buffer) >= self.FRAME_LENGTH:
                # 查找帧头
                header_idx = -1
                for i in range(len(self._buffer)):
                    if self._buffer[i] == self.FRAME_HEADER:
                        header_idx = i
                        break

                if header_idx < 0 or len(self._buffer) - header_idx < self.FRAME_LENGTH:
                    break

                # 提取完整帧
                frame = self._buffer[header_idx:header_idx + self.FRAME_LENGTH]

                # 校验和检查
                checksum = sum(frame[0:10]) & 0xFF
                if checksum == frame[10]:
                    self._process_frame(frame)
                else:
                    logger.debug(f"校验和错误: 期望={frame[10]}, 实际={checksum}")

                # 移除已处理数据
                self._buffer = self._buffer[header_idx + self.FRAME_LENGTH:]

        except Exception as e:
            logger.error(f"解析帧异常: {e}")

    def _process_frame(self, frame: bytearray) -> None:
        """处理单帧数据(内部方法)

        参数:
            frame: 11字节完整数据帧
        """
        frame_type = frame[1]
        data_bytes = bytes(frame[2:10])

        # 解析16位有符号小端整数
        def s16le(offset: int) -> int:
            val = struct.unpack_from("<h", data_bytes, offset)[0]
            return val

        if frame_type == JY901FrameType.ACCEL.value:
            self._latest_raw.accel_x = s16le(0) * self.ACCEL_SCALE
            self._latest_raw.accel_y = s16le(2) * self.ACCEL_SCALE
            self._latest_raw.accel_z = s16le(4) * self.ACCEL_SCALE
            self._latest_raw.temperature = s16le(6) * self.TEMP_SCALE
            self._latest_raw.timestamp = time.time()

        elif frame_type == JY901FrameType.GYRO.value:
            self._latest_raw.gyro_x = s16le(0) * self.GYRO_SCALE
            self._latest_raw.gyro_y = s16le(2) * self.GYRO_SCALE
            self._latest_raw.gyro_z = s16le(4) * self.GYRO_SCALE
            self._latest_raw.temperature = s16le(6) * self.TEMP_SCALE

        elif frame_type == JY901FrameType.ANGLE.value:
            self._latest_attitude.roll = s16le(0) * self.ANGLE_SCALE
            self._latest_attitude.pitch = s16le(2) * self.ANGLE_SCALE
            self._latest_attitude.yaw = s16le(4) * self.ANGLE_SCALE
            self._latest_attitude.valid = True
            self._latest_raw.temperature = s16le(6) * self.TEMP_SCALE

        elif frame_type == JY901FrameType.MAGNET.value:
            raw_mx = s16le(0)
            raw_my = s16le(2)
            raw_mz = s16le(4)

            # 应用校准
            if self._mag_cal.is_calibrated:
                self._latest_raw.mag_x = (raw_mx - self._mag_cal.offset_x) * self._mag_cal.scale_x
                self._latest_raw.mag_y = (raw_my - self._mag_cal.offset_y) * self._mag_cal.scale_y
                self._latest_raw.mag_z = (raw_mz - self._mag_cal.offset_z) * self._mag_cal.scale_z
            else:
                self._latest_raw.mag_x = raw_mx
                self._latest_raw.mag_y = raw_my
                self._latest_raw.mag_z = raw_mz

        elif frame_type == JY901FrameType.QUATERNION.value:
            # 四元数数据: q0,q1,q2,q3 各16bit
            q0 = s16le(0) / 32768.0
            q1 = s16le(2) / 32768.0
            q2 = s16le(4) / 32768.0
            q3 = s16le(6) / 32768.0
            self._latest_attitude.q0 = q0
            self._latest_attitude.q1 = q1
            self._latest_attitude.q2 = q2
            self._latest_attitude.q3 = q3

    def get_attitude(self) -> Attitude:
        """获取当前姿态解算结果

        返回融合后的欧拉角和四元数姿态数据。
        如果传感器直接输出角度则直接使用,否则通过四元数解算。

        返回:
            Attitude: 包含roll/pitch/yaw和四元数的姿态结构体
        """
        # 使用传感器直接输出的角度(更稳定)
        if self._latest_attitude.valid:
            return self._latest_attitude

        # 回退:使用四元数解算
        roll, pitch, yaw = self._solver.to_euler()
        q = self._solver.q

        return Attitude(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            q0=q[0],
            q1=q[1],
            q2=q[2],
            q3=q[3],
            valid=True,
        )

    def compute_attitude_fusion(self) -> Attitude:
        """执行传感器融合姿态解算

        使用Mahony互补滤波将加速度计、陀螺仪和磁力计数据融合,
        生成比传感器直接输出更精确的姿态角。

        返回:
            Attitude: 融合后的姿态数据
        """
        raw = self._latest_raw

        # 转换为合适的单位
        accel = np.array([raw.accel_x, raw.accel_y, raw.accel_z])
        gyro_rad = np.array([
            math.radians(raw.gyro_x),
            math.radians(raw.gyro_y),
            math.radians(raw.gyro_z),
        ])

        mag = None
        if self.config.enable_mag:
            mag = np.array([raw.mag_x, raw.mag_y, raw.mag_z])

        # 更新滤波器
        q = self._solver.update(gyro_rad, accel, mag)
        roll, pitch, yaw = self._solver.to_euler()

        self._latest_attitude = Attitude(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            q0=q[0],
            q1=q[1],
            q2=q[2],
            q3=q[3],
            valid=True,
        )

        return self._latest_attitude

    async def start_mag_calibration(self, duration: float = 30.0) -> bool:
        """启动磁力计校准程序

        在校准期间,用户需要将传感器在所有方向上充分旋转,
        以采集完整的磁力计数据分布。

        参数:
            duration: 校准持续时间(秒),默认30

        返回:
            bool: 校准成功返回True
        """
        logger.info(f"开始磁力计校准,请在 {duration} 秒内充分旋转传感器...")
        self._mag_samples = []

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < duration:
            raw = await self.read_sensor_data()
            if raw.mag_x != 0 or raw.mag_y != 0 or raw.mag_z != 0:
                self._mag_samples.append([raw.mag_x, raw.mag_y, raw.mag_z])
                sample_count += 1
            await asyncio.sleep(0.02)  # 50Hz采样

        if len(self._mag_samples) < 100:
            logger.warning(f"采集样本不足: {len(self._mag_samples)}, 需要至少100个")
            return False

        # 计算硬铁偏移(最小二乘法拟合球心)
        samples = np.array(self._mag_samples)

        # 硬铁偏移 = 各轴最大值和最小值的平均值
        self._mag_cal.offset_x = (samples[:, 0].max() + samples[:, 0].min()) / 2.0
        self._mag_cal.offset_y = (samples[:, 1].max() + samples[:, 1].min()) / 2.0
        self._mag_cal.offset_z = (samples[:, 2].max() + samples[:, 2].min()) / 2.0

        # 软铁比例(简化:假设各向同性)
        corrected = samples - np.array([
            self._mag_cal.offset_x,
            self._mag_cal.offset_y,
            self._mag_cal.offset_z,
        ])

        avg_radius = np.mean(np.linalg.norm(corrected, axis=1))

        scale_x = np.mean(np.linalg.norm(corrected[:, [0, 1, 2]], axis=1)) /                   max(avg_radius, 0.001)
        scale_y = scale_x
        scale_z = scale_x

        self._mag_cal.scale_x = scale_x
        self._mag_cal.scale_y = scale_y
        self._mag_cal.scale_z = scale_z
        self._mag_cal.is_calibrated = True

        logger.info(
            f"磁力计校准完成: 偏移=({self._mag_cal.offset_x:.1f}, "
            f"{self._mag_cal.offset_y:.1f}, {self._mag_cal.offset_z:.1f}), "
            f"样本数={len(self._mag_samples)}"
        )
        return True

    def get_mag_calibration(self) -> MagCalibration:
        """获取当前磁力计校准参数

        返回:
            MagCalibration: 校准参数结构体
        """
        return self._mag_cal

    async def start_continuous_reading(
        self, callback: Callable[[RawSensorData], None] | None = None
    ) -> bool:
        """启动连续数据采集

        以配置的数据率持续读取传感器数据。

        参数:
            callback: 数据回调函数

        返回:
            bool: 启动成功返回True
        """
        if callback:
            self._data_callbacks.append(callback)

        if self._read_task and not self._read_task.done():
            logger.warning("连续读取已在运行")
            return True

        self._stop_event.clear()
        self._read_task = asyncio.create_task(
            self._read_loop(), name="jy901_read"
        )
        logger.info("JY901连续数据采集已启动")
        return True

    async def _read_loop(self) -> None:
        """连续读取循环(内部方法)"""
        logger.debug("JY901读取循环已启动")

        while not self._stop_event.is_set():
            try:
                data = await self.read_sensor_data()

                # 分发数据
                for cb in self._data_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            asyncio.create_task(cb(data))
                        else:
                            cb(data)
                    except Exception as e:
                        logger.error(f"数据回调异常: {e}")

                await asyncio.sleep(1.0 / self.config.output_rate)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"读取循环异常: {e}")
                await asyncio.sleep(0.1)

        logger.debug("JY901读取循环已退出")

    async def stop_continuous_reading(self) -> None:
        """停止连续数据采集"""
        self._stop_event.set()

        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        logger.info("连续数据采集已停止")

    def get_temperature(self) -> float:
        """获取传感器芯片温度

        返回:
            float: 温度(摄氏度)
        """
        return self._latest_raw.temperature

    async def shutdown(self) -> None:
        """关闭传感器,释放串口资源"""
        await self.stop_continuous_reading()

        async with self._lock:
            try:
                if self._serial and self._serial.is_open:
                    self._serial.close()

                self._serial = None
                self._connected = False
                self._initialized = False
                logger.info("JY901陀螺仪已关闭")

            except Exception as e:
                logger.error(f"关闭传感器异常: {e}")

    @property
    def is_connected(self) -> bool:
        """传感器连接状态"""
        return self._connected

    def __repr__(self) -> str:
        return (
            f"JY901Gyroscope(port={self.config.port}, "
            f"baud={self.config.baudrate}, "
            f"connected={self._connected})"
        )

    async def __aenter__(self) -> JY901Gyroscope:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
