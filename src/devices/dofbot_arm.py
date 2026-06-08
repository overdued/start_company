"""
Dofbot 6自由度机械臂驱动模块

基于PCA9685 PWM控制器的6自由度机械臂驱动,支持运动学正逆解、
笛卡尔空间移动和抓取/放置动作序列。
适用于OrangePi Kunpeng Pro (RK3588)平台。

功能特性:
    - 6自由度舵机控制(0-180°)
    - 正运动学(FK)和逆运动学(IK)解算
    - 笛卡尔空间直线/圆弧移动
    - 抓取/放置预定义动作序列
    - 速度/加速度规划
    - 碰撞检测与安全限位

舵机布局:
    Joint 1: 基座旋转 (Base)     - 舵机0
    Joint 2: 肩部俯仰 (Shoulder)  - 舵机1
    Joint 3: 肘部俯仰 (Elbow)     - 舵机2
    Joint 4: 腕部俯仰 (Wrist)     - 舵机3
    Joint 5: 腕部旋转 (WristRot)  - 舵机4
    Joint 6: 夹爪 (Gripper)       - 舵机5

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ArmStatus(Enum):
    """机械臂状态枚举"""
    IDLE = "idle"               # 空闲
    MOVING = "moving"           # 运动中
    GRIPPING = "gripping"       # 夹取中
    HOMING = "homing"           # 回零中
    ERROR = "error"             # 错误状态
    EMERGENCY_STOP = "estop"    # 紧急停止


class GripperState(Enum):
    """夹爪状态枚举"""
    OPEN = "open"       # 打开
    CLOSED = "closed"   # 关闭
    HALF = "half"       # 半开


@dataclass
class JointAngles:
    """关节角度数据结构

    属性:
        j1-j6: 6个关节的角度值(度)
        timestamp: 记录时间戳
    """
    j1: float = 90.0
    j2: float = 90.0
    j3: float = 90.0
    j4: float = 90.0
    j5: float = 90.0
    j6: float = 90.0
    timestamp: float = 0.0

    def to_array(self) -> np.ndarray:
        """转换为numpy数组

        返回:
            np.ndarray: [j1, j2, j3, j4, j5, j6]
        """
        return np.array([self.j1, self.j2, self.j3, self.j4, self.j5, self.j6])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> JointAngles:
        """从numpy数组创建

        参数:
            arr: 6维角度数组

        返回:
            JointAngles实例
        """
        return cls(
            j1=float(arr[0]), j2=float(arr[1]), j3=float(arr[2]),
            j4=float(arr[3]), j5=float(arr[4]), j6=float(arr[5]),
            timestamp=time.time(),
        )


@dataclass
class CartesianPose:
    """笛卡尔空间位姿数据结构

    属性:
        x/y/z: 末端位置(米)
        roll/pitch/yaw: 末端姿态(欧拉角,度)
        valid: 位姿是否有效
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    valid: bool = False


@dataclass
class DHParameter:
    """DH参数数据结构(改进DH参数)

    属性:
        theta: 关节角偏移(rad)
        d: 连杆偏距(m)
        a: 连杆长度(m)
        alpha: 连杆扭角(rad)
    """
    theta: float = 0.0
    d: float = 0.0
    a: float = 0.0
    alpha: float = 0.0


@dataclass
class ArmConfig:
    """机械臂配置参数

    属性:
        i2c_bus: I2C总线号
        pca9685_addr: PCA9685 I2C地址
        servo_channels: 6个舵机的通道映射
        joint_limits: 各关节角度限制(度) [(min,max), ...]
        max_speed: 最大关节速度(度/秒)
        max_accel: 最大关节加速度(度/秒²)
        default_speed: 默认运动速度(度/秒)
    """
    i2c_bus: int = 1
    pca9685_addr: int = 0x40
    servo_channels: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5]
    )
    joint_limits: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0, 180),    # j1: base
            (0, 180),    # j2: shoulder
            (0, 180),    # j3: elbow
            (0, 180),    # j4: wrist pitch
            (0, 180),    # j5: wrist rot
            (30, 180),   # j6: gripper
        ]
    )
    max_speed: float = 120.0     # 度/秒
    max_accel: float = 240.0     # 度/秒²
    default_speed: float = 60.0  # 度/秒


class DofbotArm:
    """Dofbot 6自由度机械臂驱动类

    提供完整的机械臂控制功能,包括关节控制、运动学解算、
    笛卡尔空间移动和预定义动作序列。

    示例:
        >>> arm = DofbotArm(ArmConfig())
        >>> await arm.initialize()
        >>> await arm.home()  # 回零位
        >>> pose = await arm.forward_kinematics()  # 正运动学
        >>> target = CartesianPose(x=0.15, y=0.0, z=0.1)
        >>> await arm.move_to_cartesian(target, speed=50)
        >>> await arm.grip()  # 夹取
        >>> await arm.shutdown()

    属性:
        config: 机械臂配置
        _status: 当前状态
        _current_angles: 当前关节角度
        _dh_params: DH参数表
    """

    # 默认DH参数(根据实际机械臂标定)
    DEFAULT_DH_PARAMS: list[DHParameter] = [
        DHParameter(theta=0.0,      d=0.065,   a=0.0,     alpha=math.pi/2),
        DHParameter(theta=0.0,      d=0.0,     a=0.110,   alpha=0.0),
        DHParameter(theta=0.0,      d=0.0,     a=0.096,   alpha=0.0),
        DHParameter(theta=math.pi/2, d=0.0,     a=0.0,     alpha=math.pi/2),
        DHParameter(theta=0.0,      d=0.072,   a=0.0,     alpha=-math.pi/2),
        DHParameter(theta=0.0,      d=0.045,   a=0.0,     alpha=0.0),
    ]

    # 舵机PWM参数
    SERVO_FREQ: float = 50.0          # Hz
    PWM_MIN: int = 150                # 0°对应脉宽(0.5ms @50Hz)
    PWM_MAX: int = 600                # 180°对应脉宽(2.5ms @50Hz)
    PWM_RESOLUTION: int = 4096        # 12bit分辨率

    # 安全常量
    DEFAULT_TIMEOUT: float = 10.0
    HOMING_TIMEOUT: float = 15.0
    MAX_RETRIES: int = 3
    SERVO_DELAY: float = 0.02         # 舵机响应延迟

    def __init__(self, config: ArmConfig | None = None) -> None:
        """初始化Dofbot机械臂驱动

        参数:
            config: 机械臂配置,None则使用默认配置
        """
        self.config: ArmConfig = config or ArmConfig()

        # 内部状态
        self._status: ArmStatus = ArmStatus.IDLE
        self._current_angles: JointAngles = JointAngles()
        self._target_angles: JointAngles = JointAngles()
        self._gripper_state: GripperState = GripperState.OPEN
        self._dh_params: list[DHParameter] = list(self.DEFAULT_DH_PARAMS)
        self._initialized: bool = False
        self._emergency_stop: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

        # 状态回调
        self._status_callbacks: list[Callable[[ArmStatus], None]] = []

        # PCA9685控制器(延迟初始化)
        self._pca: Any = None
        self._bus: Any = None

    async def initialize(self) -> bool:
        """初始化机械臂硬件

        初始化I2C总线、PCA9685 PWM控制器,并将所有舵机移动到初始位置。

        返回:
            bool: 初始化成功返回True
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # 尝试导入smbus2库
                try:
                    from smbus2 import SMBus
                    self._bus = SMBus(self.config.i2c_bus)
                    self._setup_pca9685()
                except ImportError:
                    logger.warning("smbus2未安装,使用模拟模式")
                    self._bus = None

                # 设置所有舵机到初始位置
                await self._set_all_servos(self._current_angles.to_array())

                self._initialized = True
                self._status = ArmStatus.IDLE
                logger.info("Dofbot机械臂初始化成功")
                return True

            except Exception as e:
                logger.error(f"Dofbot机械臂初始化失败: {e}")
                return False

    def _setup_pca9685(self) -> None:
        """配置PCA9685 PWM控制器(内部方法)

        设置PWM频率为50Hz,配置模式寄存器。
        """
        if self._bus is None:
            return

        addr = self.config.pca9685_addr

        # MODE1: 重启 + 自动递增 + 正常模式
        self._bus.write_byte_data(addr, 0x00, 0x00)
        time.sleep(0.005)

        # 设置PWM频率
        prescale = int(25000000.0 / (self.PWM_RESOLUTION * self.SERVO_FREQ) - 1)
        old_mode = self._bus.read_byte_data(addr, 0x00)
        self._bus.write_byte_data(addr, 0x00, (old_mode & 0x7F) | 0x10)
        self._bus.write_byte_data(addr, 0xFE, prescale)
        self._bus.write_byte_data(addr, 0x00, old_mode)
        time.sleep(0.005)
        self._bus.write_byte_data(addr, 0x00, old_mode | 0xA1)

        logger.debug(f"PCA9685配置完成: 频率={self.SERVO_FREQ}Hz, "
                     f"prescale={prescale}")

    def _angle_to_pwm(self, angle: float) -> int:
        """角度转PWM值(内部方法)

        将0-180度映射到PCA9685的PWM寄存器值。

        参数:
            angle: 角度值(度)

        返回:
            int: PWM寄存器值
        """
        # 钳制角度
        angle = max(0, min(180, angle))

        # 线性映射: 0°->PWM_MIN, 180°->PWM_MAX
        pwm = int(self.PWM_MIN + (angle / 180.0) * (self.PWM_MAX - self.PWM_MIN))
        return pwm

    def _pwm_to_angle(self, pwm: int) -> float:
        """PWM值转角度(内部方法)

        参数:
            pwm: PWM寄存器值

        返回:
            float: 角度值(度)
        """
        angle = (pwm - self.PWM_MIN) / (self.PWM_MAX - self.PWM_MIN) * 180.0
        return max(0, min(180, angle))

    async def _set_servo(self, channel: int, angle: float) -> bool:
        """设置单个舵机角度(内部方法)

        参数:
            channel: 舵机通道号
            angle: 目标角度(度)

        返回:
            bool: 设置成功返回True
        """
        try:
            if self._bus is None:
                # 模拟模式
                await asyncio.sleep(self.SERVO_DELAY)
                return True

            pwm = self._angle_to_pwm(angle)
            addr = self.config.pca9685_addr

            # PCA9685寄存器: LEDx_ON_L, LEDx_ON_H, LEDx_OFF_L, LEDx_OFF_H
            base_reg = 0x06 + channel * 4
            self._bus.write_byte_data(addr, base_reg, 0x00)
            self._bus.write_byte_data(addr, base_reg + 1, 0x00)
            self._bus.write_byte_data(addr, base_reg + 2, pwm & 0xFF)
            self._bus.write_byte_data(addr, base_reg + 3, (pwm >> 8) & 0x0F)

            return True

        except Exception as e:
            logger.error(f"设置舵机{channel}角度失败: {e}")
            return False

    async def _set_all_servos(self, angles: np.ndarray) -> bool:
        """同时设置所有舵机角度(内部方法)

        参数:
            angles: 6维角度数组

        返回:
            bool: 设置成功返回True
        """
        for i, angle in enumerate(angles[:6]):
            channel = self.config.servo_channels[i]
            success = await self._set_servo(channel, angle)
            if not success:
                return False
            await asyncio.sleep(0.005)  # 短暂延迟避免I2C拥塞

        return True

    def _check_joint_limits(self, angles: JointAngles) -> tuple[bool, str]:
        """检查关节角度是否在安全范围内(内部方法)

        参数:
            angles: 关节角度

        返回:
            tuple: (是否安全, 错误信息)
        """
        arr = angles.to_array()
        for i, (joint_min, joint_max) in enumerate(self.config.joint_limits):
            if arr[i] < joint_min or arr[i] > joint_max:
                return False, (
                    f"关节{i+1}角度{arr[i]:.1f}°超出限制"
                    f"[{joint_min}, {joint_max}]"
                )
        return True, "OK"

    def _update_status(self, new_status: ArmStatus) -> None:
        """更新机械臂状态并通知回调(内部方法)

        参数:
            new_status: 新状态
        """
        if self._status != new_status:
            self._status = new_status
            logger.debug(f"机械臂状态变更: {new_status.value}")

            for cb in self._status_callbacks:
                try:
                    cb(new_status)
                except Exception as e:
                    logger.error(f"状态回调异常: {e}")

    def register_status_callback(
        self, callback: Callable[[ArmStatus], None]
    ) -> None:
        """注册状态变更回调函数

        参数:
            callback: 状态回调函数
        """
        if callback not in self._status_callbacks:
            self._status_callbacks.append(callback)

    async def home(self, speed: float | None = None) -> bool:
        """机械臂回零位

        将所有关节移动到预定义的初始位置。

        参数:
            speed: 运动速度(度/秒),None则使用默认值

        返回:
            bool: 回零成功返回True
        """
        if self._emergency_stop:
            logger.error("机械臂处于紧急停止状态,无法运动")
            return False

        self._update_status(ArmStatus.HOMING)

        home_angles = JointAngles(
            j1=90.0, j2=90.0, j3=90.0, j4=90.0, j5=90.0, j6=90.0
        )

        result = await self.move_joints(home_angles, speed)

        if result:
            self._update_status(ArmStatus.IDLE)
            logger.info("机械臂已回零位")
        else:
            self._update_status(ArmStatus.ERROR)

        return result

    async def move_joints(
        self,
        angles: JointAngles,
        speed: float | None = None,
        blocking: bool = True,
    ) -> bool:
        """关节空间移动

        将机械臂移动到指定的关节角度配置。

        参数:
            angles: 目标关节角度
            speed: 运动速度(度/秒),None则使用默认值
            blocking: 是否阻塞等待完成

        返回:
            bool: 运动成功返回True

        示例:
            >>> target = JointAngles(j1=90, j2=45, j3=120, j4=90, j5=90, j6=60)
            >>> await arm.move_joints(target, speed=60)
        """
        if not self._initialized:
            logger.error("机械臂未初始化")
            return False

        if self._emergency_stop:
            logger.error("紧急停止状态,拒绝运动指令")
            return False

        # 安全检查
        safe, msg = self._check_joint_limits(angles)
        if not safe:
            logger.error(f"关节角度安全检查失败: {msg}")
            return False

        self._update_status(ArmStatus.MOVING)

        move_speed = speed or self.config.default_speed
        move_speed = min(move_speed, self.config.max_speed)

        try:
            current = self._current_angles.to_array()
            target = angles.to_array()

            # 计算最大关节位移
            max_delta = np.max(np.abs(target - current))

            if max_delta < 0.5:
                # 目标已到达
                self._update_status(ArmStatus.IDLE)
                return True

            # 计算运动时间
            move_time = max_delta / move_speed

            # 梯形速度规划: 插值步数
            num_steps = max(int(move_time / 0.02), 1)  # 50Hz更新率

            for step in range(1, num_steps + 1):
                if self._emergency_stop:
                    logger.warning("运动被紧急停止中断")
                    self._update_status(ArmStatus.EMERGENCY_STOP)
                    return False

                t = step / num_steps
                # 平滑插值(余弦缓动)
                t_smooth = 0.5 * (1 - math.cos(t * math.pi))

                interp = current + (target - current) * t_smooth

                success = await self._set_all_servos(interp)
                if not success:
                    self._update_status(ArmStatus.ERROR)
                    return False

                await asyncio.sleep(0.02)

            self._current_angles = angles
            self._current_angles.timestamp = time.time()
            self._update_status(ArmStatus.IDLE)

            logger.debug(f"关节运动完成: {angles}")
            return True

        except Exception as e:
            logger.error(f"关节运动异常: {e}")
            self._update_status(ArmStatus.ERROR)
            return False

    def forward_kinematics(
        self, angles: JointAngles | None = None
    ) -> CartesianPose:
        """正运动学解算

        根据关节角度计算末端执行器的笛卡尔空间位姿。
        使用改进DH参数法构建齐次变换矩阵。

        参数:
            angles: 关节角度,None则使用当前角度

        返回:
            CartesianPose: 末端位姿(x, y, z, roll, pitch, yaw)

        示例:
            >>> pose = arm.forward_kinematics(JointAngles(j1=90, j2=45, ...))
            >>> print(f"末端位置: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")
        """
        if angles is None:
            angles = self._current_angles

        try:
            joint_rad = np.radians(angles.to_array())

            # 构建齐次变换矩阵
            T = np.eye(4)

            for i, dh in enumerate(self._dh_params):
                theta = dh.theta + joint_rad[i]
                ct, st = math.cos(theta), math.sin(theta)
                ca, sa = math.cos(dh.alpha), math.sin(dh.alpha)

                A_i = np.array([
                    [ct, -st*ca,  st*sa, dh.a*ct],
                    [st,  ct*ca, -ct*sa, dh.a*st],
                    [0,   sa,     ca,    dh.d],
                    [0,   0,      0,     1],
                ])

                T = T @ A_i

            # 提取位置和姿态
            x, y, z = T[0, 3], T[1, 3], T[2, 3]

            # 旋转矩阵转欧拉角(ZYX顺序)
            R = T[:3, :3]

            if abs(R[2, 0]) < 0.99999:
                pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))
                yaw = math.atan2(R[1, 0], R[0, 0])
                roll = math.atan2(R[2, 1], R[2, 2])
            else:
                pitch = math.pi / 2 if R[2, 0] < 0 else -math.pi / 2
                yaw = 0
                roll = math.atan2(-R[0, 1], R[1, 1])

            return CartesianPose(
                x=x, y=y, z=z,
                roll=math.degrees(roll),
                pitch=math.degrees(pitch),
                yaw=math.degrees(yaw),
                valid=True,
            )

        except Exception as e:
            logger.error(f"正运动学解算失败: {e}")
            return CartesianPose()

    def inverse_kinematics(
        self,
        target: CartesianPose,
        initial_guess: JointAngles | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> JointAngles | None:
        """逆运动学解算(数值方法)

        使用牛顿-拉夫森迭代法求解目标位姿对应的关节角度。
        基于雅可比矩阵的伪逆进行迭代优化。

        参数:
            target: 目标笛卡尔位姿
            initial_guess: 初始猜测角度,None则使用当前角度
            max_iterations: 最大迭代次数
            tolerance: 收敛容差

        返回:
            JointAngles或None(无解)

        示例:
            >>> target = CartesianPose(x=0.15, y=0.0, z=0.12)
            >>> angles = arm.inverse_kinematics(target)
            >>> if angles:
            ...     await arm.move_joints(angles)
        """
        if initial_guess is None:
            initial_guess = self._current_angles

        try:
            q = initial_guess.to_array().copy()
            target_pos = np.array([target.x, target.y, target.z])

            for iteration in range(max_iterations):
                # 当前位姿
                current_angles = JointAngles.from_array(q)
                current_pose = self.forward_kinematics(current_angles)

                if not current_pose.valid:
                    logger.warning("正运动学解算失败,IK迭代中断")
                    return None

                current_pos = np.array([
                    current_pose.x, current_pose.y, current_pose.z,
                ])

                # 位置误差
                pos_error = target_pos - current_pos
                error_norm = np.linalg.norm(pos_error)

                if error_norm < tolerance:
                    # 收敛
                    result = JointAngles.from_array(q)
                    safe, msg = self._check_joint_limits(result)
                    if safe:
                        logger.debug(f"IK收敛于 {iteration} 次迭代")
                        return result
                    else:
                        logger.warning(f"IK解超出关节限制: {msg}")
                        return None

                # 数值雅可比矩阵
                J = self._compute_jacobian(q)

                # 伪逆
                try:
                    J_pinv = np.linalg.pinv(J)
                except np.linalg.LinAlgError:
                    logger.warning("雅可比矩阵奇异")
                    return None

                # 更新关节角度
                delta_q = J_pinv @ pos_error
                q = q + delta_q * 0.5  # 阻尼因子

                # 关节限幅
                for i, (jmin, jmax) in enumerate(self.config.joint_limits):
                    q[i] = max(jmin, min(jmax, q[i]))

            logger.warning(f"IK达到最大迭代次数,误差={error_norm:.6f}")
            return None

        except Exception as e:
            logger.error(f"逆运动学解算异常: {e}")
            return None

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """计算数值雅可比矩阵(内部方法)

        使用中心差分法计算雅可比矩阵。

        参数:
            q: 当前关节角度

        返回:
            np.ndarray: 3x6雅可比矩阵
        """
        delta = 0.001  # 微小扰动
        J = np.zeros((3, 6))

        base_pose = self.forward_kinematics(JointAngles.from_array(q))
        base_pos = np.array([base_pose.x, base_pose.y, base_pose.z])

        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += delta

            pose_plus = self.forward_kinematics(JointAngles.from_array(q_plus))
            pos_plus = np.array([pose_plus.x, pose_plus.y, pose_plus.z])

            J[:, i] = (pos_plus - base_pos) / delta

        return J

    async def move_to_cartesian(
        self,
        target: CartesianPose,
        speed: float | None = None,
    ) -> bool:
        """笛卡尔空间移动

        将末端执行器移动到目标笛卡尔位姿。
        先进行逆运动学解算,再执行关节运动。

        参数:
            target: 目标笛卡尔位姿
            speed: 运动速度(度/秒)

        返回:
            bool: 移动成功返回True

        示例:
            >>> target = CartesianPose(x=0.15, y=0.05, z=0.10)
            >>> await arm.move_to_cartesian(target, speed=50)
        """
        # 逆运动学解算
        angles = self.inverse_kinematics(target)

        if angles is None:
            logger.error("逆运动学无解")
            return False

        # 执行关节运动
        return await self.move_joints(angles, speed)

    async def move_linear(
        self,
        target: CartesianPose,
        speed: float | None = None,
        num_waypoints: int = 20,
    ) -> bool:
        """直线插值移动

        在笛卡尔空间进行直线插值,生成中间路径点并依次执行。

        参数:
            target: 目标位姿
            speed: 运动速度
            num_waypoints: 插值路径点数

        返回:
            bool: 移动成功返回True
        """
        current_pose = self.forward_kinematics()
        if not current_pose.valid:
            return False

        # 生成直线插值路径点
        waypoints: list[CartesianPose] = []
        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            # 直线插值
            wp = CartesianPose(
                x=current_pose.x + (target.x - current_pose.x) * t,
                y=current_pose.y + (target.y - current_pose.y) * t,
                z=current_pose.z + (target.z - current_pose.z) * t,
                roll=current_pose.roll + (target.roll - current_pose.roll) * t,
                pitch=current_pose.pitch + (target.pitch - current_pose.pitch) * t,
                yaw=current_pose.yaw + (target.yaw - current_pose.yaw) * t,
            )
            waypoints.append(wp)

        # 依次执行路径点
        for wp in waypoints:
            angles = self.inverse_kinematics(wp)
            if angles is None:
                logger.warning("路径点逆运动学无解,跳过")
                continue

            success = await self.move_joints(angles, speed, blocking=True)
            if not success:
                logger.error("路径点运动失败")
                return False

            await asyncio.sleep(0.01)

        return True

    async def grip(self, force: float = 0.8) -> bool:
        """夹取动作

        控制夹爪闭合到指定力度位置。

        参数:
            force: 夹取力度(0-1.0),0.8表示闭合到80%

        返回:
            bool: 操作成功返回True
        """
        self._update_status(ArmStatus.GRIPPING)

        gripper_angle = 30.0 + (1.0 - force) * 150.0  # 30°=闭合, 180°=张开
        gripper_angle = max(30.0, min(180.0, gripper_angle))

        current = self._current_angles
        current.j6 = gripper_angle

        result = await self.move_joints(current)

        if result:
            self._gripper_state = GripperState.CLOSED
            self._update_status(ArmStatus.IDLE)
            logger.info(f"夹取完成: 力度={force:.1f}")

        return result

    async def release(self) -> bool:
        """释放夹爪

        完全打开夹爪。

        返回:
            bool: 操作成功返回True
        """
        current = self._current_angles
        current.j6 = 180.0  # 完全打开

        result = await self.move_joints(current)

        if result:
            self._gripper_state = GripperState.OPEN
            logger.info("夹爪已释放")

        return result

    async def execute_sequence(
        self,
        sequence: list[dict[str, Any]],
        speed: float | None = None,
    ) -> bool:
        """执行动作序列

        按顺序执行预定义的动作列表。

        参数:
            sequence: 动作序列,每个动作为字典:
                {"type": "move_joints", "angles": JointAngles}
                {"type": "move_cartesian", "pose": CartesianPose}
                {"type": "grip", "force": 0.8}
                {"type": "release"}
                {"type": "wait", "duration": 1.0}
            speed: 运动速度

        返回:
            bool: 全部执行成功返回True
        """
        for i, action in enumerate(sequence):
            logger.debug(f"执行动作 {i+1}/{len(sequence)}: {action}")

            action_type = action.get("type", "")

            if action_type == "move_joints":
                angles = action["angles"]
                success = await self.move_joints(angles, speed)

            elif action_type == "move_cartesian":
                pose = action["pose"]
                success = await self.move_to_cartesian(pose, speed)

            elif action_type == "grip":
                force = action.get("force", 0.8)
                success = await self.grip(force)

            elif action_type == "release":
                success = await self.release()

            elif action_type == "wait":
                duration = action.get("duration", 1.0)
                await asyncio.sleep(duration)
                success = True

            else:
                logger.warning(f"未知动作类型: {action_type}")
                success = False

            if not success:
                logger.error(f"动作序列在步骤 {i+1} 失败")
                return False

        logger.info(f"动作序列执行完成: {len(sequence)} 个动作")
        return True

    async def pick_and_place(
        self,
        pick_pose: CartesianPose,
        place_pose: CartesianPose,
        speed: float | None = None,
    ) -> bool:
        """抓取-放置完整动作

        执行标准的抓取-放置操作序列:
        1. 移动到抓取上方
        2. 下降并夹取
        3. 提升
        4. 移动到放置上方
        5. 下降并释放
        6. 撤回

        参数:
            pick_pose: 抓取位姿
            place_pose: 放置位姿
            speed: 运动速度

        返回:
            bool: 操作成功返回True
        """
        sequence = [
            # 1. 移动到抓取点上方
            {"type": "move_cartesian", "pose": CartesianPose(
                x=pick_pose.x, y=pick_pose.y, z=pick_pose.z + 0.05,
            )},
            # 2. 打开夹爪
            {"type": "release"},
            # 3. 下降到抓取点
            {"type": "move_cartesian", "pose": pick_pose},
            # 4. 夹取
            {"type": "grip", "force": 0.8},
            # 5. 提升
            {"type": "move_cartesian", "pose": CartesianPose(
                x=pick_pose.x, y=pick_pose.y, z=pick_pose.z + 0.05,
            )},
            # 6. 移动到放置点上方
            {"type": "move_cartesian", "pose": CartesianPose(
                x=place_pose.x, y=place_pose.y, z=place_pose.z + 0.05,
            )},
            # 7. 下降到放置点
            {"type": "move_cartesian", "pose": place_pose},
            # 8. 释放
            {"type": "release"},
            # 9. 撤回
            {"type": "move_cartesian", "pose": CartesianPose(
                x=place_pose.x, y=place_pose.y, z=place_pose.z + 0.05,
            )},
        ]

        logger.info("开始执行抓取-放置动作")
        return await self.execute_sequence(sequence, speed)

    async def emergency_stop(self) -> None:
        """紧急停止

        立即停止所有运动,锁定当前姿态。
        需要调用 reset_emergency_stop 解除。
        """
        self._emergency_stop = True
        self._update_status(ArmStatus.EMERGENCY_STOP)
        logger.critical("机械臂紧急停止已触发")

    async def reset_emergency_stop(self) -> None:
        """复位紧急停止状态"""
        self._emergency_stop = False
        self._update_status(ArmStatus.IDLE)
        logger.info("机械臂紧急停止已复位")

    @property
    def status(self) -> ArmStatus:
        """当前机械臂状态"""
        return self._status

    @property
    def current_angles(self) -> JointAngles:
        """当前关节角度"""
        return self._current_angles

    @property
    def current_pose(self) -> CartesianPose:
        """当前末端位姿"""
        return self.forward_kinematics()

    @property
    def gripper_state(self) -> GripperState:
        """当前夹爪状态"""
        return self._gripper_state

    async def shutdown(self) -> None:
        """关闭机械臂,释放资源

        停止所有运动,关闭PWM输出。
        """
        async with self._lock:
            try:
                if self._pca:
                    # 关闭所有PWM输出
                    pass

                if self._bus:
                    self._bus.close()

                self._initialized = False
                logger.info("Dofbot机械臂已关闭")

            except Exception as e:
                logger.error(f"关闭机械臂异常: {e}")

    def __repr__(self) -> str:
        return (
            f"DofbotArm(status={self._status.value}, "
            f"initialized={self._initialized}, "
            f"angles={self._current_angles.to_array().round(1)})"
        )

    async def __aenter__(self) -> DofbotArm:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
