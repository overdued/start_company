"""
情感表达输出模块

多通道情感表达输出驱动,支持OLED表情显示、机械臂姿态映射、
TTS语调控制等多通道同步输出。
适用于OrangePi Kunpeng Pro (RK3588)平台。

功能特性:
    - OLED显示屏控制(SSD1306 I2C)
    - 表情图案库(高兴、悲伤、关心、惊讶等)
    - 机械臂姿态映射(情感→姿态)
    - TTS语调控制
    - 多通道同步输出
    - 情感过渡动画

情感映射:
    HAPPY:    OLED微笑表情 + 机械臂抬手 + TTS欢快语调
    SAD:      OLED悲伤表情 + 机械臂下垂 + TTS缓慢语调
    CARING:   OLED关心表情 + 机械臂前倾 + TTS温柔语调
    SURPRISED:OLED惊讶表情 + 机械臂举起 + TTS升调
    NEUTRAL:  OLED中性表情 + 机械臂自然 + TTS正常语调
    ANGRY:    OLED生气表情 + 机械臂收紧 + TTS急促语调
    SLEEPY:   OLED困倦表情 + 机械臂低垂 + TTS慵懒语调

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
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """情感类型枚举

    定义系统支持的所有情感表达类型。
    """
    NEUTRAL = "neutral"       # 中性/平静
    HAPPY = "happy"           # 高兴/开心
    SAD = "sad"               # 悲伤/难过
    CARING = "caring"         # 关心/温柔
    SURPRISED = "surprised"   # 惊讶/意外
    ANGRY = "angry"           # 生气/愤怒
    SLEEPY = "sleepy"         # 困倦/疲惫
    EXCITED = "excited"       # 兴奋/激动
    WORRIED = "worried"       # 担心/忧虑
    GRATEFUL = "grateful"     # 感激/感谢


class OutputChannel(Enum):
    """输出通道枚举"""
    OLED = "oled"             # OLED显示屏
    ARM = "arm"               # 机械臂姿态
    TTS = "tts"               # TTS语调
    LED = "led"               # RGB LED颜色
    ALL = "all"               # 所有通道


@dataclass
class EmotionConfig:
    """情感表达配置

    属性:
        oled_i2c_bus: OLED I2C总线号
        oled_address: OLED I2C地址
        oled_width: OLED宽度(像素)
        oled_height: OLED高度(像素)
        transition_duration: 情感过渡时间(秒)
        enable_oled: 是否启用OLED
        enable_arm: 是否启用机械臂姿态映射
        enable_tts_prosody: 是否启用TTS语调控制
        enable_led: 是否启用LED
    """
    oled_i2c_bus: int = 1
    oled_address: int = 0x3C
    oled_width: int = 128
    oled_height: int = 64
    transition_duration: float = 0.5
    enable_oled: bool = True
    enable_arm: bool = True
    enable_tts_prosody: bool = True
    enable_led: bool = True


@dataclass
class EmotionExpression:
    """情感表达定义

    定义一种情感在所有输出通道上的具体表现。

    属性:
        emotion: 情感类型
        oled_face: OLED表情图案(点阵数据)
        arm_angles: 机械臂关节角度
        tts_speed: TTS语速(0.5-2.0)
        tts_pitch: TTS音调偏移(-10 to +10)
        tts_volume: TTS音量(0.0-1.0)
        led_color: RGB LED颜色
        led_brightness: LED亮度
    """
    emotion: EmotionType = EmotionType.NEUTRAL
    oled_face: list[list[int]] = field(default_factory=list)
    arm_angles: dict[str, float] = field(default_factory=dict)
    tts_speed: float = 1.0
    tts_pitch: float = 0.0
    tts_volume: float = 0.8
    led_color: tuple[int, int, int] = (255, 255, 255)
    led_brightness: float = 0.5


class EmotionOutput:
    """情感表达输出类

    管理多通道情感输出,将抽象的情感类型映射为具体的硬件动作。
    支持OLED表情、机械臂姿态、TTS语调和LED颜色的同步输出。

    示例:
        >>> emotion = EmotionOutput(EmotionConfig())
        >>> await emotion.initialize()
        >>> 
        >>> # 表达高兴
        >>> await emotion.express(EmotionType.HAPPY, intensity=0.8)
        >>> 
        >>> # 表达关心
        >>> await emotion.express(EmotionType.CARING, intensity=0.6)
        >>> 
        >>> # 恢复中性
               >>> await emotion.express(EmotionType.NEUTRAL)
        >>> 
        >>> await emotion.shutdown()

    属性:
        config: 情感表达配置
        _current_emotion: 当前情感状态
    """

    # SSD1306 OLED命令常量
    SSD1306_COMMAND: int = 0x00
    SSD1306_DATA: int = 0x40
    SSD1306_SETCONTRAST: int = 0x81
    SSD1306_DISPLAYALLON_RESUME: int = 0xA4
    SSD1306_NORMALDISPLAY: int = 0xA6
    SSD1306_DISPLAYON: int = 0xAF
    SSD1306_SETDISPLAYOFFSET: int = 0xD3
    SSD1306_SETCOMPINS: int = 0xDA
    SSD1306_SETVCOMDETECT: int = 0xDB
    SSD1306_SETDISPLAYCLOCKDIV: int = 0xD5
    SSD1306_SETPRECHARGE: int = 0xD9
    SSD1306_SETMULTIPLEX: int = 0xA8
    SSD1306_SETSTARTLINE: int = 0x40
    SSD1306_MEMORYMODE: int = 0x20
    SSD1306_COLUMNADDR: int = 0x21
    SSD1306_PAGEADDR: int = 0x22
    SSD1306_COMSCANDEC: int = 0xC8
    SSD1306_SEGREMAP: int = 0xA1
    SSD1306_CHARGEPUMP: int = 0x8D

    def __init__(self, config: EmotionConfig | None = None) -> None:
        """初始化情感表达输出

        参数:
            config: 配置参数,None则使用默认配置
        """
        self.config: EmotionConfig = config or EmotionConfig()

        # 状态
        self._initialized: bool = False
        self._current_emotion: EmotionType = EmotionType.NEUTRAL
        self._current_intensity: float = 0.5
        self._lock: asyncio.Lock = asyncio.Lock()

        # OLED对象(延迟初始化)
        self._oled_bus: Any = None
        self._framebuffer: list[int] = [0] * (
            self.config.oled_width * self.config.oled_height // 8
        )

        # 情感定义
        self._expressions: dict[EmotionType, EmotionExpression] = {}
        self._build_expression_library()

        # 回调
        self._emotion_callbacks: list[Callable[[EmotionType, float], None]] = []

        # 过渡动画任务
        self._transition_task: asyncio.Task | None = None

    async def initialize(self) -> bool:
        """初始化情感输出硬件

        初始化OLED显示屏和其他输出设备。

        返回:
            bool: 初始化成功返回True
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # 初始化OLED
                if self.config.enable_oled:
                    try:
                        from smbus2 import SMBus
                        self._oled_bus = SMBus(self.config.oled_i2c_bus)
                        self._init_oled()
                        self._clear_oled()
                        logger.debug("OLED初始化成功")
                    except ImportError:
                        logger.warning("smbus2未安装,OLED不可用")
                        self._oled_bus = None
                    except Exception as e:
                        logger.warning(f"OLED初始化失败: {e}")
                        self._oled_bus = None

                # 显示初始表情(中性)
                await self._display_face(EmotionType.NEUTRAL)

                self._initialized = True
                logger.info("情感表达输出初始化成功")
                return True

            except Exception as e:
                logger.error(f"情感表达初始化失败: {e}")
                return False

    def _init_oled(self) -> None:
        """初始化SSD1306 OLED(内部方法)"""
        if self._oled_bus is None:
            return

        addr = self.config.oled_address
        init_sequence = [
            (self.SSD1306_COMMAND, self.SSD1306_DISPLAYOFF),
            (self.SSD1306_COMMAND, self.SSD1306_SETDISPLAYCLOCKDIV, 0x80),
            (self.SSD1306_COMMAND, self.SSD1306_SETMULTIPLEX, 0x3F),
            (self.SSD1306_COMMAND, self.SSD1306_SETDISPLAYOFFSET, 0x00),
            (self.SSD1306_COMMAND, self.SSD1306_SETSTARTLINE | 0x00),
            (self.SSD1306_COMMAND, self.SSD1306_CHARGEPUMP, 0x14),
            (self.SSD1306_COMMAND, self.SSD1306_MEMORYMODE, 0x00),
            (self.SSD1306_COMMAND, self.SSD1306_SEGREMAP | 0x01),
            (self.SSD1306_COMMAND, self.SSD1306_COMSCANDEC),
            (self.SSD1306_COMMAND, self.SSD1306_SETCOMPINS, 0x12),
            (self.SSD1306_COMMAND, self.SSD1306_SETCONTRAST, 0xCF),
            (self.SSD1306_COMMAND, self.SSD1306_SETPRECHARGE, 0xF1),
            (self.SSD1306_COMMAND, self.SSD1306_SETVCOMDETECT, 0x40),
            (self.SSD1306_COMMAND, self.SSD1306_DISPLAYALLON_RESUME),
            (self.SSD1306_COMMAND, self.SSD1306_NORMALDISPLAY),
            (self.SSD1306_COMMAND, self.SSD1306_DISPLAYON),
        ]

        for cmd in init_sequence:
            if len(cmd) == 2:
                self._oled_bus.write_byte_data(addr, cmd[0], cmd[1])
            else:
                self._oled_bus.write_byte_data(addr, cmd[0], cmd[1])

    def _clear_oled(self) -> None:
        """清屏(内部方法)"""
        if self._oled_bus is None:
            return

        addr = self.config.oled_address
        w, h = self.config.oled_width, self.config.oled_height
        pages = h // 8

        # 设置寻址范围
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, self.SSD1306_COLUMNADDR)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, 0)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, w - 1)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, self.SSD1306_PAGEADDR)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, 0)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, pages - 1)

        # 写入空数据
        for page in range(pages):
            for col in range(0, w, 16):
                data = [self.SSD1306_DATA] + [0x00] * min(16, w - col)
                self._oled_bus.write_i2c_block_data(addr, data[0], data[1:])

    def _build_expression_library(self) -> None:
        """构建情感表达库(内部方法)

        定义每种情感在各输出通道上的表现。
        """
        # 中性表情
        self._expressions[EmotionType.NEUTRAL] = EmotionExpression(
            emotion=EmotionType.NEUTRAL,
            arm_angles={"j1": 90, "j2": 90, "j3": 90, "j4": 90, "j5": 90},
            tts_speed=1.0,
            tts_pitch=0.0,
            tts_volume=0.8,
            led_color=(255, 255, 255),
            led_brightness=0.5,
        )

        # 高兴表情
        self._expressions[EmotionType.HAPPY] = EmotionExpression(
            emotion=EmotionType.HAPPY,
            arm_angles={"j1": 90, "j2": 60, "j3": 120, "j4": 60, "j5": 90},
            tts_speed=1.2,
            tts_pitch=2.0,
            tts_volume=0.9,
            led_color=(255, 255, 0),
            led_brightness=0.8,
        )

        # 悲伤表情
        self._expressions[EmotionType.SAD] = EmotionExpression(
            emotion=EmotionType.SAD,
            arm_angles={"j1": 90, "j2": 120, "j3": 60, "j4": 120, "j5": 90},
            tts_speed=0.7,
            tts_pitch=-2.0,
            tts_volume=0.5,
            led_color=(100, 100, 255),
            led_brightness=0.3,
        )

        # 关心表情
        self._expressions[EmotionType.CARING] = EmotionExpression(
            emotion=EmotionType.CARING,
            arm_angles={"j1": 90, "j2": 75, "j3": 105, "j4": 75, "j5": 90},
            tts_speed=0.85,
            tts_pitch=-1.0,
            tts_volume=0.6,
            led_color=(255, 200, 200),
            led_brightness=0.6,
        )

        # 惊讶表情
        self._expressions[EmotionType.SURPRISED] = EmotionExpression(
            emotion=EmotionType.SURPRISED,
            arm_angles={"j1": 90, "j2": 30, "j3": 150, "j4": 30, "j5": 90},
            tts_speed=1.3,
            tts_pitch=4.0,
            tts_volume=1.0,
            led_color=(255, 165, 0),
            led_brightness=1.0,
        )

        # 生气表情
        self._expressions[EmotionType.ANGRY] = EmotionExpression(
            emotion=EmotionType.ANGRY,
            arm_angles={"j1": 90, "j2": 100, "j3": 80, "j4": 100, "j5": 45},
            tts_speed=1.4,
            tts_pitch=1.0,
            tts_volume=1.0,
            led_color=(255, 0, 0),
            led_brightness=1.0,
        )

        # 困倦表情
        self._expressions[EmotionType.SLEEPY] = EmotionExpression(
            emotion=EmotionType.SLEEPY,
            arm_angles={"j1": 90, "j2": 135, "j3": 45, "j4": 135, "j5": 90},
            tts_speed=0.6,
            tts_pitch=-3.0,
            tts_volume=0.4,
            led_color=(128, 0, 128),
            led_brightness=0.2,
        )

        # 兴奋表情
        self._expressions[EmotionType.EXCITED] = EmotionExpression(
            emotion=EmotionType.EXCITED,
            arm_angles={"j1": 90, "j2": 45, "j3": 135, "j4": 45, "j5": 135},
            tts_speed=1.5,
            tts_pitch=3.0,
            tts_volume=1.0,
            led_color=(0, 255, 255),
            led_brightness=1.0,
        )

        # 担心表情
        self._expressions[EmotionType.WORRIED] = EmotionExpression(
            emotion=EmotionType.WORRIED,
            arm_angles={"j1": 90, "j2": 80, "j3": 100, "j4": 80, "j5": 90},
            tts_speed=0.8,
            tts_pitch=-1.5,
            tts_volume=0.55,
            led_color=(255, 180, 100),
            led_brightness=0.4,
        )

        # 感激表情
        self._expressions[EmotionType.GRATEFUL] = EmotionExpression(
            emotion=EmotionType.GRATEFUL,
            arm_angles={"j1": 90, "j2": 70, "j3": 110, "j4": 70, "j5": 90},
            tts_speed=0.9,
            tts_pitch=1.0,
            tts_volume=0.7,
            led_color=(255, 215, 0),
            led_brightness=0.7,
        )

    async def express(
        self,
        emotion: EmotionType,
        intensity: float = 0.5,
        duration: float | None = None,
    ) -> bool:
        """表达指定情感

        在所有启用的输出通道上同步输出指定情感。

        参数:
            emotion: 情感类型
            intensity: 情感强度(0.0-1.0)
            duration: 表达持续时间(秒),None则持续到下次切换

        返回:
            bool: 表达成功返回True

        示例:
            >>> await emotion.express(EmotionType.HAPPY, intensity=0.8)
            >>> await emotion.express(EmotionType.CARING, intensity=0.6, duration=3.0)
        """
        if not self._initialized:
            logger.error("情感输出未初始化")
            return False

        intensity = max(0.0, min(1.0, intensity))

        async with self._lock:
            try:
                # 取消正在进行的过渡动画
                if self._transition_task and not self._transition_task.done():
                    self._transition_task.cancel()

                expression = self._expressions.get(emotion)
                if expression is None:
                    logger.warning(f"未定义的情感: {emotion.value}")
                    return False

                # OLED输出
                if self.config.enable_oled:
                    await self._display_face(emotion)

                # 机械臂姿态
                if self.config.enable_arm:
                    await self._set_arm_pose(expression.arm_angles, intensity)

                # LED颜色
                if self.config.enable_led:
                    await self._set_led(
                        expression.led_color,
                        expression.led_brightness * intensity,
                    )

                self._current_emotion = emotion
                self._current_intensity = intensity

                # 通知回调
                for cb in self._emotion_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            asyncio.create_task(cb(emotion, intensity))
                        else:
                            cb(emotion, intensity)
                    except Exception as e:
                        logger.error(f"情感回调异常: {e}")

                logger.debug(f"情感表达: {emotion.value}, 强度={intensity:.1f}")
                return True

            except Exception as e:
                logger.error(f"情感表达异常: {e}")
                return False

    async def _display_face(self, emotion: EmotionType) -> None:
        """在OLED上显示表情(内部方法)

        参数:
            emotion: 情感类型
        """
        face_data = self._generate_face_bitmap(emotion)

        if self._oled_bus is not None:
            self._draw_bitmap(face_data)
        else:
            # 模拟模式:记录日志
            logger.debug(f"OLED显示表情: {emotion.value}")

    def _generate_face_bitmap(
        self, emotion: EmotionType
    ) -> list[list[int]]:
        """生成表情位图(内部方法)

        为指定情感生成OLED点阵图案。

        参数:
            emotion: 情感类型

        返回:
            list: 128x64点阵数据
        """
        w, h = self.config.oled_width, self.config.oled_height
        bitmap = [[0] * w for _ in range(h)]

        # 绘制脸部轮廓(圆形)
        cx, cy = w // 2, h // 2
        radius = min(cx, cy) - 2

        for y in range(h):
            for x in range(w):
                dx, dy = x - cx, y - cy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < radius:
                    bitmap[y][x] = 0
                elif dist < radius + 2:
                    bitmap[y][x] = 1

        # 根据情感绘制五官
        if emotion == EmotionType.HAPPY:
            self._draw_happy_face(bitmap, cx, cy)
        elif emotion == EmotionType.SAD:
            self._draw_sad_face(bitmap, cx, cy)
        elif emotion == EmotionType.SURPRISED:
            self._draw_surprised_face(bitmap, cx, cy)
        elif emotion == EmotionType.ANGRY:
            self._draw_angry_face(bitmap, cx, cy)
        elif emotion == EmotionType.CARING:
            self._draw_caring_face(bitmap, cx, cy)
        elif emotion == EmotionType.SLEEPY:
            self._draw_sleepy_face(bitmap, cx, cy)
        else:
            self._draw_neutral_face(bitmap, cx, cy)

        return bitmap

    def _draw_happy_face(
        self, bitmap: list[list[int]], cx: int, cy: int
    ) -> None:
        """绘制高兴表情(内部方法)"""
        # 眼睛(弯月形)
        for x in range(cx - 20, cx - 8):
            bitmap[cy - 8][x] = 1
        for x in range(cx + 8, cx + 20):
            bitmap[cy - 8][x] = 1
        # 嘴巴(大笑)
        for angle in range(20, 161):
            rad = math.radians(angle)
            mx = cx + int(15 * math.cos(rad))
            my = cy + int(10 * math.sin(rad))
            if 0 <= my < len(bitmap) and 0 <= mx < len(bitmap[0]):
                bitmap[my][mx] = 1

    def _draw_sad_face(
        self, bitmap: list[list[int]], cx: int, cy: int
    ) -> None:
        """绘制悲伤表情(内部方法)"""
        # 眼睛(倒弯月)
        for x in range(cx - 20, cx - 8):
            bitmap[cy - 6][x] = 1
        for x in range(cx + 8, cx + 20):
            bitmap[cy - 6][x] = 1
        # 眼泪
        bitmap[cy - 2][cx - 14] = 1
        bitmap[cy][cx - 14] = 1
        bitmap[cy - 2][cx + 14] = 1
        bitmap[cy][cx + 14] = 1
        # 嘴巴(下弯)
        for angle in range(200, 341):
            rad = math.radians(angle)
            mx = cx + int(12 * math.cos(rad))
            my = cy + int(6 * math.sin(rad))
            if 0 <= my < len(bitmap) and 0 <= mx < len(bitmap[0]):
                bitmap[my][mx] = 1

    def _draw_surprised_face(
        self, bitmap: list[list[int]], cx: int, cy: int
    ) -> None:
        """绘制惊讶表情(内部方法)"""
        # 大眼睛(圆形)
        for angle in range(360):
            rad = math.radians(angle)
            for eye_x, eye_y in [(cx - 14, cy - 8), (cx + 14, cy - 8)]:
                ex = eye_x + int(4 * math.cos(rad))
                ey = eye_y + int(5 * math.sin(rad))
                if 0 <= ey < len(bitmap) and 0 <= ex < len(bitmap[0]):
                    bitmap[ey][ex] = 1
        # 嘴巴(O形)
        for angle in range(360):
            rad = math.radians(angle)
            mx = cx + int(6 * math.cos(rad))
            my = cy + int(8 * math.sin(rad)) + 8
            if 0 <= my < len(bitmap) and 0 <= mx < len(bitmap[0]):
                bitmap[my][mx] = 1

    def _draw_angry_face(
        self, bitmap: list[list[int]], cx: int, cy: int
    ) -> None:
        """绘制生气表情(内部方法)"""
        # 眉毛(倒八字)
        for i in range(10):
            bitmap[cy - 14][cx - 18 + i] = 1
            bitmap[cy - 12 - i // 3][cx - 18 + i] = 1
            bitmap[cy - 14][cx + 8 + i] = 1
            bitmap[cy - 12 - i // 3][cx + 8 + i] = 1
        # 眼睛(横线)
        for x in range(cx - 16, cx - 6):
            bitmap[cy - 6][x] = 1
        for x in range(cx + 6, cx + 16):
            bitmap[cy - 6][x] = 1
        # 嘴巴(一字)
        for x in range(cx - 10, cx + 10):
            bitmap[cy + 10][x] = 1

    def _draw_caring_face(
        self, bitmap: list[list[int]], cx: int, cy: int
    ) -> None:
        """绘制关心表情(内部方法)"""
        # 温柔的眼睛
        for x in range(cx - 18, cx - 6):
            bitmap[cy - 8][x] = 1
        for x in range(cx + 6, cx + 18):
            bitmap[cy - 8][x] = 1
        # 微笑
        for angle in range(30, 151):
            rad = math.radians(angle)
            mx = cx + int(12 * math.cos(rad))
            my = cy + int(7 * math.sin(rad)) + 2
            if 0 <= my < len(bitmap) and 0 <= mx < len(bitmap[0]):
                bitmap[my][mx] = 1

    def _draw_sleepy_face(
        self, bitmap: list[list[int]], cx: int, cy: int
    ) -> None:
        """绘制困倦表情(内部方法)"""
        # 闭着的眼睛(横线)
        for x in range(cx - 16, cx - 6):
            bitmap[cy - 8][x] = 1
        for x in range(cx + 6, cx + 16):
            bitmap[cy - 8][x] = 1
        # 小嘴
        for x in range(cx - 4, cx + 4):
            bitmap[cy + 8][x] = 1
        # Zzz
        bitmap[cy - 16][cx + 18] = 1
        bitmap[cy - 14][cx + 20] = 1
        bitmap[cy - 12][cx + 22] = 1

    def _draw_neutral_face(
        self, bitmap: list[list[int]], cx: int, cy: int
    ) -> None:
        """绘制中性表情(内部方法)"""
        # 圆形眼睛
        for eye_x in [cx - 14, cx + 14]:
            for angle in range(360):
                rad = math.radians(angle)
                ex = eye_x + int(3 * math.cos(rad))
                ey = cy - 8 + int(4 * math.sin(rad))
                if 0 <= ey < len(bitmap) and 0 <= ex < len(bitmap[0]):
                    bitmap[ey][ex] = 1
        # 直线嘴巴
        for x in range(cx - 8, cx + 8):
            bitmap[cy + 8][x] = 1

    def _draw_bitmap(self, bitmap: list[list[int]]) -> None:
        """绘制位图到OLED(内部方法)

        将二维点阵数据发送到SSD1306 OLED。

        参数:
            bitmap: 二维点阵数据
        """
        if self._oled_bus is None:
            return

        addr = self.config.oled_address
        w, h = self.config.oled_width, self.config.oled_height
        pages = h // 8

        # 设置寻址范围
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, self.SSD1306_COLUMNADDR)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, 0)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, w - 1)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, self.SSD1306_PAGEADDR)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, 0)
        self._oled_bus.write_byte_data(addr, self.SSD1306_COMMAND, pages - 1)

        # 按页发送数据
        for page in range(pages):
            for col in range(0, w, 16):
                chunk = [self.SSD1306_DATA]
                for c in range(col, min(col + 16, w)):
                    byte_val = 0
                    for row in range(8):
                        y = page * 8 + row
                        if y < h and bitmap[y][c]:
                            byte_val |= (1 << row)
                    chunk.append(byte_val)
                self._oled_bus.write_i2c_block_data(addr, chunk[0], chunk[1:])

    async def _set_arm_pose(
        self, angles: dict[str, float], intensity: float
    ) -> None:
        """设置机械臂姿态(内部方法)

        参数:
            angles: 目标关节角度
            intensity: 情感强度
        """
        # 这里将与DofbotArm交互
        # 目前只记录目标姿态,由上层调用者执行
        logger.debug(f"机械臂姿态映射: {angles}, 强度={intensity}")

    async def _set_led(
        self, color: tuple[int, int, int], brightness: float
    ) -> None:
        """设置LED颜色(内部方法)

        参数:
            color: RGB颜色
            brightness: 亮度
        """
        r = int(color[0] * brightness)
        g = int(color[1] * brightness)
        b = int(color[2] * brightness)
        logger.debug(f"LED颜色: RGB({r},{g},{b})")

    def get_expression(self, emotion: EmotionType) -> EmotionExpression | None:
        """获取指定情感的表达定义

        参数:
            emotion: 情感类型

        返回:
            EmotionExpression或None
        """
        return self._expressions.get(emotion)

    def get_tts_prosody(self, emotion: EmotionType) -> dict[str, float]:
        """获取TTS语调参数

        返回指定情感对应的TTS语速、音调和音量参数。

        参数:
            emotion: 情感类型

        返回:
            dict: {"speed": float, "pitch": float, "volume": float}
        """
        expr = self._expressions.get(emotion)
        if expr:
            return {
                "speed": expr.tts_speed,
                "pitch": expr.tts_pitch,
                "volume": expr.tts_volume,
            }
        return {"speed": 1.0, "pitch": 0.0, "volume": 0.8}

    def get_current_emotion(self) -> EmotionType:
        """获取当前情感状态

        返回:
            EmotionType: 当前情感
        """
        return self._current_emotion

    def register_callback(
        self, callback: Callable[[EmotionType, float], None]
    ) -> None:
        """注册情感变更回调

        参数:
            callback: 回调函数,接收(emotion, intensity)
        """
        if callback not in self._emotion_callbacks:
            self._emotion_callbacks.append(callback)

    async def transition(
        self,
        from_emotion: EmotionType,
        to_emotion: EmotionType,
        duration: float = 1.0,
    ) -> bool:
        """情感过渡动画

        从一种情感平滑过渡到另一种情感。

        参数:
            from_emotion: 起始情感
            to_emotion: 目标情感
            duration: 过渡时间(秒)

        返回:
            bool: 过渡完成返回True
        """
        steps = int(duration / 0.05)  # 20Hz更新

        for i in range(1, steps + 1):
            t = i / steps
            # 使用余弦缓动
            t_smooth = 0.5 * (1 - math.cos(t * math.pi))

            # 这里可以实现更复杂的插值
            if t_smooth > 0.5:
                await self.express(to_emotion, intensity=t_smooth)
            else:
                await self.express(from_emotion, intensity=1.0 - t_smooth)

            await asyncio.sleep(0.05)

        await self.express(to_emotion, intensity=1.0)
        return True

    async def shutdown(self) -> None:
        """关闭情感输出模块"""
        async with self._lock:
            try:
                # 清屏
                if self._oled_bus:
                    self._clear_oled()
                    self._oled_bus = None

                self._initialized = False
                logger.info("情感表达输出已关闭")

            except Exception as e:
                logger.error(f"关闭情感输出异常: {e}")

    def __repr__(self) -> str:
        return (
            f"EmotionOutput(emotion={self._current_emotion.value}, "
            f"intensity={self._current_intensity:.1f})"
        )

    async def __aenter__(self) -> EmotionOutput:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
