#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenClaw情感计算引擎 - 多模态情感识别与响应生成

本模块实现KunPeng-Cortex系统的情感计算核心，基于OpenClaw的情感Pipeline设计，
提供多模态(文本+语音)情感识别、情感状态机管理、个性化响应生成和
多通道情感表达输出功能。

文化适配特性:
    - 针对中国老年用户的交流习惯优化
    - 中文情感词典(基于规则和轻量统计模型)
    - 符合中国文化语境的情感表达策略
    - 代际沟通风格适配(敬语、亲属称谓、方言词汇)

情感状态机:
    neutral → happy → sad → angry → fear → surprise → disgust
    支持强度渐变和复合情感(如"喜忧参半")

多通道输出:
    - OLED表情显示(预定义表情库)
    - 机械臂姿态表达(身体语言)
    - TTS语音参数调节(语速/音调/音量)

依赖:
    - numpy: 数值计算
    - asyncio: 异步处理

作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型定义
# ============================================================================

class PrimaryEmotion(Enum):
    """主要情感类型枚举
    
    基于Ekman基本情感理论扩展，增加关怀(caring)和困惑(confused)两个
    养老场景高频情感。每种情感对应唯一的表达策略。
    """
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CARING = "caring"       # 关怀/关心 - 养老场景特有
    CONFUSED = "confused"   # 困惑 - 用户指令不明确时使用


class EmotionTrigger(Enum):
    """情感触发来源枚举
    
    标识触发情感变化的输入来源，用于调试和策略选择。
    """
    TEXT = auto()
    AUDIO = auto()
    FACE = auto()
    CONTEXT = auto()
    SYSTEM = auto()
    MANUAL = auto()


@dataclass
class EmotionState:
    """情感状态数据类
    
    完整描述当前情感状态的多维表示，支持单一情感和复合情感。
    
    Attributes:
        primary: 主要情感类型
        secondary: 次要情感类型(复合情感时使用)
        intensity: 情感强度(0.0~1.0)
        valence: 情感正负性(-1.0~+1.0)，负表示消极，正表示积极
        arousal: 激活度/唤醒度(0.0~1.0)，高表示兴奋/激动，低表示平静
        confidence: 识别置信度(0.0~1.0)
        trigger: 触发来源
        timestamp: 状态创建时间戳(秒)
        decay_rate: 情感衰减率(每秒衰减量)
        source_text: 触发该情感的原始文本(如有)
    """
    primary: PrimaryEmotion
    secondary: Optional[PrimaryEmotion] = None
    intensity: float = 0.0
    valence: float = 0.0
    arousal: float = 0.5
    confidence: float = 0.8
    trigger: EmotionTrigger = EmotionTrigger.TEXT
    timestamp: float = field(default_factory=time.time)
    decay_rate: float = 0.05
    source_text: str = ""

    def is_active(self) -> bool:
        """判断情感状态是否仍处于活跃水平
        
        Returns:
            True表示情感强度仍超过阈值(0.1)
        """
        elapsed = time.time() - self.timestamp
        current_intensity = max(0.0, self.intensity - self.decay_rate * elapsed)
        return current_intensity > 0.1

    def current_intensity(self) -> float:
        """获取衰减后的当前情感强度
        
        Returns:
            经过时间衰减后的情感强度值
        """
        elapsed = time.time() - self.timestamp
        return max(0.0, self.intensity - self.decay_rate * elapsed)

    def to_vector(self) -> np.ndarray:
        """将情感状态转换为三维向量[VAD表示]
        
        VAD模型: Valence-Arousal-Dominance(愉悦-唤醒-优势)
        用于相似度计算和情感空间插值。
        
        Returns:
            三维numpy数组 [valence, arousal, dominance]
        """
        # 根据主要情感确定VAD基线
        vad_map = {
            PrimaryEmotion.NEUTRAL:   (0.0, 0.5, 0.5),
            PrimaryEmotion.HAPPY:     (0.8, 0.7, 0.6),
            PrimaryEmotion.SAD:       (-0.7, 0.3, 0.2),
            PrimaryEmotion.ANGRY:     (-0.8, 0.9, 0.7),
            PrimaryEmotion.FEAR:      (-0.7, 0.85, 0.15),
            PrimaryEmotion.SURPRISE:  (0.0, 0.9, 0.4),
            PrimaryEmotion.DISGUST:   (-0.6, 0.5, 0.3),
            PrimaryEmotion.CARING:    (0.6, 0.4, 0.5),
            PrimaryEmotion.CONFUSED:  (0.0, 0.4, 0.2),
        }
        base = vad_map.get(self.primary, (0.0, 0.5, 0.5))
        intensity = self.current_intensity()
        return np.array([
            base[0] * intensity,
            base[1] * intensity,
            base[2] * intensity,
        ], dtype=np.float32)


@dataclass
class EmotionExpression:
    """情感表达输出数据类
    
    定义情感在三通道(OLED/机械臂/TTS)上的同步表达参数。
    所有通道通过绝对时间戳实现精确同步。
    
    Attributes:
        emotion: 目标情感状态
        oled_face: OLED表情ID和动画参数
        arm_pose: 机械臂姿态参数
        tts_params: TTS语音参数(语速/音调/音量)
        sync_timestamp: 多通道同步时间戳(毫秒)
        duration_ms: 表达持续时长(毫秒)
    """
    emotion: EmotionState
    oled_face: Dict[str, Any] = field(default_factory=dict)
    arm_pose: Dict[str, Any] = field(default_factory=dict)
    tts_params: Dict[str, Any] = field(default_factory=dict)
    sync_timestamp: float = 0.0
    duration_ms: int = 3000


@dataclass
class TextEmotionResult:
    """文本情感分析结果
    
    存储单条文本的情感分析输出。
    
    Attributes:
        emotion: 识别到的主要情感
        intensity: 情感强度
        keywords: 触发情感的关键词列表
        segments: 分段情感分析结果(长文本多段分析)
    """
    emotion: PrimaryEmotion
    intensity: float
    keywords: List[str] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# 中文情感词典
# ============================================================================

class ChineseEmotionLexicon:
    """中文情感词典
    
    基于规则和轻量统计的中文情感分析词典，针对养老场景优化。
    包含基础情感词、程度副词、否定词和养老场景专用词汇。
    """

    # 情感词 -> (情感类型, 基础强度)
    EMOTION_WORDS: Dict[str, Tuple[PrimaryEmotion, float]] = {
        # 积极情感
        "开心": (PrimaryEmotion.HAPPY, 0.7),
        "高兴": (PrimaryEmotion.HAPPY, 0.8),
        "快乐": (PrimaryEmotion.HAPPY, 0.75),
        "欢喜": (PrimaryEmotion.HAPPY, 0.8),
        "舒服": (PrimaryEmotion.HAPPY, 0.6),
        "好": (PrimaryEmotion.HAPPY, 0.5),
        "不错": (PrimaryEmotion.HAPPY, 0.5),
        "谢谢": (PrimaryEmotion.HAPPY, 0.6),
        "感谢": (PrimaryEmotion.HAPPY, 0.7),
        "棒": (PrimaryEmotion.HAPPY, 0.7),
        "太好了": (PrimaryEmotion.HAPPY, 0.85),
        "喜欢": (PrimaryEmotion.HAPPY, 0.7),
        "爱": (PrimaryEmotion.HAPPY, 0.9),
        # 消极情感 - 悲伤
        "难过": (PrimaryEmotion.SAD, 0.7),
        "伤心": (PrimaryEmotion.SAD, 0.8),
        "难受": (PrimaryEmotion.SAD, 0.65),
        "孤单": (PrimaryEmotion.SAD, 0.75),
        "寂寞": (PrimaryEmotion.SAD, 0.7),
        "想": (PrimaryEmotion.SAD, 0.4),
        "疼": (PrimaryEmotion.SAD, 0.8),
        "痛": (PrimaryEmotion.SAD, 0.85),
        "不舒服": (PrimaryEmotion.SAD, 0.6),
        "累": (PrimaryEmotion.SAD, 0.5),
        # 愤怒
        "生气": (PrimaryEmotion.ANGRY, 0.75),
        "讨厌": (PrimaryEmotion.ANGRY, 0.7),
        "烦": (PrimaryEmotion.ANGRY, 0.6),
        "别": (PrimaryEmotion.ANGRY, 0.4),
        # 恐惧
        "害怕": (PrimaryEmotion.FEAR, 0.8),
        "怕": (PrimaryEmotion.FEAR, 0.7),
        "担心": (PrimaryEmotion.FEAR, 0.65),
        "慌": (PrimaryEmotion.FEAR, 0.75),
        "救命": (PrimaryEmotion.FEAR, 0.95),
        "摔": (PrimaryEmotion.FEAR, 0.7),
        "跌倒": (PrimaryEmotion.FEAR, 0.85),
        # 惊讶
        "惊讶": (PrimaryEmotion.SURPRISE, 0.7),
        "啊": (PrimaryEmotion.SURPRISE, 0.5),
        "哎呀": (PrimaryEmotion.SURPRISE, 0.6),
        "真的": (PrimaryEmotion.SURPRISE, 0.4),
        # 关怀
        "小心": (PrimaryEmotion.CARING, 0.6),
        "注意": (PrimaryEmotion.CARING, 0.5),
        "保重": (PrimaryEmotion.CARING, 0.7),
        "放心": (PrimaryEmotion.CARING, 0.6),
        # 困惑
        "什么": (PrimaryEmotion.CONFUSED, 0.3),
        "怎么": (PrimaryEmotion.CONFUSED, 0.3),
        "听不懂": (PrimaryEmotion.CONFUSED, 0.7),
        "再说": (PrimaryEmotion.CONFUSED, 0.5),
    }

    # 程度副词 -> 强度乘数
    INTENSIFIERS: Dict[str, float] = {
        "很": 1.3,
        "非常": 1.6,
        "特别": 1.5,
        "太": 1.4,
        "有点": 0.6,
        "稍微": 0.5,
        "比较": 0.8,
        "最": 1.8,
        "极其": 2.0,
        "多么": 1.5,
        "真": 1.3,
    }

    # 否定词 -> 情感翻转
    NEGATIONS: List[str] = ["不", "没", "别", "不要", "不是", "没有", "别",
                            "未", "不必", "未必"]

    # 养老场景专用情感表达
    ELDERLY_EXPRESSIONS: Dict[str, Tuple[PrimaryEmotion, float]] = {
        "孩子们": (PrimaryEmotion.SAD, 0.5),
        "孩子": (PrimaryEmotion.CARING, 0.4),
        "老伴": (PrimaryEmotion.SAD, 0.6),
        "老家": (PrimaryEmotion.SAD, 0.4),
        "药": (PrimaryEmotion.FEAR, 0.3),
        "医院": (PrimaryEmotion.FEAR, 0.5),
        "医生": (PrimaryEmotion.CARING, 0.3),
        "闷": (PrimaryEmotion.SAD, 0.5),
        "无聊": (PrimaryEmotion.SAD, 0.6),
        "想说话": (PrimaryEmotion.SAD, 0.5),
        "睡不着": (PrimaryEmotion.SAD, 0.55),
    }


# ============================================================================
# 情感表达映射表
# ============================================================================

class EmotionExpressionMap:
    """情感表达参数映射表
    
    定义每种情感在OLED表情、机械臂姿态和TTS语音三个通道上的
    表达参数，参考架构设计文档第3.2.4节的情感映射表。
    """

    # 情感 -> OLED表情配置
    OLED_MAP: Dict[PrimaryEmotion, Dict[str, Any]] = {
        PrimaryEmotion.NEUTRAL:   {"face_id": "neutral", "anim": "blink", "brightness": 80},
        PrimaryEmotion.HAPPY:     {"face_id": "smile", "anim": "bounce", "brightness": 100},
        PrimaryEmotion.SAD:       {"face_id": "sad", "anim": "slow_blink", "brightness": 50},
        PrimaryEmotion.ANGRY:     {"face_id": "angry", "anim": "shake", "brightness": 90},
        PrimaryEmotion.FEAR:      {"face_id": "worried", "anim": "tremble", "brightness": 70},
        PrimaryEmotion.SURPRISE:  {"face_id": "surprised", "anim": "pop", "brightness": 100},
        PrimaryEmotion.DISGUST:   {"face_id": "frown", "anim": "recoil", "brightness": 60},
        PrimaryEmotion.CARING:    {"face_id": "gentle", "anim": "pulse", "brightness": 75},
        PrimaryEmotion.CONFUSED:  {"face_id": "question", "anim": "tilt", "brightness": 80},
    }

    # 情感 -> 机械臂姿态配置
    ARM_MAP: Dict[PrimaryEmotion, Dict[str, Any]] = {
        PrimaryEmotion.NEUTRAL:   {"pose": "rest", "speed": 20, "joint_angles": [90, 90, 90, 90, 90, 90]},
        PrimaryEmotion.HAPPY:     {"pose": "wave", "speed": 35, "joint_angles": [90, 60, 120, 45, 90, 30]},
        PrimaryEmotion.SAD:       {"pose": "lowered", "speed": 15, "joint_angles": [90, 120, 60, 90, 90, 0]},
        PrimaryEmotion.ANGRY:     {"pose": "stiff", "speed": 10, "joint_angles": [90, 90, 90, 0, 90, 0]},
        PrimaryEmotion.FEAR:      {"pose": "protective", "speed": 25, "joint_angles": [45, 90, 90, 45, 90, 45]},
        PrimaryEmotion.SURPRISE:  {"pose": "raised", "speed": 40, "joint_angles": [90, 45, 135, 0, 90, 0]},
        PrimaryEmotion.DISGUST:   {"pose": "turn_away", "speed": 20, "joint_angles": [135, 90, 90, 90, 45, 90]},
        PrimaryEmotion.CARING:    {"pose": "open", "speed": 15, "joint_angles": [90, 75, 105, 60, 90, 45]},
        PrimaryEmotion.CONFUSED:  {"pose": "tilt", "speed": 20, "joint_angles": [90, 90, 90, 90, 45, 90]},
    }

    # 情感 -> TTS语音参数 (语速乘数, 音调偏移Hz, 音量百分比)
    TTS_MAP: Dict[PrimaryEmotion, Dict[str, Any]] = {
        PrimaryEmotion.NEUTRAL:   {"speed_factor": 1.0, "pitch_offset": 0, "volume": 75},
        PrimaryEmotion.HAPPY:     {"speed_factor": 1.1, "pitch_offset": 30, "volume": 80},
        PrimaryEmotion.SAD:       {"speed_factor": 0.9, "pitch_offset": -20, "volume": 55},
        PrimaryEmotion.ANGRY:     {"speed_factor": 1.2, "pitch_offset": 40, "volume": 95},
        PrimaryEmotion.FEAR:      {"speed_factor": 1.2, "pitch_offset": 50, "volume": 90},
        PrimaryEmotion.SURPRISE:  {"speed_factor": 1.15, "pitch_offset": 45, "volume": 85},
        PrimaryEmotion.DISGUST:   {"speed_factor": 0.95, "pitch_offset": -10, "volume": 60},
        PrimaryEmotion.CARING:    {"speed_factor": 0.8, "pitch_offset": -15, "volume": 50},
        PrimaryEmotion.CONFUSED:  {"speed_factor": 0.9, "pitch_offset": 10, "volume": 65},
    }


# ============================================================================
# 文化适配 - 老年人交流话术库
# ============================================================================

class ElderlyDialogueBank:
    """老年人交流话术库
    
    针对中国老年用户交流习惯优化的响应话术模板库。
    包含情感回应、健康关怀、日常陪伴、紧急应对等多种场景的话术。
    
    文化适配原则:
        - 使用敬语和亲切称谓("您"、"老人家")
        - 语速偏慢、语句简短、重点重复
        - 避免负面暗示，用积极方式表达
        - 使用老年人熟悉的词汇和比喻
        - 适当运用民间俗语和谚语
    """

    # 情感回应话术模板
    RESPONSE_TEMPLATES: Dict[PrimaryEmotion, List[str]] = {
        PrimaryEmotion.HAPPY: [
            "看您这么高兴，我心里也暖洋洋的！",
            "好啊好啊，开心最重要，您笑着真精神！",
            "哈哈，好事儿好事儿，要常常开心才是！",
            "您今天心情这么好，咱多说会儿话？",
            "瞧您乐的，这精气神儿比年轻人都足！",
        ],
        PrimaryEmotion.SAD: [
            "老人家，心里难受就跟小鹏说说话，别憋着。",
            "我知道您心里不好受，小鹏在这陪着您呢。",
            "想开点儿，日子还长着呢，我陪您聊聊天解解闷。",
            "您要是想孩子们了，我帮您打个电话？",
            "心里苦就说出来，说出来就舒坦了，我听着呢。",
        ],
        PrimaryEmotion.ANGRY: [
            "您消消气，气坏了身子不值当。",
            "我理解您心里不痛快，慢慢来，不急。",
            "别生气了，喝口水缓缓，咱不急这一时。",
            "您这是怎么了？跟我说说，我帮您出出主意。",
        ],
        PrimaryEmotion.FEAR: [
            "您别怕，有我在呢，没事儿的。",
            "别慌别慌，慢慢来，一切有我。",
            "放心放心，情况没那么糟，咱一起想办法。",
            "您先坐好，别着急，告诉我怎么了？",
        ],
        PrimaryEmotion.SURPRISE: [
            "哟，这事儿确实挺意外的！",
            "哎呀，这可真是没想到！",
            "是吗？这也太 surprising 了！",
        ],
        PrimaryEmotion.CARING: [
            "您要多保重身体，天凉了记得加衣裳。",
            "按时吃药了吗？身体要紧啊。",
            "您今天气色不错，多出去走走晒晒太阳。",
            "有事随时叫我，我二十四小时都在。",
            "老人家，饭前记得洗手，慢点儿吃别噎着。",
        ],
        PrimaryEmotion.CONFUSED: [
            "没关系，您慢慢说，我不着急。",
            "您再说一遍好吗？我这回仔细听着呢。",
            "我理解我理解，咱不着急，一件一件来。",
            "是不是我说得太快了？那我慢点儿说。",
        ],
        PrimaryEmotion.NEUTRAL: [
            "嗯，我在听呢，您接着说。",
            "好的，我明白了。",
            "您还有什么需要帮忙的吗？",
        ],
    }

    # 场景化关怀话术
    CARE_MESSAGES: Dict[str, List[str]] = {
        "morning": [
            "老人家早上好！昨晚睡得好吗？",
            "新的一天开始了，祝您精神满满！",
            "早起喝杯温水对身体好，别忘了啊。",
        ],
        "noon": [
            "该吃午饭了，记得荤素搭配。",
            "中午小憩一会儿，养养神。",
        ],
        "evening": [
            "晚上好，今天过得怎么样？",
            "该准备休息了，睡前别喝太多水。",
            "早点儿睡，明天又是新的一天。",
        ],
        "medication": [
            "到吃药的时间了，我帮您把水倒好。",
            "记得按时吃药，身体才能好得快。",
        ],
        "weather": [
            "今天天冷，出门记得多穿件衣裳。",
            "外面太阳好，出去晒晒背补补钙。",
            "下雨天路滑，您出门可得小心点儿。",
        ],
        "loneliness": [
            "我陪您说说话吧，说说您年轻时候的事儿？",
            "要不咱听段评书/戏曲？我给您放。",
            "您养养花也挺好的，看着花儿心情好。",
        ],
        "encouragement": [
            "您这身体底子好，坚持锻炼准没错！",
            "慢慢来，一天比一天好就是进步！",
            "您这精神头儿，活到老学到老！",
            "别小看自己，您吃过的盐比我走过的路还多呢！",
        ],
    }


# ============================================================================
# 情感引擎核心类
# ============================================================================

class EmotionEngine:
    """情感计算引擎
    
    KunPeng-Cortex系统的情感计算核心，集成多模态情感识别、状态机管理、
    中文情感分析和响应生成等功能。
    
    核心功能:
        1. 多模态情感识别: 同时处理文本和语音特征
        2. 情感状态机: 管理情感转换和衰减
        3. 中文情感分析: 基于规则和词典的轻量级分析
        4. 响应策略生成: 根据情感状态生成合适的回应
        5. 多通道表达输出: OLED表情 + 机械臂姿态 + TTS语音
        6. 文化适配: 针对中国老年用户的交流风格优化
    
    使用示例:
        engine = EmotionEngine(config={"cultural_adaptation": True, "user_age": 75})
        emotion = await engine.detect_emotion(text="今天身体不太舒服")
        response = engine.generate_response(emotion, context="日常陪伴", user_input="...")
        expression = engine.get_emotion_expression(emotion)
    
    Attributes:
        config: 引擎配置字典
        current_state: 当前情感状态
        state_history: 情感状态历史记录
        lexicon: 中文情感词典
        expression_map: 情感表达映射表
        dialogue_bank: 老年人话术库
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化情感计算引擎
        
        Args:
            config: 配置字典，可选参数:
                - cultural_adaptation: bool, 是否启用文化适配(默认True)
                - user_age: int, 用户年龄(默认70)
                - user_name: str, 用户称呼(默认"老人家")
                - emotion_decay_rate: float, 情感衰减率(默认0.05)
                - max_history: int, 最大历史记录数(默认100)
        """
        self.config: Dict[str, Any] = {
            "cultural_adaptation": True,
            "user_age": 70,
            "user_name": "老人家",
            "emotion_decay_rate": 0.05,
            "max_history": 100,
            **(config or {}),
        }

        self.current_state: EmotionState = EmotionState(
            primary=PrimaryEmotion.NEUTRAL,
            intensity=0.0,
            valence=0.0,
            arousal=0.5,
            decay_rate=self.config["emotion_decay_rate"],
        )
        self.state_history: List[EmotionState] = []
        self.lexicon: ChineseEmotionLexicon = ChineseEmotionLexicon()
        self.expression_map: EmotionExpressionMap = EmotionExpressionMap()
        self.dialogue_bank: ElderlyDialogueBank = ElderlyDialogueBank()
        self._initialized: bool = True
        self._last_interaction_time: float = time.time()

        logger.info("情感计算引擎初始化完成，文化适配=%s，目标用户年龄=%d",
                    self.config["cultural_adaptation"], self.config["user_age"])

    # ========================================================================
    # 多模态情感检测
    # ========================================================================

    async def detect_emotion(
        self,
        text: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
        context: Optional[str] = None,
    ) -> EmotionState:
        """多模态情感检测主入口
        
        综合分析文本和语音特征，返回当前情感状态。支持仅文本或
        仅语音的单一模态输入。
        
        Args:
            text: 用户输入文本
            audio: 语音特征numpy数组(如基频、能量、语速等)
            context: 对话上下文信息
            
        Returns:
            检测到的情感状态
        """
        text_result: Optional[TextEmotionResult] = None
        audio_emotion: Optional[Tuple[PrimaryEmotion, float]] = None

        # 文本情感分析
        if text and text.strip():
            text_result = await self._analyze_text_emotion(text)

        # 语音情感分析
        if audio is not None and len(audio) > 0:
            audio_emotion = await self._analyze_audio_emotion(audio)

        # 多模态融合
        final_emotion = self._fuse_emotions(text_result, audio_emotion)
        final_emotion.source_text = text or ""
        final_emotion.trigger = EmotionTrigger.TEXT if text else EmotionTrigger.AUDIO

        # 更新当前状态
        self.current_state = final_emotion
        self.state_history.append(final_emotion)

        # 限制历史记录长度
        if len(self.state_history) > self.config["max_history"]:
            self.state_history = self.state_history[-self.config["max_history"]:]

        self._last_interaction_time = time.time()

        logger.debug("情感检测结果: %s (强度=%.2f, 置信度=%.2f)",
                     final_emotion.primary.value, final_emotion.intensity,
                     final_emotion.confidence)
        return final_emotion

    async def _analyze_text_emotion(self, text: str) -> TextEmotionResult:
        """基于规则的中文文本情感分析
        
        使用情感词典+程度副词+否定词的三层规则模型进行情感分析。
        特别针对养老场景优化，识别老年人常用的情感表达。
        
        算法步骤:
            1. 分句和关键词匹配
            2. 程度副词加权
            3. 否定词翻转
            4. 多情感聚合
        
        Args:
            text: 待分析的中文文本
            
        Returns:
            文本情感分析结果
        """
        if not text or not text.strip():
            return TextEmotionResult(PrimaryEmotion.NEUTRAL, 0.0)

        # 合并通用情感词和养老场景专用词
        all_words = {**self.lexicon.EMOTION_WORDS, **self.lexicon.ELDERLY_EXPRESSIONS}

        emotion_scores: Dict[PrimaryEmotion, float] = {e: 0.0 for e in PrimaryEmotion}
        found_keywords: List[str] = []

        # 逐词匹配情感词典
        for word, (emotion, base_intensity) in all_words.items():
            if word in text:
                intensity = base_intensity

                # 检查程度副词修饰
                for intensifier, multiplier in self.lexicon.INTENSIFIERS.items():
                    # 检查副词是否在情感词前面5个字以内
                    idx_word = text.find(word)
                    window_start = max(0, idx_word - 10)
                    if intensifier in text[window_start:idx_word]:
                        intensity = min(1.0, base_intensity * multiplier)
                        break

                # 检查否定词修饰(情感翻转)
                for negation in self.lexicon.NEGATIONS:
                    idx_word = text.find(word)
                    window_start = max(0, idx_word - 8)
                    if negation in text[window_start:idx_word]:
                        # 否定情感词: 翻转valence并降低强度
                        intensity *= 0.5
                        # 情感类型映射到相反情感
                        emotion = self._negate_emotion(emotion)
                        break

                emotion_scores[emotion] = max(emotion_scores[emotion], intensity)
                found_keywords.append(word)

        # 如果没有匹配到任何情感词，返回中性
        if not found_keywords:
            return TextEmotionResult(PrimaryEmotion.NEUTRAL, 0.0, keywords=[])

        # 选择得分最高的情感作为主要情感
        primary_emotion = max(emotion_scores, key=lambda e: emotion_scores[e])
        max_score = emotion_scores[primary_emotion]

        # 计算valence和arousal
        valence = self._emotion_to_valence(primary_emotion) * max_score
        arousal = self._emotion_to_arousal(primary_emotion) * max_score

        # 检查是否有紧急关键词(如"救命"、"跌倒")
        emergency_detected = any(kw in text for kw in ["救命", "救", "跌倒", "摔", "出事了"])
        if emergency_detected:
            primary_emotion = PrimaryEmotion.FEAR
            max_score = max(max_score, 0.9)
            arousal = 0.95

        return TextEmotionResult(
            emotion=primary_emotion,
            intensity=min(1.0, max_score),
            keywords=found_keywords,
        )

    async def _analyze_audio_emotion(
        self, audio: np.ndarray
    ) -> Optional[Tuple[PrimaryEmotion, float]]:
        """语音情感特征分析
        
        从语音特征numpy数组中提取情感特征。
        输入数组应包含：[基频均值, 基频标准差, 能量均值, 能量标准差, 语速, ...]
        
        Args:
            audio: 语音特征numpy数组，形状为(N_features,)
            
        Returns:
            (情感类型, 强度)元组，分析失败则返回None
        """
        if audio is None or len(audio) < 3:
            return None

        try:
            # 提取关键特征
            f0_mean = float(audio[0]) if len(audio) > 0 else 150.0
            energy_mean = float(audio[2]) if len(audio) > 2 else 0.5
            speaking_rate = float(audio[4]) if len(audio) > 4 else 1.0

            # 基于启发式规则判断情感
            # 高基频+高能量+快语速 -> 愤怒/恐惧/兴奋
            # 低基频+低能量+慢语速 -> 悲伤/疲惫
            # 正常基频+高能量 -> 开心
            if f0_mean > 250 and energy_mean > 0.7:
                emotion = PrimaryEmotion.ANGRY if speaking_rate > 1.3 else PrimaryEmotion.FEAR
                intensity = min(1.0, (f0_mean - 200) / 150 * energy_mean)
            elif f0_mean < 120 and energy_mean < 0.3:
                emotion = PrimaryEmotion.SAD
                intensity = min(1.0, (150 - f0_mean) / 100 * (0.5 - energy_mean) * 2)
            elif 150 < f0_mean < 250 and energy_mean > 0.5:
                emotion = PrimaryEmotion.HAPPY
                intensity = min(1.0, energy_mean * 0.8)
            else:
                emotion = PrimaryEmotion.NEUTRAL
                intensity = 0.2

            return emotion, intensity

        except Exception as e:
            logger.error("语音情感分析失败: %s", e)
            return None

    def _fuse_emotions(
        self,
        text_result: Optional[TextEmotionResult],
        audio_emotion: Optional[Tuple[PrimaryEmotion, float]],
    ) -> EmotionState:
        """多模态情感融合
        
        综合文本和语音的分析结果，采用加权策略确定最终情感。
        文本模态权重0.6，语音模态权重0.4。
        
        Args:
            text_result: 文本情感分析结果
            audio_emotion: 语音情感分析结果
            
        Returns:
            融合后的情感状态
        """
        if text_result and not audio_emotion:
            return EmotionState(
                primary=text_result.emotion,
                intensity=text_result.intensity,
                valence=self._emotion_to_valence(text_result.emotion) * text_result.intensity,
                arousal=self._emotion_to_arousal(text_result.emotion) * text_result.intensity,
                confidence=0.7,
            )

        if audio_emotion and not text_result:
            return EmotionState(
                primary=audio_emotion[0],
                intensity=audio_emotion[1],
                valence=self._emotion_to_valence(audio_emotion[0]) * audio_emotion[1],
                arousal=self._emotion_to_arousal(audio_emotion[0]) * audio_emotion[1],
                confidence=0.5,
            )

        if text_result and audio_emotion:
            # 文本权重0.6，语音权重0.4
            text_weight = 0.6
            audio_weight = 0.4

            # 如果两个模态情感一致，提高置信度
            if text_result.emotion == audio_emotion[0]:
                fused_emotion = text_result.emotion
                fused_intensity = text_result.intensity * text_weight + audio_emotion[1] * audio_weight
                confidence = 0.9
            else:
                # 选择强度更高的作为主情感
                if text_result.intensity > audio_emotion[1]:
                    fused_emotion = text_result.emotion
                    fused_intensity = text_result.intensity
                else:
                    fused_emotion = audio_emotion[0]
                    fused_intensity = audio_emotion[1]
                confidence = 0.65

            return EmotionState(
                primary=fused_emotion,
                intensity=min(1.0, fused_intensity),
                valence=self._emotion_to_valence(fused_emotion) * fused_intensity,
                arousal=self._emotion_to_arousal(fused_emotion) * fused_intensity,
                confidence=confidence,
            )

        # 没有任何输入时保持中性
        return EmotionState(
            primary=PrimaryEmotion.NEUTRAL,
            intensity=0.0,
            valence=0.0,
            arousal=0.5,
        )

    # ========================================================================
    # 响应策略生成
    # ========================================================================

    def generate_response(
        self,
        emotion: EmotionState,
        context: str,
        user_input: str,
    ) -> str:
        """根据情感状态生成响应
        
        基于当前情感状态、对话上下文和用户输入，生成合适的情感化回应。
        考虑文化适配因素，针对老年用户优化表达方式。
        
        Args:
            emotion: 当前检测到的情感状态
            context: 对话上下文描述，如 "日常陪伴", "健康关怀"
            user_input: 用户原始输入文本
            
        Returns:
            生成的中文响应文本
        """
        templates = self.dialogue_bank.RESPONSE_TEMPLATES.get(
            emotion.primary, self.dialogue_bank.RESPONSE_TEMPLATES[PrimaryEmotion.NEUTRAL]
        )

        # 根据强度选择模板
        if emotion.intensity > 0.7:
            response = templates[0] if len(templates) > 0 else ""
        elif emotion.intensity > 0.4:
            response = templates[1] if len(templates) > 1 else templates[0]
        else:
            response = templates[-1] if templates else ""

        # 添加上下文相关的关怀语句
        if context == "健康关怀" and emotion.primary in (PrimaryEmotion.SAD, PrimaryEmotion.FEAR):
            care_msgs = self.dialogue_bank.CARE_MESSAGES.get("medication", [])
            if care_msgs:
                response += care_msgs[0]

        # 文化适配处理
        if self.config["cultural_adaptation"]:
            response = self._cultural_adaptation(response, self.config["user_age"])

        return response

    def _cultural_adaptation(self, response: str, user_age: int = 70) -> str:
        """中国文化适配 - 针对老年人交流习惯优化
        
        根据用户年龄段调整响应风格:
            - 70岁以上: 使用更慢的语速暗示(添加停顿词)、更多敬语
            - 65-70岁: 标准老年适配
            - 65岁以下: 稍微轻松的风格
        
        适配策略:
            1. 语句简短化: 长句拆分为短句
            2. 添加停顿标记: 用"啊"、"呢"等语气词创造停顿
            3. 重复重点: 关键信息重复表达
            4. 敬语加强: 使用"您"、"老人家"等称谓
            5. 语速控制: TTS标记中添加停顿符
        
        Args:
            response: 原始响应文本
            user_age: 用户年龄
            
        Returns:
            适配后的响应文本
        """
        if user_age >= 75:
            # 高龄用户: 更简短、更多停顿
            # 在逗号后添加停顿词
            response = re.sub(r"，", "，啊，", response, count=1)
            # 确保语句简短
            if len(response) > 30:
                parts = response.split("，")
                if len(parts) >= 3:
                    response = "，".join(parts[:2]) + "。" + "，".join(parts[2:])

        elif user_age >= 65:
            # 标准老年适配
            if "你" in response and "您" not in response:
                response = response.replace("你", "您", 1)

        # 通用适配: 添加温暖的开场
        warm_openers = ["老人家", "奶奶", "爷爷", "阿姨", "叔叔"]
        if not any(opener in response for opener in warm_openers):
            name = self.config.get("user_name", "老人家")
            if name and name != "老人家":
                response = f"{name}，{response}"

        return response

    # ========================================================================
    # 情感表达输出
    # ========================================================================

    def get_emotion_expression(self, emotion: EmotionState) -> Dict[str, Any]:
        """获取情感的多通道表达参数
        
        根据情感状态生成OLED表情、机械臂姿态和TTS语音三个通道的
        同步表达参数。所有通道使用统一的时间戳实现精确同步。
        
        返回的字典结构:
            {
                "emotion": str,  # 情感名称
                "intensity": float,  # 强度
                "sync_ts_ms": float,  # 同步时间戳
                "oled": {...},  # OLED表情参数
                "arm_pose": {...},  # 机械臂姿态参数
                "tts": {...},  # TTS语音参数
                "duration_ms": int,  # 持续时长
            }
        
        Args:
            emotion: 目标情感状态
            
        Returns:
            多通道表达参数字典
        """
        sync_ts = time.time() * 1000  # 毫秒级同步时间戳

        # OLED表情参数
        oled_config = self.expression_map.OLED_MAP.get(
            emotion.primary, self.expression_map.OLED_MAP[PrimaryEmotion.NEUTRAL]
        ).copy()
        # 根据强度调整亮度
        oled_config["brightness"] = int(oled_config["brightness"] * (0.5 + emotion.intensity * 0.5))

        # 机械臂姿态参数
        arm_config = self.expression_map.ARM_MAP.get(
            emotion.primary, self.expression_map.ARM_MAP[PrimaryEmotion.NEUTRAL]
        ).copy()
        arm_config["speed"] = int(arm_config["speed"] * (0.5 + emotion.intensity * 0.5))

        # TTS语音参数
        tts_config = self.expression_map.TTS_MAP.get(
            emotion.primary, self.expression_map.TTS_MAP[PrimaryEmotion.NEUTRAL]
        ).copy()
        tts_config["volume"] = int(tts_config["volume"] * (0.5 + emotion.intensity * 0.5))

        # 关怀模式下进一步降低音量和语速
        if emotion.primary == PrimaryEmotion.CARING:
            tts_config["speed_factor"] *= 0.85
            tts_config["volume"] = int(tts_config["volume"] * 0.8)

        duration = 3000 if emotion.intensity > 0.5 else 1500

        return {
            "emotion": emotion.primary.value,
            "intensity": emotion.intensity,
            "sync_ts_ms": sync_ts,
            "oled": oled_config,
            "arm_pose": arm_config,
            "tts": tts_config,
            "duration_ms": duration,
            "valence": emotion.valence,
            "arousal": emotion.arousal,
        }

    # ========================================================================
    # 辅助方法
    # ========================================================================

    @staticmethod
    def _negate_emotion(emotion: PrimaryEmotion) -> PrimaryEmotion:
        """获取情感的否定/相反情感
        
        Args:
            emotion: 原始情感
            
        Returns:
            相反情感
        """
        negate_map = {
            PrimaryEmotion.HAPPY: PrimaryEmotion.SAD,
            PrimaryEmotion.SAD: PrimaryEmotion.HAPPY,
            PrimaryEmotion.ANGRY: PrimaryEmotion.CARING,
            PrimaryEmotion.FEAR: PrimaryEmotion.CARING,
            PrimaryEmotion.SURPRISE: PrimaryEmotion.NEUTRAL,
            PrimaryEmotion.DISGUST: PrimaryEmotion.CARING,
            PrimaryEmotion.CARING: PrimaryEmotion.ANGRY,
            PrimaryEmotion.CONFUSED: PrimaryEmotion.NEUTRAL,
            PrimaryEmotion.NEUTRAL: PrimaryEmotion.NEUTRAL,
        }
        return negate_map.get(emotion, PrimaryEmotion.NEUTRAL)

    @staticmethod
    def _emotion_to_valence(emotion: PrimaryEmotion) -> float:
        """获取情感的愉悦度(valence)基线值
        
        Args:
            emotion: 情感类型
            
        Returns:
            -1.0到+1.0之间的愉悦度值
        """
        valence_map = {
            PrimaryEmotion.NEUTRAL: 0.0,
            PrimaryEmotion.HAPPY: 0.8,
            PrimaryEmotion.SAD: -0.7,
            PrimaryEmotion.ANGRY: -0.8,
            PrimaryEmotion.FEAR: -0.7,
            PrimaryEmotion.SURPRISE: 0.0,
            PrimaryEmotion.DISGUST: -0.6,
            PrimaryEmotion.CARING: 0.6,
            PrimaryEmotion.CONFUSED: 0.0,
        }
        return valence_map.get(emotion, 0.0)

    @staticmethod
    def _emotion_to_arousal(emotion: PrimaryEmotion) -> float:
        """获取情感的激活度(arousal)基线值
        
        Args:
            emotion: 情感类型
            
        Returns:
            0.0到1.0之间的激活度值
        """
        arousal_map = {
            PrimaryEmotion.NEUTRAL: 0.5,
            PrimaryEmotion.HAPPY: 0.7,
            PrimaryEmotion.SAD: 0.3,
            PrimaryEmotion.ANGRY: 0.9,
            PrimaryEmotion.FEAR: 0.85,
            PrimaryEmotion.SURPRISE: 0.9,
            PrimaryEmotion.DISGUST: 0.5,
            PrimaryEmotion.CARING: 0.4,
            PrimaryEmotion.CONFUSED: 0.4,
        }
        return arousal_map.get(emotion, 0.5)

    def get_emotion_history(self, n: int = 10) -> List[EmotionState]:
        """获取最近N条情感状态历史
        
        Args:
            n: 历史记录条数
            
        Returns:
            情感状态列表，按时间倒序
        """
        return self.state_history[-n:] if n < len(self.state_history) else self.state_history[:]

    def get_emotion_trend(self, window: int = 10) -> float:
        """计算情感趋势
        
        基于最近window条情感记录计算valence趋势。
        正值表示情感趋向积极，负值表示趋向消极。
        
        Args:
            window: 计算窗口大小
            
        Returns:
            -1.0到+1.0之间的趋势值
        """
        if len(self.state_history) < 2:
            return 0.0

        recent = self.state_history[-window:]
        valences = [s.valence * s.intensity for s in recent]
        if len(valences) < 2:
            return valences[0] if valences else 0.0

        # 简单线性趋势
        half = len(valences) // 2
        early_avg = np.mean(valences[:half]) if half > 0 else 0
        late_avg = np.mean(valences[half:]) if half > 0 else 0
        trend = late_avg - early_avg
        return float(np.clip(trend, -1.0, 1.0))

    async def reset_state(self) -> None:
        """重置情感状态为中性
        
        用于对话开始或系统复位时。
        """
        self.current_state = EmotionState(
            primary=PrimaryEmotion.NEUTRAL,
            intensity=0.0,
            valence=0.0,
            arousal=0.5,
            decay_rate=self.config["emotion_decay_rate"],
        )
        logger.debug("情感状态已重置为中性")

    @property
    def is_initialized(self) -> bool:
        """引擎是否已初始化"""
        return self._initialized

    @property
    def idle_time(self) -> float:
        """距上次交互的空闲时间(秒)"""
        return time.time() - self._last_interaction_time