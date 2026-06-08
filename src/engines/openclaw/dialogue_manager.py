#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenClaw对话管理器 - 多轮对话管理与意图识别

本模块实现KunPeng-Cortex系统的对话管理核心，提供多轮对话状态追踪(DST)、
意图识别、槽位填充、对话策略选择等功能。针对养老场景深度优化，内置
老年人高频话术库和场景化对话模板。

核心功能:
    1. 意图分类: 控制指令、查询、情感交流、紧急求助四大类
    2. 槽位填充: 提取地点、物品、时间、人物等关键信息
    3. 对话状态追踪: 维护对话上下文和状态转换
    4. 对话策略: 澄清、确认、主动建议、转接人工
    5. 养老场景话术库: 覆盖健康、起居、娱乐、安全等场景

设计参考:
    - OpenClaw对话管理Pipeline
    - 养老场景对话需求分析
    - 中文自然语言理解轻量方案

依赖:
    - asyncio: 异步处理
    - re: 正则表达式模式匹配
    - dataclasses: 数据模型

作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型定义
# ============================================================================

class IntentType(Enum):
    """意图类型枚举
    
    养老场景下的四大类意图分类，覆盖老年人与智能助手的典型交互模式。
    """
    # 控制指令类 - 控制硬件设备
    CONTROL_DEVICE = "control_device"       # 控制设备(开灯/关空调等)
    CONTROL_ACTION = "control_action"       # 执行动作(递东西/拿水等)
    CONTROL_CANCEL = "control_cancel"       # 取消当前操作
    CONTROL_PAUSE = "control_pause"         # 暂停当前操作

    # 查询类 - 信息查询
    QUERY_STATUS = "query_status"           # 查询状态(天气/电量/时间)
    QUERY_INFO = "query_info"               # 查询信息(药品/养生知识)
    QUERY_PERSON = "query_person"           # 找人/联系家人

    # 情感交流类 - 情感互动
    CHAT_GREETING = "chat_greeting"         # 问候/打招呼
    CHAT_EMOTION = "chat_emotion"           # 情感表达/倾诉
    CHAT_COMPANION = "chat_companion"       # 陪伴/闲聊
    CHAT_STORY = "chat_story"               # 讲故事/听评书
    CHAT_SING = "chat_sing"                 # 唱歌/听戏

    # 紧急求助类 - 安全相关
    EMERGENCY_FALL = "emergency_fall"       # 跌倒检测/求救
    EMERGENCY_MEDICAL = "emergency_medical" # 医疗急救
    EMERGENCY_FIRE = "emergency_fire"       # 火灾/燃气泄漏
    EMERGENCY_INTRUDER = "emergency_intruder" # 入侵检测

    # 系统类
    SYSTEM_HELP = "system_help"             # 请求帮助
    SYSTEM_SETTING = "system_setting"       # 系统设置
    SYSTEM_SHUTDOWN = "system_shutdown"     # 关机
    UNKNOWN = "unknown"                     # 未知意图
    NO_INPUT = "no_input"                   # 无输入/静音


class DialogueState(Enum):
    """对话状态枚举
    
    对话状态机定义，用于多轮对话管理。
    """
    IDLE = "idle"                           # 空闲等待
    GREETING = "greeting"                   # 问候阶段
    INTENT_CLARIFICATION = "clarification"  # 意图澄清
    SLOT_FILLING = "slot_filling"          # 槽位填充中
    CONFIRMATION = "confirmation"          # 等待用户确认
    ACTION_EXECUTING = "executing"         # 动作执行中
    EMOTION_SUPPORT = "emotion_support"    # 情感支持模式
    EMERGENCY = "emergency"                # 紧急处理模式
    RECOVERY = "recovery"                  # 错误恢复
    END = "end"                            # 对话结束


class DialoguePolicy(Enum):
    """对话策略枚举
    
    对话策略用于决定在特定状态下系统应采取的行动。
    """
    CLARIFY = "clarify"         # 请求澄清
    CONFIRM = "confirm"         # 请求确认
    INFORM = "inform"          # 提供信息
    EXECUTE = "execute"        # 执行操作
    PROPOSE = "propose"        # 主动建议
    APOLOGIZE = "apologize"    # 道歉
    ESCALATE = "escalate"      # 升级/转人工
    WAIT = "wait"             # 等待
    GREET = "greet"           # 问候
    BYE = "bye"               # 告别


@dataclass
class Slot:
    """槽位数据类
    
    对话中的关键信息槽位，用于存储从用户输入中提取的实体。
    
    Attributes:
        name: 槽位名称(如 "location", "object", "time", "person")
        value: 槽位值
        confidence: 提取置信度
        source: 来源位置(在原文中的起止索引)
        required: 是否为必需槽位
        filled: 是否已填充
    """
    name: str
    value: Optional[str] = None
    confidence: float = 0.0
    source: Optional[Tuple[int, int]] = None
    required: bool = False
    filled: bool = False

    def fill(self, value: str, confidence: float = 0.8,
             source: Optional[Tuple[int, int]] = None) -> None:
        """填充槽位值
        
        Args:
            value: 槽位值
            confidence: 置信度
            source: 来源位置
        """
        self.value = value
        self.confidence = confidence
        self.source = source
        self.filled = True

    def clear(self) -> None:
        """清空槽位"""
        self.value = None
        self.confidence = 0.0
        self.source = None
        self.filled = False


@dataclass
class DialogueContext:
    """对话上下文数据类
    
    维护多轮对话的上下文信息，包括对话历史、已填充槽位、当前状态等。
    
    Attributes:
        session_id: 对话会话ID
        state: 当前对话状态
        intent: 当前识别意图
        slots: 已填充槽位字典
        history: 对话历史(最多保存100轮)
        turn_count: 当前轮数
        emotion_state: 情感状态引用
        user_profile: 用户画像缓存
        last_action: 上次执行动作
        last_response: 上次系统回复
        start_time: 对话开始时间戳
    """
    session_id: str
    state: DialogueState = DialogueState.IDLE
    intent: IntentType = IntentType.UNKNOWN
    slots: Dict[str, Slot] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    emotion_state: Optional[Any] = None
    user_profile: Dict[str, Any] = field(default_factory=dict)
    last_action: Optional[str] = None
    last_response: str = ""
    start_time: float = field(default_factory=time.time)

    def add_turn(self, role: str, text: str, intent: Optional[IntentType] = None) -> None:
        """添加对话轮次
        
        Args:
            role: 角色("user"或"system")
            text: 对话内容
            intent: 意图(用户轮次)
        """
        self.history.append({
            "role": role,
            "text": text,
            "intent": intent.value if intent else None,
            "timestamp": time.time(),
            "turn": self.turn_count,
        })
        if role == "user":
            self.turn_count += 1

    def get_slot(self, name: str) -> Optional[Slot]:
        """获取指定槽位
        
        Args:
            name: 槽位名称
            
        Returns:
            Slot对象或None
        """
        return self.slots.get(name)

    def set_slot(self, name: str, value: str, confidence: float = 0.8) -> None:
        """设置槽位值
        
        Args:
            name: 槽位名称
            value: 槽位值
            confidence: 置信度
        """
        if name not in self.slots:
            self.slots[name] = Slot(name=name)
        self.slots[name].fill(value, confidence)

    def get_recent_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近N轮对话历史
        
        Args:
            n: 轮数
            
        Returns:
            对话历史列表
        """
        return self.history[-n:] if self.history else []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "intent": self.intent.value,
            "slots": {k: {"value": v.value, "filled": v.filled} for k, v in self.slots.items()},
            "turn_count": self.turn_count,
            "duration_sec": time.time() - self.start_time,
        }


@dataclass
class DialogueResponse:
    """对话响应数据类
    
    封装系统对用户的完整响应。
    
    Attributes:
        text: 响应文本
        intent: 识别的意图
        state: 当前对话状态
        slots_filled: 本次填充的槽位
        action: 待执行的动作
        emotion_expression: 情感表达参数
        policy: 使用的对话策略
        confidence: 整体置信度
    """
    text: str
    intent: IntentType
    state: DialogueState
    slots_filled: Dict[str, str] = field(default_factory=dict)
    action: Optional[str] = None
    emotion_expression: Optional[Dict[str, Any]] = None
    policy: DialoguePolicy = DialoguePolicy.INFORM
    confidence: float = 0.8


# ============================================================================
# 意图识别模式库
# ============================================================================

class IntentPatternLibrary:
    """意图识别模式库
    
    基于正则表达式和关键词匹配的轻量级意图识别，
    针对养老场景优化，覆盖老年人常用表达方式。
    """

    PATTERNS: Dict[IntentType, List[str]] = {
        # 控制指令
        IntentType.CONTROL_DEVICE: [
            r"(打开|开|启动).{0,5}(灯|空调|风扇|电视|收音机|音响|窗帘|窗户)",
            r"(关闭|关|停).{0,5}(灯|空调|风扇|电视|收音机|音响|窗帘|窗户)",
            r"把.{0,3}(灯|空调|风扇|电视).{0,3}(打开|开|关了|关|关掉)",
            r"(调|调大|调小|调高|调低).{0,3}(温度|音量|亮度)",
        ],
        IntentType.CONTROL_ACTION: [
            r"(帮我|给我|给我拿|递|拿过来|拿过来).{0,10}(水|杯子|药|眼镜|遥控器|拐杖|报纸|书|纸巾)",
            r"(拿|取|递).{0,5}(水|杯子|药|眼镜|遥控器|拐杖)",
            r"我(要|想).{0,3}(喝水|吃药|休息|看电视|听戏|上厕所)",
            r"(扶|搀).{0,3}(我|一下)",
        ],
        IntentType.CONTROL_CANCEL: [
            r"(不用|别|取消|算了|不要|停).{0,5}(了|吧)",
            r"不(要|用|想).{0,3}(了|做|动)",
            r"算了.{0,3}吧",
        ],

        # 查询类
        IntentType.QUERY_STATUS: [
            r"(现在|当前).{0,3}(几点|什么|几号|星期几|什么天气|温度)",
            r"(今天|明天|后天).{0,5}(天气|温度|几度|下雨|下雪)",
            r"(还有|剩余).{0,3}(多少|几).{0,3}(电|电量|水)",
            r"(几点|什么时候|几点钟).{0,3}(吃饭|吃药|睡觉|出门)",
        ],
        IntentType.QUERY_INFO: [
            r"(什么|怎么|怎样).{0,5}(吃|喝|用|做|预防|治疗|保养)",
            r"(告诉我|说说|讲讲).{0,5}(养生|健康|保健|注意事项)",
            r"(这个药|药片|胶囊).{0,3}(怎么|多少|几次|什么时候)",
        ],
        IntentType.QUERY_PERSON: [
            r"(给|找|联系).{0,3}(儿子|女儿|孙子|孙女|孩子|老伴|医生|护工|小张|小王).{0,3}(打电话|视频|联系)",
            r"我想.{0,3}(儿子|女儿|孙子|孙女|孩子|老伴)",
            r"(叫|喊).{0,3}(医生|护士|护工|人来|人)",
        ],

        # 情感交流
        IntentType.CHAT_GREETING: [
            r"(你好|您好|早上好|中午好|晚上好|哈喽|嗨|在吗)",
            r"(小鹏|机器人|喂).{0,3}(你好|在吗|醒醒)",
        ],
        IntentType.CHAT_EMOTION: [
            r"(我|心里).{0,5}(难受|不开心|烦|闷|堵得慌|不是滋味)",
            r"(想|想念).{0,3}(老伴|孩子|孙子|老家|以前)",
            r"(孤独|寂寞|无聊|没劲|没意思)",
            r"(开心|高兴|太好了|好|舒服).{0,3}(啊|呢|呀)",
        ],
        IntentType.CHAT_COMPANION: [
            r"(陪|跟).{0,3}(我|我说).{0,3}(说话|聊聊天|说说话|聊会天)",
            r"(无聊|闷).{0,3}(陪我|说说话|聊聊天)",
            r"(讲个故事|说个笑话|聊聊天)",
        ],
        IntentType.CHAT_STORY: [
            r"(讲|说).{0,3}(故事|评书|相声|历史|往事)",
            r"(我想|给我).{0,3}(听|讲).{0,3}(故事|评书|书)",
            r"(三国演义|西游记|红楼梦|水浒传|岳飞|杨家将)",
        ],
        IntentType.CHAT_SING: [
            r"(唱|放|听).{0,3}(歌|戏|京剧|越剧|黄梅戏|评剧|豫剧|梆子)",
            r"(东方红|茉莉花|小白菜|茉莉花|我的祖国)",
            r"(戏曲|老歌|红歌|民歌)",
        ],

        # 紧急求助
        IntentType.EMERGENCY_FALL: [
            r"(救命|救我|来人啊|快来人|摔倒了|跌倒了|起不来|动不了|帮帮我)",
            r"(我摔|摔倒|跌倒|滑倒).{0,3}(了|啦)",
            r"(疼|痛|动不了|不能动|站不起来)",
        ],
        IntentType.EMERGENCY_MEDICAL: [
            r"(叫|快叫|拨打|打).{0,3}(120|急救|救护车|医生)",
            r"(心脏|胸|头|肚子).{0,3}(疼|痛|不舒服|难受)",
            r"(喘|呼吸|气).{0,3}(不上来|困难|急促)",
            r"(血压|血糖).{0,3}(高|低|不舒服)",
        ],
        IntentType.EMERGENCY_FIRE: [
            r"(着火|起火了|冒烟|煤气泄漏|燃气|火灾)",
            r"(119|火警).{0,3}(快|赶紧|帮忙)",
        ],
        IntentType.EMERGENCY_INTRUDER: [
            r"(有小偷|进贼|坏人|陌生人|撬门|小偷|抢劫)",
            r"(110|报警).{0,3}(快|赶紧)",
        ],

        # 系统类
        IntentType.SYSTEM_HELP: [
            r"(你|你能|怎么).{0,5}(做什么|干什么|会什么|有什么功能|怎么用)",
            r"(帮助|帮忙|指导|说明|教程|使用说明)",
        ],
        IntentType.SYSTEM_SETTING: [
            r"(设置|调整|修改|设定).{0,3}(音量|声音|语速|亮度|时间)",
            r"(声音|音量).{0,3}(大|小|高一点|低一点|调大|调小)",
        ],
        IntentType.SYSTEM_SHUTDOWN: [
            r"(关机|休息|闭嘴|别说话|睡觉|休眠)",
        ],
    }

    # 意图优先级(高优先级优先匹配)
    INTENT_PRIORITY: List[IntentType] = [
        IntentType.EMERGENCY_FALL,
        IntentType.EMERGENCY_MEDICAL,
        IntentType.EMERGENCY_FIRE,
        IntentType.EMERGENCY_INTRUDER,
        IntentType.CONTROL_ACTION,
        IntentType.CONTROL_DEVICE,
        IntentType.QUERY_PERSON,
        IntentType.QUERY_STATUS,
        IntentType.QUERY_INFO,
        IntentType.CHAT_GREETING,
        IntentType.CHAT_EMOTION,
        IntentType.CHAT_COMPANION,
        IntentType.CHAT_SING,
        IntentType.CHAT_STORY,
        IntentType.SYSTEM_HELP,
        IntentType.SYSTEM_SETTING,
        IntentType.SYSTEM_SHUTDOWN,
        IntentType.CONTROL_CANCEL,
        IntentType.UNKNOWN,
    ]


# ============================================================================
# 槽位提取模式库
# ============================================================================

class SlotPatternLibrary:
    """槽位提取模式库
    
    定义从用户输入中提取关键实体的正则表达式模式。
    """

    SLOT_PATTERNS: Dict[str, List[str]] = {
        "location": [
            r"(客厅|卧室|厨房|卫生间|阳台|门口|桌上|床头|沙发|椅子)",
            r"在.{0,2}(客厅|卧室|厨房|卫生间|阳台|门口|桌上|床头)",
        ],
        "object": [
            r"(水杯|杯子|药|药片|眼镜|遥控器|拐杖|报纸|书|纸巾|毛巾|衣服|手机|钥匙)",
            r"(拿|递|给我|给我拿).{0,3}(.*?)(?:过来|一下|给我|$)",
        ],
        "time": [
            r"(早上|中午|晚上|现在|马上|立刻|等一会儿|半小时|一小时|明天|后天)",
            r"(\d{1,2})\s*[点:：]\s*(\d{0,2})",
        ],
        "person": [
            r"(儿子|女儿|孙子|孙女|外孙|外孙女|老伴|老伴儿|孩子|小张|小王|老李)",
            r"给.{0,2}(.*?)(?:打|视频|联系|电话)",
        ],
        "device": [
            r"(灯|空调|风扇|电视|收音机|音响|窗帘|窗户|加湿器|净化器)",
        ],
        "action": [
            r"(打开|开|关闭|关|调大|调小|调高|调低|递|拿|扶|送)",
        ],
        "degree": [
            r"(一点|一些|稍微|很|非常|特别|太|最|比较|大|小|高|低)",
        ],
    }


# ============================================================================
# 养老场景话术库
# ============================================================================

class ElderlyDialogueBank:
    """养老场景话术库
    
    针对养老场景的丰富话术模板，覆盖健康、起居、娱乐、安全等场景。
    """

    # 澄清话术
    CLARIFICATION_TEMPLATES: List[str] = [
        "老人家，您刚才说的是{question}吗？",
        "我没太听懂，您是想要{question}对吗？",
        "您慢点儿说，是不是要{question}？",
        "不好意思啊，您再说一遍好吗？",
        "我年纪轻耳朵不太好使(笑)，您再说清楚点儿？",
    ]

    # 确认话术
    CONFIRMATION_TEMPLATES: List[str] = [
        "好的，我这就{action}，您看行吗？",
        "明白了，是要{action}，对吧？",
        "我这就去{action}，稍等一下啊。",
        "好嘞，{action}，马上就好！",
    ]

    # 执行反馈
    EXECUTION_TEMPLATES: List[str] = [
        "已经帮您{action}了，还有什么需要的吗？",
        "{action}完成了，您看这样行吗？",
        "{action}弄好了，还需要别的吗？",
    ]

    # 主动建议
    PROACTIVE_TEMPLATES: Dict[str, List[str]] = {
        "medication": [
            "到吃药的时间了，我帮您把药和水准备好？",
            "该吃药啦，饭前饭后别忘了，需要我提醒您吗？",
        ],
        "weather": [
            "今天外面{weather}，出门记得多穿点儿。",
            "天气{weather}，您注意保暖/防暑啊。",
        ],
        "activity": [
            "坐久了不好，起来活动活动筋骨？",
            "该活动活动了，我陪您做几个简单的保健操？",
        ],
        "hydration": [
            "多喝点水对身体好，我帮您倒杯温水？",
            "一上午没喝水了吧？来，喝几口润润嗓子。",
        ],
    }

    # 错误恢复
    RECOVERY_TEMPLATES: List[str] = [
        "抱歉我没听清，您能再说一遍吗？",
        "哎哟，我这脑子没转过来，您再说说？",
        "不好意思啊老人家，我没明白，麻烦您再说一下。",
        "对不住对不住，您再说一遍，这次我仔细听着。",
    ]

    # 问候话术(按时间段)
    GREETING_TEMPLATES: Dict[str, List[str]] = {
        "morning": [
            "早上好！老人家昨晚睡得好吗？",
            "早啊！新的一天开始了，祝您精神满满！",
            "早上好！今天气色不错啊，有什么想吃的吗？",
        ],
        "noon": [
            "中午好！该吃午饭了，今天想吃点什么？",
            "午安！午休一会儿养养神，对身体好。",
        ],
        "afternoon": [
            "下午好！下午阳光好，出去晒晒太阳补补钙？",
            "下午好！要不我给您放段评书解解闷？",
        ],
        "evening": [
            "晚上好！今天过得怎么样？",
            "晚上好！该准备休息了，睡前别喝太多水啊。",
        ],
        "night": [
            "夜深了，早点儿休息，晚安！",
            "该睡了，我在这儿守着您，有事随时叫我。",
        ],
    }


# ============================================================================
# 对话管理器核心类
# ============================================================================

class DialogueManager:
    """对话管理器
    
    KunPeng-Cortex系统的对话管理核心，实现多轮对话的状态追踪、
    意图识别、槽位填充和策略选择。
    
    核心功能:
        1. 意图分类: 基于规则和模式匹配的轻量意图识别
        2. 槽位填充: 从用户输入中提取关键实体
        3. 对话状态追踪: 维护对话上下文
        4. 对话策略: 澄清、确认、主动建议等
        5. 养老场景话术: 覆盖高频场景的响应模板
    
    使用示例:
        engine = EmotionEngine()
        dm = DialogueManager(emotion_engine=engine)
        response = await dm.process_turn("帮我开灯", session_id="sess_001")
    
    Attributes:
        emotion_engine: 情感计算引擎引用
        intent_patterns: 意图识别模式库
        slot_patterns: 槽位提取模式库
        dialogue_bank: 话术库
        sessions: 会话字典 {session_id: DialogueContext}
    """

    def __init__(self, emotion_engine: Optional[Any] = None) -> None:
        """初始化对话管理器
        
        Args:
            emotion_engine: 情感计算引擎实例，用于情感感知对话
        """
        self.emotion_engine = emotion_engine
        self.intent_patterns = IntentPatternLibrary()
        self.slot_patterns = SlotPatternLibrary()
        self.dialogue_bank = ElderlyDialogueBank()
        self.sessions: Dict[str, DialogueContext] = {}
        self._session_timeout_sec: float = 300.0  # 5分钟会话超时
        self._max_turns: int = 50  # 最大对话轮数

        logger.info("对话管理器初始化完成")

    # ========================================================================
    # 核心处理流程
    # ========================================================================

    async def process_turn(self, user_input: str, session_id: str) -> DialogueResponse:
        """处理单轮对话
        
        对话处理的主入口，完成意图识别、槽位填充、策略选择和响应生成
        的完整流程。
        
        处理流程:
            1. 获取或创建对话上下文
            2. 检查会话状态(超时/轮数限制)
            3. 意图识别
            4. 槽位填充
            5. 对话策略选择
            6. 响应生成
            7. 更新对话状态
        
        Args:
            user_input: 用户输入文本
            session_id: 会话唯一标识
            
        Returns:
            DialogueResponse对象
        """
        start_time = time.monotonic()

        # 1. 获取或创建对话上下文
        context = self._get_or_create_session(session_id)
        context.add_turn("user", user_input)

        # 2. 检查紧急意图(优先处理)
        intent = self.classify_intent(user_input)

        if self._is_emergency_intent(intent):
            return await self._handle_emergency(context, intent, user_input)

        # 3. 情感分析(如果引擎可用)
        emotion_state = None
        if self.emotion_engine:
            try:
                emotion_state = await self.emotion_engine.detect_emotion(text=user_input)
                context.emotion_state = emotion_state
            except Exception as e:
                logger.warning("情感分析失败: %s", e)

        # 4. 槽位填充
        slots = self.extract_slots(user_input, intent.value)
        for slot_name, slot_value in slots.items():
            context.set_slot(slot_name, slot_value)

        # 5. 对话策略选择
        policy = self._select_policy(context, intent, slots)

        # 6. 生成响应
        response_text = await self._generate_response(context, intent, slots, policy, emotion_state)

        # 7. 更新对话状态
        context.state = self._next_state(context, intent, policy)
        context.last_response = response_text
        context.add_turn("system", response_text, intent)

        elapsed = (time.monotonic() - start_time) * 1000
        logger.debug("对话处理完成: intent=%s policy=%s 耗时=%.2fms",
                     intent.value, policy.value, elapsed)

        return DialogueResponse(
            text=response_text,
            intent=intent,
            state=context.state,
            slots_filled=slots,
            policy=policy,
            emotion_expression=(
                self.emotion_engine.get_emotion_expression(emotion_state)
                if emotion_state and self.emotion_engine else None
            ),
        )

    # ========================================================================
    # 意图识别
    # ========================================================================

    def classify_intent(self, text: str) -> IntentType:
        """意图分类
        
        基于正则表达式和关键词匹配的多模式意图识别。
        按优先级顺序匹配，返回第一个命中的意图。
        
        支持四大类意图:
            - 控制指令: 控制设备、执行动作、取消/暂停
            - 查询: 状态查询、信息查询、找人
            - 情感交流: 问候、情感表达、陪伴、故事、戏曲
            - 紧急求助: 跌倒、医疗、火灾、入侵
        
        Args:
            text: 用户输入文本
            
        Returns:
            识别的意图类型
        """
        if not text or not text.strip():
            return IntentType.NO_INPUT

        text = text.strip()
        best_intent = IntentType.UNKNOWN
        best_score = 0.0

        # 按优先级顺序匹配
        for intent in self.intent_patterns.INTENT_PRIORITY:
            patterns = self.intent_patterns.PATTERNS.get(intent, [])
            if not patterns:
                continue

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    # 计算匹配分数: 完整匹配加分
                    score = len(match.group(0)) / len(text) if text else 0
                    if match.group(0) == text:
                        score += 0.5  # 完全匹配额外加分

                    if score > best_score:
                        best_score = score
                        best_intent = intent

                    # 紧急意图立即返回
                    if self._is_emergency_intent(intent):
                        return intent

                    break  # 该意图已匹配，尝试下一个意图

        logger.debug("意图识别: text='%s' -> intent=%s (score=%.2f)",
                     text[:30], best_intent.value, best_score)
        return best_intent

    # ========================================================================
    # 槽位填充
    # ========================================================================

    def extract_slots(self, text: str, intent: str) -> Dict[str, str]:
        """槽位提取
        
        从用户输入中提取关键实体信息，包括地点、物品、时间、人物等。
        基于正则表达式模式匹配实现。
        
        Args:
            text: 用户输入文本
            intent: 当前意图(用于选择提取策略)
            
        Returns:
            槽位字典 {slot_name: slot_value}
        """
        slots: Dict[str, str] = {}
        if not text:
            return slots

        for slot_name, patterns in self.slot_patterns.SLOT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    value = match.group(0)
                    # 清理提取的值
                    value = value.strip("给拿把把过来看一下给递")
                    if value:
                        slots[slot_name] = value
                    break

        return slots

    # ========================================================================
    # 对话策略
    # ========================================================================

    def _select_policy(
        self,
        context: DialogueContext,
        intent: IntentType,
        slots: Dict[str, str],
    ) -> DialoguePolicy:
        """选择对话策略
        
        基于当前对话状态、意图和槽位填充情况选择最优对话策略。
        
        策略选择规则:
            - 意图置信度低 -> 澄清(CLARIFY)
            - 缺少必需槽位 -> 槽位追问(INFORM)
            - 需要确认的操作 -> 确认(CONFIRM)
            - 闲聊/情感 -> 问候/陪伴(GREET)
            - 紧急 -> 立即执行(EXECUTE)
        
        Args:
            context: 对话上下文
            intent: 当前意图
            slots: 已填充槽位
            
        Returns:
            选定的对话策略
        """
        # 紧急意图直接执行
        if self._is_emergency_intent(intent):
            return DialoguePolicy.EXECUTE

        # 首次问候
        if intent == IntentType.CHAT_GREETING and context.turn_count <= 2:
            return DialoguePolicy.GREET

        # 意图不明确时澄清
        if intent == IntentType.UNKNOWN:
            return DialoguePolicy.CLARIFY

        # 取消操作
        if intent == IntentType.CONTROL_CANCEL:
            return DialoguePolicy.APOLOGIZE

        # 情感交流
        if intent in (IntentType.CHAT_EMOTION, IntentType.CHAT_COMPANION):
            return DialoguePolicy.INFORM

        # 控制类: 缺少关键槽位时追问
        if intent in (IntentType.CONTROL_DEVICE, IntentType.CONTROL_ACTION):
            if "device" not in slots and "object" not in slots:
                return DialoguePolicy.CLARIFY
            return DialoguePolicy.CONFIRM

        # 查询类: 直接提供信息
        if intent in (IntentType.QUERY_STATUS, IntentType.QUERY_INFO, IntentType.QUERY_PERSON):
            return DialoguePolicy.INFORM

        # 系统类
        if intent == IntentType.SYSTEM_HELP:
            return DialoguePolicy.INFORM
        if intent == IntentType.SYSTEM_SHUTDOWN:
            return DialoguePolicy.BYE

        return DialoguePolicy.INFORM

    # ========================================================================
    # 响应生成
    # ========================================================================

    async def _generate_response(
        self,
        context: DialogueContext,
        intent: IntentType,
        slots: Dict[str, str],
        policy: DialoguePolicy,
        emotion_state: Optional[Any],
    ) -> str:
        """生成系统响应文本
        
        根据对话策略和意图生成合适的中文响应。
        
        Args:
            context: 对话上下文
            intent: 当前意图
            slots: 已填充槽位
            policy: 对话策略
            emotion_state: 情感状态
            
        Returns:
            响应文本
        """
        if policy == DialoguePolicy.CLARIFY:
            templates = self.dialogue_bank.CLARIFICATION_TEMPLATES
            question = self._generate_clarification_question(intent, slots)
            return templates[context.turn_count % len(templates)].format(question=question)

        elif policy == DialoguePolicy.CONFIRM:
            templates = self.dialogue_bank.CONFIRMATION_TEMPLATES
            action = self._describe_action(intent, slots)
            return templates[context.turn_count % len(templates)].format(action=action)

        elif policy == DialoguePolicy.GREET:
            return self._generate_greeting()

        elif policy == DialoguePolicy.EXECUTE:
            if self._is_emergency_intent(intent):
                return self._generate_emergency_response(intent)
            return self._generate_execution_response(intent, slots)

        elif policy == DialoguePolicy.APOLOGIZE:
            return "好的，没问题，您说怎样就怎样。"

        elif policy == DialoguePolicy.BYE:
            return "好的，我这就休息去，有事随时叫我啊，老人家保重！"

        elif policy == DialoguePolicy.INFORM:
            return self._generate_inform_response(intent, slots, emotion_state)

        elif policy == DialoguePolicy.PROPOSE:
            return self._generate_proactive_response("activity")

        # 默认响应
        return "嗯，我在听呢，您接着说。"

    def _generate_greeting(self) -> str:
        """生成问候响应
        
        根据当前时间段选择合适的话术。
        
        Returns:
            问候文本
        """
        hour = time.localtime().tm_hour
        if 5 <= hour < 11:
            period = "morning"
        elif 11 <= hour < 14:
            period = "noon"
        elif 14 <= hour < 18:
            period = "afternoon"
        elif 18 <= hour < 22:
            period = "evening"
        else:
            period = "night"

        templates = self.dialogue_bank.GREETING_TEMPLATES.get(period, ["您好！"])
        return templates[0]

    def _generate_execution_response(self, intent: IntentType, slots: Dict[str, str]) -> str:
        """生成执行反馈响应
        
        Args:
            intent: 意图
            slots: 槽位
            
        Returns:
            执行反馈文本
        """
        action = self._describe_action(intent, slots)
        templates = self.dialogue_bank.EXECUTION_TEMPLATES
        return templates[0].format(action=action)

    def _generate_inform_response(
        self,
        intent: IntentType,
        slots: Dict[str, str],
        emotion_state: Optional[Any],
    ) -> str:
        """生成信息响应
        
        Args:
            intent: 意图
            slots: 槽位
            emotion_state: 情感状态
            
        Returns:
            信息响应文本
        """
        if intent == IntentType.CHAT_COMPANION:
            return "好嘞，我陪着您呢，想聊点什么？您说说年轻时候的事儿？"

        elif intent == IntentType.CHAT_STORY:
            return "好嘞，我给您讲一段三国演义的故事？还是您想听评书？"

        elif intent == IntentType.CHAT_SING:
            return "好啊，我给您放段京剧《红灯记》选段？还是来首老歌？"

        elif intent == IntentType.QUERY_STATUS:
            return self._generate_status_response(slots)

        elif intent == IntentType.SYSTEM_HELP:
            return ("我能帮您做不少事呢！比如开灯关灯、递东西、"
                    "提醒您吃药、陪您聊天、讲故事听戏、查询天气时间，"
                    "有急事还能帮您联系家人或叫救护车。您需要我做什么？")

        elif intent == IntentType.CHAT_EMOTION and emotion_state:
            if self.emotion_engine:
                return self.emotion_engine.generate_response(
                    emotion_state, "日常陪伴", ""
                )
            return "我明白我明白，有什么心事跟我说说，我听着呢。"

        elif intent == IntentType.QUERY_PERSON:
            person = slots.get("person", "")
            if person:
                return f"好的，我这就帮您联系{person}，您稍等啊。"
            return "好的，您想联系谁？告诉我名字我帮您拨号。"

        return "我明白了，还有什么需要我帮忙的吗？"

    def _generate_status_response(self, slots: Dict[str, str]) -> str:
        """生成状态查询响应
        
        Args:
            slots: 槽位
            
        Returns:
            状态信息文本
        """
        hour = time.localtime().tm_hour
        minute = time.localtime().tm_min

        # 天气查询(模拟)
        if "天气" in str(slots.values()):
            return "今天天气不错，晴转多云，温度18到26度，适合出去走走。"

        # 时间查询
        return f"现在时间是{hour}点{minute}分。"

    def _generate_emergency_response(self, intent: IntentType) -> str:
        """生成紧急响应
        
        紧急情况的立即响应话术。
        
        Args:
            intent: 紧急意图
            
        Returns:
            紧急响应文本
        """
        responses = {
            IntentType.EMERGENCY_FALL: "您别动！我立刻联系急救和家人！",
            IntentType.EMERGENCY_MEDICAL: "坚持住！我马上拨打120！",
            IntentType.EMERGENCY_FIRE: "赶紧撤离！我立即报警！",
            IntentType.EMERGENCY_INTRUDER: "别出声！我立刻报警！",
        }
        return responses.get(intent, "紧急情况！立即处理！")

    def _generate_proactive_response(self, category: str) -> str:
        """生成主动建议响应
        
        Args:
            category: 建议类别
            
        Returns:
            建议文本
        """
        templates = self.dialogue_bank.PROACTIVE_TEMPLATES.get(category, [])
        if templates:
            return templates[0]
        return "老人家，您看需要我帮什么忙吗？"

    # ========================================================================
    # 状态管理
    # ========================================================================

    def _next_state(
        self,
        context: DialogueContext,
        intent: IntentType,
        policy: DialoguePolicy,
    ) -> DialogueState:
        """计算下一对话状态
        
        基于当前状态和策略决定状态转换。
        
        Args:
            context: 当前上下文
            intent: 意图
            policy: 策略
            
        Returns:
            下一状态
        """
        if self._is_emergency_intent(intent):
            return DialogueState.EMERGENCY

        if policy == DialoguePolicy.CLARIFY:
            return DialogueState.INTENT_CLARIFICATION

        if policy == DialoguePolicy.CONFIRM:
            return DialogueState.CONFIRMATION

        if policy == DialoguePolicy.EXECUTE:
            return DialogueState.ACTION_EXECUTING

        if policy == DialoguePolicy.GREET:
            return DialogueState.GREETING

        if policy == DialoguePolicy.BYE:
            return DialogueState.END

        if intent == IntentType.CHAT_EMOTION:
            return DialogueState.EMOTION_SUPPORT

        if context.turn_count >= self._max_turns:
            return DialogueState.END

        return DialogueState.IDLE

    def _is_emergency_intent(self, intent: IntentType) -> bool:
        """判断是否为紧急意图
        
        Args:
            intent: 意图
            
        Returns:
            True表示紧急意图
        """
        return intent in {
            IntentType.EMERGENCY_FALL,
            IntentType.EMERGENCY_MEDICAL,
            IntentType.EMERGENCY_FIRE,
            IntentType.EMERGENCY_INTRUDER,
        }

    # ========================================================================
    # 会话管理
    # ========================================================================

    def _get_or_create_session(self, session_id: str) -> DialogueContext:
        """获取或创建对话会话
        
        如果会话存在且未超时则返回现有会话，否则创建新会话。
        
        Args:
            session_id: 会话ID
            
        Returns:
            对话上下文
        """
        if session_id in self.sessions:
            context = self.sessions[session_id]
            # 检查超时
            last_time = context.history[-1]["timestamp"] if context.history else context.start_time
            if time.time() - last_time > self._session_timeout_sec:
                logger.info("会话 %s 超时，创建新会话", session_id)
                del self.sessions[session_id]
            else:
                return context

        # 创建新会话
        context = DialogueContext(session_id=session_id)
        self.sessions[session_id] = context
        logger.info("创建新会话: %s", session_id)
        return context

    def get_session(self, session_id: str) -> Optional[DialogueContext]:
        """获取会话上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            对话上下文或None
        """
        return self.sessions.get(session_id)

    def end_session(self, session_id: str) -> None:
        """结束会话
        
        清理会话资源。
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("会话已结束: %s", session_id)

    def cleanup_expired_sessions(self) -> List[str]:
        """清理过期会话
        
        Returns:
            已清理的会话ID列表
        """
        now = time.time()
        expired = []
        for sid, ctx in list(self.sessions.items()):
            last_time = ctx.history[-1]["timestamp"] if ctx.history else ctx.start_time
            if now - last_time > self._session_timeout_sec:
                expired.append(sid)
                del self.sessions[sid]
        if expired:
            logger.info("清理%d个过期会话", len(expired))
        return expired

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _generate_clarification_question(self, intent: IntentType, slots: Dict[str, str]) -> str:
        """生成澄清问题
        
        Args:
            intent: 模糊意图
            slots: 部分槽位
            
        Returns:
            澄清问题文本
        """
        if not slots:
            return "做什么"
        if "device" not in slots and "object" not in slots:
            return "开/关什么"
        return "这个意思"

    def _describe_action(self, intent: IntentType, slots: Dict[str, str]) -> str:
        """描述要执行的动作
        
        Args:
            intent: 意图
            slots: 槽位
            
        Returns:
            动作描述文本
        """
        device = slots.get("device", slots.get("object", ""))
        action = slots.get("action", "")

        if intent == IntentType.CONTROL_DEVICE:
            if action in ("打开", "开"):
                return f"把{device}打开"
            elif action in ("关闭", "关", "关掉"):
                return f"把{device}关上"
            return f"操作{device}"

        elif intent == IntentType.CONTROL_ACTION:
            return f"把{device}拿过来" if device else "帮您拿"

        elif intent == IntentType.QUERY_PERSON:
            person = slots.get("person", "")
            return f"联系{person}" if person else "联系"

        return "处理"

    def get_active_session_count(self) -> int:
        """获取活跃会话数
        
        Returns:
            活跃会话数量
        """
        return len(self.sessions)

    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息
        
        Returns:
            统计字典
        """
        return {
            "active_sessions": len(self.sessions),
            "total_turns": sum(ctx.turn_count for ctx in self.sessions.values()),
            "session_timeout_sec": self._session_timeout_sec,
            "max_turns": self._max_turns,
        }