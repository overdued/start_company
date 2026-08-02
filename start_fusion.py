#!/usr/bin/env python3
"""
KunPeng-Hermes 融合系统 — 智能对话终端 v2.0

集成:
  - Kimi/DeepSeek API（通过 Anthropic 协议）
  - Hermes Bridge: 记忆系统 + Skill 管理 + 进化引擎 + 会话搜索
  - KunPeng-Cortex: 意图识别 + 情感计算 + 硬件控制

启动:
  source venv_kunpeng/bin/activate
  cd /home/openEuler/agent_xia/kunpeng-cortex/project
  export ANTHROPIC_API_KEY="sk-kimi-..."
  export ANTHROPIC_BASE_URL="https://api.kimi.com/coding"
  PYTHONPATH=src python3 start_fusion.py
"""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# ── KunPeng 引擎 ──
try:
    from engines.openclaw.dialogue_manager import DialogueManager
    from engines.openclaw.emotion_engine import EmotionEngine
    KUNPENG_OK = True
except Exception as e:
    print(f"[WARN] KunPeng 引擎不可用: {e}")
    KUNPENG_OK = False

# ── Hermes Bridge ──
try:
    from hermes_bridge.memory_tool import KunpengMemoryStore
    from hermes_bridge.skill_manager import SkillManager
    from hermes_bridge.session_search import SessionSearchDB
    from hermes_bridge.evolution_engine import EvolutionEngine
    from hermes_bridge.memos_adapter import MemosStore
    HERMES_OK = True
except Exception as e:
    print(f"[WARN] Hermes Bridge 不可用: {e}")
    HERMES_OK = False

# ── API 客户端 ──
# 支持两种协议: Anthropic (Kimi) 和 OpenAI (Qwen/DeepSeek)
_api_type = None
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ============================================================
# 硬件状态
# ============================================================
class HardwareStatus:
    def __init__(self):
        self.camera = [d for d in self._devs("video") if d != "/dev/video1"]  # video1 为 USB 扬声器接口
        self.gpio = os.path.exists("/dev/gpiochip0")
        self.i2c = self._devs("i2c-")
        self.uart = self._devs("ttyS") + self._devs("ttyUSB")
        self.npu = os.path.exists("/dev/rknpu")
        # 音频：USB 扬声器 + 四通道麦克风
        self.speaker = os.path.exists("/dev/snd/pcmC0D0p")
        self.mic = os.path.exists("/dev/snd/pcmC0D0c")
        self.tts = bool(subprocess.run(["which", "espeak-ng"], capture_output=True).returncode == 0)
    def _devs(self, prefix):
        try: return [f"/dev/{d}" for d in os.listdir("/dev") if d.startswith(prefix)]
        except: return []
    def prompt(self):
        return (
            f"- 📷 摄像头: {'✅' if self.camera else '❌'} {self.camera}\n"
            f"- 🔈 扬声器: {'✅' if self.speaker else '❌'}  🎤 麦克风: {'✅' if self.mic else '❌'}  🗣️ TTS: {'✅' if self.tts else '❌'}\n"
            f"- ⚡ GPIO: {'✅' if self.gpio else '❌'}\n"
            f"- 🔌 I2C: {'✅' if self.i2c else '❌'} {len(self.i2c)} buses\n"
            f"- 🔗 UART: {'✅' if self.uart else '❌'}\n"
            f"- 🧠 NPU: {'✅' if self.npu else '❌'}"
        )

class HardwareExecutor:
    def __init__(self, hw): self.hw = hw; self.dir = Path.home() / "Pictures"; self.dir.mkdir(exist_ok=True)
    def take_photo(self):
        if not self.hw.camera: return False, "无摄像头"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        f = self.dir / f"capture_{ts}.jpg"
        devs = sorted(self.hw.camera, key=lambda d: (d != "/dev/video0", d))
        for d in devs:
            try:
                r = subprocess.run(["fswebcam","-d",d,"-r","640x480","--no-banner",str(f)],
                                   capture_output=True, text=True, timeout=15)
                if r.returncode == 0 and f.exists(): return True, f"照片已保存: {f}"
            except: continue
        return False, "拍照失败"
    def list_photos(self):
        try:
            fs = sorted(self.dir.glob("capture_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
            return True, "\n".join(f"  📸 {p.name} ({p.stat().st_size//1024}KB)" for p in fs[:10]) if fs else "暂无照片"
        except Exception as e: return False, str(e)
    def files(self, path="~"):
        try:
            p = Path(path).expanduser()
            if not p.exists(): return False, f"路径不存在: {p}"
            items = [f"  {'📁' if i.is_dir() else '📄'} {i.name}" for i in sorted(p.iterdir())[:30]]
            return True, f"{p}:\n" + "\n".join(items)
        except Exception as e: return False, str(e)
    def get_photo_base64(self, filepath: str = None) -> tuple[bool, str, str]:
        """读取照片为 base64。filepath 为 None 则读最新照片。返回 (ok, path, b64data)"""
        try:
            if filepath:
                p = Path(filepath)
            else:
                fs = sorted(self.dir.glob("capture_*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
                p = fs[0] if fs else None
            if not p or not p.exists(): return False, "", ""
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return True, str(p), b64
        except Exception: return False, "", ""
    def speak(self, text: str) -> tuple[bool, str]:
        """TTS 语音播报（edge-tts 微软云自然语音 → espeak-ng 离线兜底）"""
        if not self.hw.speaker: return False, "扬声器不可用"
        mp3_path = "/tmp/tts_output.mp3"
        wav_path = "/tmp/tts_output.wav"
        try:
            # 优先 edge-tts 微软云自然语音
            r = subprocess.run(
                ["edge-tts", "--voice", "zh-CN-XiaoxiaoNeural", "--text", text,
                 "--write-media", mp3_path],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0 and Path(mp3_path).exists():
                # 转 WAV 播放
                subprocess.run(["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1",
                                wav_path], capture_output=True, timeout=10)
            else:
                raise RuntimeError("edge-tts failed, using espeak fallback")
        except Exception:
            # 兜底：espeak-ng 离线机械音
            try:
                subprocess.run(["espeak-ng", "-w", wav_path, "-v", "zh", "-s", "150", text],
                               capture_output=True, text=True, timeout=10)
            except Exception:
                return False, "TTS 生成失败（edge-tts 和 espeak-ng 均不可用）"
        try:
            subprocess.run(["fuser", "-k", "/dev/snd/pcmC0D0p"], capture_output=True, timeout=2)
            r2 = subprocess.run(["aplay", "-q", wav_path], capture_output=True, text=True, timeout=10)
            if r2.returncode != 0:
                r2 = subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
                                    capture_output=True, text=True, timeout=10)
            return (r2.returncode == 0, f"播报: {text[:50]}")
        except FileNotFoundError as e:
            return False, f"缺少工具: {e}"


# ============================================================
# AI 工具定义（OpenAI function calling 格式）
# ============================================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "add_reminder",
            "description": "为用户添加一条提醒事项。当用户提到要做某事、需要提醒、别忘了等意图时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "提醒时间，如 '14:00'、'下午3点'、'晚上8点'。如果用户没指定时间则留空"},
                    "title": {"type": "string", "description": "提醒标题，简短描述要做的事"},
                    "desc": {"type": "string", "description": "提醒详情或备注"},
                    "icon": {"type": "string", "description": "提醒图标 emoji，如 💊🚶🍽️📞😴🛒", "default": "📌"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "查询今天的提醒事项列表",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_device",
            "description": "开关智能家居设备（灯、空调、电视、窗帘等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string", "description": "房间名称，如 '客厅'、'卧室'、'厨房'"},
                    "device": {"type": "string", "description": "设备名称，如 '主灯'、'空调'、'电视'"},
                    "action": {"type": "string", "enum": ["on", "off", "toggle"], "description": "操作：on=打开, off=关闭, toggle=切换"},
                },
                "required": ["device"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_photo",
            "description": "用摄像头拍一张照片",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_data",
            "description": "获取用户的健康数据（心率、血压、血氧、体温等）",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取当前天气和环境信息（温度、湿度、天气状况）",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_car",
            "description": "控制小车移动。支持前进、后退、左转、右转、停止。当用户说'前进'、'开过去'、'往左走'、'停下来'等移动指令时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["forward", "backward", "left", "right", "stop"], "description": "移动方向: forward=前进, backward=后退, left=左转, right=右转, stop=停止"},
                    "speed": {"type": "integer", "description": "速度百分比 0-100，默认50", "default": 50, "minimum": 5, "maximum": 100},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_arm",
            "description": "控制机械臂。可以移动指定关节到目标角度，或执行预设动作（归位、夹取、释放、递水）。当用户说'帮我拿'、'夹住'、'放手'、'机械臂动一下'等指令时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["move_joint", "preset"], "description": "操作类型: move_joint=移动关节, preset=预设动作"},
                    "joint": {"type": "string", "enum": ["j1","j2","j3","j4","j5","j6"], "description": "关节编号 j1=底座 j2=肩 j3=肘 j4=腕俯仰 j5=腕旋转 j6=夹爪"},
                    "angle": {"type": "integer", "description": "目标角度 0-180（夹爪 0-90）", "minimum": 0, "maximum": 180},
                    "preset": {"type": "string", "enum": ["home","grip","release","wave","drink"], "description": "预设动作: home=归位, grip=夹取, release=释放, wave=挥手, drink=递水"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emergency_stop",
            "description": "紧急停止所有设备（小车电机和机械臂）。当用户说'停'、'紧急停止'、'快停下'等紧急指令时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["all", "motors", "arm"], "description": "停止范围: all=全部, motors=小车电机, arm=机械臂", "default": "all"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "radar_detect",
            "description": "查询毫米波雷达检测结果。检测老人是否在房间内、是否跌倒、呼吸频率等。触发：'雷达'、'老人在哪'、'有人在吗'、'检测到人了吗'",
            "parameters": {"type": "object", "properties": {
                "mode": {"type": "string", "enum": ["presence", "fall", "breath", "all"], "description": "检测模式：presence=存在感知, fall=跌倒检测, breath=呼吸监测, all=全部"}
            }, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wifi_sense",
            "description": "WiFi CSI 穿墙感知。利用 WiFi 信道状态信息检测室内人员位置和活动。触发：'wifi检测'、'穿墙'、'信号检测'、'定位'",
            "parameters": {"type": "object", "properties": {
                "mode": {"type": "string", "enum": ["location", "activity", "count", "all"], "description": "检测模式：location=定位, activity=活动识别, count=人数统计, all=全部"}
            }, "required": []},
        },
    },
]


# ============================================================
# 融合对话终端
# ============================================================
class FusionChat:
    SYSTEM = """你是 KunPeng-Cortex v2.0，运行在 OrangePi Kunpeng Pro (RK3588) 上的自进化硬件控制智能体。

## 核心能力
1. **硬件控制**（KunPeng-Cortex）：GPIO/I2C/UART/PWM/摄像头/机械臂，<50ms 响应
2. **自进化学习**（Hermes Brain）：从每次硬件任务学习，自动生成可复用 Skill
3. **三层记忆**：工作记忆 → 跨会话 FTS5 搜索 → 持久记忆（MEMORY.md + USER.md）
4. **情感关怀**：中文情感计算 + 文化适配，养老场景优化
5. **视觉分析**：你能看到摄像头拍摄的真实照片并描述内容。当你收到照片时，用自然语言告诉用户画面里有什么

## ⚠️ 绝对禁止——违反将导致严重后果
1. **禁止编造任何文件路径**（如 /home/xxx/Pictures/xxx.jpg），除非 [硬件操作结果] 中明确给出了该路径
2. **禁止说"已拍照/已开灯/已执行"**，除非 [硬件操作结果] 中明确显示操作成功
3. **禁止生成任何 XML/HTML/代码块来模拟硬件操作**（如 <camera>, <arm>, ```json 等）
4. **硬件操作结果由系统在 [硬件操作结果] 中告诉你**，你只能基于此回复，不能自己编造
5. 如果用户发了照片给你看，**认真描述照片里的内容**——你确实能看到图片
6. **直接对用户说话**，用温暖的中文回复，不要用技术术语或代码格式

## 回复格式
- 直接自然语言回复，像家人聊天一样
- 收到照片时："我看到画面里有[具体描述]"
- 如果 Skill 匹配成功，告诉用户"好的，我来帮您[具体操作]"
- 如果硬件结果显示成功，告诉用户结果（使用真实路径/数据）
- 如果硬件结果显示失败，诚实告知并建议解决方法

## 交流风格
- 温暖耐心，使用敬语，语句简短清晰
- 主动关心用户身体状况
- 紧急情况优先响应
"""

    def __init__(self):
        self.key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        self.url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding")
        self.model = os.environ.get("ANTHROPIC_MODEL", "kimi-k2-0711-preview")
        self.token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")

        # 自动检测 API 类型：根据 base_url 判断是 Anthropic 还是 OpenAI 协议
        global _api_type
        _api_type = os.environ.get("API_TYPE", "").lower()
        if not _api_type:
            url_lower = self.url.lower()
            # Kimi/DeepSeek/Claude 均走 Anthropic Messages 协议
            if "anthropic" in url_lower or "kimi" in url_lower or "moonshot" in url_lower:
                _api_type = "anthropic"
            else:
                _api_type = "openai"  # Qwen/DashScope 用 OpenAI 协议

        # Vision model: use env var if set, otherwise auto-detect
        # Kimi k2 / Claude / Qwen-VL support vision; DeepSeek v4 text models do not
        self.vision_model = os.environ.get("VISION_MODEL", "")
        if not self.vision_model:
            if "deepseek" in self.model.lower():
                self.vision_model = ""  # DeepSeek text-only, disable vision
            else:
                self.vision_model = self.model  # Kimi/Claude/Qwen use same model for vision

        if not self.key:
            print("[ERROR] 未设置 ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN"); sys.exit(1)

        # 初始化客户端
        if _api_type == "openai":
            if not _HAS_OPENAI:
                print("[ERROR] 使用 OpenAI 协议需要 openai 库: pip install openai")
                sys.exit(1)
            self.client = openai.OpenAI(api_key=self.key, base_url=self.url)
            print(f"[INFO] 使用 OpenAI 协议 (Qwen 兼容), 模型: {self.model}")
        else:
            if not _HAS_ANTHROPIC:
                print("[ERROR] 使用 Anthropic 协议需要 anthropic 库: pip install anthropic")
                sys.exit(1)
            client_kwargs = {"api_key": self.key, "base_url": self.url}
            if self.token:
                client_kwargs["auth_token"] = self.token
            self.client = anthropic.Anthropic(**client_kwargs)
            print(f"[INFO] 使用 Anthropic 协议, 模型: {self.model}")

        self.hw = HardwareStatus()
        self.hw_exe = HardwareExecutor(self.hw)

        self.dm = None; self.em = None
        if KUNPENG_OK:
            try:
                self.em = EmotionEngine(config={"cultural_adaptation":True, "user_age":70})
                self.dm = DialogueManager(emotion_engine=self.em)
            except: pass

        self.store = None; self.mgr = None; self.db = None; self.ev = None; self.memos = None
        if HERMES_OK:
            try:
                self.store = KunpengMemoryStore(memory_dir="data/memories")
                self.mgr = SkillManager(skill_dir="data/skills")
                self.db = SessionSearchDB(db_path="data/sessions.db")
                self.ev = EvolutionEngine(self.mgr, self.store, self.db)
                self.memos = MemosStore(db_path="data/memos.db")
            except Exception as e:
                print(f"[WARN] Hermes Bridge 初始化失败: {e}")

        self.msgs: list = []
        self.sid = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.last_photo: str = ""  # 最近一次拍照的路径
        self.tool_handlers: dict = {}  # 外部注册的工具回调: {name: callable}
        self.tool_results: list = []   # 最近一次对话中执行的工具结果

    def system_prompt(self) -> str:
        p = self.SYSTEM
        if self.store:
            snap = self.store.get_snapshot_for_prompt()
            if snap["memory"]: p += f"\n\n【环境记忆】\n{snap['memory']}"
            if snap["user"]:   p += f"\n\n【用户画像】\n{snap['user']}"
        if self.mgr:
            skills = self.mgr.list_skills(detail_level=0)
            if skills:
                lines = ["\n\n【已加载 Skill】"]
                for s in skills:
                    lines.append(f"- {s['name']} [{s['category']}]: {s['description'][:80]}")
                p += "\n".join(lines)
        if self.memos:
            ctx = self.memos.get_recent_context(limit=5)
            if ctx:
                p += ctx
        p += f"\n\n【硬件状态】\n{self.hw.prompt()}"
        return p

    def local_analysis(self, inp: str) -> str:
        parts = []
        if self.dm:
            try:
                intent = self.dm.classify_intent(inp)
                slots = self.dm.extract_slots(inp, intent.value)
                parts.append(f"[意图] {intent.value}")
                if slots: parts.append(f"[槽位] {slots}")
            except: pass
        if self.em:
            try:
                emotion = asyncio.run(self.em.detect_emotion(text=inp))
                parts.append(f"[情感] {emotion.primary.value}({emotion.intensity:.2f})")
            except: pass
        return "\n".join(parts)

    def hw_action(self, inp: str) -> tuple[bool, str]:
        kw = inp.lower()
        if any(k in kw for k in ("拍照","拍张","拍个照","拍张照片")):
            return self.hw_exe.take_photo()
        if any(k in kw for k in ("照片","图片","拍的照","看看照片")):
            return self.hw_exe.list_photos()
        if any(k in kw for k in ("文件","目录","有什么","列出")):
            return self.hw_exe.files("~")
        return False, ""

    def skill_match(self, inp: str) -> str:
        if self.mgr:
            skill = self.mgr.match_skill(inp)
            if skill:
                return f"[Skill 匹配] {skill['name']}: {skill['description'][:120]}"
        return ""

    def execute_tool(self, name: str, arguments: dict) -> str:
        """执行一个工具调用，返回 JSON 字符串结果"""
        import json as _json

        # 优先用外部注册的回调（server.py 注册的）
        if name in self.tool_handlers:
            try:
                result = self.tool_handlers[name](arguments)
                return _json.dumps(result, ensure_ascii=False)
            except Exception as e:
                return _json.dumps({"error": str(e)}, ensure_ascii=False)

        # 内置工具：拍照
        if name == "take_photo":
            ok, msg = self.hw_exe.take_photo()
            if ok:
                import re
                m = re.search(r'(/[\w/]+\.jpg)', msg)
                if m:
                    self.last_photo = m.group(1)
            return _json.dumps({"success": ok, "message": msg, "path": self.last_photo if ok else ""}, ensure_ascii=False)

        return _json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

    def call_api(self) -> dict:
        """调用 API，自动适配 Anthropic 或 OpenAI 协议。

        返回 dict:
          - {"type": "text", "content": "..."}           普通文本回复
          - {"type": "tool_calls", "tool_calls": [...]}   需要执行工具
        """
        system_prompt = self.system_prompt()

        if _api_type == "openai":
            # ── OpenAI 协议 (Qwen/DashScope) ──
            api_msgs = [{"role": "system", "content": system_prompt}]
            for m in self.msgs:
                role = m.get("role", "user")
                if role == "system": continue
                content = m.get("content", "")
                img_data = m.get("image")
                if img_data and isinstance(content, str):
                    content_blocks = [
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{img_data[1]};base64,{img_data[0]}"
                        }}
                    ]
                    content = content_blocks
                msg_entry = {"role": role if role in ("user", "assistant", "tool") else "user", "content": content}
                # Preserve tool_call_id for tool response messages
                if role == "tool" and "tool_call_id" in m:
                    msg_entry["tool_call_id"] = m["tool_call_id"]
                # Preserve tool_calls for assistant messages
                if role == "assistant" and "tool_calls" in m:
                    msg_entry["tool_calls"] = m["tool_calls"]
                    if not content:
                        msg_entry["content"] = None
                api_msgs.append(msg_entry)
            try:
                print(f"[API] Sending request: model={self.model}, msgs={len(api_msgs)}, tools={len(TOOL_DEFINITIONS)}")
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=api_msgs,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                )
                choice = resp.choices[0]
                msg = choice.message
                print(f"[API] Response: finish_reason={choice.finish_reason}, has_tool_calls={bool(msg.tool_calls)}")

                # Check if the model wants to call tools
                if msg.tool_calls and len(msg.tool_calls) > 0:
                    print(f"[API] Tool calls: {[tc.function.name for tc in msg.tool_calls]}")
                    tool_calls_data = []
                    for tc in msg.tool_calls:
                        tool_calls_data.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })
                    return {"type": "tool_calls", "tool_calls": tool_calls_data, "assistant_content": msg.content}
                else:
                    text = msg.content or ""
                    print(f"[API] Text reply ({len(text)} chars): {text[:80]}")
                    return {"type": "text", "content": text}
            except Exception as e:
                print(f"[API ERROR] {type(e).__name__}: {e}")
                return {"type": "text", "content": f"[API 调用失败] {e}"}
        else:
            # ── Anthropic 协议 (Kimi/Claude) ──
            api_msgs = []
            for m in self.msgs:
                role = m.get("role", "user")
                if role == "system": continue
                content = m.get("content", "")
                img_data = m.get("image")
                if img_data and isinstance(content, str):
                    content_blocks = [{"type": "text", "text": content}]
                    content_blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": img_data[1], "data": img_data[0]}
                    })
                    content = content_blocks
                api_msgs.append({"role": role if role in ("user", "assistant") else "user", "content": content})
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=4096,
                    system=system_prompt, messages=api_msgs,
                )
                text_blocks = [b for b in resp.content if hasattr(b, 'text')]
                text = text_blocks[0].text if text_blocks else str(resp.content[0])
                return {"type": "text", "content": text}
            except Exception as e:
                return {"type": "text", "content": f"[API 调用失败] {e}"}

    def chat(self, inp: str, hw_result: tuple[bool, str] = None, photo_path: str = None) -> str:
        # 本地分析
        local = self.local_analysis(inp)
        skill_info = self.skill_match(inp)
        # hw_result 由 run() 传入（避免重复调用）；如未传入则自行检测
        if hw_result is None:
            hw_result = self.hw_action(inp)
        if photo_path is None:
            photo_path = self.last_photo  # 使用最新拍照

        # 构建消息
        user_msg = inp
        extras = []
        if local: extras.append(local)
        if skill_info: extras.append(skill_info)
        if hw_result and hw_result[1]:
            hw_ok, hw_res = hw_result
            extras.append(f"[硬件操作结果] 操作{'成功' if hw_ok else '失败'}，真实结果如下，严禁编造：{hw_res}")
        if extras: user_msg += "\n\n[系统分析]\n" + "\n".join(extras)

        # 检测视觉相关请求 —— 如果有最近拍摄的照片，一并发送给模型
        vision_keywords = ("看看","照片","图片","拍的照","拍照","画面","看到","拍摄","摄像头")
        is_vision = any(k in inp for k in vision_keywords)
        msg_entry = {"role": "user", "content": user_msg}

        if is_vision and photo_path:
            ok, path, b64 = self.hw_exe.get_photo_base64(photo_path)
            if ok:
                msg_entry["image"] = (b64, "image/jpeg")
                msg_entry["content"] = user_msg + f"\n[系统] 以下是你刚才拍摄的照片 ({path})，请描述画面内容"

        self.msgs.append(msg_entry)

        # 持久化
        if self.db:
            try: self.db.append_interaction(self.sid, "user", inp)
            except: pass

        # ── 工具调用循环 ──
        self.tool_results = []
        max_tool_rounds = 5  # 防止无限循环
        for round_idx in range(max_tool_rounds):
            result = self.call_api()

            if result["type"] == "text":
                # 最终文本回复
                reply = result["content"]
                self.msgs.append({"role": "assistant", "content": reply})
                break
            elif result["type"] == "tool_calls":
                # AI 要求调用工具
                tool_calls = result["tool_calls"]

                # 先把 assistant 的 tool_calls 消息加入历史
                assistant_msg = {
                    "role": "assistant",
                    "content": result.get("assistant_content") or "",
                    "tool_calls": tool_calls,
                }
                self.msgs.append(assistant_msg)

                # 执行每个工具调用
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        import json as _json
                        func_args = _json.loads(tc["function"]["arguments"])
                    except:
                        func_args = {}

                    print(f"  \033[35m[Tool] {func_name}({_json.dumps(func_args, ensure_ascii=False)})\033[0m")

                    tool_result = self.execute_tool(func_name, func_args)
                    self.tool_results.append({"tool": func_name, "args": func_args, "result": tool_result})

                    # 把工具结果加入消息历史
                    self.msgs.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tc["id"],
                    })
            else:
                reply = str(result)
                self.msgs.append({"role": "assistant", "content": reply})
                break
        else:
            reply = "抱歉，我处理请求时遇到了问题，请再试一次。"

        # 持久化 + 进化记录
        if self.db:
            try: self.db.append_interaction(self.sid, "agent", reply, tool_name=skill_info if skill_info else None)
            except: pass
        if self.ev:
            try: self.ev.record_task({"success": True, "tool_calls": 3 if skill_info else 1, "skill_used": skill_info.split("]")[1].split(":")[0].strip() if skill_info else None, "elapsed_ms": 100})
            except: pass

        # === Memos 自动记录：检测用户输入中的关键信息 ===
        if self.memos:
            try:
                self._auto_log_to_memos(inp, reply)
            except: pass

        if len(self.msgs) > 40:
            self.msgs = self.msgs[-40:]

        return reply

    def _auto_log_to_memos(self, inp: str, reply: str) -> None:
        """智能自动记录——当用户提到关键信息时自动保存到 Memos（30分钟去重）"""
        triggers = {
            "药": ("health", "用药"), "吃药": ("health", "用药"), "血压": ("health", "用药"),
            "不喜欢": ("用户偏好", "偏好"), "喜欢": ("用户偏好", "偏好"),
            "想要": ("客户需求", "需求"), "需要": ("客户需求", "需求"),
            "希望": ("客户需求", "功能建议"), "能帮我": ("客户需求", "功能建议"),
            "建议": ("客户需求", "功能建议"),
            "坏了": ("问题反馈", "问题"), "不行": ("问题反馈", "问题"),
            "记住": ("manual", "重要"), "记下来": ("manual", "重要"),
            "摔倒": ("紧急", "跌倒"), "不舒服": ("health", "身体不适"),
        }
        import time as _time
        for kw, (cat, tag) in triggers.items():
            if kw in inp:
                recent = self.memos.search(kw)
                if recent:
                    if _time.time() - recent[0].get("created_ts", 0) < 1800:
                        continue
                self.memos.create(content=inp[:200], tags=[tag], category=cat, source="auto_detect")
                return

    def run(self):
        print("\n" + "=" * 60)
        print("  🧠 KunPeng-Hermes 融合系统 v2.2")
        print("=" * 60)
        print(f"  API : {self.url}")
        print(f"  Model: {self.model}")
        print(f"  📷: {'✅' if self.hw.camera else '❌'}  🔈: {'✅' if self.hw.speaker else '❌'}  🎤: {'✅' if self.hw.mic else '❌'}  ⚡GPIO: {'✅' if self.hw.gpio else '❌'}")
        if self.mgr:
            ss = self.mgr.get_stats()
            print(f"  🎯 Skill: {ss['total']}  📝 记忆: {len(self.store.memory_entries)+len(self.store.user_entries)} 条")
        if self.memos:
            print(f"  📋 Memos: {self.memos.count()} 条记录")
        print("-" * 60)
        print("  指令: /quit /clear /skills /memory /photo /photos /hw /status /memos /memo /profile /speaker /speak")
        print("=" * 60)

        while True:
            try:
                inp = input("\n\033[32m你 >\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！"); break
            if not inp: continue

            if inp.startswith("/"):
                cmd = inp[1:].lower()
                if cmd in ("quit","exit","q"): print("\n再见！"); break
                elif cmd == "clear": self.msgs = []; print("[已清空]"); continue
                elif cmd == "skills":
                    if self.mgr:
                        for s in self.mgr.list_skills():
                            print(f"  🎯 {s['name']} [{s['category']}]: {s['description'][:80]}")
                    else: print("  (Skill 系统未加载)"); continue
                elif cmd == "memory":
                    if self.store:
                        snap = self.store.get_snapshot_for_prompt()
                        print(f"【环境记忆】({len(snap['memory'])}字)\n{snap['memory'][:400]}")
                        print(f"\n【用户画像】({len(snap['user'])}字)\n{snap['user'][:400]}")
                    else: print("  (记忆系统未加载)"); continue
                elif cmd == "photo":
                    ok, res = self.hw_exe.take_photo()
                    print(f"  {'✅' if ok else '❌'} {res}")
                    if ok:
                        import re
                        m = re.search(r'(/[\w/]+\.jpg)', res)
                        if m: self.last_photo = m.group(1)
                    continue
                elif cmd == "photos":
                    ok, res = self.hw_exe.list_photos()
                    print(res); continue
                elif cmd == "hw":
                    print(self.hw.prompt()); continue
                elif cmd == "status":
                    print(f"  消息数: {len(self.msgs)}  会话: {self.sid}")
                    if self.mgr: print(f"  Skill: {self.mgr.get_stats()}")
                    if self.ev: print(f"  进化: {self.ev.get_stats()}")
                    if self.memos: print(f"  Memos: {self.memos.get_stats()}")
                    continue
                elif cmd == "memos":
                    if self.memos:
                        ms = self.memos.list(limit=10)
                        if ms:
                            for m in ms:
                                tags = ", ".join(m.get("tags", [])) if m.get("tags") else ""
                                ts = datetime.fromtimestamp(m["updated_ts"]).strftime("%m/%d %H:%M")
                                print(f"  📋 [{ts}] {m['content'][:80]}{' #'+tags if tags else ''}")
                        else: print("  (暂无记录)")
                    else: print("  (Memos 未加载)"); continue
                elif cmd.startswith("memo "):
                    content = inp[6:]
                    if self.memos and content:
                        r = self.memos.auto_log(content)
                        print(f"  📋 [{r.get('category')}] #{', '.join(r.get('tags', []))}: {content[:50]}")
                    continue
                elif cmd.startswith("memos "):
                    query = inp[7:]
                    if self.memos and query:
                        for m in self.memos.search(query):
                            tags = ", ".join(m.get("tags", [])) if m.get("tags") else ""
                            ts = datetime.fromtimestamp(m["updated_ts"]).strftime("%m/%d %H:%M")
                            print(f"  📋 [{ts}] {m['content'][:100]}{' #'+tags if tags else ''}")
                    continue
                elif cmd == "profile":
                    if self.memos:
                        p = self.memos.get_user_profile()
                        print(f"  📋 总记录: {p['total_memos']}  ❤️ 偏好: {len(p['preferences'])}  💊 健康: {len(p['health'])}  📝 需求: {len(p['requirements'])}")
                    continue
                elif cmd == "export":
                    if self.memos:
                        print(self.memos.export_recent(30)[:2000])
                    continue
                elif cmd == "speaker":
                    ok, res = self.hw_exe.speak("你好，我是鲲鹏智能助手小鲲")
                    print(f"  {'✅' if ok else '❌'} {res}"); continue
                elif cmd.startswith("speak "):
                    ok, res = self.hw_exe.speak(inp[7:])
                    print(f"  {'✅' if ok else '❌'} {res}"); continue
                elif cmd == "help":
                    print("  /quit /clear /skills /memory /photo /photos /hw /status /help")
                    print("  📋 Memos: /memo <内容> | /memos [关键词] | /profile | /export")
                    print("  🔊 音频: /speaker 测试 | /speak <文字> TTS播报")
                    continue

            try:
                # 显示分析（只调用一次硬件操作）
                local = self.local_analysis(inp)
                skill_info = self.skill_match(inp)
                hw_result = self.hw_action(inp)  # 只执行一次
                if local: print(f"  \033[36m{local}\033[0m")
                if skill_info: print(f"  \033[35m{skill_info}\033[0m")
                if hw_result[1]: print(f"  \033[35m[硬件] {'✅' if hw_result[0] else '❌'} {hw_result[1][:100]}\033[0m")

                # 如果是拍照操作且成功，记录最新照片路径
                if hw_result[0] and "照片已保存" in hw_result[1]:
                    import re
                    m = re.search(r'(/[\w/]+\.jpg)', hw_result[1])
                    if m: self.last_photo = m.group(1)

                reply = self.chat(inp, hw_result=hw_result, photo_path=self.last_photo)
                print(f"\n\033[33mKunPeng >\033[0m {reply}")
            except Exception as e:
                print(f"\n[错误] {e}")
                traceback.print_exc()


def main():
    FusionChat().run()

if __name__ == "__main__":
    main()
