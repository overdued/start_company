#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KunPeng-Cortex 智能对话终端 (Kimi API + 真实硬件版)

集成 Kimi API + KunPeng-Cortex 本地意图识别/情感引擎/硬件控制，
提供真正可操作硬件的交互式对话体验。

启动方式:
    source /home/openEuler/agent_xia/venv_kunpeng/bin/activate
    cd /home/openEuler/agent_xia/kunpeng-cortex/project
    PYTHONPATH=src python3 kunpeng_chat.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 设置项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# ---------------------------------------------------------------------------
# 导入 KunPeng-Cortex 核心模块
# ---------------------------------------------------------------------------
try:
    from engines.openclaw.dialogue_manager import DialogueManager, DialogueState, IntentType
    from engines.openclaw.emotion_engine import EmotionEngine
    KUNPENG_AVAILABLE = True
except ImportError as e:
    print(f"[警告] KunPeng-Cortex 引擎导入失败: {e}")
    KUNPENG_AVAILABLE = False

# ---------------------------------------------------------------------------
# 导入 Anthropic 客户端
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    print("[错误] 未安装 anthropic 客户端。请执行: pip install anthropic")
    sys.exit(1)


# =============================================================================
# 硬件状态检测
# =============================================================================

class HardwareStatus:
    """硬件状态检测器"""

    def __init__(self):
        self.camera = self._check_camera()
        self.gpio = self._check_gpio()
        self.i2c = self._check_i2c()
        self.npu = self._check_npu()
        self.uart = self._check_uart()

    def _check_camera(self) -> dict:
        devices = []
        try:
            for d in os.listdir('/dev'):
                if d.startswith('video'):
                    devices.append(f'/dev/{d}')
        except Exception:
            pass
        return {"available": len(devices) > 0, "devices": devices}

    def _check_gpio(self) -> dict:
        available = os.path.exists('/dev/gpiochip0')
        return {"available": available, "backend": "libgpiod" if available else None}

    def _check_i2c(self) -> dict:
        buses = []
        try:
            for d in os.listdir('/dev'):
                if d.startswith('i2c-'):
                    buses.append(d)
        except Exception:
            pass
        return {"available": len(buses) > 0, "buses": buses}

    def _check_npu(self) -> dict:
        return {"available": os.path.exists('/dev/rknpu')}

    def _check_uart(self) -> dict:
        ports = []
        try:
            for d in os.listdir('/dev'):
                if d.startswith('ttyS') or d.startswith('ttyUSB'):
                    ports.append(f'/dev/{d}')
        except Exception:
            pass
        return {"available": len(ports) > 0, "ports": ports}

    def to_prompt(self) -> str:
        """生成硬件状态描述，用于 system prompt"""
        lines = ["## 当前硬件状态（实时检测）"]
        lines.append(f"- 摄像头: {'可用' if self.camera['available'] else '不可用'} {self.camera.get('devices', [])}")
        lines.append(f"- GPIO: {'可用' if self.gpio['available'] else '不可用'} ({self.gpio.get('backend', 'none')})")
        lines.append(f"- I2C: {'可用' if self.i2c['available'] else '不可用'} {self.i2c.get('buses', [])}")
        lines.append(f"- UART: {'可用' if self.uart['available'] else '不可用'} {self.uart.get('ports', [])}")
        lines.append(f"- NPU: {'可用' if self.npu['available'] else '不可用'}")
        return "\n".join(lines)


# =============================================================================
# 硬件操作执行器
# =============================================================================

class HardwareExecutor:
    """真实硬件操作执行器"""

    def __init__(self, hw_status: HardwareStatus):
        self.hw = hw_status
        self.photo_dir = Path.home() / "Pictures"
        self.photo_dir.mkdir(exist_ok=True)

    def take_photo(self) -> tuple[bool, str]:
        """拍照并保存"""
        if not self.hw.camera["available"]:
            return False, "摄像头未检测到，无法拍照"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.photo_dir / f"capture_{timestamp}.jpg"

        # 优先尝试 video0（通常是主捕获设备）
        devices = self.hw.camera["devices"]
        # 排序让 video0 优先
        devices_sorted = sorted(devices, key=lambda d: (d != "/dev/video0", d))

        for device in devices_sorted:
            try:
                result = subprocess.run(
                    ["fswebcam", "-d", device, "-r", "640x480", "--no-banner", str(filename)],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0 and filename.exists():
                    return True, f"照片已保存: {filename}"
            except FileNotFoundError:
                return False, "未安装 fswebcam，请执行: sudo dnf install fswebcam"
            except Exception:
                continue

        # 所有设备都失败，返回最后一个错误
        return False, f"所有摄像头设备都无法拍照，请检查摄像头连接"

    def list_photos(self) -> tuple[bool, str]:
        """列出已保存的照片"""
        try:
            files = sorted(self.photo_dir.glob("capture_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                return True, "尚未保存任何照片"
            lines = [f"已保存 {len(files)} 张照片:"]
            for f in files[:10]:
                size = f.stat().st_size
                size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                lines.append(f"  {f.name} ({size_str})")
            return True, "\n".join(lines)
        except Exception as e:
            return False, f"读取照片目录失败: {e}"

    def control_gpio(self, action: str, pin: int | None = None) -> tuple[bool, str]:
        """控制 GPIO"""
        if not self.hw.gpio["available"]:
            return False, "GPIO 不可用"
        # 实际 GPIO 操作需要 gpiod 库，这里返回说明
        return True, f"GPIO 控制: {action} (pin={pin}) — 需要进一步配置具体引脚"

    def list_files(self, path: str = "~") -> tuple[bool, str]:
        """列出目录文件"""
        try:
            target = Path(path).expanduser()
            if not target.exists():
                return False, f"路径不存在: {target}"
            items = []
            for item in sorted(target.iterdir()):
                icon = "📁" if item.is_dir() else "📄"
                items.append(f"  {icon} {item.name}")
            return True, f"{target} 下的内容:\n" + "\n".join(items[:30])
        except Exception as e:
            return False, f"读取目录失败: {e}"


# =============================================================================
# KunPeng-Cortex 对话终端
# =============================================================================

class KunPengChat:
    """KunPeng-Cortex 智能对话终端（真实硬件集成版）"""

    BASE_PROMPT = """你是 KunPeng-Cortex 智能助手，运行在 OrangePi Kunpeng Pro (RK3588) 嵌入式平台上。

## 核心能力
1. 智能家居控制：理解语音指令控制灯光、空调、窗帘等设备（需 GPIO 连接继电器）
2. 养老辅助：药品提醒、健康监测、紧急呼叫、情感陪伴
3. 机械臂操作：递取物品、辅助起居（需 UART/I2C 连接机械臂）
4. 多模态感知：摄像头拍照、超声波测距、陀螺仪姿态

## 交流风格
- 对老年用户使用温暖、耐心的语气
- 使用中文回复，语句简短清晰
- 主动关心用户身体状况
- 遇到紧急情况优先响应

## ⚠️ 极其重要的规则

1. **绝对禁止编造**：你不能虚构任何文件路径、硬件状态或操作结果。
2. **硬件操作由系统执行**：当用户要求拍照、控制设备等操作时，系统会调用真实硬件执行，你只需要根据执行结果回复用户。
3. **诚实告知限制**：如果某项硬件不可用，直接告诉用户原因和解决建议。
4. **不要假装操作成功**：收到系统返回的成功/失败结果后再回复。
"""

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding")
        self.model = os.environ.get("ANTHROPIC_MODEL", "kimi-k2-0711-preview")

        if not self.api_key:
            print("[错误] 未设置 ANTHROPIC_API_KEY 环境变量")
            print("请执行: export ANTHROPIC_API_KEY='your-key-here'")
            sys.exit(1)

        # 初始化硬件状态
        self.hw_status = HardwareStatus()
        self.hw_executor = HardwareExecutor(self.hw_status)

        # 初始化 Anthropic 客户端
        self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)

        # 初始化 KunPeng-Cortex 引擎
        self.dialogue_manager = None
        self.emotion_engine = None
        self._init_kunpeng_engines()

        # 对话历史
        self.messages: list[dict] = []
        self.session_id = "kunpeng_session_001"

    def _init_kunpeng_engines(self) -> None:
        if not KUNPENG_AVAILABLE:
            print("[警告] KunPeng-Cortex 引擎不可用")
            return
        try:
            self.emotion_engine = EmotionEngine(config={
                "cultural_adaptation": True, "user_age": 70, "user_name": "老人家",
            })
            self.dialogue_manager = DialogueManager(emotion_engine=self.emotion_engine)
            print("[OK] KunPeng-Cortex 情感引擎 + 对话管理器 已加载")
        except Exception as e:
            print(f"[警告] KunPeng-Cortex 引擎初始化失败: {e}")

    def _build_system_prompt(self) -> str:
        """构建包含实时硬件状态的 system prompt"""
        return self.BASE_PROMPT + "\n\n" + self.hw_status.to_prompt()

    def _build_kunpeng_context(self, user_input: str) -> str:
        """本地意图/情感分析"""
        context_parts = []
        if self.dialogue_manager and self.emotion_engine:
            try:
                intent = self.dialogue_manager.classify_intent(user_input)
                context_parts.append(f"[意图识别] {intent.value}")
                slots = self.dialogue_manager.extract_slots(user_input, intent.value)
                if slots:
                    context_parts.append(f"[提取信息] {slots}")
                emotion = asyncio.run(self.emotion_engine.detect_emotion(text=user_input))
                context_parts.append(
                    f"[情感状态] {emotion.primary.value} (强度={emotion.intensity:.2f})"
                )
            except Exception as e:
                context_parts.append(f"[本地分析异常] {e}")
        return "\n".join(context_parts) if context_parts else ""

    def _execute_hardware_action(self, intent: str, slots: dict, user_input: str) -> tuple[bool, str]:
        """根据用户输入执行真实硬件操作（不依赖意图分类，直接检测关键词）"""
        text = user_input.lower()

        # 检测拍照请求
        photo_keywords = ("拍照", "拍张", "照相", "拍个照", "拍张照片", "拍一下")
        if any(kw in text for kw in photo_keywords):
            return self.hw_executor.take_photo()

        # 检测查看照片请求
        view_photo_keywords = ("照片", "图片", "刚才拍的", "拍的照", "看看照片")
        if any(kw in text for kw in view_photo_keywords):
            return self.hw_executor.list_photos()

        # 检测文件列表请求
        if any(kw in text for kw in ("文件", "目录", "有什么", "列出")):
            path = slots.get("location", "~")
            return self.hw_executor.list_files(path)

        return False, ""

    def _call_kimi(self, messages: list[dict]) -> str:
        try:
            api_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    continue
                api_role = role if role in ("user", "assistant") else "user"
                api_messages.append({"role": api_role, "content": content})

            response = self.client.messages.create(
                model=self.model, max_tokens=4096,
                system=self._build_system_prompt(), messages=api_messages,
            )
            return response.content[0].text
        except Exception as e:
            return f"[Kimi API 调用失败] {type(e).__name__}: {e}"

    def chat(self, user_input: str) -> str:
        # 1. 本地意图/情感分析
        local_context = self._build_kunpeng_context(user_input)
        if local_context:
            print(f"\n  \033[36m{local_context}\033[0m")

        # 2. 尝试执行硬件操作
        intent_str = ""
        slots = {}
        if self.dialogue_manager:
            intent = self.dialogue_manager.classify_intent(user_input)
            intent_str = intent.value
            slots = self.dialogue_manager.extract_slots(user_input, intent.value)

        hw_success, hw_result = self._execute_hardware_action(intent_str, slots, user_input)
        if hw_result:
            print(f"  \033[35m[硬件操作] {'成功' if hw_success else '失败'}: {hw_result[:100]}\033[0m")

        # 3. 构建发送给 Kimi 的消息
        user_message = user_input
        if local_context:
            user_message += f"\n\n[系统本地分析]\n{local_context}"
        if hw_result:
            user_message += f"\n\n[硬件操作结果] {'成功' if hw_success else '失败'}: {hw_result}"

        self.messages.append({"role": "user", "content": user_message})

        # 4. 调用 Kimi API
        assistant_reply = self._call_kimi(self.messages)
        self.messages.append({"role": "assistant", "content": assistant_reply})

        # 截断历史
        if len(self.messages) > 40:
            self.messages = self.messages[-40:]

        return assistant_reply

    def run_interactive(self) -> None:
        hw = self.hw_status
        print("\n" + "=" * 60)
        print("  🦅 KunPeng-Cortex 智能对话终端 (Kimi API + 硬件)")
        print("=" * 60)
        print(f"  API: {self.base_url}")
        print(f"  模型: {self.model}")
        print(f"  摄像头: {'✅' if hw.camera['available'] else '❌'} {hw.camera.get('devices', [])}")
        print(f"  GPIO: {'✅' if hw.gpio['available'] else '❌'}")
        print(f"  I2C: {'✅' if hw.i2c['available'] else '❌'} {len(hw.i2c.get('buses', []))} buses")
        print(f"  UART: {'✅' if hw.uart['available'] else '❌'}")
        print(f"  NPU: {'✅' if hw.npu['available'] else '❌'}")
        print("-" * 60)
        print("  指令: /quit 退出 | /clear 清空 | /status 状态 | /hw 硬件详情")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\033[32m你 >\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue

            # 斜杠命令
            if user_input.startswith("/"):
                cmd = user_input[1:].lower()
                if cmd in ("quit", "exit", "q"):
                    print("\n再见！KunPeng-Cortex 期待下次为您服务。"); break
                elif cmd == "clear":
                    self.messages = []; print("[已清空对话历史]"); continue
                elif cmd == "status":
                    print(f"  消息数: {len(self.messages)}  会话: {self.session_id}"); continue
                elif cmd == "hw":
                    print(self.hw_status.to_prompt()); continue
                elif cmd == "photo":
                    ok, msg = self.hw_executor.take_photo()
                    print(f"  {'✅' if ok else '❌'} {msg}"); continue
                elif cmd == "photos":
                    ok, msg = self.hw_executor.list_photos()
                    print(f"  {'✅' if ok else '❌'} {msg}"); continue
                elif cmd == "help":
                    print("  /quit    - 退出")
                    print("  /clear   - 清空历史")
                    print("  /status  - 系统状态")
                    print("  /hw      - 硬件详情")
                    print("  /photo   - 立即拍照")
                    print("  /photos  - 查看已保存照片")
                    continue

            # 处理用户输入
            try:
                reply = self.chat(user_input)
                print(f"\n\033[33mKunPeng >\033[0m {reply}\n")
            except Exception as e:
                print(f"\n[错误] {e}")
                traceback.print_exc()


def main():
    chat = KunPengChat()
    chat.run_interactive()


if __name__ == "__main__":
    main()
