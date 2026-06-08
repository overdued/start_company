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
    HERMES_OK = True
except Exception as e:
    print(f"[WARN] Hermes Bridge 不可用: {e}")
    HERMES_OK = False

# ── API 客户端 ──
try:
    import anthropic
except ImportError:
    print("[ERROR] pip install anthropic")
    sys.exit(1)


# ============================================================
# 硬件状态
# ============================================================
class HardwareStatus:
    def __init__(self):
        self.camera = self._devs("video")
        self.gpio = os.path.exists("/dev/gpiochip0")
        self.i2c = self._devs("i2c-")
        self.uart = self._devs("ttyS") + self._devs("ttyUSB")
        self.npu = os.path.exists("/dev/rknpu")
    def _devs(self, prefix):
        try: return [f"/dev/{d}" for d in os.listdir("/dev") if d.startswith(prefix)]
        except: return []
    def prompt(self):
        return (
            f"- 📷 摄像头: {'✅' if self.camera else '❌'} {self.camera}\n"
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
        self.vision_model = self.model  # Kimi k2 supports vision; DeepSeek v4 doesn't

        if not self.key:
            print("[ERROR] 未设置 ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN"); sys.exit(1)

        client_kwargs = {"api_key": self.key, "base_url": self.url}
        if self.token:
            client_kwargs["auth_token"] = self.token
        self.client = anthropic.Anthropic(**client_kwargs)

        self.hw = HardwareStatus()
        self.hw_exe = HardwareExecutor(self.hw)

        self.dm = None; self.em = None
        if KUNPENG_OK:
            try:
                self.em = EmotionEngine(config={"cultural_adaptation":True, "user_age":70})
                self.dm = DialogueManager(emotion_engine=self.em)
            except: pass

        self.store = None; self.mgr = None; self.db = None; self.ev = None
        if HERMES_OK:
            try:
                self.store = KunpengMemoryStore(memory_dir="data/memories")
                self.mgr = SkillManager(skill_dir="data/skills")
                self.db = SessionSearchDB(db_path="data/sessions.db")
                self.ev = EvolutionEngine(self.mgr, self.store, self.db)
            except Exception as e:
                print(f"[WARN] Hermes Bridge 初始化失败: {e}")

        self.msgs: list = []
        self.sid = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.last_photo: str = ""  # 最近一次拍照的路径

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

    def call_api(self) -> str:
        api_msgs = []
        for m in self.msgs:
            role = m.get("role", "user")
            if role == "system": continue
            content = m.get("content", "")
            # 如果消息带有 image 字段，构建多模态 content 数组
            img_data = m.get("image")  # (b64data, media_type)
            if img_data and isinstance(content, str):
                content_blocks = [{"type": "text", "text": content}]
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": img_data[1], "data": img_data[0]}
                })
                content = content_blocks
            api_msgs.append({"role": role if role in ("user","assistant") else "user", "content": content})
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=4096,
                system=self.system_prompt(), messages=api_msgs,
            )
            text_blocks = [b for b in resp.content if hasattr(b, 'text')]
            return text_blocks[0].text if text_blocks else str(resp.content[0])
        except Exception as e:
            return f"[API 调用失败] {e}"

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

        # 调用 API
        reply = self.call_api()
        self.msgs.append({"role": "assistant", "content": reply})

        # 持久化 + 进化记录
        if self.db:
            try: self.db.append_interaction(self.sid, "agent", reply, tool_name=skill_info if skill_info else None)
            except: pass
        if self.ev:
            try: self.ev.record_task({"success": True, "tool_calls": 3 if skill_info else 1, "skill_used": skill_info.split("]")[1].split(":")[0].strip() if skill_info else None, "elapsed_ms": 100})
            except: pass

        if len(self.msgs) > 40:
            self.msgs = self.msgs[-40:]

        return reply

    def run(self):
        print("\n" + "=" * 60)
        print("  🧠 KunPeng-Hermes 融合系统 v2.0")
        print("=" * 60)
        print(f"  API : {self.url}")
        print(f"  Model: {self.model}")
        print(f"  📷: {'✅' if self.hw.camera else '❌'}  ⚡GPIO: {'✅' if self.hw.gpio else '❌'}  🔌I2C: {'✅' if self.hw.i2c else '❌'}")
        if self.mgr:
            ss = self.mgr.get_stats()
            print(f"  🎯 Skill: {ss['total']}  📝 记忆: {len(self.store.memory_entries)+len(self.store.user_entries)} 条")
        print("-" * 60)
        print("  指令: /quit /clear /skills /memory /photo /photos /hw /status")
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
                    continue
                elif cmd == "help":
                    print("  /quit /clear /skills /memory /photo /photos /hw /status /help")
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
