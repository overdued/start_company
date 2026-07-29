#!/usr/bin/env python3
"""service_main.py — 声纹语音服务入口

集成 FusionChat (v2.0 智能体) + VoiceService (声纹管道)

命令行指令:
  /enroll <名字>   — 用最近一次说话人声音注册新用户
  /users           — 列出已注册用户
  /quit            — 退出
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voice.service import VoiceService


def on_voice_event(event: str, payload: dict) -> None:
    """语音事件显示"""
    icons = {
        "wake": "🔔", "listening": "👂", "asr": "📝",
        "identified": "👤", "pending": "❓", "new_speaker": "🆕",
        "multi_speaker": "👥", "reply": "💬", "enrolled": "✅",
    }
    icon = icons.get(event, "•")
    if event == "wake":
        print(f"\n{icon} 唤醒: {payload.get('keyword')}")
    elif event == "asr":
        print(f"{icon} 识别: {payload.get('text')}")
    elif event == "identified":
        print(f"{icon} 身份: {payload.get('user')} (相似度 {payload.get('similarity', 0):.2f})")
    elif event == "pending":
        print(f"{icon} 待定: 相似度 {payload.get('similarity', 0):.2f}")
    elif event == "new_speaker":
        print(f"{icon} 新说话人 (相似度 {payload.get('similarity', 0):.2f}) — 输入 /enroll <名字> 注册")
    elif event == "multi_speaker":
        print(f"{icon} 检测到多人场景")
    elif event == "reply":
        print(f"{icon} 小鲲: {payload.get('reply', '')[:200]}")
    elif event == "enrolled":
        print(f"{icon} 已注册: {payload.get('user')} (ID={payload.get('id')})")


def main() -> int:
    no_chat = os.environ.get("NO_CHAT", "") == "1"

    chat = None
    if not no_chat:
        try:
            from start_fusion import FusionChat
            chat = FusionChat()
            print("[OK] FusionChat v2.0 智能体已加载")
        except Exception as e:
            print(f"[WARN] FusionChat 不可用 ({e})，仅运行声纹识别")

    service = VoiceService(chat=chat, on_event=on_voice_event)

    print("=" * 60)
    print("  🎤 KunPeng 声纹语音交互服务 v2.2")
    print("=" * 60)
    print(f"  唤醒词: 小鲲小鲲 / 你好小鲲")
    print(f"  声纹库: {service.db.count()} 个注册用户")
    print(f"  LLM: {'已连接' if chat else '未连接 (仅声纹)'}")
    print("-" * 60)
    print("  指令: /enroll <名字> | /users | /quit")
    print("=" * 60)

    if not service.start():
        print("[ERROR] 服务启动失败")
        return 1

    try:
        while True:
            try:
                inp = input().strip()
            except EOFError:
                break
            if not inp:
                continue
            if inp in ("/quit", "/exit", "/q"):
                break
            elif inp.startswith("/enroll "):
                name = inp[8:].strip()
                if name:
                    if service.enroll_current(name):
                        pass  # 事件已打印
                    else:
                        print("  ⚠️ 无可用声纹锚点，请先唤醒并说话")
            elif inp == "/users":
                users = service.db.list_users()
                if users:
                    for u in users:
                        print(f"  👤 {u['name']} (ID={u['id']}, 样本={u['sample_count']})")
                else:
                    print("  (无注册用户)")
            elif inp == "/status":
                print(f"  状态: {service.state}  用户: {service.db.count()}  待定: {service.db.get_stats()['pending']}")
    except KeyboardInterrupt:
        pass
    finally:
        print("\n正在关闭...")
        service.stop()
        print("再见！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
