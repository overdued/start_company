#!/usr/bin/env python3
"""service.py — 声纹语音交互服务主循环

状态机:
  [待机] → VAD 检测语音 → [唤醒监听]
    ├─ KWS 命中唤醒词 → [ASR + 声纹]
    │   ├─ 单人 → 声纹识别 → 身份判定 → FusionChat 对话 → TTS 播报
    │   └─ 多人 → 提示多人场景（本期不分离）
    └─ 未命中 → 返回 [待机]

所有 LLM/工具决策均通过 v2.0 智能体 (FusionChat) 完成。
"""

from __future__ import annotations

import os
import sys
import time
import threading
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voice.capture import MicCapture, TARGET_RATE, BLOCK_MS
from voice.vad import create_vad, EnergyVAD
from voice.kws import create_spotter
from voice.asr import create_asr
from voice.voiceprint import create_extractor, VoiceprintDB, VERIFY_THRESHOLD, PENDING_THRESHOLD
from voice.tracker import WakeTracker, MultiSpeakerDetector


class VoiceService:
    """声纹语音交互服务"""

    def __init__(
        self,
        chat=None,
        model_dir: str = "models/sherpa-kws",
        data_dir: str = "data/voice",
        wake_window: float = 15.0,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        """
        Args:
            chat: FusionChat 实例（可选，无则只识别不对话）
            model_dir: sherpa-onnx KWS 模型目录
            data_dir: 声纹数据目录
            wake_window: 唤醒后跟踪窗口（秒）
            on_event: 事件回调 (event_name, payload) — 用于推送 Web companion
        """
        self.chat = chat
        self.on_event = on_event or (lambda n, p: None)

        # 子系统（懒加载/容错）
        self.vad = create_vad()
        self.spotter = create_spotter(model_dir=model_dir)
        self.asr = create_asr()
        self.extractor = create_extractor()
        self.db = VoiceprintDB(data_dir=data_dir)
        self.tracker = WakeTracker(window_seconds=wake_window)
        self.multi_detector = MultiSpeakerDetector()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state = "standby"  # standby | listening | processing
        self._audio_buf: list = []
        self._last_voice_ts = 0.0

        caps = {
            "vad": type(self.vad).__name__,
            "kws": "sherpa-onnx" if self.spotter else "unavailable",
            "asr": "SenseVoiceSmall" if self.asr else "unavailable",
            "voiceprint": "CAM++" if self.extractor else "unavailable",
        }
        print(f"[VoiceService] 能力: {caps}")

    # ── 生命周期 ──

    def start(self) -> bool:
        if self._running:
            return True
        self.mic = MicCapture(on_block=self._on_audio_block)
        if not self.mic.start():
            print("[VoiceService] 麦克风启动失败")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        print(f"[VoiceService] 已启动 (采集模式: {self.mic.mode})")
        return True

    def stop(self):
        self._running = False
        if hasattr(self, 'mic'):
            self.mic.stop()
        if self._thread:
            self._thread.join(timeout=2)

    # ── 音频处理 ──

    def _on_audio_block(self, block: np.ndarray):
        """30ms 音频块回调：持续累积语音，长停顿后一次性处理"""
        energy = float(np.sqrt(np.mean(block ** 2)))
        now = time.time()
        if not hasattr(self, '_last_level_ts'): self._last_level_ts = 0
        if now - self._last_level_ts > 2.0:
            level = min(10, int(energy * 200))
            bar = "█" * level + "░" * (10 - level)
            tag = "🔊 说话中" if self._audio_buf else ("🔊" if energy > 0.02 else "...")
            print(f"\r  🎤 [{bar}] {tag}", end="", flush=True)
            self._last_level_ts = now

        is_speech = energy > 0.003  # 极低阈值，不漏过任何声音
        if is_speech:
            self._audio_buf.append(block.copy())
            self._last_voice_ts = now
        elif self._audio_buf:
            # 静音超过 1.5s → 处理累积的完整句子
            if now - self._last_voice_ts > 1.5:
                seg = np.concatenate(self._audio_buf)
                self._audio_buf = []
                if self._state == "standby" and len(seg) > TARGET_RATE * 0.5:
                    threading.Thread(target=self._process_segment, args=(seg,), daemon=True).start()

    def _main_loop(self):
        """主循环：超时兜底处理长时间累积的语音"""
        while self._running:
            if self._audio_buf and time.time() - self._last_voice_ts > 4.0:
                seg = np.concatenate(self._audio_buf)
                self._audio_buf = []
                if self._state == "standby" and len(seg) > TARGET_RATE * 0.5:
                    threading.Thread(target=self._process_segment, args=(seg,), daemon=True).start()
            time.sleep(0.2)

    # ── 核心流程 ──

    def _process_segment(self, wav: np.ndarray):
        """处理一段语音：能量/跟踪窗口唤醒 → ASR → 声纹 → 对话"""
        duration = len(wav) / TARGET_RATE
        energy = float(np.sqrt(np.mean(wav ** 2)))
        print(f"\n  [DEBUG] 语音段: {duration:.1f}s, 能量={energy:.4f}")

        # 接受任意长度的高能量语音（说话中断大于1秒才切段）
        if duration < 0.3 and energy < 0.01:
            print(f"  [DEBUG] 太短且能量低，跳过")
            return

        # 1. 唤醒检测（跟踪窗口内免唤醒，否则能量唤醒）
        woke = False
        if self.tracker.is_within_window():
            woke = True
        elif energy > 0.005:
            woke = True
            self._emit("wake", {"keyword": "能量唤醒"})

        if not woke:
            print(f"  [DEBUG] 未唤醒 (跟踪窗口={self.tracker.is_within_window()})")
            return

        print(f"  [DEBUG] ✅ 已唤醒，开始 ASR...")
        self._state = "processing"
        self._emit("listening", {})

        # 2. ASR 转写
        if not self.asr:
            print(f"  [DEBUG] ❌ ASR 未加载")
            self._state = "standby"
            return
        text, dur = self.asr.transcribe(wav)
        print(f"  [DEBUG] ASR 结果: '{text}' ({dur:.1f}s)")
        if not text:
            self._state = "standby"
            print(f"  [DEBUG] ASR 返回空，跳过")
            return
        self._emit("asr", {"text": text, "duration": dur})

        # 3. 声纹识别
        identity = self._identify_speaker(wav)

        # 4. 多人检测（滑窗，本期仅提示）
        if self.extractor and len(wav) > TARGET_RATE * 3:
            try:
                if self.multi_detector.detect(wav, self.extractor):
                    self._emit("multi_speaker", {"text": text})
            except Exception:
                pass

        # 5. 对话（通过 v2.0 智能体决策）
        if self.chat:
            user_name = identity.get("name") if identity else None
            prompt = text if not user_name else f"[说话人: {user_name}] {text}"
            try:
                reply = self.chat.chat(prompt)
            except Exception as e:
                reply = f"抱歉，处理时出错了: {e}"
            self._emit("reply", {"text": text, "reply": reply, "user": user_name})

            # 身份 → USER.md / Memos 个性化注入
            self._inject_identity(identity, text)

            # 6. TTS 播报（独立线程，不阻塞语音管道）
            if hasattr(self.chat, 'hw_exe') and self.chat.hw.speaker:
                reply_text = reply[:100]
                threading.Thread(
                    target=lambda: self.chat.hw_exe.speak(reply_text),
                    daemon=True
                ).start()

        self.tracker.refresh()
        self._state = "standby"

    def _inject_identity(self, identity: Optional[Dict[str, Any]], text: str) -> None:
        """身份 → USER.md / Memos 个性化注入

        - 识别出已知用户：将其交互写入 Memos（带用户名标签）
        - 识别出新用户：创建基础 profile（下次交互生效）
        """
        if not self.chat or not getattr(self.chat, 'memos', None):
            return
        try:
            if identity:
                name = identity.get("name", "未知")
                self.chat.memos.create(
                    content=f"[声纹:{name}] {text[:150]}",
                    tags=["声纹", name],
                    category="voice_interaction",
                    source="voiceprint",
                )
            else:
                # 未识别用户：记录为新用户交互
                self.chat.memos.create(
                    content=f"[声纹:未识别] {text[:150]}",
                    tags=["声纹", "未识别"],
                    category="voice_interaction",
                    source="voiceprint",
                )
        except Exception:
            pass

    def _identify_speaker(self, wav: np.ndarray) -> Optional[Dict[str, Any]]:
        """声纹识别：返回用户信息或 None"""
        if not self.extractor:
            return None
        emb = self.extractor.extract(wav)

        # 跟踪窗口内：与锚点比对
        if self.tracker.is_within_window() and self.tracker.check_continuation(emb):
            user = self.tracker.anchor_user
            self.tracker.refresh(emb)
            return user

        # 全库识别
        user, sim, status = self.db.identify(emb)
        if status == "matched":
            self.tracker.start(emb, user)
            self._emit("identified", {"user": user.get("name"), "similarity": sim})
            return user
        elif status == "pending":
            self.db.add_pending(emb, sim)
            user2, sim2, status2 = self.db.resolve_pending(emb)
            if status2 == "matched":
                self.tracker.start(emb, user2)
                self._emit("identified", {"user": user2.get("name"), "similarity": sim2})
                return user2
            self._emit("pending", {"similarity": sim})
            return None
        else:
            self._emit("new_speaker", {"similarity": sim})
            # 新说话人：暂不自动注册，等待 /enroll 指令或连续出现
            self.tracker.start(emb, None)
            return None

    # ── 管理接口 ──

    def enroll_current(self, name: str, role: str = "user") -> bool:
        """用当前跟踪锚点注册新用户"""
        if self.tracker.anchor_emb is None:
            return False
        uid = self.db.enroll(name, self.tracker.anchor_emb, role)
        user = {"id": uid, "name": name, "role": role}
        self.tracker.anchor_user = user
        self._emit("enrolled", {"user": name, "id": uid})
        return True

    def enroll_from_file(self, name: str, wav_path: str, role: str = "user") -> bool:
        """从音频文件注册用户"""
        if not self.extractor:
            return False
        import wave
        with wave.open(wav_path, 'rb') as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            if sr != TARGET_RATE:
                ratio = sr / TARGET_RATE
                idx = np.arange(0, len(data), ratio)
                idx = idx[idx < len(data)]
                data = np.interp(idx, np.arange(len(data)), data)
        emb = self.extractor.extract(data)
        uid = self.db.enroll(name, emb, role)
        self._emit("enrolled", {"user": name, "id": uid})
        return True

    def _emit(self, event: str, payload: dict):
        try:
            self.on_event(event, payload)
        except Exception:
            pass

    @property
    def state(self) -> str:
        return self._state
