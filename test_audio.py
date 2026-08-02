#!/usr/bin/env python3
"""麦克风→扬声器回声测试"""
import subprocess, sys, os

print("=" * 50)
print("  🎤 麦克风 → 🔊 扬声器 回声测试")
print("=" * 50)
print("  录音 3 秒后自动播放...")
print()

duration = 3
wav_file = "/tmp/echo_test.wav"

# 1. 录音
print(f"🎤 录音中 ({duration}秒)... 请对着麦克风说话！")
r = subprocess.run(
    ["arecord", "-D", "plughw:0,0", "-d", str(duration),
     "-f", "S16_LE", "-r", "16000", "-c", "1", wav_file],
    capture_output=True, text=True, timeout=10
)
if r.returncode != 0 or not os.path.exists(wav_file):
    print(f"❌ 录音失败: {r.stderr.strip()}")
    sys.exit(1)

size = os.path.getsize(wav_file)
print(f"✅ 录制完成: {size} bytes")

# 2. 播放
print(f"🔊 播放中...")
r2 = subprocess.run(
    ["aplay", "-q", wav_file],
    capture_output=True, text=True, timeout=10
)
if r2.returncode != 0:
    r2 = subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_file],
        capture_output=True, text=True, timeout=10
    )

if r2.returncode == 0:
    print("✅ 播放完成！你应该听到了刚才说的话。")
else:
    print(f"⚠️ 播放异常: {r2.stderr.strip()[:100]}")

print()
print("  如果听到了自己的声音 → 麦克风和扬声器都正常")
print("  如果没听到 → 扬声器可能有问题")
print("  如果录音是 44 bytes → 麦克风没插好或静音")
