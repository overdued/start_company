#!/bin/bash
# 麦克风→扬声器 回声测试
DUR=${1:-4}
WAV=/tmp/echo_test.wav

pkill -9 arecord 2>/dev/null; sleep 0.5

echo "🎤 请对着麦克风说话 (${DUR}秒)..."
arecord -D plughw:1,0 -d "$DUR" -f S16_LE -r 16000 -c 1 "$WAV" 2>/dev/null
SIZE=$(stat -c%s "$WAV" 2>/dev/null)

echo "文件: ${SIZE} bytes"
# 用 ffprobe 检查音频内容
source ~/agent_xia/venv_kunpeng/bin/activate 2>/dev/null
python3 -c "
import wave, numpy as np
w = wave.open('$WAV', 'rb')
data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
energy = float(np.sqrt(np.mean(data.astype(np.float32)**2)))
peak = np.max(np.abs(data))
dur = len(data)/w.getframerate()
print(f'时长:{dur:.1f}s 峰值:{peak} 能量:{energy:.1f}')
if peak < 100:
    print('❌ 录音几乎是静音！检查麦克风是否静音或插错了孔')
elif energy < 50:
    print('⚠️ 音量很低，可能听不清')
else:
    print('✅ 录音正常')
"

echo "🔊 播放中 (用 ffplay 保底)..."
ffplay -nodisp -autoexit -loglevel quiet "$WAV" 2>/dev/null
echo "✅ 完成"
echo ""
echo "如果只听到滴一声 → 录音内容就是静音，请检查:"
echo "  1. 麦克风是否插在 USB 口且被系统识别"
echo "  2. 说话时音量条是否有反应"
