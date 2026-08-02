#!/bin/bash
# =============================================================================
# KunPeng-Hermes Fusion — 一键启动（Web服务 + 声纹语音交互）
# =============================================================================
# 启动后: 直接对着麦克风说"小鲲小鲲"或"你好小鲲"开始对话
# 同时 Web 桌面宠和 App 可连接 http://<IP>:8765
# =============================================================================
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="/home/openEuler/agent_xia/venv_kunpeng"

if [ ! -d "$VENV_PATH" ]; then echo -e "${RED}虚拟环境不存在: $VENV_PATH${NC}"; exit 1; fi

# ── API 配置 ──
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$ANTHROPIC_AUTH_TOKEN" ]; then
    export ANTHROPIC_API_KEY="sk-kimi-QU0PI1vCYSttBnUjoUhBcv6ye0pQdPXcE1i4n9jMpxBxzWgzbOUbeNK5s7yEIrH8"
    export ANTHROPIC_BASE_URL="https://api.kimi.com/coding"
    export ANTHROPIC_MODEL="kimi-k2-0711-preview"
    export API_TYPE="anthropic"
    echo -e "${GREEN}[OK] Kimi API (默认)${NC}"
fi

cd "$SCRIPT_DIR"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src"

# ── 0. 清理旧进程 ──
pkill -9 -f arecord 2>/dev/null
pkill -9 -f "voice.service" 2>/dev/null
pkill -9 -f "web/server" 2>/dev/null
sleep 2

# ── 1. Web 服务（后台）──
echo -e "${GREEN}[OK] Web 服务启动: http://$(hostname -I | awk '{print $1}'):8765${NC}"
WEB_PORT=8765 nohup python3 src/web/server.py > /tmp/kunpeng_web.log 2>&1 &
WEB_PID=$!
trap "kill $WEB_PID 2>/dev/null; exit 0" INT TERM

sleep 3
if ! curl -s http://127.0.0.1:8765/health > /dev/null 2>&1; then
    echo -e "${RED}[ERROR] Web 服务启动失败，看日志: tail /tmp/kunpeng_web.log${NC}"
    exit 1
fi

# ── 2. 获取局域网 IP ──
LAN_IP=$(ip addr show | grep 'inet ' | grep -v '127\|172' | head -1 | awk '{print $2}' | cut -d/ -f1)
echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   🧠 KunPeng-Hermes Fusion v2.3             ║"
echo "  ╠══════════════════════════════════════════════╣"
echo "  ║   🎤 唤醒词: 小鲲小鲲 / 你好小鲲             ║"
echo "  ║   📱 App:   ws://${LAN_IP}:8765/ws     ║"
echo "  ║   🖥️  桌宠:  http://${LAN_IP}:8765/companion ║"
echo "  ║   🌐 官网:  http://${LAN_IP}:8765/site       ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""
echo "  直接对着麦克风说话即可唤醒，不需要打字！"
echo ""

# ── 3. 声纹语音交互（前台）──
python3 -m voice.service_main
