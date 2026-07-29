#!/bin/bash
# =============================================================================
# KunPeng 声纹语音交互服务 — 启动脚本
# =============================================================================
# 用法:
#   ./start_voice.sh              # 启动声纹语音交互（Kimi API）
#   API=deepseek ./start_voice.sh # 使用 DeepSeek API
#   NO_CHAT=1 ./start_voice.sh    # 仅声纹识别，不调用 LLM 对话
# =============================================================================

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="/home/openEuler/agent_xia/venv_kunpeng"

if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}[错误] 虚拟环境不存在: $VENV_PATH${NC}"
    exit 1
fi

# API 配置
if [ "$API" = "deepseek" ]; then
    export ANTHROPIC_AUTH_TOKEN="sk-79826c6bff974b52bac5f0cdb3ec9cdf"
    export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
    export ANTHROPIC_MODEL="deepseek-v4-pro"
    echo -e "${GREEN}[OK] DeepSeek API${NC}"
else
    export ANTHROPIC_API_KEY="sk-kimi-QU0PI1vCYSttBnUjoUhBcv6ye0pQdPXcE1i4n9jMpxBxzWgzbOUbeNK5s7yEIrH8"
    export ANTHROPIC_BASE_URL="https://api.kimi.com/coding"
    export ANTHROPIC_MODEL="kimi-k2-0711-preview"
    export API_TYPE="anthropic"
    echo -e "${GREEN}[OK] Kimi API${NC}"
fi

cd "$SCRIPT_DIR"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src"

echo -e "${GREEN}[OK] 启动声纹语音服务...${NC}"
echo "  唤醒词: 小鲲小鲲 / 你好小鲲"
echo "  声纹库: $SCRIPT_DIR/data/voice/"
echo ""

python3 -m voice.service_main "$@"
