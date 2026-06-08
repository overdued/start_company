#!/bin/bash
# =============================================================================
# KunPeng-Cortex 智能对话终端启动脚本
# =============================================================================
# 用法: ./start_chat.sh
# =============================================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
SRC_DIR="$PROJECT_DIR/src"
VENV_PATH="/home/openEuler/agent_xia/venv_kunpeng"

# 检查虚拟环境
if [ ! -d "$VENV_PATH" ]; then
    echo "[错误] 虚拟环境不存在: $VENV_PATH"
    echo "请先创建虚拟环境并安装依赖"
    exit 1
fi

# 检查 API Key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}[提示] 未设置 ANTHROPIC_API_KEY 环境变量${NC}"
    echo "请执行以下命令设置 API Key:"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    echo "  export ANTHROPIC_BASE_URL='https://api.kimi.com/coding'"
    exit 1
fi

echo -e "${GREEN}[OK] 虚拟环境检查通过${NC}"
echo -e "${GREEN}[OK] API Key 已配置${NC}"
echo ""

# 激活虚拟环境并启动对话
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$SRC_DIR"

cd "$PROJECT_DIR"
python3 kunpeng_chat.py
