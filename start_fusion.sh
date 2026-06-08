#!/bin/bash
# =============================================================================
# KunPeng-Hermes 融合系统 — 一键启动脚本
# =============================================================================
# 用法:
#   方式1: 使用 Kimi API（默认）
#     ./start_fusion.sh
#
#   方式2: 使用 DeepSeek API
#     API=deepseek ./start_fusion.sh
# =============================================================================

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="/home/openEuler/agent_xia/venv_kunpeng"

# 检查虚拟环境
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}[错误] 虚拟环境不存在: $VENV_PATH${NC}"
    exit 1
fi

# 选择 API
if [ "$API" = "deepseek" ]; then
    export ANTHROPIC_AUTH_TOKEN="sk-2118b50413354007a3d891e3c2357274"
    export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
    export ANTHROPIC_MODEL="deepseek-v4-pro"
    echo -e "${GREEN}[OK] 使用 DeepSeek API（纯文本，无视觉）${NC}"
else
    export ANTHROPIC_API_KEY="sk-kimi-PCTC17MehMX0uSyTuK0Qua7c5eoEWELgdazk2dnpN93qcqjtmlbv4Mxm4o7mIQE4"
    export ANTHROPIC_BASE_URL="https://api.kimi.com/coding"
    export ANTHROPIC_MODEL="kimi-k2-0711-preview"
    echo -e "${GREEN}[OK] 使用 Kimi API（支持图像识别）${NC}"
fi

echo -e "${GREEN}[OK] 虚拟环境: $VENV_PATH${NC}"
echo ""

# 进入项目目录，激活环境，启动融合终端
cd "$SCRIPT_DIR"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src"

python3 start_fusion.py
