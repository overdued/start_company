#!/bin/bash
# ============================================================
# 小鲲 Web HMI 启动脚本
# ============================================================
# 启动前后端分离的 Web HMI 系统
# 前端: http://localhost:8765 (hengxiang-hmi.html)
# 后端: WebSocket ws://localhost:8765/ws
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 环境检查 ──
echo "============================================================"
echo "  🐟 小鲲 Web HMI 系统"
echo "============================================================"

# 检查 Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "[ERROR] 未找到 Python，请先安装 Python 3.9+"
    exit 1
fi

echo "  Python: $($PYTHON --version)"

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    echo "  虚拟环境: venv/"
    source venv/bin/activate
elif [ -d "venv_kunpeng" ]; then
    echo "  虚拟环境: venv_kunpeng/"
    source venv_kunpeng/bin/activate
fi

# 检查必要依赖
$PYTHON -c "import fastapi" 2>/dev/null || {
    echo "[INFO] 安装 FastAPI 和 uvicorn..."
    pip install fastapi uvicorn[standard]
}
$PYTHON -c "import dotenv" 2>/dev/null || {
    echo "[INFO] 安装 python-dotenv..."
    pip install python-dotenv
}
$PYTHON -c "import openai" 2>/dev/null || {
    echo "[INFO] 安装 openai (Qwen/DashScope 兼容)..."
    pip install openai
}

# API 配置 — 优先从 .env 文件加载
if [ -f ".env" ]; then
    echo "  📄 加载 .env 配置..."
    set -a  # 自动 export 所有变量
    # 跳过注释和空行，source 剩余内容
    grep -v '^\s*#' .env | grep -v '^\s*$' > /tmp/_kunpeng_env_$$ 2>/dev/null || true
    source /tmp/_kunpeng_env_$$ 2>/dev/null || true
    rm -f /tmp/_kunpeng_env_$$
    set +a
    echo "  ✅ MODEL: ${ANTHROPIC_MODEL:-not set}"
    echo "  ✅ URL  : ${ANTHROPIC_BASE_URL:-not set}"
    _key="${ANTHROPIC_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}"
    if [ -n "$_key" ]; then
        echo "  ✅ KEY  : ${_key:0:6}****${_key: -4}"
    else
        echo "  ⚠ KEY  : 未设置"
    fi
elif [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$ANTHROPIC_AUTH_TOKEN" ]; then
    echo ""
    echo "  ⚠ 未找到 .env 文件，也未设置 API Key"
    echo "  设置方法: 复制 .env.example 为 .env 并填入 API Key"
    echo "  或: export ANTHROPIC_AUTH_TOKEN='sk-...'"
    echo ""
fi

# 端口配置
WEB_PORT=${WEB_PORT:-8765}

echo "────────────────────────────────────────────────────────────"
echo "  🌐 Web HMI: http://localhost:$WEB_PORT"
echo "  📡 WebSocket: ws://localhost:$WEB_PORT/ws"
echo "────────────────────────────────────────────────────────────"
echo "  按 Ctrl+C 停止服务"
echo "============================================================"

# 设置 PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export WEB_PORT=$WEB_PORT

# 启动 Web 服务器
cd "$SCRIPT_DIR"
exec $PYTHON src/web/server.py
