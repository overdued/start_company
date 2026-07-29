# 小鲲 Web HMI — 前后端分离架构

**居家养老 AI 助手** · 基于 FastAPI + WebSocket 的触屏 HMI 系统

## 架构概览

```
┌─────────────────────────────────────────┐
│  触摸屏（HDMI/MIPI 接 RK3588）          │
│  浏览器全屏打开 http://localhost:8765    │
├─────────────────────────────────────────┤
│                                         │
│  前端 HTML/CSS/JS（hengxiang-hmi.html） │
│  800×480 横屏 · 暖色主题 · 6 个页面     │
│                                         │
│         ↕ WebSocket JSON 协议           │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│  后端 Python（src/web/server.py）       │
│  FastAPI + Uvicorn                      │
│  ┌───────────┐ ┌──────────┐ ┌────────┐ │
│  │ FusionChat│ │ Hermes   │ │ HAL    │ │
│  │ AI 对话   │ │ 记忆/Skill│ │ GPIO   │ │
│  │ 情感计算  │ │ 自进化   │ │ I2C    │ │
│  └───────────┘ └──────────┘ │ Camera │ │
│                             └────────┘ │
└─────────────────────────────────────────┘
```

## 快速启动

### 1. 安装依赖

```bash
cd start_company
pip install fastapi uvicorn[standard] python-multipart
```

### 2. 配置 API（可选，不配则 AI 对话用回显模式）

```bash
# Kimi API（支持视觉识别）
export ANTHROPIC_API_KEY="sk-kimi-..."
export ANTHROPIC_BASE_URL="https://api.kimi.com/coding"

# 或 DeepSeek API（纯文本）
export ANTHROPIC_AUTH_TOKEN="sk-..."
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
```

### 3. 启动

```bash
# 方式一：启动脚本（推荐）
bash start_web.sh

# 方式二：直接运行
PYTHONPATH=src python src/web/server.py

# 自定义端口
WEB_PORT=9000 python src/web/server.py
```

### 4. 打开浏览器

访问 `http://localhost:8765` 即可看到 HMI 界面。

桌宠也可以单独作为桌面 surface 访问：

```text
http://localhost:8765/companion
```

该地址仍使用同一份 `hmi-ws.html`，只显示共享桌宠和陪伴状态，不加载完整 HMI 的聊天、提醒和设备界面。没有图形桌面壳时，可用 Chromium app 模式验证：

```bash
chromium --app=http://127.0.0.1:8765/companion
```

在嵌入式触屏上，可以用 Chromium 全屏模式：

```bash
chromium-browser --kiosk --noerrdialogs http://localhost:8765
```

## 页面说明

| 页面 | 功能 |
|------|------|
| 启动页 | 品牌动画 + 进度条加载 |
| 首页 | 问候语、环境数据（温度/湿度/天气）、今日摘要、快捷按钮 |
| 会话 | AI 对话（文字/语音），消息气泡实时滚动 |
| 提醒 | 今日提醒、用药管理（7天记录）、健康数据（血压/血糖图表）、SOS 紧急联系 |
| 家居 | 按房间分区控制设备（灯/摄像头/传感器）、摄像头预览、4种情景模式 |
| 设置 | 个人信息、设备设置（音量/亮度）、系统信息与硬件状态 |

## WebSocket 协议

所有消息为 JSON 格式：

### 前端 → 后端

```json
// 发送聊天消息
{"action": "chat.send", "data": {"text": "帮我开一下客厅的灯"}}

// 控制设备
{"action": "device.toggle", "data": {"device": "主灯"}}

// 拍照
{"action": "photo.take", "data": {}}

// 执行情景模式
{"action": "scene.execute", "data": {"scene": "起床模式"}}

// 获取提醒列表
{"action": "reminders.get", "data": {}}

// 获取健康数据
{"action": "health.get", "data": {}}

// 获取系统信息
{"action": "system.info", "data": {}}

// 紧急呼叫
{"action": "sos.call", "data": {"type": "son"}}
```

### 后端 → 前端

```json
// AI 回复
{"action": "chat.reply", "data": {"text": "好的，客厅灯已经帮您打开了", "hw_result": "客厅主灯已打开"}}

// 设备状态变更
{"action": "device.state", "data": {"device": "主灯", "on": true, "status": "已开启 · 70%"}}

// 拍照结果
{"action": "photo.result", "data": {"success": true, "path": "/home/openEuler/Pictures/capture_20260608.jpg"}}

// 情景模式结果
{"action": "scene.result", "data": {"scene": "起床模式", "success": true, "message": "起床模式已启动：卧室灯渐亮中..."}}

// 环境状态推送（每30秒自动广播）
{"action": "status.push", "data": {"temp": 26, "humidity": 60, "weather": "晴"}}
```

## 文件结构

```
start_company/
├── src/web/
│   ├── __init__.py        ← Web 模块
│   └── server.py          ← ★ FastAPI 后端服务（HTTP + WebSocket）
├── docs/
│   ├── hengxiang-hmi.html ← HMI 原型（纯前端，mock 数据）
│   ├── hmi-ws.html        ← ★ HMI 前端（对接 WebSocket）
│   ├── hmi.html           ← 早期完整 HMI 原型（16页面版）
│   └── loading-page.html  ← 启动加载页
├── start_web.sh           ← ★ Web HMI 启动脚本
├── start_fusion.py        ← 原始 CLI 对话终端（后端复用）
├── backup_*/              ← 改造前的文件备份
└── requirements.txt       ← Python 依赖（已更新）
```

## 与 CLI 模式的区别

| 特性 | CLI 模式 (start_fusion.py) | Web HMI (start_web.sh) |
|------|---------------------------|------------------------|
| 界面 | 终端文字 | 图形触屏 |
| 输入 | 键盘输入 | 触屏按钮 + 语音 |
| 部署 | SSH/终端 | 浏览器全屏 |
| 多用户 | 单用户 | 多客户端同时连接 |
| 状态推送 | 无 | 自动广播环境数据 |

## 硬件平台

**OrangePi Kunpeng Pro (RK3588)** — 8核 ARM64, 16GB RAM

| 硬件 | 状态 | 说明 |
|------|------|------|
| 摄像头 | ✅ | USB / 奥比中光深度相机 |
| GPIO | ✅ | libgpiod |
| I2C | ✅ | 10 条总线 |
| UART | ✅ | 6 端口 |
| NPU | ❌ | 待驱动 |

## 开发说明

### 后端扩展

在 `src/web/server.py` 中的 `_ACTION_HANDLERS` 字典注册新的 WebSocket action 处理器：

```python
async def handle_my_action(websocket, data):
    """处理自定义操作"""
    result = await do_something(data)
    await websocket.send_json({"action": "my.result", "data": result})

_ACTION_HANDLERS["my.action"] = handle_my_action
```

### 前端扩展

在 `docs/hmi-ws.html` 的 `handleServerMsg()` 函数中添加新的 action 分支：

```javascript
case 'my.result':
    updateMyUI(data);
    break;
```

---

**文档版本**: v1.0
**最后更新**: 2026-06-08
**作者**: zhao-yuhang11234
