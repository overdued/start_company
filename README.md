# KunPeng-Hermes Fusion v2.2

**自进化硬件控制智能体** — OrangePi Kunpeng Pro (RK3588) 上的养老家居 AI 助手

[![Gitee](https://img.shields.io/badge/Gitee-start__company-red)](https://gitee.com/zhao-yuhang11234/start_company)
[![Version](https://img.shields.io/badge/version-v2.2-blue)]()
[![Tests](https://img.shields.io/badge/tests-14%2F14-brightgreen)]()

---

## 什么是 KunPeng-Hermes Fusion？

将 **KunPeng-Cortex 硬件控制能力** 与 **Hermes Agent 自进化学习能力** 融合，打造"会学习的硬件控制智能体"。v2.2 新增**声纹识别 + 全双工语音交互 + Web 桌宠**。

```
每次完成一次硬件任务 → 自动学习 → 生成可复用 Skill → 下次直接复用（2-3x 更快）
```

## 核心特性

| 特性 | 来源 | 说明 |
|------|------|------|
| 🎯 **Skill 自动创建** | Hermes | 复杂硬件任务后自动生成 SKILL.md，下次复用 2-3x 加速 |
| 🧠 **三层记忆系统** | Hermes | MEMORY.md（环境）+ USER.md（用户画像）+ SQLite FTS5（会话搜索） |
| 🔄 **自进化引擎** | Hermes | 每15任务自检，成功率<70%自动重写 Skill，Token 降低60% |
| 📷 **真实视觉识别** | Kimi k2 | 真实摄像头拍照 + AI 图像描述（非编造） |
| 🎤 **声纹识别** | CAM++ | 192维声纹向量，识别"谁在说话"，新用户自动待定注册 |
| 🗣️ **语音交互** | sherpa-onnx + SenseVoice | 唤醒词"小鲲小鲲"/"你好小鲲"（拼音模糊匹配），全链路本地 ASR |
| 📋 **Memos 智能笔记** | Fusion | 自动分类/打标签/用户画像，客户需求追踪 |
| 🖥️ **Web 桌宠** | FastAPI | 隐私安全 HMI，浏览器访问桌面宠物，工具调用（提醒/设备/小车/机械臂） |
| ⚡ **硬件控制** | KunPeng | GPIO/I2C/UART/PWM/摄像头/机械臂，<50ms 响应 |
| 💬 **情感关怀** | KunPeng | 中文情感计算 + 老年用户文化适配 |
| 🔒 **防编造机制** | Fusion | Frozen Snapshot 注入真实硬件状态，禁止模型虚构任何结果 |

## 硬件平台

**OrangePi Kunpeng Pro (RK3588)** — 8核 ARM64, 16GB RAM, 6-8 TOPS NPU

| 硬件 | 状态 | 设备 |
|------|------|------|
| 摄像头 | ✅ | /dev/video0 (Aveo SP2820W) |
| 四通道麦克风 | ✅ | HK MIC USB-Audio (2ch@48k→16k mono) |
| USB 扬声器 | ✅ | hw:0,0 (TTS 语音播报) |
| GPIO | ✅ | libgpiod |
| I2C | ✅ | 10 条总线 |
| UART | ✅ | 6 端口 (含 FT2232×2) |
| NPU | ❌ | 待驱动 |

## 快速开始

### 语音交互（推荐）

```bash
git clone https://gitee.com/zhao-yuhang11234/start_company.git
cd start_company
./start_voice.sh
```

对着四通道麦克风说 **"小鲲小鲲"** 唤醒，然后说指令。系统会：
1. 声纹识别"你是谁"（注册过的用户自动识别）
2. ASR 转写你的指令
3. v2.0 智能体决策并回复
4. 扬声器 TTS 播报

新用户说几句话后输入 `/enroll 王大爷` 注册声纹。

### 对话终端

```bash
./start_fusion.sh              # Kimi API（支持视觉识别）
API=deepseek ./start_fusion.sh  # DeepSeek（纯文本）
```

### Web 桌宠

```bash
./start_web.sh
# 浏览器访问 http://<板子IP>:8765/companion
```

### 交互命令

| 命令 | 功能 |
|------|------|
| 自然语言输入 | 自动意图识别 + Skill 匹配 + AI 对话 |
| `/photo` | 真实拍照（fswebcam） |
| `/skills` | 查看已加载 Skill |
| `/memory` | 查看记忆快照 |
| `/hw` | 硬件状态 |
| `/status` | 系统统计 |
| `/clear` | 清空对话历史 |

### 对话示例

```
你 > 帮我拍张照片
[硬件] ✅ 照片已保存: /home/openEuler/Pictures/capture_20260608_172203.jpg
KunPeng > 好的，照片拍好了。

你 > 看看照片里有什么
KunPeng > 画面里是卧室，床底下有几双鞋，地面是浅灰色瓷砖，
         左边门边有只洞洞鞋，中间地上有一团白色纸巾...
         王大爷，要我帮您收拾一下吗？
```

## 项目结构

```
start_company/
├── start_fusion.sh          ← 一键启动（推荐）
├── start_fusion.py          ← 融合对话终端
├── kunpeng_chat.py          ← 原始对话终端
├── requirements.txt
├── config/                  ← 系统配置
├── src/
│   ├── hermes_bridge/       ← ★ 融合桥接层（新增）
│   │   ├── memory_tool.py   ← 三层记忆系统
│   │   ├── skill_manager.py ← Skill 自动创建
│   │   ├── session_search.py← FTS5 会话搜索
│   │   ├── evolution_engine.py ← 自进化引擎
│   │   └── __init__.py
│   ├── core/                ← 调度器/规划器/上下文
│   ├── engines/             ← Claude Code / OpenClaw 引擎
│   ├── hal/                 ← 硬件抽象层 (GPIO/I2C/UART)
│   ├── devices/             ← 设备驱动 (机械臂/传感器)
│   ├── interaction/         ← 语音/表情交互
│   └── utils/               ← 配置/日志/监控
├── data/
│   ├── memories/            ← MEMORY.md + USER.md
│   └── skills/              ← Skill 存储
└── docs/                    ← 部署文档
```

## 技术架构

```
用户输入（语音/文本/APP）
    ↓
┌─────────────────────────────────────────┐
│  决策层（Hermes Brain）                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ 三层记忆  │ │ Skill系统 │ │ 自进化    │ │
│  │ Frozen   │ │ 自动创建  │ │ 15任务自检│ │
│  │ Snapshot │ │ Progressive│ │ 成功率优化│ │
│  └──────────┘ └──────────┘ └──────────┘ │
├─────────────────────────────────────────┤
│  融合调度层（Orchestrator v2）            │
│  Skill匹配 → 意图识别 → 任务路由 → 执行  │
├─────────────────────────────────────────┤
│  硬件执行层（KunPeng-Cortex HAL）         │
│  GPIO │ I2C │ UART │ Camera │ 机械臂     │
└─────────────────────────────────────────┘
```

## 部署

详见 [docs/deploy_kunpeng_pro.md](docs/deploy_kunpeng_pro.md)

## 许可证

MIT License

---

**作者**: zhao-yuhang11234  
**仓库**: https://gitee.com/zhao-yuhang11234/start_company  
**硬件**: OrangePi Kunpeng Pro (RK3588) / openEuler 22.03
