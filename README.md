# KunPeng-Hermes Fusion

**自进化硬件控制智能体** — OrangePi Kunpeng Pro (RK3588) 上的养老家居 AI 助手

[![Gitee](https://img.shields.io/badge/Gitee-start__company-red)](https://gitee.com/zhao-yuhang11234/start_company)

---

## 什么是 KunPeng-Hermes Fusion？

将 **KunPeng-Cortex 硬件控制能力** 与 **Hermes Agent 自进化学习能力** 融合，打造"会学习的硬件控制智能体"。

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
| ⚡ **硬件控制** | KunPeng | GPIO/I2C/UART/PWM/摄像头/机械臂，<50ms 响应 |
| 💬 **情感关怀** | KunPeng | 中文情感计算 + 老年用户文化适配 |
| 🔒 **防编造机制** | Fusion | Frozen Snapshot 注入真实硬件状态，禁止模型虚构任何结果 |

## 硬件平台

**OrangePi Kunpeng Pro (RK3588)** — 8核 ARM64, 16GB RAM, 6-8 TOPS NPU

| 硬件 | 状态 | 设备 |
|------|------|------|
| 摄像头 | ✅ | /dev/video0, /dev/video1 |
| GPIO | ✅ | libgpiod |
| I2C | ✅ | 10 条总线 |
| UART | ✅ | 6 端口 |
| NPU | ❌ | 待驱动 |

## 快速开始

```bash
cd /home/openEuler/agent_xia/start_company
bash start_fusion.sh              # Kimi API（支持视觉识别）
API=deepseek bash start_fusion.sh  # DeepSeek（纯文本）
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
