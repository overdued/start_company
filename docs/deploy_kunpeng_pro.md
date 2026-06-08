# KunPeng-Hermes Fusion — 香橙派鲲鹏Pro 部署指南

## 一、硬件准备

| 项目 | 要求 |
|------|------|
| 开发板 | OrangePi Kunpeng Pro (RK3588) |
| 系统 | openEuler 22.03 (LTS-SP4) |
| 内存 | ≥8GB（推荐 16GB）|
| 存储 | ≥10GB 可用（含模型和依赖）|
| 摄像头 | USB 摄像头（可选，视觉功能需要）|
| 网络 | 可访问 PyPI 和 API 服务 |

## 二、系统环境检查

```bash
# 确认系统版本
cat /etc/os-release
# PRETTY_NAME="openEuler 22.03 (LTS-SP4)"

# Python 版本
python3 --version
# Python 3.9.9

# 硬件检测
ls /dev/video*      # 摄像头
ls /dev/gpiochip*   # GPIO
ls /dev/i2c-*       # I2C
ls /dev/ttyS*       # UART
```

## 三、安装步骤

### 3.1 安装系统依赖

```bash
# 基础工具
sudo dnf install -y git wget curl gcc make python3-pip

# 摄像头
sudo dnf install -y fswebcam ffmpeg

# 硬件库
sudo dnf install -y libgpiod i2c-tools

# Python 3.11（Hermes 兼容）
sudo dnf install -y openssl-devel zlib-devel libffi-devel \
  sqlite-devel readline-devel ncurses-devel gdbm-devel xz-devel \
  tk-devel uuid-devel bzip2-devel

# 编译安装 Python 3.11.9
cd /tmp
wget https://mirrors.tuna.tsinghua.edu.cn/python/3.11.9/Python-3.11.9.tar.xz
tar xf Python-3.11.9.tar.xz && cd Python-3.11.9
./configure --prefix=/opt/python3.11 --with-ensurepip=install
make -j8 && sudo make install
```

### 3.2 克隆项目

```bash
mkdir -p /home/openEuler/agent_xia
cd /home/openEuler/agent_xia
git clone https://gitee.com/zhao-yuhang11234/start_company.git
```

### 3.3 创建虚拟环境

```bash
cd /home/openEuler/agent_xia/start_company
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3.4 安装 Python 依赖

```bash
pip install -r requirements.txt
pip install anthropic

# 如 RPi.GPIO 编译失败，跳过即可（RK3588 用 gpiod）
# pip install -r requirements.txt 2>&1 | grep -v "RPi.GPIO"
```

### 3.5 配置 API

```bash
# 方式一：Kimi API（推荐 — 支持视觉识别）
export ANTHROPIC_API_KEY="sk-kimi-your-key-here"
export ANTHROPIC_BASE_URL="https://api.kimi.com/coding"

# 方式二：DeepSeek API（纯文本，更快）
export ANTHROPIC_AUTH_TOKEN="sk-your-deepseek-key"
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_MODEL="deepseek-v4-pro"
```

### 3.6 启动

```bash
cd /home/openEuler/agent_xia/start_company

# 默认 Kimi（视觉识别）
bash start_fusion.sh

# 或 DeepSeek
API=deepseek bash start_fusion.sh
```

## 四、验证

启动后应看到：

```
============================================================
  🧠 KunPeng-Hermes 融合系统 v2.0
============================================================
  API : https://api.kimi.com/coding
  Model: kimi-k2-0711-preview
  📷: ✅  ⚡GPIO: ✅  🔌I2C: ✅
  🎯 Skill: 3  📝 记忆: 9 条
------------------------------------------------------------
  指令: /quit /clear /skills /memory /photo /photos /hw /status
============================================================
```

### 验证清单

- [ ] `📷: ✅` 摄像头可用
- [ ] `🎯 Skill: 3` Skill 系统加载正常
- [ ] 输入 `开灯` → `Skill 匹配: light-control`
- [ ] 输入 `/photo` → 真实拍照成功
- [ ] 输入 `照片里有什么` → AI 描述画面内容（Kimi API）
- [ ] 输入 `/skills` → 显示 3 个预设 Skill
- [ ] 输入 `/memory` → 显示环境记忆 + 用户画像

## 五、硬件配置（可选）

### 摄像头

```bash
# 确认设备
ls /dev/video*

# 授权
sudo chmod 666 /dev/video0 /dev/video1
```

### GPIO 测试

```bash
# 安装 gpiod
pip install gpiod

# 测试
python3 -c "import gpiod; print(gpiod.__version__)"
```

### I2C 测试

```bash
i2cdetect -l
# 应输出 i2c-0 ~ i2c-13
```

## 六、自定义

### 修改用户画像

编辑 `data/memories/USER.md`：

```markdown
§ 基本信息
- 姓名: 您的用户名
- 年龄: 75岁

§ 偏好
- 语言: 简短清晰

§ 用药
- 药品名: 时间
```

### 添加新 Skill

```python
from hermes_bridge.skill_manager import SkillManager
mgr = SkillManager(skill_dir="data/skills")
mgr.create_skill(
    name="window-control",
    description="控制窗帘。Trigger on: '开窗帘','关窗帘'",
    trigger_conditions=["开窗帘", "关窗帘"],
    category="smart-home",
    procedure="1. 解析目标\n2. PWM信号\n3. 确认位置"
)
```

### 修改系统提示词

编辑 `start_fusion.py` 中的 `FusionChat.SYSTEM` 变量。

## 七、常见问题

**Q: 启动报 `未设置 ANTHROPIC_API_KEY`**
A: 先 export API Key，或写入 `~/.bashrc`

**Q: 摄像头不工作**
A: `sudo chmod 666 /dev/video0` 并安装 fswebcam

**Q: Skill 匹配失败**
A: 确认 `data/skills/*/SKILL.md` 存在，检查 YAML 格式

**Q: DeepSeek 视觉失败**
A: DeepSeek v4-pro 是纯文本模型，使用 Kimi API 获得视觉能力

**Q: Python 3.9 兼容问题**
A: 融合模块（hermes_bridge）在 3.9 下运行正常。Hermes CLI 需 3.11

---

**文档版本**: v2.0  
**最后更新**: 2026-06-08  
**仓库**: https://gitee.com/zhao-yuhang11234/start_company
