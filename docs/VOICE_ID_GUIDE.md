# KunPeng 声纹识别与语音交互指南

> 基于 FunASR + CAM++ + sherpa-onnx 的全链路本地语音服务
> 部署平台: OrangePi Kunpeng Pro (RK3588) / openEuler 22.03

---

## 一、架构

```
HK MIC (4通道麦克风)
    ↓ 2ch@48kHz → mono@16kHz
┌──────────────┐
│  VAD 语音检测 │  FSMN-VAD / 能量阈值兜底
└──────┬───────┘
       ↓ 语音段
┌──────────────┐
│  KWS 唤醒词   │  sherpa-onnx Zipformer (3.3M)
│  小鲲小鲲     │  拼音模糊匹配 (肖坤/小框等)
│  你好小鲲     │
└──────┬───────┘
       ↓ 命中
┌──────────────┐
│  ASR 转写     │  SenseVoiceSmall (234M, RTF≈0.47)
└──────┬───────┘
       ↓ 文本
┌──────────────┐
│  声纹识别     │  CAM++ 192维嵌入 (7.2M, RTF≈0.13)
│  余弦≥0.35    │  SQLite3 + NumPy 注册库
└──────┬───────┘
       ↓ 身份
┌──────────────────────┐
│  FusionChat v2.0     │  LLM 决策 (Kimi/DeepSeek)
│  记忆/Skill/Memos    │  身份注入 → 个性化回复
└──────┬───────────────┘
       ↓ 回复
┌──────────────┐
│  TTS 播报     │  espeak-ng → USB 扬声器
└──────────────┘
```

## 二、快速开始

```bash
cd /home/openEuler/agent_xia/start_company
./start_voice.sh                # Kimi API（默认）
API=deepseek ./start_voice.sh   # DeepSeek API
NO_CHAT=1 ./start_voice.sh      # 仅声纹识别，不调 LLM
```

启动后对着麦克风说 **"小鲲小鲲"** 或 **"你好小鲲"** 唤醒，然后说出指令。

## 三、用户注册

### 方式一：对话中注册
唤醒后随意说几句话，系统提示 `🆕 新说话人`，然后输入：
```
/enroll 王大爷
```
最近说话人的声纹锚点将被注册。

### 方式二：从音频文件注册
```python
from voice.service import VoiceService
service = VoiceService()
service.enroll_from_file("王大爷", "path/to/voice.wav")
```

### 查看已注册用户
```
/users
```

## 四、声纹阈值调优

配置文件: `voice_config.yaml`

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `verify_threshold` | 0.35 | 身份验证阈值。调高减少误识（认亲），调低减少拒识 |
| `pending_threshold` | 0.25 | 待定区间下界。0.25~0.35 之间的声音进入待定池 |
| `enrollment_duration` | 3.0s | 注册所需最小音频长度 |

**注意**: espeak 等合成语音的相似度天然很高（>0.85 即使不同音高），真实人声的区分度大得多。建议用真实家庭成员声音校准阈值：
- 若家人常被误判为新用户 → 调低到 0.30
- 若陌生人被误识为家人 → 调高到 0.40

## 五、唤醒词定制

编辑 `models/sherpa-kws/keywords_raw.txt`（每行一个唤醒词），然后重新生成 tokens：

```bash
cd models/sherpa-kws
# 格式: 拼音 tokens @中文 (无声调拼音按 tokens.txt 拆分)
# 参考已有格式: x iǎo k ūn x iǎo k ūn @小鲲小鲲
```

当前唤醒词:
- `x iǎo k ūn x iǎo k ūn @小鲲小鲲`
- `n ǐ h ǎo x iǎo k ūn @你好小鲲`

## 六、API 参考

```python
from voice.service import VoiceService
from start_fusion import FusionChat

chat = FusionChat()
service = VoiceService(chat=chat, on_event=lambda e, p: print(e, p))
service.start()

# 状态查询
service.state                    # standby | listening | processing
service.db.count()               # 注册用户数
service.db.get_stats()           # 完整统计

# 管理
service.enroll_current("王大爷")  # 用锚点注册
service.db.list_users()          # 用户列表

# 停止
service.stop()
```

### 事件回调

| 事件 | 载荷 | 说明 |
|------|------|------|
| `wake` | `{keyword}` | 唤醒词命中 |
| `listening` | `{}` | 开始接收指令 |
| `asr` | `{text, duration}` | 转写完成 |
| `identified` | `{user, similarity}` | 身份确认 |
| `pending` | `{similarity}` | 待定（0.25~0.35） |
| `new_speaker` | `{similarity}` | 新说话人 |
| `multi_speaker` | `{text}` | 多人场景检测 |
| `reply` | `{text, reply, user}` | 智能体回复 |
| `enrolled` | `{user, id}` | 注册成功 |

## 七、隐私说明

- 所有声纹向量仅存本地 `data/voice/`（SQLite3 + .npy）
- 原始音频不落盘（仅内存处理）
- Web companion 只接收身份标签和语义事件，不接收音频
- 声纹数据可通过删除 `data/voice/` 目录完全清除

## 八、硬件参考

| 组件 | 路径 | 说明 |
|------|------|------|
| 四通道麦克风 | HK MIC (hw:0,0) | 2ch@48kHz 采集，软件降混 |
| USB 扬声器 | hw:0,0 (pcmC0D0p) | espeak-ng TTS 播报 |
| 摄像头 | /dev/video0 | 拍照/视觉识别 |

## 九、已知限制

1. **MossFormer2 人声分离未启用**：RK3588 CPU 算力不足以实时分离，本期仅做滑窗多人检测提示。多人场景建议分先后说话。
2. **espeak TTS 音质**：机械音较重，可后续替换 edge-tts（需网络）或本地 VITS 模型。
3. **KWS 对合成音敏感**：自然语音唤醒效果最佳；机械音（如手机播放）可能漏唤醒。
