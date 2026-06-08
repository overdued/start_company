---
name: fall-detection
description: >
  Use when detecting a user fall or when the user calls for emergency help.
  Trigger on: "我摔倒了", "救命", "help", "emergency", "fall down",
  "摔了", "跌倒了", "快来帮我", "出事了".
  Hardware required: orbbec_astra (depth camera), microphone.
  Priority: CRITICAL — overrides all other tasks.
version: 1.0.0
author: kunpeng-hermes
license: MIT
metadata:
  hermes:
    tags: [emergency, safety, elderly-care, critical]
    category: emergency
    priority: 100  # 最高优先级
    auto_created: false
    disable-model-invocation: true  # 必须用户显式触发或系统自动检测
---

# 跌倒检测与紧急救援流程

## When to Use
- 用户语音呼救（"我摔倒了"/"救命"等）
- 深度相机检测到人体姿态异常（突然倒地）
- 长时间无活动 + 异常姿态
- 用户手动按下紧急按钮

## CRITICAL: 优先级规则
- 此skill具有最高优先级（100），可以中断任何正在进行的任务
- 触发后立即执行，不需要用户二次确认
- 执行期间忽略所有其他非紧急输入

## Procedure

### Step 1: 紧急响应（< 1秒内）
```python
# 立即停止所有硬件运动
await emergency_stop_all()

# 播放安抚语音
await tts.speak("检测到紧急情况，正在处理，请保持冷静", 
                 emotion="calm", volume=1.0)

# 记录时间戳
incident_time = now()
```

### Step 2: 确认情况
```python
# 摄像头转向用户方向
camera.pan_to("last_known_user_position")

# 拍摄现场照片
capture = await camera.capture(save_path=f"emergency/{incident_time}.jpg")

# 深度分析人体姿态
pose = depth_camera.analyze_pose(capture)

if pose.is_fallen:
    severity = assess_severity(pose)
    # severity: minor / moderate / severe
else:
    # 可能是误报，但用户确实呼叫了
    severity = "unknown"

# 询问用户状态（如果用户还能回应）
response = await voice.listen(timeout=10.0)
if response:
    user_status = parse_emergency_response(response)
```

### Step 3: 通知紧急联系人
```python
# 获取紧急联系人（USER.md）
emergency_contact = memory_get("user")["emergency_contact"]

# 发送通知
await notify_emergency_contact(
    contact=emergency_contact,
    message=f"紧急情况！{user_name} 在 {location} 可能发生跌倒。"
            f"时间: {incident_time}。"
            f"严重程度: {severity}。"
            f"现场照片: {capture.path}",
    include_photo=True
)

# 如果配置了多个联系人，依次通知
for contact in emergency_contacts:
    await notify(contact, message)
    await asyncio.sleep(2)  # 避免同时发送
```

### Step 4: 现场处理
```python
if severity in ["moderate", "severe", "unknown"]:
    # 保持陪伴，持续监测
    await tts.speak("我已经通知了您的家人，他们正在赶来。" +
                    "请不要乱动，我会一直陪着您。", 
                    emotion="caring")
    
    # 持续监测用户状态（每30秒）
    while True:
        await asyncio.sleep(30)
        new_capture = await camera.capture()
        new_pose = depth_camera.analyze_pose(new_capture)
        
        if not new_pose.is_fallen:
            # 用户可能自己起来了
            await tts.speak("您站起来了吗？感觉怎么样？")
            response = await voice.listen(timeout=10.0)
            if "没事" in response or "好了" in response:
                await tts.speak("太好了，如果还有不舒服请及时就医。")
                break
        
        # 检查是否超过10分钟未恢复
        if (now() - incident_time) > 600:
            await notify_emergency_contact(
                contact=emergency_contacts[0],
                message=f"提醒：{user_name} 的紧急情况已持续10分钟，请确认是否已处理。"
            )
```

### Step 5: 事后记录
```python
# 记录到MEMORY.md
memory_add("memory", 
    f"跌倒事件: {incident_time}, 位置: {location}, " +
    f"严重程度: {severity}, 处理结果: {result}",
    category="emergency")

# 如果位置是新发现的危险区域
if is_new_risk_area(location):
    memory_add("memory",
        f"⚠️ 跌倒风险区域: {location} — 用户曾在此跌倒",
        category="risk")
    
    # 建议安全措施
    await tts.speak("我发现这里容易滑倒，建议铺上防滑垫。")
```

## Pitfalls

### 误报处理
- **症状**: 用户正常蹲下或坐下被误判为跌倒
- **修复**:
  1. 结合语音确认：询问用户是否摔倒
  2. 姿态连续性检查：摔倒前是否有失衡动作
  3. 深度变化速率：正常坐下 vs 突然倒地速度不同
  4. 误报后记录到memory，避免同类误报

### 网络通知失败
- **症状**: 无法发送通知给紧急联系人
- **修复**:
  1. 使用备用通信方式（短信、电话）
  2. 本地保存紧急记录
  3. 尝试连接到邻居WiFi
  4. 如果都失败，提高语音音量持续呼救

### 用户无法回应
- **症状**: 用户意识不清或无法说话
- **修复**:
  1. 立即通知所有紧急联系人
  2. 持续监测生命体征（如果有可穿戴设备）
  3. 如果配置了120/911自动拨打，执行
  4. 不要移动用户（除非有进一步危险）

## Verification
- [ ] 紧急停止在1秒内执行
- [ ] 现场照片已拍摄并保存
- [ ] 紧急联系人已通知（至少1人）
- [ ] 用户状态已确认（或正在持续监测）
- [ ] 事件已记录到MEMORY.md

## Related Skills
- `emergency/emergency-stop` — 通用紧急停止
- `emergency/health-monitor` — 健康监测
- `hardware/medicine-fetch` — 取药
