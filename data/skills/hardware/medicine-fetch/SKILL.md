---
name: medicine-fetch
description: >
  Use when the user asks to fetch medicine from a table, cabinet, or other surface.
  Trigger on: "拿药", "取药", "把药拿过来", "帮我拿药", "取一下药",
  "medicine", "pills", "drugs", "fetch medicine", "get medicine".
  Hardware required: dofbot_arm, orbbec_astra (or camera), hc_sr04 (optional).
  This is a high-frequency task for elderly care scenarios.
version: 1.0.0
author: kunpeng-hermes
license: MIT
metadata:
  hermes:
    tags: [hardware, arm, medicine, elderly-care]
    category: hardware
    auto_created: false
    requires_toolsets: [gpio, i2c, uart]
    hardware:
      - dofbot_arm
      - orbbec_astra
      - hc_sr04
---

# 取药任务标准流程

## When to Use
当用户要求从桌上、抽屉、柜子或其他平面取药时触发。这是居家养老场景中的高频任务。

## Before Starting
1. 确认机械臂已上电且处于home位置
2. 确认摄像头正常工作
3. 确认目标药品在家庭布局中有记录（MEMORY.md）
4. 如果不确定药品位置，先询问用户

## Procedure

### Step 1: 目标识别与定位
```python
# 捕获当前画面
capture = camera.capture()

# 使用YOLO检测药品盒
results = yolo_detect(capture, target_classes=["medicine_box", "bottle", "pill_container"])

if not results:
    # fallback: 询问用户
    ask_user("请问药放在哪里？请指向药盒")
    # 等待用户指示后重新检测
    
# 获取3D位置（深度相机）
if depth_camera_available:
    position = get_3d_position(results[0].bbox, depth_frame)
else:
    # fallback: 超声波辅助定位
    position = estimate_position(results[0].bbox, camera_params)
```

### Step 2: 路径规划与导航
```python
# 如果目标距离 > 50cm，先导航
if position.distance > 0.5:
    # 使用超声波扫描障碍物
    obstacles = hc_sr04.scan(full_sweep=True)
    if obstacles:
        path = plan_avoidance_path(position, obstacles)
    else:
        path = direct_path(position)
    
    # 执行导航
    await motor.navigate(path, speed=0.3)
```

### Step 3: 机械臂抓取
```python
# 移动到预抓取位置（上方10cm）
pre_grasp = [position.x, position.y, position.z + 0.10]
await arm.move_to(pre_grasp, speed=0.3)

# 缓慢下降
await arm.move_down(0.10, speed=0.1)

# 打开夹爪
await arm.gripper.open()

# 微调位置（视觉反馈）
fine_tune = visual_servo_adjustment(capture, target_center)
if fine_tune.magnitude > 0.02:
    await arm.move_relative(fine_tune, speed=0.05)

# 闭合夹爪（适度力度）
await arm.gripper.close(force=0.6)

# 验证抓取（轻提上拉测试）
await arm.move_up(0.03, speed=0.1)
is_grasped = check_grasp_stability()
if not is_grasped:
    # 重试：增加力度
    await arm.gripper.close(force=0.8)
    retry_count += 1
    if retry_count > 3:
        return fail("抓取失败，请检查物品位置")
```

### Step 4: 提取并送达
```python
# 提升到安全高度
await arm.move_up(0.15, speed=0.2)

# 移动到递送位置（用户面前，预设位置）
delivery_pos = get_delivery_position()
await arm.move_to(delivery_pos, speed=0.3)

# 缓慢下降
await arm.move_down(0.05, speed=0.1)

# 松开夹爪
await arm.gripper.open()

# 撤回
await arm.move_up(0.10, speed=0.2)
await arm.move_to("home", speed=0.3)
```

### Step 5: 确认与记录
```python
# 语音提示
await tts.speak("药已经拿过来了，请慢用", emotion="caring")

# 记录到MEMORY.md（如药品位置有变化）
if position_changed:
    memory_add("memory", f"药品新位置: {target} @ {position}", category="layout")

# 如果是定期用药，提醒
if is_scheduled_medicine():
    await tts.speak("记得按时服药哦")
```

## Pitfalls

### 抓取失败
- **症状**: 夹爪闭合后物品滑落
- **修复**: 
  1. 增加夹爪力度: `gripper.close(force=0.8)`
  2. 调整抓取角度: 从侧面改为从上方
  3. 检查物品是否过重（>500g需双手模式）
  4. 超过3次失败 → 通知用户手动操作

### 深度测量失败
- **症状**: 深度相机无法获取有效深度
- **修复**:
  1. fallback到超声波测距
  2. 使用已知参考物（如A4纸大小）估算距离
  3. 从侧面多角度拍摄获取深度

### 药品识别失败
- **症状**: YOLO未检测到药品盒
- **修复**:
  1. 检查是否被遮挡
  2. 扩大搜索区域
  3. 询问用户确认位置
  4. fallback: 扫描整个桌面，让用户语音确认

### 障碍物阻挡
- **症状**: 导航路径上有障碍
- **修复**:
  1. 超声波扫描绕行路径
  2. 如果绕不过去，请求用户清理路径
  3. 紧急情况下可以轻推小障碍物（<200g）

### 安全区域外
- **症状**: 目标位置超出预设安全区域
- **修复**:
  1. 拒绝操作，通知用户
  2. 如果是新安全区域需求，更新配置并重新校准

## Hardware Parameters
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| arm_speed | 0.3 | 0.1-0.5 | 机械臂移动速度 |
| gripper_force | 0.6 | 0.3-0.9 | 夹爪力度 |
| fine_tune_speed | 0.05 | 0.02-0.1 | 微调速度 |
| max_retry | 3 | 1-5 | 最大重试次数 |
| safety_zone_x | [-200, 200] | mm | X轴安全范围 |
| safety_zone_y | [100, 350] | mm | Y轴安全范围 |
| safety_zone_z | [0, 200] | mm | Z轴安全范围 |
| max_weight | 500 | g | 最大抓取重量 |

## Verification
- [ ] 摄像头画面清晰，能识别目标
- [ ] 深度测量有效（或fallback成功）
- [ ] 目标在安全区域内
- [ ] 夹爪成功闭合，稳定性测试通过
- [ ] 递送到位，物品放置平稳
- [ ] 机械臂安全回到home位置

## Related Skills
- `hardware/object-fetch` — 通用取物
- `hardware/emergency-stop` — 紧急停止
- `smart-home/light-control` — 灯光控制
