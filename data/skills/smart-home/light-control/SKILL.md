---
name: light-control
description: >
  Use when the user asks to control lights: turn on/off, dim, change color, set brightness.
  Trigger on: "开灯", "关灯", "亮一点", "暗一点", "灯", "light",
  "turn on the light", "dim", "brightness", "color".
  Hardware required: gpio (relay module) or UART (smart bulb).
  This is the most frequently used smart home skill.
version: 1.0.0
author: kunpeng-hermes
license: MIT
metadata:
  hermes:
    tags: [smart-home, light, gpio, uart, elderly-care]
    category: smart-home
    auto_created: false
    requires_toolsets: [gpio]
---

# 灯光控制标准流程

## When to Use
当用户要求控制灯光时触发，包括开关、亮度调节、颜色变化等。

## Slot Extraction
从用户指令中提取以下参数：
- **location**: 哪个房间的灯（客厅/卧室/厨房/...）
- **action**: 操作（开/关/调亮/调暗/变色）
- **brightness**: 亮度级别（0-100%，可选）
- **color**: 颜色（可选）

## Procedure

### 解析用户意图
```python
# 示例解析
"把客厅的灯开一下" → location=客厅, action=开
"卧室灯暗一点" → location=卧室, action=调暗
"关灯" → location=当前房间, action=关
```

### 获取灯光配置
```python
# 从MEMORY.md读取灯光映射
light_map = memory_search("灯光", store="memory")
# 格式: 客厅主灯 → GPIO_12
#       客厅氛围灯 → GPIO_13
#       卧室灯 → GPIO_14

gpio_pin = light_map[location]["pin"]
current_state = gpio_read(gpio_pin)
```

### 执行操作
```python
if action == "开":
    gpio_write(gpio_pin, 1)
    # 如果是PWM调光灯，设置默认亮度
    if light_map[location].get("pwm"):
        pwm_set(light_map[location]["pwm_channel"], 70.0)  # 默认70%
    await tts.speak(f"{location}的灯已打开")

elif action == "关":
    gpio_write(gpio_pin, 0)
    await tts.speak(f"{location}的灯已关闭")

elif action in ["调亮", " brighter"]:
    current = pwm_get(light_map[location]["pwm_channel"])
    new_brightness = min(current + 20, 100)
    pwm_set(light_map[location]["pwm_channel"], new_brightness)
    await tts.speak(f"{location}的灯已调亮到{new_brightness}%")

elif action in ["调暗", "dimmer"]:
    current = pwm_get(light_map[location]["pwm_channel"])
    new_brightness = max(current - 20, 10)  # 最低10%
    pwm_set(light_map[location]["pwm_channel"], new_brightness)
    await tts.speak(f"{location}的灯已调暗到{new_brightness}%")
```

### 情景模式
```python
# 如果用户说"睡觉"或"晚安"
if intent in ["sleep", "goodnight"]:
    # 关闭所有灯，保留卧室夜灯
    for room, config in light_map.items():
        if room == "卧室" and config.get("night_mode"):
            pwm_set(config["pwm_channel"], 10.0)  # 夜灯10%
        else:
            gpio_write(config["pin"], 0)
    await tts.speak("晚安，卧室留了小夜灯", emotion="gentle")

# 如果用户说"起床"或"早上好"
if intent in ["wake_up", "good_morning"]:
    # 渐亮卧室灯
    for brightness in range(10, 71, 5):
        pwm_set(light_map["卧室"]["pwm_channel"], brightness)
        await asyncio.sleep(0.5)  # 每0.5秒增加5%
    await tts.speak("早上好，灯已经慢慢亮起来了", emotion="gentle")
```

## Pitfalls
- **GPIO反转**: 有些继电器模块是低电平触发（写0=开），确认硬件接线
- **PWM频率**: LED灯建议1kHz，伺服电机50Hz，不要混用
- **过载保护**: 继电器最大电流10A，不要接大功率设备

## Hardware Parameters
| 参数 | 默认值 | 说明 |
|------|--------|------|
| pwm_freq | 1000 | LED调光频率(Hz) |
| default_brightness | 70 | 默认亮度(%) |
| night_brightness | 10 | 夜灯亮度(%) |
| fade_step | 5 | 渐变步长(%) |
| fade_delay | 0.5 | 渐变间隔(秒) |

## Related Skills
- `smart-home/sensor-monitor` — 传感器监测
- `smart-home/scene-mode` — 情景模式
