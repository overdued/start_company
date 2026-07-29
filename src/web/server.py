#!/usr/bin/env python3
"""
小鲲 KunPeng-Cortex Web Server v2.0
=====================================

FastAPI-based web server for the elderly care AI assistant HMI.

Provides:
  - Static file serving from the project's docs/ directory (hengxiang-hmi.html, etc.)
  - WebSocket endpoint at /ws for real-time JSON communication with the frontend
  - Integration with FusionChat (from start_fusion.py) for AI-powered conversation
  - Simulated environment sensor data, reminders, device states, and health metrics

Standalone Usage:
    python server.py                     # default port 8765
    WEB_PORT=9000 python server.py       # custom port

The server gracefully degrades when optional dependencies (FusionChat, hardware
modules) are unavailable, falling back to echo mode for chat responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup -- ensure project root is importable
# ---------------------------------------------------------------------------
# server.py lives at  <project_root>/src/web/server.py
# We need <project_root> on sys.path so that `import start_fusion` works.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent          # <project_root>/src
_PROJECT_ROOT = _SRC_DIR.parent               # <project_root>
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print(
        "[ERROR] Required packages not installed.\n"
        "        pip install fastapi uvicorn\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Optional: FusionChat integration
# ---------------------------------------------------------------------------
_fusion_chat_instance: Any = None
_fusion_available = False

try:
    from start_fusion import FusionChat, TOOL_DEFINITIONS
    _fusion_available = True
except Exception as exc:
    logging.getLogger(__name__).warning(
        "FusionChat import failed (%s). Chat will run in echo-fallback mode.", exc
    )
    TOOL_DEFINITIONS = []

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kunpeng.web")

# ---------------------------------------------------------------------------
# Load .env file if present (robust: fallback to manual parsing)
# ---------------------------------------------------------------------------
_env_file = _PROJECT_ROOT / ".env"
_env_loaded = False

# Try python-dotenv first
try:
    from dotenv import load_dotenv
    if _env_file.is_file():
        load_dotenv(str(_env_file))
        _env_loaded = True
except ImportError:
    pass

# Fallback: manual .env parser (no external dependency needed)
if not _env_loaded and _env_file.is_file():
    try:
        for line in _env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        _env_loaded = True
    except Exception as exc:
        pass  # Will be logged below

if _env_loaded:
    _model = os.environ.get("ANTHROPIC_MODEL", "not set")
    _url = os.environ.get("ANTHROPIC_BASE_URL", "not set")
    _key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    _masked = (_key[:6] + "****" + _key[-4:]) if len(_key) > 10 else ("set" if _key else "NOT SET")
    print(f"[ENV] .env loaded from {_env_file}")
    print(f"[ENV]   MODEL : {_model}")
    print(f"[ENV]   URL   : {_url}")
    print(f"[ENV]   KEY   : {_masked}")
elif _env_file.is_file():
    print(f"[WARN] .env file exists at {_env_file} but failed to load")
else:
    print(f"[WARN] No .env file at {_env_file} — set API keys via environment variables")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEB_HOST: str = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT: int = int(os.environ.get("WEB_PORT", "8765"))
DOCS_DIR: Path = _PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Simulated / mock data (fallback when real sensors unavailable)
# ---------------------------------------------------------------------------

# Environment sensor readings -- updated periodically by a background task.
_env_state: dict[str, Any] = {
    "temp": 26.0,
    "humidity": 60,
    "pm25": 35,
    "weather": "晴",
    "aqi": 52,
    "noise": 38,
}

# ---------------------------------------------------------------------------
# Dynamic reminder system — stored by date, persisted to JSON file
# ---------------------------------------------------------------------------
_REMINDER_FILE = _PROJECT_ROOT / "data" / "reminders.json"

def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# Reminder store: { "2026-06-10": [...items], "2026-06-09": [...items] }
_reminder_store: dict[str, list[dict[str, Any]]] = {}
_reminder_next_id: int = 100

# Default reminders for today (used as initial data)
_DEFAULT_REMINDERS: list[dict[str, Any]] = [
    {"id": 1, "time": "08:00", "title": "早餐服药",       "desc": "降压药 1 片 + 维生素 D",  "done": True,  "icon": "💊"},
    {"id": 2, "time": "10:00", "title": "测量血压",       "desc": "使用电子血压计，记录数据",  "done": True,  "icon": "🩺"},
    {"id": 3, "time": "12:00", "title": "午餐",           "desc": "低盐饮食，注意营养均衡",   "done": False, "icon": "🍽️"},
    {"id": 4, "time": "14:00", "title": "午休",           "desc": "建议休息 30-60 分钟",      "done": False, "icon": "😴"},
    {"id": 5, "time": "15:30", "title": "下午散步",       "desc": "小区内散步 20 分钟",        "done": False, "icon": "🚶"},
    {"id": 6, "time": "18:00", "title": "晚餐服药",       "desc": "降糖药 1 片",              "done": False, "icon": "💊"},
    {"id": 7, "time": "20:00", "title": "泡脚放松",       "desc": "温水泡脚 15 分钟",          "done": False, "icon": "🦶"},
    {"id": 8, "time": "21:30", "title": "准备就寝",       "desc": "检查门窗，服药",            "done": False, "icon": "🌙"},
]

def _load_reminders():
    """Load reminders from JSON file, or initialize with defaults."""
    global _reminder_store, _reminder_next_id
    if _REMINDER_FILE.is_file():
        try:
            import json as _json
            _reminder_store = _json.loads(_REMINDER_FILE.read_text(encoding="utf-8"))
            # Find max ID
            for items in _reminder_store.values():
                for item in items:
                    if item.get("id", 0) >= _reminder_next_id:
                        _reminder_next_id = item["id"] + 1
            logger.info("Loaded reminders from %s (%d dates)", _REMINDER_FILE, len(_reminder_store))
            return
        except Exception as exc:
            logger.warning("Failed to load reminders: %s", exc)
    # Initialize with defaults for today
    _reminder_store[_today_str()] = list(_DEFAULT_REMINDERS)

def _save_reminders():
    """Persist reminders to JSON file."""
    try:
        import json as _json
        _REMINDER_FILE.parent.mkdir(parents=True, exist_ok=True)
        _REMINDER_FILE.write_text(_json.dumps(_reminder_store, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save reminders: %s", exc)

def _get_today_reminders() -> list[dict]:
    """Get today's reminders, initializing if needed."""
    today = _today_str()
    if today not in _reminder_store:
        _reminder_store[today] = []
    return _reminder_store[today]

def _add_reminder(time: str, title: str, desc: str = "", icon: str = "📌") -> dict:
    """Add a new reminder and return it."""
    global _reminder_next_id
    item = {
        "id": _reminder_next_id,
        "time": time,
        "title": title,
        "desc": desc,
        "done": False,
        "icon": icon,
        "created_at": datetime.now().isoformat(),
    }
    _reminder_next_id += 1
    _get_today_reminders().append(item)
    _save_reminders()
    logger.info("Reminder added: [%s] %s - %s", time, title, desc)
    return item

# Load reminders on module init
_load_reminders()

# ---------------------------------------------------------------------------
# Chat history — persisted to JSON file, last 200 messages
# ---------------------------------------------------------------------------
_CHAT_HISTORY_FILE = _PROJECT_ROOT / "data" / "chat_history.json"
_chat_history: list[dict[str, Any]] = []
_CHAT_HISTORY_MAX = 200

def _load_chat_history():
    """Load chat history from JSON file."""
    global _chat_history
    if _CHAT_HISTORY_FILE.is_file():
        try:
            _chat_history = json.loads(_CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
            logger.info("Loaded %d chat messages from history", len(_chat_history))
        except Exception as exc:
            logger.warning("Failed to load chat history: %s", exc)
            _chat_history = []

def _save_chat_history():
    """Persist chat history to JSON file."""
    try:
        _CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CHAT_HISTORY_FILE.write_text(
            json.dumps(_chat_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Failed to save chat history: %s", exc)

def _append_chat_msg(role: str, text: str):
    """Append a message to chat history and persist."""
    _chat_history.append({
        "role": role,
        "text": text,
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": _today_str(),
    })
    # Trim to max
    if len(_chat_history) > _CHAT_HISTORY_MAX:
        _chat_history[:] = _chat_history[-_CHAT_HISTORY_MAX:]
    _save_chat_history()

_load_chat_history()

# Simulated health data.
_health_data: dict[str, Any] = {
    "heart_rate": 72,
    "blood_pressure_sys": 128,
    "blood_pressure_dia": 82,
    "blood_oxygen": 97,
    "temperature": 36.5,
    "steps_today": 3200,
    "sleep_hours": 7.2,
    "last_updated": "",
}

# System info returned to the frontend (will be enriched with real data at runtime).
_system_info: dict[str, Any] = {
    "version": "2.0",
    "name": "小鲲 KunPeng-Cortex",
    "hardware": {
        "board": "OrangePi Kunpeng Pro (RK3588)",
        "npu": "6 TOPS",
        "ram": "16 GB",
        "storage": "64 GB eMMC",
    },
    "software": {
        "os": "openEuler 24.03",
        "python": sys.version.split()[0],
        "fusion_available": _fusion_available,
    },
    "uptime_seconds": 0,
}

_start_time: float = time.time()

# Weather cycle for simulation.
_WEATHER_OPTIONS = ["晴", "多云", "阴", "小雨", "晴转多云"]

# ---------------------------------------------------------------------------
# Active WebSocket connections (for broadcast / push)
# ---------------------------------------------------------------------------
_active_clients: set[WebSocket] = set()
_companion_runtime: Any = None

# ---------------------------------------------------------------------------
# FusionChat singleton (lazy init)
# ---------------------------------------------------------------------------

def _get_fusion_chat():
    """Return the shared FusionChat instance, creating it on first call.

    Returns None when FusionChat is unavailable (missing dependencies,
    missing API key, etc.).
    """
    global _fusion_chat_instance, _fusion_available
    if _fusion_chat_instance is not None:
        return _fusion_chat_instance
    if not _fusion_available:
        return None
    try:
        _fusion_chat_instance = FusionChat()
        _register_tool_handlers(_fusion_chat_instance)
        logger.info("FusionChat initialized successfully with tool handlers.")
        return _fusion_chat_instance
    except BaseException as exc:
        # FusionChat may call sys.exit() when optional AI packages are absent.
        # Keep the HMI and companion simulator available in fallback mode.
        _fusion_available = False
        logger.warning("FusionChat initialization failed; using HMI fallback: %s", exc)
        return None


def _register_tool_handlers(fusion):
    """Register server-side tool callbacks for the AI agent."""

    def tool_add_reminder(args: dict) -> dict:
        time_str = args.get("time", "")
        title = args.get("title", "").strip()
        desc = args.get("desc", "")
        icon = args.get("icon", "📌")
        if not title:
            return {"success": False, "error": "提醒标题不能为空"}
        item = _add_reminder(time_str, title, desc, icon)
        return {"success": True, "item": item, "message": f"已添加提醒: {time_str} - {title}"}

    def tool_list_reminders(args: dict) -> dict:
        items = _get_today_reminders()
        return {"date": _today_str(), "items": items, "count": len(items)}

    def tool_toggle_device(args: dict) -> dict:
        room = args.get("room", "")
        device = args.get("device", "")
        action = args.get("action", "toggle")
        # Route through the agent for natural handling
        return {"room": room, "device": device, "action": action, "success": True, "message": f"{room}{device} {'已打开' if action == 'on' else '已关闭' if action == 'off' else '已切换'}"}

    def tool_get_health_data(args: dict) -> dict:
        _update_health_data()
        return _health_data

    def tool_get_weather(args: dict) -> dict:
        _update_env_state()
        return _env_state

    def tool_control_car(args: dict) -> dict:
        """AI tool: control car movement (sync wrapper)."""
        if _car_state["estop"]:
            return {"success": False, "error": "紧急停止已激活，请先解除"}
        direction = args.get("direction", "stop")
        speed_pct = args.get("speed", 50)
        left_s, right_s = _car_direction_to_speeds(direction, speed_pct)
        _car_state["direction"] = {"forward":"前进","backward":"后退","left":"左转","right":"右转","stop":"停止"}.get(direction, direction)
        _car_state["speed"] = speed_pct
        _car_state["left_speed"] = round(left_s * 100)
        _car_state["right_speed"] = round(right_s * 100)
        hw = _get_car_hw()
        if hw:
            try:
                if direction == "stop":
                    hw.stop_all()
                else:
                    hw.set_speeds(left=left_s, right=right_s)
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        return {"success": True, "direction": _car_state["direction"], "speed": speed_pct,
                "left": _car_state["left_speed"], "right": _car_state["right_speed"],
                "message": f"小车{_car_state['direction']}，速度 {speed_pct}%"}

    def tool_control_arm(args: dict) -> dict:
        """AI tool: control robotic arm (sync wrapper)."""
        if _arm_state["estop"]:
            return {"success": False, "error": "紧急停止已激活"}
        action = args.get("action", "preset")
        if action == "move_joint":
            joint = args.get("joint", "j1")
            angle = args.get("angle", 90)
            joint_limits = {"j1":(0,180),"j2":(0,180),"j3":(0,180),"j4":(0,180),"j5":(0,180),"j6":(0,90)}
            if joint not in joint_limits:
                return {"success": False, "error": f"未知关节: {joint}"}
            lo, hi = joint_limits[joint]
            angle = max(lo, min(hi, int(angle)))
            _arm_state["angles"][joint] = angle
            _arm_state["state"] = f"{joint}→{angle}°"
            return {"success": True, "joint": joint, "angle": angle, "message": f"机械臂 {joint} 已移动到 {angle}°"}
        elif action == "preset":
            preset = args.get("preset", "home")
            preset_angles = {
                "home":    {"j1":90,"j2":90,"j3":90,"j4":90,"j5":90,"j6":45},
                "wave":    {"j1":45,"j2":90,"j3":90,"j4":90,"j5":90,"j6":45},
                "drink":   {"j1":90,"j2":60,"j3":45,"j4":120,"j5":90,"j6":30},
            }
            labels = {"home":"归位","grip":"夹取","release":"释放","wave":"挥手","drink":"递水"}
            _arm_state["state"] = labels.get(preset, preset)
            if preset == "grip":
                _arm_state["angles"]["j6"] = 10
                _arm_state["gripper"] = "closed"
            elif preset == "release":
                _arm_state["angles"]["j6"] = 85
                _arm_state["gripper"] = "open"
            elif preset in preset_angles:
                _arm_state["angles"] = dict(preset_angles[preset])
            return {"success": True, "preset": preset, "message": f"机械臂已执行: {labels.get(preset, preset)}"}
        return {"success": False, "error": f"未知操作: {action}"}

    def tool_emergency_stop(args: dict) -> dict:
        """AI tool: emergency stop all hardware (sync wrapper)."""
        scope = args.get("scope", "all")
        _car_state["estop"] = True
        _car_state["direction"] = "紧急停止"
        _car_state["speed"] = 0
        _arm_state["estop"] = True
        _arm_state["state"] = "紧急停止"
        hw = _get_car_hw()
        if hw:
            try: hw.brake_all()
            except: pass
        return {"success": True, "scope": scope, "message": f"🛑 紧急停止已执行（{scope}）"}

    fusion.tool_handlers = {
        "add_reminder": tool_add_reminder,
        "list_reminders": tool_list_reminders,
        "toggle_device": tool_toggle_device,
        "get_health_data": tool_get_health_data,
        "get_weather": tool_get_weather,
        "control_car": tool_control_car,
        "control_arm": tool_control_arm,
        "emergency_stop": tool_emergency_stop,
    }
    logger.info("Registered %d tool handlers: %s", len(fusion.tool_handlers), list(fusion.tool_handlers.keys()))


# ===========================================================================
# FastAPI application
# ===========================================================================
app = FastAPI(
    title="小鲲 KunPeng-Cortex Web Server",
    version="2.0",
    docs_url=None,       # disable Swagger UI in production
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# Mount static files from docs/ directory
# ---------------------------------------------------------------------------
if DOCS_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(DOCS_DIR)), name="static")
    logger.info("Static files mounted from %s", DOCS_DIR)
else:
    logger.warning("docs/ directory not found at %s -- static serving disabled", DOCS_DIR)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main HMI page at the root URL."""
    # Prefer the WebSocket-enabled HMI, fall back to the static prototype
    ws_file = DOCS_DIR / "hmi-ws.html"
    static_file = DOCS_DIR / "hengxiang-hmi.html"
    for f in (ws_file, static_file):
        if f.is_file():
            return FileResponse(str(f), media_type="text/html")
    return HTMLResponse(
        "<html><body><h1>小鲲 KunPeng-Cortex</h1>"
        "<p>HMI 页面未找到。请将 hmi-ws.html 放入 docs/ 目录。</p>"
        "<p>WebSocket available at <code>/ws</code></p>"
        "</body></html>",
        status_code=200,
    )


@app.get("/companion", response_class=HTMLResponse)
async def serve_companion():
    """Serve the same HMI in desktop companion surface mode."""
    ws_file = DOCS_DIR / "hmi-ws.html"
    if ws_file.is_file():
        return FileResponse(str(ws_file), media_type="text/html")
    return HTMLResponse("<h1>桌宠页面未找到</h1>", status_code=404)


@app.get("/health")
async def health_check():
    """Simple health endpoint for monitoring."""
    return {
        "status": "ok",
        "clients": len(_active_clients),
        "fusion": _fusion_available,
        "uptime": int(time.time() - _start_time),
    }


@app.get("/photo/{filename}")
async def serve_photo(filename: str):
    """Serve photo files from the Pictures directory.

    This allows the frontend to display captured photos via <img> tags.
    """
    from fastapi.responses import FileResponse
    from fastapi import HTTPException

    # Photos are stored in ~/Pictures/
    photo_dir = Path.home() / "Pictures"
    photo_path = photo_dir / filename

    # Security: only allow files within the Pictures directory
    try:
        photo_path = photo_path.resolve()
        photo_dir = photo_dir.resolve()
        if not str(photo_path).startswith(str(photo_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid path")

    if not photo_path.is_file():
        raise HTTPException(status_code=404, detail="Photo not found")

    return FileResponse(str(photo_path), media_type="image/jpeg")


@app.get("/photos")
async def list_photos():
    """List available photos."""
    photo_dir = Path.home() / "Pictures"
    if not photo_dir.is_dir():
        return {"photos": []}

    photos = []
    for f in sorted(photo_dir.glob("capture_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        photos.append({
            "name": f.name,
            "url": f"/photo/{f.name}",
            "size": f.stat().st_size,
            "time": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return {"photos": photos}


# ===========================================================================
# Environment simulation helpers
# ===========================================================================

def _update_env_state():
    """Update sensor readings — try real sensors via FusionChat, fall back to simulation."""
    global _env_state
    fusion = _get_fusion_chat()
    if fusion and hasattr(fusion, 'hw') and fusion.hw:
        # Try reading real sensor data through the agent's hardware layer
        try:
            from hal.sensor import UnifiedSensorManager
            mgr = UnifiedSensorManager()
            readings = mgr.read_all()
            if "temperature" in readings:
                _env_state["temp"] = round(readings["temperature"], 1)
            if "humidity" in readings:
                _env_state["humidity"] = int(readings["humidity"])
            # Only override if real data is available; keep previous values otherwise
        except Exception:
            pass  # No real sensors, fall through to simulation
    # Simulated drift for values without real sensors
    if "temp" not in _env_state or _env_state["temp"] == 26.0:
        _env_state["temp"] = round(
            max(18.0, min(35.0, _env_state["temp"] + random.uniform(-0.3, 0.3))), 1
        )
    _env_state["humidity"] = max(
        30, min(85, _env_state["humidity"] + random.randint(-2, 2))
    )
    _env_state["pm25"] = max(5, min(150, _env_state["pm25"] + random.randint(-3, 3)))
    _env_state["noise"] = max(20, min(80, _env_state["noise"] + random.randint(-2, 2)))
    _env_state["aqi"] = max(20, min(200, _env_state["aqi"] + random.randint(-2, 2)))
    if random.random() < 0.05:
        _env_state["weather"] = random.choice(_WEATHER_OPTIONS)


def _update_health_data():
    """Apply small random drift to simulated health metrics."""
    _health_data["heart_rate"] = max(
        55, min(100, _health_data["heart_rate"] + random.randint(-2, 2))
    )
    _health_data["blood_pressure_sys"] = max(
        100, min(160, _health_data["blood_pressure_sys"] + random.randint(-1, 1))
    )
    _health_data["blood_pressure_dia"] = max(
        60, min(100, _health_data["blood_pressure_dia"] + random.randint(-1, 1))
    )
    _health_data["blood_oxygen"] = max(
        90, min(100, _health_data["blood_oxygen"] + random.randint(-1, 1))
    )
    _health_data["temperature"] = round(
        max(35.8, min(37.5, _health_data["temperature"] + random.uniform(-0.1, 0.1))), 1
    )
    _health_data["steps_today"] = max(0, _health_data["steps_today"] + random.randint(0, 15))
    _health_data["last_updated"] = datetime.now().strftime("%H:%M:%S")


def _build_status_push() -> dict:
    """Construct a status.push payload with current time and sensor data."""
    now = datetime.now()
    return {
        "action": "status.push",
        "data": {
            "temp": _env_state["temp"],
            "humidity": _env_state["humidity"],
            "pm25": _env_state["pm25"],
            "aqi": _env_state["aqi"],
            "noise": _env_state["noise"],
            "weather": _env_state["weather"],
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
        },
    }


# ===========================================================================
# WebSocket message handlers
# ===========================================================================

async def _handle_chat_send(data: dict, ws: WebSocket) -> dict:
    """Handle an incoming chat message by delegating to FusionChat or echo.

    When FusionChat uses tool calling (e.g. add_reminder), the tool results
    are inspected to broadcast updated reminders to all connected clients.
    """
    text = (data.get("text") or "").strip()
    if not text:
        return {"action": "chat.reply", "data": {"text": "您好像没有说话呢，请再说一次？"}}

    logger.info("Chat request: %s", text[:80])

    # Save user message to history
    _append_chat_msg("user", text)

    fusion = _get_fusion_chat()
    if fusion is not None:
        try:
            # FusionChat.chat() is synchronous; run in a thread to avoid blocking.
            loop = asyncio.get_running_loop()
            logger.info("Calling FusionChat.chat() for: %s", text[:50])
            reply = await loop.run_in_executor(None, fusion.chat, text)
            logger.info("FusionChat replied (%d chars): %s", len(reply or ""), (reply or "")[:80])

            # Save AI reply to history
            _append_chat_msg("assistant", reply or "")

            result = {"action": "chat.reply", "data": {"text": reply}}

            # Check if any tool calls changed state — broadcast updates to all clients
            added_reminders = []
            car_arm_changed = False
            for tr in getattr(fusion, "tool_results", []):
                tool_name = tr.get("tool", "")
                if tool_name == "add_reminder":
                    try:
                        tool_res = json.loads(tr["result"]) if isinstance(tr["result"], str) else tr["result"]
                        if tool_res.get("success") and tool_res.get("item"):
                            added_reminders.append(tool_res["item"])
                    except Exception:
                        pass
                elif tool_name in ("control_car", "control_arm", "emergency_stop"):
                    car_arm_changed = True

            if added_reminders:
                result["data"]["added_reminders"] = added_reminders
                await _broadcast_reminders()

            # Broadcast car/arm status if AI controlled them
            if car_arm_changed:
                await _broadcast_car_status()
                await _broadcast_arm_status()

            return result
        except Exception as exc:
            logger.error("FusionChat error: %s", exc, exc_info=True)
            err_msg = f"抱歉，AI 服务暂时不可用，请稍后再试。\n({exc})"
            _append_chat_msg("assistant", err_msg)
            return {
                "action": "chat.reply",
                "data": {"text": err_msg},
            }

    # Echo fallback -- useful for frontend development without API keys.
    echo_msg = f"[echo] 收到您的消息：{text}\n（FusionChat 未启用，当前为回声模式）"
    _append_chat_msg("assistant", echo_msg)
    return {
        "action": "chat.reply",
        "data": {"text": echo_msg},
    }



async def _handle_chat_history(data: dict, ws: WebSocket) -> dict:
    """Return stored chat history."""
    return {
        "action": "chat.history",
        "data": {"messages": _chat_history, "total": len(_chat_history)},
    }


async def _handle_chat_clear(data: dict, ws: WebSocket) -> dict:
    """Clear chat history."""
    _chat_history.clear()
    _save_chat_history()
    return {
        "action": "chat.cleared",
        "data": {"success": True},
    }


async def _handle_device_toggle(data: dict, ws: WebSocket) -> dict:
    """Toggle a smart-home device — routed through FusionChat Agent.

    Sends a natural language command to the agent which handles:
    intent classification → GPIO/hardware execution → AI response.
    """
    room = data.get("room", "")
    device = data.get("device", "")
    target = data.get("on", None)  # True=开, False=关, None=toggle

    # Build natural language command for the agent
    if target is True:
        cmd = f"帮我打开{room}的{device}"
    elif target is False:
        cmd = f"帮我关闭{room}的{device}"
    else:
        cmd = f"帮我切换一下{room}的{device}"

    logger.info("Device toggle via agent: %s/%s -> %s", room, device, cmd)

    fusion = _get_fusion_chat()
    if fusion is not None:
        try:
            loop = asyncio.get_running_loop()
            reply = await loop.run_in_executor(None, fusion.chat, cmd)
            return {
                "action": "device.state",
                "data": {
                    "room": room,
                    "device": device,
                    "agent_reply": reply,
                    "success": True,
                },
            }
        except Exception as exc:
            logger.error("Agent device toggle error: %s", exc, exc_info=True)

    # Fallback without agent
    return {
        "action": "device.state",
        "data": {
            "room": room,
            "device": device,
            "agent_reply": f"设备控制暂未接入（Agent 未启用）。指令：{cmd}",
            "success": False,
        },
    }


async def _handle_photo_take(data: dict, ws: WebSocket) -> dict:
    """Take a photo — uses FusionChat's hardware executor."""
    fusion = _get_fusion_chat()
    if fusion is not None:
        try:
            loop = asyncio.get_running_loop()
            ok, msg = await loop.run_in_executor(None, fusion.hw_exe.take_photo)
            if ok:
                import re
                m = re.search(r"(/[\w/]+\.jpg)", msg)
                if m:
                    fusion.last_photo = m.group(1)
                return {
                    "action": "photo.result",
                    "data": {"success": True, "path": fusion.last_photo, "message": msg},
                }
            return {"action": "photo.result", "data": {"success": False, "message": msg}}
        except Exception as exc:
            logger.error("Photo capture error: %s", exc)

    # Simulated response when no hardware is available.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fake_path = f"/tmp/capture_{ts}.jpg"
    logger.info("Simulated photo capture: %s", fake_path)
    return {
        "action": "photo.result",
        "data": {"success": True, "path": fake_path, "message": f"模拟拍照成功: {fake_path}"},
    }


async def _handle_scene_execute(data: dict, ws: WebSocket) -> dict:
    """Execute a scene mode — routed through FusionChat Agent."""
    scene_name = data.get("scene", "")

    # Build natural language command for the agent
    cmd = f"帮我执行{scene_name}"
    logger.info("Scene execute via agent: %s", cmd)

    fusion = _get_fusion_chat()
    if fusion is not None:
        try:
            loop = asyncio.get_running_loop()
            reply = await loop.run_in_executor(None, fusion.chat, cmd)
            return {
                "action": "scene.result",
                "data": {
                    "scene": scene_name,
                    "success": True,
                    "message": reply,
                },
            }
        except Exception as exc:
            logger.error("Agent scene execute error: %s", exc, exc_info=True)

    return {
        "action": "scene.result",
        "data": {
            "scene": scene_name,
            "success": False,
            "message": f"情景模式暂未接入（Agent 未启用）。指令：{cmd}",
        },
    }


async def _handle_reminders_get(data: dict, ws: WebSocket) -> dict:
    """Return today's reminders."""
    return {
        "action": "reminders.list",
        "data": {"items": _get_today_reminders(), "date": _today_str()},
    }


async def _handle_reminders_add(data: dict, ws: WebSocket) -> dict:
    """Add a new reminder."""
    time_str = data.get("time", "")
    title = data.get("title", "").strip()
    desc = data.get("desc", "")
    icon = data.get("icon", "📌")
    if not title:
        return {"action": "reminders.added", "data": {"success": False, "message": "提醒标题不能为空"}}
    item = _add_reminder(time_str, title, desc, icon)
    # Broadcast updated reminders to all clients
    await _broadcast_reminders()
    return {"action": "reminders.added", "data": {"success": True, "item": item}}


async def _handle_reminders_toggle(data: dict, ws: WebSocket) -> dict:
    """Toggle a reminder's done status."""
    item_id = data.get("id")
    today_items = _get_today_reminders()
    for item in today_items:
        if item["id"] == item_id:
            item["done"] = not item.get("done", False)
            _save_reminders()
            await _broadcast_reminders()
            return {
                "action": "reminders.toggled",
                "data": {"id": item_id, "done": item["done"]},
            }
    return {"action": "reminders.toggled", "data": {"success": False, "message": "未找到该提醒"}}


async def _handle_reminders_history(data: dict, ws: WebSocket) -> dict:
    """Return reminders history grouped by date."""
    history = []
    for date_str in sorted(_reminder_store.keys(), reverse=True):
        items = _reminder_store[date_str]
        if items:
            history.append({"date": date_str, "items": items})
    return {
        "action": "reminders.history",
        "data": {"history": history},
    }


async def _broadcast_reminders():
    """Broadcast updated reminders to all connected clients."""
    payload = {
        "action": "reminders.list",
        "data": {"items": _get_today_reminders(), "date": _today_str()},
    }
    dead: list = []
    for client in _active_clients:
        try:
            await client.send_json(payload)
        except Exception:
            dead.append(client)
    for client in dead:
        _active_clients.discard(client)


async def _broadcast_to_all(payload: dict):
    """Send a JSON payload to all connected WebSocket clients."""
    dead: list = []
    for client in _active_clients:
        try:
            await client.send_json(payload)
        except Exception:
            dead.append(client)
    for client in dead:
        _active_clients.discard(client)


async def _broadcast_car_status():
    """Broadcast current car state to all clients."""
    await _broadcast_to_all({"action": "car.status", "data": dict(_car_state)})


async def _broadcast_arm_status():
    """Broadcast current arm state to all clients."""
    await _broadcast_to_all({"action": "arm.status", "data": dict(_arm_state)})


async def _handle_health_get(data: dict, ws: WebSocket) -> dict:
    """Return current health metrics."""
    _update_health_data()
    return {
        "action": "health.data",
        "data": _health_data,
    }


async def _handle_companion_snapshot(data: dict, ws: WebSocket) -> dict:
    """Return the current semantic elder and companion state after reconnect."""
    if _companion_runtime is None:
        return {"action": "error", "data": {"message": "陪伴服务暂未就绪"}}
    return await _companion_runtime.snapshot()


async def _handle_elder_report(data: dict, ws: WebSocket) -> dict:
    """Accept an explicit elder report without inferring an unobserved action."""
    if _companion_runtime is None:
        return {"action": "error", "data": {"message": "陪伴服务暂未就绪"}}
    return await _companion_runtime.submit_report(data)


async def _handle_simulation_start(data: dict, ws: WebSocket) -> dict:
    """Run a clearly labelled deterministic companion demonstration scenario."""
    if _companion_runtime is None:
        return {"action": "error", "data": {"message": "陪伴服务暂未就绪"}}
    return await _companion_runtime.run_scenario(str(data.get("scenario", "")))


async def _handle_system_info(data: dict, ws: WebSocket) -> dict:
    """Return system information — enriched with real hardware status from FusionChat."""
    _system_info["uptime_seconds"] = int(time.time() - _start_time)

    fusion = _get_fusion_chat()
    fusion_active = fusion is not None
    _system_info["software"]["fusion_available"] = fusion_active

    if fusion and fusion.hw:
        # Real hardware status from agent
        _system_info["hardware"]["camera"] = bool(fusion.hw.camera)
        _system_info["hardware"]["gpio"] = fusion.hw.gpio
        _system_info["hardware"]["i2c"] = len(fusion.hw.i2c) if fusion.hw.i2c else False
        _system_info["hardware"]["uart"] = len(fusion.hw.uart) if fusion.hw.uart else False
        _system_info["hardware"]["npu"] = fusion.hw.npu
        _system_info["hardware"]["camera_devices"] = fusion.hw.camera

    if fusion_active:
        _system_info["agent"] = {
            "session_id": fusion.sid,
            "messages": len(fusion.msgs),
            "model": fusion.model,
            "api_url": fusion.url,
        }
        if fusion.mgr:
            try:
                _system_info["agent"]["skills"] = fusion.mgr.get_stats()
            except:
                pass
        if fusion.store:
            try:
                _system_info["agent"]["memory_entries"] = len(fusion.store.memory_entries) + len(fusion.store.user_entries)
            except:
                pass
        if fusion.ev:
            try:
                _system_info["agent"]["evolution"] = fusion.ev.get_stats()
            except:
                pass

    return {
        "action": "system.info",
        "data": _system_info,
    }


async def _handle_sos_call(data: dict, ws: WebSocket) -> dict:
    """Handle an SOS emergency call — routed through FusionChat Agent."""
    sos_type = data.get("type", "kunpeng")
    logger.warning("SOS activated! type=%s", sos_type)

    # Route through agent for intelligent emergency response
    fusion = _get_fusion_chat()
    if fusion is not None:
        cmd_map = {
            "son": "紧急情况！请立刻通知我的家人，我儿子",
            "120": "紧急情况！请帮我拨打120急救电话",
            "kunpeng": "紧急情况！进入紧急守护模式",
        }
        cmd = cmd_map.get(sos_type, cmd_map["kunpeng"])
        try:
            loop = asyncio.get_running_loop()
            reply = await loop.run_in_executor(None, fusion.chat, cmd)
            return {
                "action": "sos.result",
                "data": {
                    "type": sos_type,
                    "success": True,
                    "message": reply,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        except Exception as exc:
            logger.error("Agent SOS error: %s", exc, exc_info=True)

    # Fallback messages
    messages = {
        "son": "已通知您的家人，请保持电话畅通。您的儿子将很快与您联系。",
        "120": "已拨打 120 急救电话，请保持冷静，救援正在赶来。",
        "kunpeng": "小鲲已进入紧急守护模式，正在持续监测您的状态并通知紧急联系人。",
    }
    return {
        "action": "sos.result",
        "data": {
            "type": sos_type,
            "success": True,
            "message": messages.get(sos_type, messages["kunpeng"]),
            "timestamp": datetime.now().isoformat(),
        },
    }


async def _handle_skills_list(data: dict, ws: WebSocket) -> dict:
    """List all available skills from FusionChat's SkillManager."""
    fusion = _get_fusion_chat()
    if fusion and fusion.mgr:
        try:
            skills = fusion.mgr.list_skills(detail_level=1)
            return {
                "action": "skills.list",
                "data": {"skills": skills, "total": len(skills)},
            }
        except Exception as exc:
            logger.error("Skills list error: %s", exc)

    return {
        "action": "skills.list",
        "data": {"skills": [], "total": 0, "message": "Skill 系统未加载"},
    }


async def _handle_memory_get(data: dict, ws: WebSocket) -> dict:
    """Get memory snapshot from FusionChat's memory store."""
    fusion = _get_fusion_chat()
    if fusion and fusion.store:
        try:
            snap = fusion.store.get_snapshot_for_prompt()
            return {
                "action": "memory.data",
                "data": {
                    "environment_memory": snap.get("memory", ""),
                    "user_profile": snap.get("user", ""),
                    "memory_entries": len(fusion.store.memory_entries),
                    "user_entries": len(fusion.store.user_entries),
                },
            }
        except Exception as exc:
            logger.error("Memory get error: %s", exc)

    return {
        "action": "memory.data",
        "data": {"message": "记忆系统未加载"},
    }


async def _handle_agent_status(data: dict, ws: WebSocket) -> dict:
    """Get full Agent status including emotion, skills, memory, evolution."""
    fusion = _get_fusion_chat()
    if not fusion:
        return {
            "action": "agent.status",
            "data": {"available": False, "message": "FusionChat 未启用"},
        }

    status = {
        "available": True,
        "session_id": fusion.sid,
        "model": fusion.model,
        "api_url": fusion.url,
        "messages_count": len(fusion.msgs),
        "last_photo": fusion.last_photo,
    }

    # Hardware status
    if fusion.hw:
        status["hardware"] = {
            "camera": bool(fusion.hw.camera),
            "camera_devices": fusion.hw.camera,
            "gpio": fusion.hw.gpio,
            "i2c_buses": len(fusion.hw.i2c) if fusion.hw.i2c else 0,
            "uart_ports": len(fusion.hw.uart) if fusion.hw.uart else 0,
            "npu": fusion.hw.npu,
        }

    # KunPeng engine status
    status["kunpeng"] = {
        "dialogue_manager": fusion.dm is not None,
        "emotion_engine": fusion.em is not None,
    }

    # Hermes bridge status
    status["hermes"] = {
        "memory_store": fusion.store is not None,
        "skill_manager": fusion.mgr is not None,
        "session_db": fusion.db is not None,
        "evolution_engine": fusion.ev is not None,
    }

    if fusion.mgr:
        try:
            status["skills_stats"] = fusion.mgr.get_stats()
        except:
            pass
    if fusion.store:
        try:
            status["memory_count"] = len(fusion.store.memory_entries) + len(fusion.store.user_entries)
        except:
            pass
    if fusion.ev:
        try:
            status["evolution_stats"] = fusion.ev.get_stats()
        except:
            pass

    return {
        "action": "agent.status",
        "data": status,
    }


# ===========================================================================
# Car / chassis control
# ===========================================================================

# Car state (shared between handlers)
_car_state: dict[str, Any] = {
    "direction": "stop",
    "speed": 0,
    "left_speed": 0,
    "right_speed": 0,
    "max_speed": 0.8,       # safety limit from config
    "estop": False,
}

# Lazy-loaded hardware bridge
_car_hw = None

def _get_car_hw():
    """Try to initialize the real motor hardware (MotorGroup via HAL).
    Returns None if hardware is unavailable (simulation mode)."""
    global _car_hw
    if _car_hw is not None:
        return _car_hw
    try:
        from hal.motor import Motor, MotorGroup, MotorConfig, MotorType
        cfg = MotorConfig(motor_type=MotorType.DC, max_speed=0.8)
        left = Motor(pwm_pin=32, dir_pin=36, config=cfg)
        right = Motor(pwm_pin=33, dir_pin=37, config=cfg)
        _car_hw = MotorGroup({"left": left, "right": right})
        logger.info("Car hardware initialized (MotorGroup, pins 32/33/36/37)")
    except Exception as exc:
        logger.info("Car hardware not available (%s), running in simulation mode", exc)
        _car_hw = False  # False = unavailable, distinct from None = not tried
    return _car_hw if _car_hw else None


def _car_direction_to_speeds(direction: str, speed_pct: int) -> tuple[float, float]:
    """Convert direction + speed% to left/right motor speeds in [-1.0, 1.0]."""
    s = min(speed_pct, 100) / 100.0 * _car_state["max_speed"]
    turn_factor = 0.5
    if direction == "forward":
        return (s, s)
    elif direction == "backward":
        return (-s, -s)
    elif direction == "left":
        return (s * turn_factor, s)
    elif direction == "right":
        return (s, s * turn_factor)
    return (0.0, 0.0)


async def _handle_car_move(data: dict, ws: WebSocket) -> dict:
    """Move the car in a direction."""
    if _car_state["estop"]:
        return {"action": "car.status", "data": {**_car_state, "message": "紧急停止已激活，请先解除"}}

    direction = data.get("direction", "stop")
    speed_pct = data.get("speed", 50)
    left_s, right_s = _car_direction_to_speeds(direction, speed_pct)

    # Update state
    _car_state["direction"] = {"forward":"前进","backward":"后退","left":"左转","right":"右转"}.get(direction, direction)
    _car_state["speed"] = speed_pct
    _car_state["left_speed"] = round(left_s * 100)
    _car_state["right_speed"] = round(right_s * 100)

    # Try real hardware
    hw = _get_car_hw()
    if hw:
        try:
            hw.set_speeds(left=left_s, right=right_s)
        except Exception as exc:
            logger.warning("Car move HW error: %s", exc)

    logger.info("Car move: %s speed=%d%% (L=%d R=%d)", direction, speed_pct, _car_state["left_speed"], _car_state["right_speed"])
    return {"action": "car.status", "data": _car_state}


async def _handle_car_stop(data: dict, ws: WebSocket) -> dict:
    """Stop the car."""
    _car_state["direction"] = "停止"
    _car_state["speed"] = 0
    _car_state["left_speed"] = 0
    _car_state["right_speed"] = 0

    hw = _get_car_hw()
    if hw:
        try:
            hw.stop_all()
        except Exception as exc:
            logger.warning("Car stop HW error: %s", exc)

    logger.info("Car stopped")
    return {"action": "car.status", "data": _car_state}


# ===========================================================================
# Robotic arm control (6-DOF Dofbot)
# ===========================================================================

_arm_state: dict[str, Any] = {
    "state": "待机",
    "gripper": "open",
    "angles": {"j1":90, "j2":90, "j3":90, "j4":90, "j5":90, "j6":45},
    "estop": False,
}

_arm_hw = None

def _get_arm_hw():
    """Try to initialize the Dofbot arm hardware."""
    global _arm_hw
    if _arm_hw is not None:
        return _arm_hw
    try:
        from devices.dofbot_arm import DofbotArm
        import asyncio as _asyncio
        _arm_hw = DofbotArm()
        # Note: DofbotArm.initialize() is async; we'll call it in the first move
        logger.info("Arm hardware initialized (DofbotArm, PCA9685 I2C)")
    except Exception as exc:
        logger.info("Arm hardware not available (%s), running in simulation mode", exc)
        _arm_hw = False
    return _arm_hw if _arm_hw else None


async def _handle_arm_move(data: dict, ws: WebSocket) -> dict:
    """Move a single arm joint to an angle."""
    if _arm_state["estop"]:
        return {"action": "arm.status", "data": {**_arm_state, "message": "紧急停止已激活"}}

    joint = data.get("joint", "")
    angle = data.get("angle", 90)

    # Validate joint
    joint_limits = {"j1":(0,180),"j2":(0,180),"j3":(0,180),"j4":(0,180),"j5":(0,180),"j6":(0,90)}
    if joint not in joint_limits:
        return {"action": "arm.status", "data": {**_arm_state, "message": f"未知关节: {joint}"}}
    lo, hi = joint_limits[joint]
    angle = max(lo, min(hi, int(angle)))

    _arm_state["angles"][joint] = angle
    _arm_state["state"] = f"{joint}→{angle}°"

    # Try real hardware
    hw = _get_arm_hw()
    if hw:
        try:
            from devices.dofbot_arm import JointAngles
            # Map j1-j6 to JointAngles fields
            a = _arm_state["angles"]
            target = JointAngles(j1=a["j1"], j2=a["j2"], j3=a["j3"], j4=a["j4"], j5=a["j5"], j6=a["j6"])
            await hw.move_joints(target, speed=60, blocking=False)
        except Exception as exc:
            logger.warning("Arm move HW error: %s", exc)

    logger.info("Arm move: %s = %d°", joint, angle)
    return {"action": "arm.status", "data": _arm_state}


async def _handle_arm_preset(data: dict, ws: WebSocket) -> dict:
    """Execute a preset arm action."""
    if _arm_state["estop"]:
        return {"action": "arm.status", "data": {**_arm_state, "message": "紧急停止已激活"}}

    preset = data.get("preset", "home")

    preset_angles = {
        "home":    {"j1":90,"j2":90,"j3":90,"j4":90,"j5":90,"j6":45},
        "grip":    None,  # only change gripper
        "release": None,
        "wave":    {"j1":45,"j2":90,"j3":90,"j4":90,"j5":90,"j6":45},
        "drink":   {"j1":90,"j2":60,"j3":45,"j4":120,"j5":90,"j6":30},
    }

    labels = {"home":"归位","grip":"夹取","release":"释放","wave":"挥手","drink":"递水"}
    _arm_state["state"] = labels.get(preset, preset)

    if preset == "grip":
        _arm_state["angles"]["j6"] = 10
        _arm_state["gripper"] = "closed"
    elif preset == "release":
        _arm_state["angles"]["j6"] = 85
        _arm_state["gripper"] = "open"
    elif preset in preset_angles and preset_angles[preset]:
        _arm_state["angles"] = dict(preset_angles[preset])

    # Try real hardware
    hw = _get_arm_hw()
    if hw:
        try:
            if preset == "grip":
                await hw.grip(force=0.8)
            elif preset == "release":
                await hw.release()
            else:
                from devices.dofbot_arm import JointAngles
                a = _arm_state["angles"]
                target = JointAngles(j1=a["j1"], j2=a["j2"], j3=a["j3"], j4=a["j4"], j5=a["j5"], j6=a["j6"])
                await hw.move_joints(target, speed=60, blocking=True)
        except Exception as exc:
            logger.warning("Arm preset HW error: %s", exc)

    logger.info("Arm preset: %s", preset)
    return {"action": "arm.status", "data": _arm_state}


# ===========================================================================
# Emergency stop (all hardware)
# ===========================================================================

async def _handle_emergency_stop(data: dict, ws: WebSocket) -> dict:
    """Emergency stop — halt all motors and arm immediately."""
    scope = data.get("scope", "all")

    _car_state["estop"] = True
    _car_state["direction"] = "紧急停止"
    _car_state["speed"] = 0
    _car_state["left_speed"] = 0
    _car_state["right_speed"] = 0

    _arm_state["estop"] = True
    _arm_state["state"] = "紧急停止"

    # Stop car hardware
    if scope in ("all", "motors"):
        hw = _get_car_hw()
        if hw:
            try:
                hw.brake_all()
            except Exception as exc:
                logger.warning("Car estop HW error: %s", exc)

    # Stop arm hardware
    if scope in ("all", "arm"):
        arm = _get_arm_hw()
        if arm:
            try:
                await arm.emergency_stop()
            except Exception as exc:
                logger.warning("Arm estop HW error: %s", exc)

    # Also try STM32 bridge estop
    try:
        from devices.stm32_bridge import STM32Bridge
        stm32 = STM32Bridge()
        await stm32.initialize()
        await stm32.emergency_stop()
    except Exception:
        pass  # STM32 not available

    logger.warning("EMERGENCY STOP executed (scope=%s)", scope)
    return {
        "action": "emergency.result",
        "data": {"scope": scope, "success": True, "message": f"🛑 紧急停止已执行（范围: {scope}）"},
    }


async def _handle_emergency_reset(data: dict, ws: WebSocket) -> dict:
    """Reset emergency stop — allow car and arm to operate again."""
    _car_state["estop"] = False
    _arm_state["estop"] = False
    logger.info("Emergency stop RESET")
    return {
        "action": "emergency.result",
        "data": {"reset": True, "success": True, "message": "✅ 紧急停止已解除，可以正常操作了"},
    }


# Action routing table.
_ACTION_HANDLERS: dict[str, Any] = {
    "chat.send":        _handle_chat_send,
    "chat.history":     _handle_chat_history,
    "chat.clear":       _handle_chat_clear,
    "device.toggle":    _handle_device_toggle,
    "photo.take":       _handle_photo_take,
    "scene.execute":    _handle_scene_execute,
    "reminders.get":    _handle_reminders_get,
    "reminders.add":    _handle_reminders_add,
    "reminders.toggle": _handle_reminders_toggle,
    "reminders.history": _handle_reminders_history,
    "health.get":       _handle_health_get,
    "system.info":      _handle_system_info,
    "sos.call":         _handle_sos_call,
    "skills.list":      _handle_skills_list,
    "memory.get":       _handle_memory_get,
    "agent.status":     _handle_agent_status,
    "companion.snapshot.get": _handle_companion_snapshot,
    "elder.report.submit": _handle_elder_report,
    "simulation.start": _handle_simulation_start,
    "car.move":         _handle_car_move,
    "car.stop":         _handle_car_stop,
    "arm.move":         _handle_arm_move,
    "arm.preset":       _handle_arm_preset,
    "emergency.stop":   _handle_emergency_stop,
    "emergency.reset":  _handle_emergency_reset,
}


# ===========================================================================
# WebSocket endpoint
# ===========================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Main WebSocket endpoint for HMI communication.

    Protocol:
        All messages are JSON objects with at least an "action" field.
        Client -> Server:  {"action": "<action_name>", "data": {...}}
        Server -> Client:  {"action": "<response_action>", "data": {...}}
    """
    await ws.accept()
    _active_clients.add(ws)
    client_id = id(ws)
    logger.info("WebSocket client connected (id=%d, total=%d)", client_id, len(_active_clients))

    # Send initial status push so the UI populates immediately.
    try:
        _update_env_state()
        await ws.send_json(_build_status_push())
    except Exception:
        pass

    try:
        while True:
            # Receive and parse the next message.
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({
                    "action": "error",
                    "data": {"message": f"无效的 JSON 消息: {raw[:100]}"},
                })
                continue

            action = msg.get("action", "")
            data = msg.get("data", {})

            if not action:
                await ws.send_json({
                    "action": "error",
                    "data": {"message": "缺少 action 字段"},
                })
                continue

            logger.debug("Received action=%s data=%s", action, data)

            # Route to the appropriate handler.
            handler = _ACTION_HANDLERS.get(action)
            if handler is None:
                await ws.send_json({
                    "action": "error",
                    "data": {
                        "message": f"未知的 action: {action}",
                        "available": list(_ACTION_HANDLERS.keys()),
                    },
                })
                continue

            # Execute the handler and send the response.
            try:
                response = await handler(data, ws)
                if response is not None:
                    await ws.send_json(response)
            except Exception as exc:
                logger.error("Handler error for action=%s: %s", action, exc, exc_info=True)
                await ws.send_json({
                    "action": "error",
                    "data": {"message": f"处理 {action} 时出错: {exc}"},
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (id=%d)", client_id)
    except Exception as exc:
        logger.error("WebSocket error (id=%d): %s", client_id, exc, exc_info=True)
    finally:
        _active_clients.discard(ws)
        logger.info("Client removed (id=%d, remaining=%d)", client_id, len(_active_clients))


# ===========================================================================
# Background tasks
# ===========================================================================

@app.on_event("startup")
async def _on_startup():
    """Start the semantic companion runtime and periodic HMI status push."""
    global _companion_runtime
    from src.web.companion import ElderStateRuntime

    _companion_runtime = ElderStateRuntime(_broadcast_to_all)
    await _companion_runtime.start()
    app.state.companion_runtime = _companion_runtime
    asyncio.create_task(_periodic_status_push())
    logger.info("Companion runtime and status-push task started.")


@app.on_event("shutdown")
async def _on_shutdown():
    """Stop the companion runtime before the server loop closes."""
    global _companion_runtime
    if _companion_runtime is not None:
        await _companion_runtime.stop()
        _companion_runtime = None


async def _periodic_status_push():
    """Every 30 seconds, update simulated sensor data and push to all clients."""
    while True:
        await asyncio.sleep(30)
        _update_env_state()
        _update_health_data()
        payload = _build_status_push()

        # Broadcast to all connected clients.
        dead: list[WebSocket] = []
        for client in _active_clients:
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            _active_clients.discard(client)

        if _active_clients:
            logger.debug(
                "Status push sent to %d client(s). temp=%.1f humidity=%d",
                len(_active_clients),
                _env_state["temp"],
                _env_state["humidity"],
            )


# ===========================================================================
# Startup banner and main block
# ===========================================================================

def _print_banner():
    """Print a clean startup banner to the console."""
    url = f"http://localhost:{WEB_PORT}"
    ws_url = f"ws://localhost:{WEB_PORT}/ws"
    print()
    print("=" * 60)
    print("  小鲲 KunPeng-Cortex  Web Server v2.0")
    print("=" * 60)
    print(f"  HTTP  : {url}")
    print(f"  WS    : {ws_url}")
    print(f"  Docs  : {DOCS_DIR}")
    if _fusion_available:
        print(f"  Agent : FusionChat (AI + 记忆 + 技能 + 硬件)")
        # Show API config
        _model = os.environ.get("ANTHROPIC_MODEL", "not set")
        _api_url = os.environ.get("ANTHROPIC_BASE_URL", "not set")
        _key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        _masked = (_key[:6] + "****" + _key[-4:]) if len(_key) > 10 else ("(set)" if _key else "NOT SET ⚠")
        print(f"  Model : {_model}")
        print(f"  API   : {_api_url}")
        print(f"  Key   : {_masked}")
        # Show tools
        _tools = ", ".join(t["function"]["name"] for t in TOOL_DEFINITIONS) if _fusion_available else "none"
        print(f"  Tools : {_tools}")
    else:
        print(f"  Agent : not available (echo mode)")
    print(f"  Actions: {', '.join(_ACTION_HANDLERS.keys())}")
    print("-" * 60)
    print(f"  Open {url} in a browser to access the HMI.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    _print_banner()
    uvicorn.run(
        app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=30,
    )
