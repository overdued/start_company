"""
配置管理模块

提供YAML配置加载/保存、JSON Schema验证、环境变量覆盖和热重载功能。
适用于OrangePi Kunpeng Pro (RK3588)平台的KunPeng-Cortex项目。

功能特性:
    - YAML配置文件的加载和保存
    - JSON Schema配置验证
    - 环境变量覆盖配置值
    - 配置文件热重载(文件变更自动加载)
    - 多级配置合并(默认+用户+环境变量)
    - 配置变更回调

配置优先级(从高到低):
    1. 环境变量(KPCORTEX_*)
    2. 用户配置文件(/etc/kpcortex/config.yaml)
    3. 项目配置文件(./config/default.yaml)
    4. 内置默认值

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """配置来源枚举"""
    DEFAULT = "default"         # 内置默认值
    FILE = "file"               # 配置文件
    ENVIRONMENT = "environment" # 环境变量
    RUNTIME = "runtime"         # 运行时修改


@dataclass
class ConfigEntry:
    """配置项数据结构

    属性:
        value: 配置值
        source: 配置来源
        description: 配置说明
        schema_type: Schema类型
    """
    value: Any = None
    source: ConfigSource = ConfigSource.DEFAULT
    description: str = ""
    schema_type: str = "string"


@dataclass
class ConfigManagerConfig:
    """配置管理器配置

    属性:
        default_config_path: 默认配置文件路径
        user_config_path: 用户配置文件路径
        system_config_path: 系统级配置文件路径
        env_prefix: 环境变量前缀
        enable_hot_reload: 是否启用热重载
        hot_reload_interval: 热重载检查间隔(秒)
        enable_schema_validation: 是否启用Schema验证
    """
    default_config_path: str = "config/default.yaml"
    user_config_path: str = "~/.config/kpcortex/config.yaml"
    system_config_path: str = "/etc/kpcortex/config.yaml"
    env_prefix: str = "KPCORTEX_"
    enable_hot_reload: bool = True
    hot_reload_interval: float = 5.0
    enable_schema_validation: bool = True


class ConfigManager:
    """配置管理器类

    提供配置加载、验证、合并和热重载功能。
    支持多级配置优先级和环境变量覆盖。

    示例:
        >>> cm = ConfigManager(ConfigManagerConfig())
        >>> await cm.initialize()
        >>> 
        >>> # 获取配置值
        >>> debug = cm.get("system.debug", False)
        >>> port = cm.get("server.port", 8765)
        >>> 
        >>> # 设置配置值
        >>> cm.set("server.port", 9000)
        >>> 
        >>> # 获取完整配置
        >>> config = cm.get_all()
        >>> 
        >>> await cm.shutdown()

    属性:
        _config: 合并后的配置字典
        _schema: JSON Schema验证规则
        _file_mtimes: 配置文件修改时间
    """

    # JSON Schema默认验证规则
    DEFAULT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "system": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "debug": {"type": "boolean"},
                    "log_level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    },
                },
            },
            "server": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                },
            },
            "hardware": {
                "type": "object",
                "properties": {
                    "i2c_bus": {"type": "integer"},
                    "uart_port": {"type": "string"},
                    "gpio_chip": {"type": "string"},
                },
            },
            "emotion": {
                "type": "object",
                "properties": {
                    "personality": {"type": "string"},
                    "response_speed": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                },
            },
        },
    }

    def __init__(self, config: ConfigManagerConfig | None = None) -> None:
        """初始化配置管理器

        参数:
            config: 配置管理器配置,None则使用默认配置
        """
        self._manager_config: ConfigManagerConfig = config or ConfigManagerConfig()

        # 配置存储
        self._config: dict[str, Any] = {}
        self._schema: dict[str, Any] = dict(self.DEFAULT_SCHEMA)

        # 文件监控
        self._file_mtimes: dict[str, float] = {}
        self._hot_reload_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

        # 变更回调
        self._change_callbacks: list[Callable[[str, Any, Any], None]] = []

        # 状态
        self._initialized: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """初始化配置管理器

        加载所有配置文件,按优先级合并,启动热重载监控。

        返回:
            bool: 初始化成功返回True
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # 1. 加载内置默认配置
                self._config = self._load_default_config()

                # 2. 加载配置文件(低优先级先加载)
                config_paths = [
                    Path(self._manager_config.default_config_path).expanduser(),
                    Path(self._manager_config.system_config_path),
                    Path(self._manager_config.user_config_path).expanduser(),
                ]

                for config_path in config_paths:
                    if config_path.exists():
                        logger.debug(f"加载配置文件: {config_path}")
                        file_config = self._load_yaml_file(str(config_path))
                        if file_config:
                            self._deep_merge(self._config, file_config)
                            self._file_mtimes[str(config_path)] =                                 config_path.stat().st_mtime

                # 3. 环境变量覆盖
                self._apply_environment_overrides()

                # 4. Schema验证
                if self._manager_config.enable_schema_validation:
                    self._validate_config()

                # 5. 启动热重载
                if self._manager_config.enable_hot_reload:
                    self._hot_reload_task = asyncio.create_task(
                        self._hot_reload_loop(), name="config_reload"
                    )

                self._initialized = True
                logger.info("配置管理器初始化成功")
                return True

            except Exception as e:
                logger.error(f"配置管理器初始化失败: {e}")
                # 使用最小默认配置
                self._config = self._load_default_config()
                self._initialized = True
                return True

    def _load_default_config(self) -> dict[str, Any]:
        """加载内置默认配置(内部方法)

        返回:
            dict: 默认配置字典
        """
        return {
            "system": {
                "name": "KunPeng-Cortex",
                "version": "1.0.0",
                "debug": False,
                "log_level": "INFO",
                "data_dir": "/var/lib/kpcortex",
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8765,
                "max_clients": 10,
            },
            "hardware": {
                "i2c_bus": 1,
                "uart_port": "/dev/ttyS2",
                "gpio_chip": "0",
                "pwm_frequency": 50,
            },
            "emotion": {
                "personality": "caring",
                "response_speed": 1.0,
                "tts_volume": 0.8,
            },
            "safety": {
                "emergency_timeout": 0.1,
                "heartbeat_interval": 0.05,
                "max_missed_heartbeats": 3,
            },
            "update": {
                "auto_check": True,
                "check_interval_hours": 24,
                "channel": "stable",
            },
        }

    def _load_yaml_file(self, filepath: str) -> dict[str, Any] | None:
        """加载YAML文件(内部方法)

        参数:
            filepath: 文件路径

        返回:
            dict或None(加载失败)
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
            return None
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"加载YAML文件失败: {filepath}: {e}")
            return None

    def _deep_merge(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """深度合并字典(内部方法)

        将override合并到base中,支持嵌套字典。

        参数:
            base: 基础字典
            override: 覆盖字典

        返回:
            dict: 合并后的字典
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _apply_environment_overrides(self) -> None:
        """应用环境变量覆盖(内部方法)

        将 KPCORTEX_* 环境变量解析并覆盖到配置中。
        格式: KPCORTEX_SECTION_KEY=value
        例如: KPCORTEX_SERVER_PORT=9000
        """
        prefix = self._manager_config.env_prefix

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 解析: KPCORTEX_SECTION_SUBSECTION_KEY
                parts = key[len(prefix):].lower().split("_")

                if len(parts) >= 2:
                    # 将值转换为适当类型
                    typed_value = self._parse_value(value)

                    # 设置嵌套配置值
                    self._set_nested_value(self._config, parts, typed_value)

                    config_path = ".".join(parts)
                    logger.debug(f"环境变量覆盖: {config_path}={typed_value}")

    def _parse_value(self, value: str) -> Any:
        """解析字符串值为适当类型(内部方法)

        参数:
            value: 字符串值

        返回:
            解析后的值(int/float/bool/str)
        """
        # 布尔值
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # 整数
        try:
            return int(value)
        except ValueError:
            pass

        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass

        # 字符串
        return value

    def _set_nested_value(
        self, config: dict[str, Any], keys: list[str], value: Any
    ) -> None:
        """设置嵌套字典值(内部方法)

        参数:
            config: 配置字典
            keys: 键路径列表
            value: 要设置的值
        """
        current = config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _validate_config(self) -> bool:
        """验证配置(内部方法)

        使用JSON Schema验证当前配置。

        返回:
            bool: 验证通过返回True
        """
        try:
            from jsonschema import validate, ValidationError

            validate(instance=self._config, schema=self._schema)
            logger.debug("配置验证通过")
            return True

        except ImportError:
            logger.debug("jsonschema未安装,跳过验证")
            return True
        except Exception as e:
            logger.warning(f"配置验证失败: {e}")
            return False

    async def _hot_reload_loop(self) -> None:
        """热重载监控循环(内部方法)

        定期检查配置文件是否变更,变更则自动重新加载。
        """
        logger.debug("配置热重载监控已启动")

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._manager_config.hot_reload_interval,
                )
            except asyncio.TimeoutError:
                pass

            # 检查文件修改时间
            for filepath, old_mtime in list(self._file_mtimes.items()):
                try:
                    new_mtime = Path(filepath).stat().st_mtime
                    if new_mtime > old_mtime:
                        logger.info(f"配置文件变更,重新加载: {filepath}")

                        new_config = self._load_yaml_file(filepath)
                        if new_config:
                            # 重新加载所有配置
                            await self.reload()

                        self._file_mtimes[filepath] = new_mtime

                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.debug(f"检查文件失败: {filepath}: {e}")

        logger.debug("配置热重载监控已退出")

    async def reload(self) -> bool:
        """重新加载配置

        重新加载所有配置文件并合并。

        返回:
            bool: 加载成功返回True
        """
        async with self._lock:
            old_config = dict(self._config)

            try:
                # 重新加载
                self._config = self._load_default_config()

                config_paths = [
                    Path(self._manager_config.default_config_path).expanduser(),
                    Path(self._manager_config.system_config_path),
                    Path(self._manager_config.user_config_path).expanduser(),
                ]

                for config_path in config_paths:
                    if config_path.exists():
                        file_config = self._load_yaml_file(str(config_path))
                        if file_config:
                            self._deep_merge(self._config, file_config)

                self._apply_environment_overrides()

                if self._manager_config.enable_schema_validation:
                    self._validate_config()

                # 通知变更
                self._notify_changes(old_config, self._config)

                logger.info("配置已重新加载")
                return True

            except Exception as e:
                logger.error(f"配置重新加载失败: {e}")
                self._config = old_config
                return False

    def _notify_changes(
        self, old: dict[str, Any], new: dict[str, Any], prefix: str = ""
    ) -> None:
        """通知配置变更(内部方法)

        对比新旧配置,调用回调通知变更项。

        参数:
            old: 旧配置
            new: 新配置
            prefix: 键前缀
        """
        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            old_val = old.get(key)
            new_val = new.get(key)
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(old_val, dict) and isinstance(new_val, dict):
                self._notify_changes(old_val, new_val, full_key)
            elif old_val != new_val:
                for cb in self._change_callbacks:
                    try:
                        cb(full_key, old_val, new_val)
                    except Exception as e:
                        logger.error(f"配置变更回调异常: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        使用点号分隔的路径获取嵌套配置值。

        参数:
            key: 配置键路径,如"server.port"
            default: 默认值(键不存在时返回)

        返回:
            配置值或默认值

        示例:
            >>> port = cm.get("server.port", 8765)
            >>> debug = cm.get("system.debug", False)
            >>> nested = cm.get("hardware.i2c.bus", 1)
        """
        keys = key.split(".")
        current = self._config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set(self, key: str, value: Any) -> bool:
        """设置配置值

        使用点号分隔的路径设置嵌套配置值。

        参数:
            key: 配置键路径
            value: 配置值

        返回:
            bool: 设置成功返回True

        示例:
            >>> cm.set("server.port", 9000)
            >>> cm.set("system.debug", True)
        """
        keys = key.split(".")

        try:
            old_value = self.get(key)
            self._set_nested_value(self._config, keys, value)

            # 通知变更
            for cb in self._change_callbacks:
                try:
                    cb(key, old_value, value)
                except Exception as e:
                    logger.error(f"配置变更回调异常: {e}")

            return True

        except Exception as e:
            logger.error(f"设置配置失败: {key}: {e}")
            return False

    def get_all(self) -> dict[str, Any]:
        """获取完整配置

        返回:
            dict: 完整的配置字典(深拷贝)
        """
        import copy
        return copy.deepcopy(self._config)

    async def save(self, filepath: str | None = None) -> bool:
        """保存配置到文件

        将当前配置保存为YAML文件。

        参数:
            filepath: 保存路径,None则使用用户配置路径

        返回:
            bool: 保存成功返回True
        """
        filepath = filepath or self._manager_config.user_config_path
        filepath = os.path.expanduser(filepath)

        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(
                    self._config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=True,
                )

            logger.info(f"配置已保存: {filepath}")
            return True

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False

    def register_change_callback(
        self, callback: Callable[[str, Any, Any], None]
    ) -> None:
        """注册配置变更回调

        当配置值发生变化时调用回调。

        参数:
            callback: 回调函数,接收(key, old_value, new_value)
        """
        if callback not in self._change_callbacks:
            self._change_callbacks.append(callback)

    def unregister_change_callback(
        self, callback: Callable[[str, Any, Any], None]
    ) -> None:
        """注销配置变更回调

        参数:
            callback: 要移除的回调函数
        """
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)

    async def shutdown(self) -> None:
        """关闭配置管理器"""
        self._stop_event.set()

        if self._hot_reload_task and not self._hot_reload_task.done():
            self._hot_reload_task.cancel()
            try:
                await self._hot_reload_task
            except asyncio.CancelledError:
                pass

        logger.info("配置管理器已关闭")

    def __getitem__(self, key: str) -> Any:
        """字典风格访问"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """字典风格设置"""
        self.set(key, value)

    def __repr__(self) -> str:
        return f"ConfigManager(configured={self._initialized})"

    async def __aenter__(self) -> ConfigManager:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
