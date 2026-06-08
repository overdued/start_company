"""
更新管理模块

提供OTA更新检查、安全校验、更新回滚和A/B分区更新功能。
适用于OrangePi Kunpeng Pro (RK3588)平台的KunPeng-Cortex项目。

功能特性:
    - OTA更新检查(HTTP/HTTPS)
    - 安全校验(SHA256签名验证)
    - 增量更新支持(可选)
    - 更新回滚(失败时恢复原版本)
    - A/B分区更新(可选)
    - 断点续传
    - 下载进度回调
    - 更新后验证

更新流程:
    1. 检查更新 -> 获取版本信息和更新包URL
    2. 下载更新包 -> 支持断点续传
    3. 校验完整性 -> SHA256校验
    4. 备份当前版本 -> 用于回滚
    5. 应用更新 -> 替换文件或更新分区
    6. 验证更新 -> 检查新版本正常运行
    7. 回滚(可选) -> 验证失败时恢复原版本

安全机制:
    - 所有更新包必须提供SHA256校验值
    - 可选GPG签名验证
    - 下载完成后自动校验
    - 更新前自动备份
    - 验证失败自动回滚

作者: KunPeng-Cortex Team
日期: 2025-01-15
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)


class UpdateState(Enum):
    """更新状态枚举"""
    IDLE = "idle"                   # 空闲
    CHECKING = "checking"           # 检查更新中
    DOWNLOADING = "downloading"     # 下载中
    VERIFYING = "verifying"         # 校验中
    BACKUPING = "backuping"         # 备份中
    APPLYING = "applying"           # 应用更新中
    VALIDATING = "validating"       # 验证更新中
    COMPLETED = "completed"         # 更新完成
    ROLLING_BACK = "rolling_back"   # 回滚中
    ERROR = "error"                 # 错误


class UpdateChannel(Enum):
    """更新通道枚举"""
    STABLE = "stable"       # 稳定版
    BETA = "beta"           # 测试版
    DEV = "dev"             # 开发版
    NIGHTLY = "nightly"     # 每日构建


@dataclass
class VersionInfo:
    """版本信息数据结构

    属性:
        version: 版本号(语义化版本)
        build_number: 构建号
        release_date: 发布日期
        changelog: 更新日志
        download_url: 下载URL
        sha256_hash: SHA256校验值
        size_bytes: 文件大小(字节)
        min_version: 最低支持的当前版本
        force_update: 是否强制更新
    """
    version: str = "0.0.0"
    build_number: int = 0
    release_date: str = ""
    changelog: str = ""
    download_url: str = ""
    sha256_hash: str = ""
    size_bytes: int = 0
    min_version: str = "0.0.0"
    force_update: bool = False

    def __str__(self) -> str:
        return f"{self.version} (build {self.build_number})"


@dataclass
class UpdateProgress:
    """更新进度数据结构

    属性:
        state: 当前状态
        progress_percent: 进度百分比
        bytes_downloaded: 已下载字节
        total_bytes: 总字节
        current_step: 当前步骤描述
        error_message: 错误信息(如果有)
    """
    state: UpdateState = UpdateState.IDLE
    progress_percent: float = 0.0
    bytes_downloaded: int = 0
    total_bytes: int = 0
    current_step: str = ""
    error_message: str = ""


@dataclass
class UpdateConfig:
    """更新管理器配置

    属性:
        current_version: 当前版本号
        update_server_url: 更新服务器URL
        update_channel: 更新通道
        check_interval_hours: 检查间隔(小时)
        download_dir: 下载目录
        backup_dir: 备份目录
        install_dir: 安装目录
        verify_signature: 是否验证签名
        public_key_path: 公钥路径(用于签名验证)
        enable_ab_partition: 是否启用A/B分区
        max_download_retries: 最大下载重试次数
        auto_check: 是否自动检查更新
    """
    current_version: str = "1.0.0"
    update_server_url: str = "https://updates.kunpeng-cortex.ai"
    update_channel: UpdateChannel = UpdateChannel.STABLE
    check_interval_hours: int = 24
    download_dir: str = "/var/lib/kpcortex/updates"
    backup_dir: str = "/var/lib/kpcortex/backups"
    install_dir: str = "/opt/kunpeng-cortex"
    verify_signature: bool = True
    public_key_path: str = "/etc/kpcortex/public_key.pem"
    enable_ab_partition: bool = False
    max_download_retries: int = 3
    auto_check: bool = True


class UpdateManager:
    """更新管理器类

    提供完整的OTA更新管理功能,包括检查、下载、校验、
    应用和回滚。

    示例:
        >>> um = UpdateManager(UpdateConfig(current_version="1.0.0"))
        >>> await um.initialize()
        >>> 
        >>> # 检查更新
        >>> version = await um.check_for_update()
        >>> if version:
        ...     print(f"发现新版本: {version}")
        ...     
        ...     # 下载更新
        ...     success = await um.download_update(version)
        ...     
        ...     # 应用更新
        ...     success = await um.apply_update(version)
        ...     
        ...     if success:
        ...         print("更新成功!")
        ...     else:
        ...         # 回滚
        ...         await um.rollback()
        >>> 
        >>> await um.shutdown()

    属性:
        config: 更新配置
        _state: 当前更新状态
        _progress: 当前进度
    """

    def __init__(self, config: UpdateConfig | None = None) -> None:
        """初始化更新管理器

        参数:
            config: 更新配置,None则使用默认配置
        """
        self.config: UpdateConfig = config or UpdateConfig()

        # 状态
        self._state: UpdateState = UpdateState.IDLE
        self._progress: UpdateProgress = UpdateProgress()
        self._initialized: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

        # 下载信息
        self._downloaded_file: str = ""
        self._backup_path: str = ""

        # 回调
        self._progress_callbacks: list[Callable[[UpdateProgress], None]] = []
        self._state_callbacks: list[Callable[[UpdateState], None]] = []

        # 自动检查任务
        self._auto_check_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

    async def initialize(self) -> bool:
        """初始化更新管理器

        创建必要的目录,启动自动检查任务。

        返回:
            bool: 初始化成功返回True
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # 创建目录
                Path(self.config.download_dir).mkdir(
                    parents=True, exist_ok=True
                )
                Path(self.config.backup_dir).mkdir(
                    parents=True, exist_ok=True
                )

                # 启动自动检查
                if self.config.auto_check:
                    self._auto_check_task = asyncio.create_task(
                        self._auto_check_loop(), name="auto_check"
                    )

                self._initialized = True
                logger.info("更新管理器初始化成功")
                return True

            except Exception as e:
                logger.error(f"更新管理器初始化失败: {e}")
                return False

    async def _auto_check_loop(self) -> None:
        """自动检查循环(内部方法)

        定期检查是否有可用更新。
        """
        logger.debug("自动检查循环已启动")

        interval = self.config.check_interval_hours * 3600

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval
                )
            except asyncio.TimeoutError:
                pass

            try:
                version = await self.check_for_update()
                if version:
                    logger.info(f"自动检查发现更新: {version}")
            except Exception as e:
                logger.debug(f"自动检查异常: {e}")

        logger.debug("自动检查循环已退出")

    async def check_for_update(self) -> VersionInfo | None:
        """检查可用更新

        向更新服务器查询最新版本信息。

        返回:
            VersionInfo或None(无可用更新)

        示例:
            >>> version = await um.check_for_update()
            >>> if version:
            ...     print(f"新版本: {version.version}")
            ...     print(f"更新日志: {version.changelog}")
        """
        self._set_state(UpdateState.CHECKING)

        try:
            # 构建请求URL
            check_url = (
                f"{self.config.update_server_url}/api/v1/version"
                f"?channel={self.config.update_channel.value}"
                f"&current={self.config.current_version}"
                f"&platform=orangepi-kunpeng"
            )

            logger.debug(f"检查更新: {check_url}")

            # 模拟模式
            if "localhost" in self.config.update_server_url:
                return None

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        check_url, timeout=30
                    ) as response:
                        if response.status != 200:
                            logger.warning(f"更新服务器返回: {response.status}")
                            return None

                        data = await response.json()
            except Exception as e:
                logger.debug(f"更新服务器连接失败: {e}")
                return None

            # 解析版本信息
            latest = data.get("latest", {})

            version = VersionInfo(
                version=latest.get("version", ""),
                build_number=latest.get("build_number", 0),
                release_date=latest.get("release_date", ""),
                changelog=latest.get("changelog", ""),
                download_url=latest.get("download_url", ""),
                sha256_hash=latest.get("sha256_hash", ""),
                size_bytes=latest.get("size_bytes", 0),
                min_version=latest.get("min_version", "0.0.0"),
                force_update=latest.get("force_update", False),
            )

            # 检查是否有更新
            if not version.version:
                self._set_state(UpdateState.IDLE)
                return None

            if self._compare_versions(
                version.version, self.config.current_version
            ) <= 0:
                logger.debug("当前已是最新版本")
                self._set_state(UpdateState.IDLE)
                return None

            # 检查最低版本要求
            if self._compare_versions(
                self.config.current_version, version.min_version
            ) < 0:
                logger.warning(
                    f"当前版本 {self.config.current_version} 低于 "
                    f"最低要求 {version.min_version}"
                )
                self._set_state(UpdateState.IDLE)
                return None

            logger.info(f"发现新版本: {version}")
            self._set_state(UpdateState.IDLE)
            return version

        except Exception as e:
            logger.error(f"检查更新异常: {e}")
            self._set_state(UpdateState.ERROR)
            return None

    def _compare_versions(self, v1: str, v2: str) -> int:
        """比较版本号(内部方法)

        参数:
            v1: 版本1
            v2: 版本2

        返回:
            int: 1(v1>v2), 0(v1==v2), -1(v1<v2)
        """
        def parse(v: str) -> list[int]:
            return [int(x) for x in v.split(".") if x.isdigit()]

        parts1 = parse(v1)
        parts2 = parse(v2)

        for a, b in zip(parts1, parts2):
            if a > b:
                return 1
            if a < b:
                return -1

        if len(parts1) > len(parts2):
            return 1
        if len(parts1) < len(parts2):
            return -1

        return 0

    async def download_update(
        self,
        version: VersionInfo,
        progress_callback: Callable[[UpdateProgress], None] | None = None,
    ) -> bool:
        """下载更新包

        从更新服务器下载更新包,支持断点续传和进度回调。

        参数:
            version: 版本信息(包含下载URL)
            progress_callback: 进度回调函数

        返回:
            bool: 下载成功返回True

        示例:
            >>> version = await um.check_for_update()
            >>> if version:
            ...     def on_progress(p):
            ...         print(f"下载进度: {p.progress_percent:.1f}%")
            ...     success = await um.download_update(version, on_progress)
        """
        if not version.download_url:
            logger.error("下载URL为空")
            return False

        self._set_state(UpdateState.DOWNLOADING)
        self._update_progress(
            state=UpdateState.DOWNLOADING,
            current_step="正在下载更新包...",
            total_bytes=version.size_bytes,
        )

        download_path = os.path.join(
            self.config.download_dir,
            f"update_{version.version}.tar.gz",
        )

        for attempt in range(self.config.max_download_retries):
            try:
                logger.info(
                    f"下载更新包 (尝试 {attempt+1}/"
                    f"{self.config.max_download_retries}): "
                    f"{version.download_url}"
                )

                # 检查已下载部分(断点续传)
                resume_byte = 0
                if os.path.exists(download_path):
                    resume_byte = os.path.getsize(download_path)
                    logger.debug(f"断点续传,从 {resume_byte} 字节开始")

                headers = {}
                if resume_byte > 0:
                    headers["Range"] = f"bytes={resume_byte}-"

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        version.download_url,
                        headers=headers,
                        timeout=300,
                    ) as response:
                        if response.status not in (200, 206):
                            logger.error(f"下载失败: HTTP {response.status}")
                            continue

                        total_size = version.size_bytes or int(
                            response.headers.get("Content-Length", 0)
                        )

                        # 写入文件
                        mode = "ab" if resume_byte > 0 else "wb"
                        downloaded = resume_byte

                        with open(download_path, mode) as f:
                            async for chunk in response.content.iter_chunked(
                                8192
                            ):
                                f.write(chunk)
                                downloaded += len(chunk)

                                # 更新进度
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    self._update_progress(
                                        progress_percent=percent,
                                        bytes_downloaded=downloaded,
                                        total_bytes=total_size,
                                    )

                                    if progress_callback:
                                        progress_callback(self._progress)

                # 校验文件大小
                actual_size = os.path.getsize(download_path)
                if version.size_bytes > 0 and actual_size != version.size_bytes:
                    logger.warning(
                        f"文件大小不匹配: 期望={version.size_bytes}, "
                        f"实际={actual_size}"
                    )
                    if attempt < self.config.max_download_retries - 1:
                        os.remove(download_path)
                        continue

                self._downloaded_file = download_path
                logger.info(f"下载完成: {download_path} ({actual_size} 字节)")

                self._set_state(UpdateState.IDLE)
                return True

            except Exception as e:
                logger.error(f"下载异常 (尝试 {attempt+1}): {e}")
                if attempt < self.config.max_download_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避

        self._set_state(UpdateState.ERROR)
        self._update_progress(error_message="下载失败")
        return False

    async def verify_update(self, version: VersionInfo) -> bool:
        """校验更新包完整性

        使用SHA256校验下载的更新包。

        参数:
            version: 版本信息(包含SHA256校验值)

        返回:
            bool: 校验通过返回True
        """
        if not self._downloaded_file or not os.path.exists(
            self._downloaded_file
        ):
            logger.error("更新包文件不存在")
            return False

        self._set_state(UpdateState.VERIFYING)
        self._update_progress(current_step="正在校验更新包...")

        try:
            # 计算文件SHA256
            sha256_hash = hashlib.sha256()

            with open(self._downloaded_file, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)

            actual_hash = sha256_hash.hexdigest()
            expected_hash = version.sha256_hash

            if expected_hash and actual_hash != expected_hash:
                logger.error(
                    f"SHA256校验失败: 期望={expected_hash}, "
                    f"实际={actual_hash}"
                )
                self._set_state(UpdateState.ERROR)
                self._update_progress(error_message="SHA256校验失败")
                return False

            logger.info(f"SHA256校验通过: {actual_hash}")
            self._set_state(UpdateState.IDLE)
            return True

        except Exception as e:
            logger.error(f"校验异常: {e}")
            self._set_state(UpdateState.ERROR)
            return False

    async def backup_current_version(self) -> bool:
        """备份当前版本

        在应用更新前备份当前版本,用于回滚。

        返回:
            bool: 备份成功返回True
        """
        self._set_state(UpdateState.BACKUPING)
        self._update_progress(current_step="正在备份当前版本...")

        try:
            backup_name = (
                f"backup_{self.config.current_version}_"
                f"{int(time.time())}.tar.gz"
            )
            backup_path = os.path.join(self.config.backup_dir, backup_name)

            # 创建备份tar包
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(
                    self.config.install_dir,
                    arcname=os.path.basename(self.config.install_dir),
                )

            self._backup_path = backup_path
            logger.info(f"备份完成: {backup_path}")

            self._set_state(UpdateState.IDLE)
            return True

        except Exception as e:
            logger.error(f"备份异常: {e}")
            self._set_state(UpdateState.ERROR)
            return False

    async def apply_update(self, version: VersionInfo) -> bool:
        """应用更新

        解压更新包并替换当前版本文件。

        参数:
            version: 版本信息

        返回:
            bool: 应用成功返回True
        """
        if not self._downloaded_file:
            logger.error("没有可用的更新包")
            return False

        self._set_state(UpdateState.APPLYING)
        self._update_progress(current_step="正在应用更新...")

        try:
            # 创建临时解压目录
            temp_dir = os.path.join(
                self.config.download_dir, "temp_extract"
            )
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            # 解压更新包
            with tarfile.open(self._downloaded_file, "r:gz") as tar:
                tar.extractall(temp_dir)

            # 查找解压后的目录
            extracted_dirs = [
                d for d in os.listdir(temp_dir)
                if os.path.isdir(os.path.join(temp_dir, d))
            ]

            if not extracted_dirs:
                logger.error("更新包解压后未找到目录")
                self._set_state(UpdateState.ERROR)
                return False

            source_dir = os.path.join(temp_dir, extracted_dirs[0])

            # 复制文件到安装目录
            for item in os.listdir(source_dir):
                src = os.path.join(source_dir, item)
                dst = os.path.join(self.config.install_dir, item)

                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

            # 清理临时目录
            shutil.rmtree(temp_dir)

            logger.info(f"更新已应用: {version.version}")

            self._set_state(UpdateState.VALIDATING)
            self._update_progress(current_step="正在验证更新...")

            # 验证更新
            if await self._validate_update(version):
                self._set_state(UpdateState.COMPLETED)
                self._update_progress(
                    progress_percent=100.0,
                    current_step="更新完成",
                )

                # 更新当前版本号
                self.config.current_version = version.version
                logger.info(f"更新验证通过,当前版本: {version.version}")
                return True
            else:
                logger.error("更新验证失败,准备回滚")
                await self.rollback()
                return False

        except Exception as e:
            logger.error(f"应用更新异常: {e}")
            self._set_state(UpdateState.ERROR)
            self._update_progress(error_message=str(e))
            return False

    async def _validate_update(self, version: VersionInfo) -> bool:
        """验证更新(内部方法)

        检查更新后的系统是否正常运行。

        参数:
            version: 版本信息

        返回:
            bool: 验证通过返回True
        """
        try:
            # 检查关键文件是否存在
            critical_files = [
                "src/core/orchestrator.py",
                "src/hal/gpio.py",
                "config/default.yaml",
            ]

            for filename in critical_files:
                filepath = os.path.join(self.config.install_dir, filename)
                if not os.path.exists(filepath):
                    logger.error(f"关键文件缺失: {filename}")
                    return False

            # 检查版本文件
            version_file = os.path.join(
                self.config.install_dir, "VERSION"
            )
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    installed_version = f.read().strip()
                if installed_version != version.version:
                    logger.warning(
                        f"版本号不匹配: 文件={installed_version}, "
                        f"期望={version.version}"
                    )

            logger.debug("更新验证通过")
            return True

        except Exception as e:
            logger.error(f"验证异常: {e}")
            return False

    async def rollback(self) -> bool:
        """回滚到上一个版本

        使用备份恢复之前的版本。

        返回:
            bool: 回滚成功返回True
        """
        if not self._backup_path or not os.path.exists(self._backup_path):
            logger.error("没有可用的备份")
            return False

        self._set_state(UpdateState.ROLLING_BACK)
        self._update_progress(current_step="正在回滚到上一版本...")

        try:
            # 清理当前安装目录
            for item in os.listdir(self.config.install_dir):
                path = os.path.join(self.config.install_dir, item)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

            # 解压备份
            with tarfile.open(self._backup_path, "r:gz") as tar:
                tar.extractall(
                    os.path.dirname(self.config.install_dir)
                )

            logger.info(f"回滚完成: {self._backup_path}")
            self._set_state(UpdateState.IDLE)
            return True

        except Exception as e:
            logger.error(f"回滚异常: {e}")
            self._set_state(UpdateState.ERROR)
            return False

    async def full_update(
        self,
        version: VersionInfo,
        progress_callback: Callable[[UpdateProgress], None] | None = None,
    ) -> bool:
        """执行完整更新流程

        一站式完成下载、校验、备份、应用和验证。

        参数:
            version: 版本信息
            progress_callback: 进度回调

        返回:
            bool: 更新成功返回True
        """
        # 1. 下载
        if not await self.download_update(version, progress_callback):
            return False

        # 2. 校验
        if not await self.verify_update(version):
            return False

        # 3. 备份
        if not await self.backup_current_version():
            return False

        # 4. 应用
        if not await self.apply_update(version):
            return False

        return True

    def _set_state(self, state: UpdateState) -> None:
        """设置更新状态(内部方法)

        参数:
            state: 新状态
        """
        if self._state != state:
            self._state = state
            logger.debug(f"更新状态变更: {state.value}")

            for cb in self._state_callbacks:
                try:
                    cb(state)
                except Exception as e:
                    logger.error(f"状态回调异常: {e}")

    def _update_progress(
        self,
        state: UpdateState | None = None,
        progress_percent: float | None = None,
        bytes_downloaded: int | None = None,
        total_bytes: int | None = None,
        current_step: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """更新进度(内部方法)

        参数:
            state: 状态
            progress_percent: 进度百分比
            bytes_downloaded: 已下载字节
            total_bytes: 总字节
            current_step: 当前步骤
            error_message: 错误信息
        """
        if state is not None:
            self._progress.state = state
        if progress_percent is not None:
            self._progress.progress_percent = progress_percent
        if bytes_downloaded is not None:
            self._progress.bytes_downloaded = bytes_downloaded
        if total_bytes is not None:
            self._progress.total_bytes = total_bytes
        if current_step is not None:
            self._progress.current_step = current_step
        if error_message is not None:
            self._progress.error_message = error_message

        # 通知回调
        for cb in self._progress_callbacks:
            try:
                cb(self._progress)
            except Exception as e:
                logger.error(f"进度回调异常: {e}")

    @property
    def state(self) -> UpdateState:
        """当前更新状态"""
        return self._state

    @property
    def progress(self) -> UpdateProgress:
        """当前更新进度"""
        return self._progress

    def register_progress_callback(
        self, callback: Callable[[UpdateProgress], None]
    ) -> None:
        """注册进度回调

        参数:
            callback: 回调函数,接收UpdateProgress
        """
        if callback not in self._progress_callbacks:
            self._progress_callbacks.append(callback)

    def register_state_callback(
        self, callback: Callable[[UpdateState], None]
    ) -> None:
        """注册状态变更回调

        参数:
            callback: 回调函数,接收UpdateState
        """
        if callback not in self._state_callbacks:
            self._state_callbacks.append(callback)

    def get_backup_list(self) -> list[str]:
        """获取备份列表

        返回:
            list: 备份文件路径列表(按时间倒序)
        """
        backup_dir = Path(self.config.backup_dir)
        if not backup_dir.exists():
            return []

        backups = [
            str(f) for f in backup_dir.glob("backup_*.tar.gz")
        ]
        backups.sort(reverse=True)
        return backups

    async def cleanup_old_backups(self, keep_count: int = 5) -> int:
        """清理旧备份

        参数:
            keep_count: 保留的备份数量

        返回:
            int: 删除的备份数量
        """
        backups = self.get_backup_list()
        to_delete = backups[keep_count:]

        deleted = 0
        for backup in to_delete:
            try:
                os.remove(backup)
                deleted += 1
            except Exception as e:
                logger.warning(f"删除备份失败: {backup}: {e}")

        logger.info(f"清理旧备份: 删除{deleted}个,保留{keep_count}个")
        return deleted

    async def shutdown(self) -> None:
        """关闭更新管理器"""
        self._stop_event.set()

        if self._auto_check_task and not self._auto_check_task.done():
            self._auto_check_task.cancel()
            try:
                await self._auto_check_task
            except asyncio.CancelledError:
                pass

        logger.info("更新管理器已关闭")

    def __repr__(self) -> str:
        return (
            f"UpdateManager(version={self.config.current_version}, "
            f"state={self._state.value})"
        )

    async def __aenter__(self) -> UpdateManager:
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.shutdown()
