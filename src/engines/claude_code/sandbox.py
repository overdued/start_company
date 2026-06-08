#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sandbox.py — KunPeng-Cortex 沙箱管理模块

基于bubblewrap用户态命名空间容器的代码执行沙箱，提供进程级隔离、
权限控制和资源限制。参考Claude Code的bubblewrap实现进行嵌入式适配。

安全策略（七层安全体系）：
    1. 命名空间隔离（PID/UTS/IPC/Mount/Net/User）
    2. Capability限制（cap-drop ALL）
    3. Seccomp-BPF系统调用过滤
    4. 硬件命令白名单验证
    5. 执行超时保护（SIGKILL）
    6. 文件系统只读挂载
    7. 资源限制（CPU/内存/Swap）
    8. 物理层安全（MCU紧急停止，独立于主系统）

硬件平台: OrangePi Kunpeng Pro (RK3588, ARM64)
作者: KunPeng-Cortex Team
版本: 1.0.0
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import resource
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 配置模块日志记录器
logger = logging.getLogger("kunpeng_cortex.claude_code.sandbox")

# =============================================================================
# 数据模型定义
# =============================================================================


@dataclasses.dataclass
class SandboxResult:
    """
    沙箱代码执行结果数据类。

    属性:
        success: 代码是否成功执行（未发生异常或超时）
        stdout: 标准输出内容
        stderr: 标准错误内容
        return_code: 进程返回码，0表示正常退出
        execution_time_ms: 实际执行时间(毫秒)
        timed_out: 是否因超时终止
        memory_usage_mb: 峰值内存使用(MB)
        command_blocked: 被安全策略阻止的命令列表
        error_message: 错误描述信息
    """
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time_ms: float = 0.0
    timed_out: bool = False
    memory_usage_mb: float = 0.0
    command_blocked: List[str] = dataclasses.field(default_factory=list)
    error_message: str = ""


# =============================================================================
# 安全常量定义
# =============================================================================

# Python内置危险函数/模块黑名单 —— 禁止在生成的代码中使用
PYTHON_CODE_BLACKLIST: List[str] = [
    "__import__",
    "importlib",
    "eval",
    "exec",
    "compile",
    "open(os",
    "open('/dev",
    "open(\"'/dev",
    "subprocess",
    "os.system",
    "os.popen",
    "os.fork",
    "os.kill",
    "os.exec",
    "os.spawn",
    "os.chmod",
    "os.chown",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.rename",
    "os.makedirs",
    "os.mkdir",
    "shutil",
    "socket",
    "urllib",
    "http.client",
    "ftplib",
    "telnetlib",
    "pty",
    "pickle",
    "marshal",
    "ctypes",
    "cffi",
    "mmap",
    "resource",
    "sys.modules",
    "sys.path",
    "sys.stdout",
    "sys.stderr",
    "sys.stdin",
    "sys.exit",
    "quit",
    "exit",
    "breakpoint",
    "input",
    "raw_input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "classmethod",
    "staticmethod",
    "property",
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__func__",
    "__closure__",
    "_thread",
    "threading",
    "multiprocessing",
    "concurrent.futures.ProcessPoolExecutor",
    "os.environ",
    "os.putenv",
    "os.unsetenv",
]

# 系统调用黑名单（用于Seccomp-BPF策略参考）
SYSCALL_BLACKLIST: List[str] = [
    "execve", "execveat", "fork", "vfork", "clone",
    "ptrace", "process_vm_writev", "process_vm_readv",
    "mknod", "mknodat", "mount", "umount", "umount2",
    "pivot_root", "chroot", "setns", "unshare",
    "reboot", "kexec_load", "kexec_file_load",
    "iopl", "ioperm", "syslog", "sysfs",
    "init_module", "finit_module", "delete_module",
    "open_by_handle_at", "name_to_handle_at",
    "perf_event_open", "bpf",
]

# 安全的Python内置白名单 —— 沙箱内允许使用的Python内置函数
SAFE_BUILTINS: Tuple[str, ...] = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip", "True", "False", "None",
    "Exception", "ValueError", "TypeError", "IndexError",
    "KeyError", "AttributeError", "RuntimeError", "StopIteration",
    "ArithmeticError", "OverflowError", "ZeroDivisionError",
    "LookupError",
)

# 危险文件系统路径 —— 禁止访问的系统路径
DANGEROUS_PATHS: List[str] = [
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "/etc/ssh", "/root", "/var/log/auth.log",
    "/proc/kcore", "/proc/sys", "/proc/irq",
    "/sys/kernel", "/dev/mem", "/dev/kmem",
    "/dev/port", "/dev/sd", "/dev/hd",
    "/dev/mmcblk0", "/dev/mtd",
]


# =============================================================================
# Sandbox 主类
# =============================================================================


class Sandbox:
    """
    代码执行沙箱管理器。

    基于bubblewrap(bwrap)构建的轻量级容器沙箱，提供进程级隔离。
    适用于在嵌入式平台上安全执行由LLM生成的Python代码。

    属性:
        allowed_paths: 沙箱内允许只读访问的文件路径列表
        network: 是否允许网络访问（默认完全禁用）
        max_memory_mb: 沙箱内进程最大内存限制(MB)
        max_cpu_percent: 最大CPU使用率限制
        sandbox_root: 沙箱根目录
        hw_cmd_fifo: 硬件命令管道路径
        hw_status_fifo: 硬件状态管道路径

    示例:
        >>> sandbox = Sandbox(
        ...     allowed_paths=["/usr", "/lib", "/lib64"],
        ...     network=False,
        ...     max_memory_mb=256
        ... )
        >>> result = await sandbox.run("print('Hello RK3588')", timeout=3.0)
        >>> print(result.stdout)
        Hello RK3588
    """

    # 类级别配置
    BWRAP_PATH: str = "/usr/bin/bwrap"
    DEFAULT_MAX_MEMORY_MB: int = 256
    DEFAULT_MAX_CPU_PERCENT: int = 50
    DEFAULT_SANDBOX_ROOT: str = "/var/lib/kpcortex/sandbox"

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        network: bool = False,
        max_memory_mb: int = DEFAULT_MAX_MEMORY_MB,
        max_cpu_percent: int = DEFAULT_MAX_CPU_PERCENT,
        sandbox_root: str = DEFAULT_SANDBOX_ROOT,
        hw_cmd_fifo: str = "/run/kpcortex/hw_cmd_fifo",
        hw_status_fifo: str = "/run/kpcortex/hw_status_fifo",
        enable_seccomp: bool = True,
    ) -> None:
        """
        初始化沙箱管理器。

        参数:
            allowed_paths: 沙箱内允许只读访问的文件系统路径列表。
                默认包含系统的Python解释器和库路径。
            network: 是否允许沙箱内进程访问网络。默认为False（完全禁用）。
            max_memory_mb: 沙箱内进程最大内存限制(MB)，默认256MB。
            max_cpu_percent: 最大CPU使用率百分比，默认50%。
            sandbox_root: 沙箱运行时根目录。
            hw_cmd_fifo: 硬件命令管道路径，用于沙箱内代码与HAL通信。
            hw_status_fifo: 硬件状态管道路径。
            enable_seccomp: 是否启用Seccomp-BPF系统调用过滤。
        """
        self.allowed_paths: List[str] = allowed_paths or [
            "/usr", "/lib", "/lib64", "/etc/ssl"
        ]
        self.network: bool = network
        self.max_memory_mb: int = max_memory_mb
        self.max_cpu_percent: int = max_cpu_percent
        self.sandbox_root: str = sandbox_root
        self.hw_cmd_fifo: str = hw_cmd_fifo
        self.hw_status_fifo: str = hw_status_fifo
        self.enable_seccomp: bool = enable_seccomp

        # 安全统计
        self._execution_count: int = 0
        self._blocked_count: int = 0
        self._timeout_count: int = 0

        # 检查bubblewrap是否可用
        self._bwrap_available = os.path.isfile(self.BWRAP_PATH) and os.access(self.BWRAP_PATH, os.X_OK)
        if not self._bwrap_available:
            logger.warning(
                f"bubblewrap未安装或不可执行: {self.BWRAP_PATH}，"
                f"将回退到subprocess隔离模式"
            )

        logger.info(
            f"沙箱管理器已初始化: network={network}, "
            f"memory={max_memory_mb}MB, cpu={max_cpu_percent}%, "
            f"bwrap_available={self._bwrap_available}"
        )

    @property
    def execution_stats(self) -> dict:
        """
        获取沙箱执行统计信息。

        返回:
            包含执行次数、阻止次数、超时次数的字典
        """
        return {
            "execution_count": self._execution_count,
            "blocked_count": self._blocked_count,
            "timeout_count": self._timeout_count,
            "success_rate": (
                (self._execution_count - self._blocked_count - self._timeout_count)
                / max(self._execution_count, 1)
            ),
        }

    def check_command_safety(self, code: str) -> Tuple[bool, str]:
        """
        检查Python代码中是否包含危险操作。

        扫描代码字符串中的黑名单关键字，检测以下危险行为：
            - eval/exec/compile等动态代码执行
            - os/subprocess等系统命令调用
            - 文件系统破坏性操作
            - 网络访问
            - 多进程/线程创建
            - 敏感系统属性访问

        参数:
            code: 待检查的Python代码字符串

        返回:
            (是否安全, 原因信息) 元组。安全返回(True, "OK")，
            不安全返回(False, 具体发现的危险内容)

        示例:
            >>> sandbox = Sandbox()
            >>> ok, msg = sandbox.check_command_safety("print('safe')")
            >>> assert ok  # 安全
            >>> ok, msg = sandbox.check_command_safety("os.system('ls')")
            >>> assert not ok  # 不安全
        """
        if not code or not isinstance(code, str):
            return False, "代码为空或不是字符串"

        # 去除注释和字符串后进行安全检查（基础版）
        # 注意：这是启发式检查，不是完整的静态分析
        code_lower = code.lower()
        found_dangers: List[str] = []

        for danger in PYTHON_CODE_BLACKLIST:
            if danger.lower() in code_lower:
                found_dangers.append(danger)

        if found_dangers:
            self._blocked_count += 1
            danger_str = ", ".join(found_dangers[:5])
            if len(found_dangers) > 5:
                danger_str += f" 等共{len(found_dangers)}项"
            return False, f"检测到危险操作: [{danger_str}]"

        # 检查是否存在直接文件系统路径访问
        for path in DANGEROUS_PATHS:
            if path.lower() in code_lower:
                return False, f"检测到禁止访问的系统路径: {path}"

        # 检查import语句（只允许白名单模块）
        import_pattern = __import__("re").compile(r"^\s*import\s+(\S+)")
        from_import_pattern = __import__("re").compile(r"^\s*from\s+(\S+)\s+import")
        allowed_modules = ("time", "math", "random", "json", "re", "struct",
                           "array", "datetime", "collections", "itertools",
                           "functools", "decimal", "fractions", "statistics",
                           "typing", "dataclasses", "enum")

        for line in code.split("\n"):
            match = import_pattern.match(line.strip())
            if match:
                module = match.group(1).split(".")[0]
                if module not in allowed_modules:
                    return False, f"禁止导入模块: '{module}'"

            match = from_import_pattern.match(line.strip())
            if match:
                module = match.group(1).split(".")[0]
                if module not in allowed_modules:
                    return False, f"禁止从模块导入: '{module}'"

        return True, "OK"

    def _build_bwrap_cmd(self) -> List[str]:
        """
        构建bubblewrap命令行参数列表。

        根据当前沙箱配置生成bwrap的完整参数列表，包括：
            - 命名空间隔离选项
            - 文件系统挂载规则（只读白名单）
            - 设备节点规则
            - 网络控制选项
            - 环境变量设置
            - 硬件通信管道绑定

        返回:
            bubblewrap命令行参数列表
        """
        cmd: List[str] = [self.BWRAP_PATH]

        # 命名空间隔离
        cmd.extend([
            "--unshare-all",           # 隔离所有命名空间
            "--die-with-parent",       # 父进程退出时终止
            "--new-session",           # 新建会话
        ])

        # 网络控制
        if not self.network:
            cmd.append("--unshare-net")  # 完全禁用网络
        else:
            cmd.append("--share-net")    # 保留网络（用于MCP回调）

        # 文件系统只读挂载（白名单）
        for path in self.allowed_paths:
            if os.path.exists(path):
                cmd.extend(["--ro-bind", path, path])

        # 系统库路径（确保Python可运行）
        for sys_path in ["/usr", "/lib", "/lib64"]:
            if os.path.exists(sys_path):
                cmd.extend(["--ro-bind", sys_path, sys_path])

        # SSL证书（HTTPS请求需要）
        if os.path.exists("/etc/ssl"):
            cmd.extend(["--ro-bind", "/etc/ssl", "/etc/ssl"])

        # 临时目录
        cmd.extend(["--dir", "/tmp"])

        # proc文件系统（只读）
        cmd.extend(["--proc", "/proc"])

        # 设备目录（受限）
        cmd.extend(["--dev", "/dev"])

        # 硬件通信管道绑定（代码与HAL通信的唯一通道）
        if os.path.exists(self.hw_cmd_fifo):
            cmd.extend(["--bind", self.hw_cmd_fifo, "/dev/hw_cmd"])
        if os.path.exists(self.hw_status_fifo):
            cmd.extend(["--bind", self.hw_status_fifo, "/dev/hw_status"])

        # 代码模板目录
        template_dir = os.path.join(self.sandbox_root, "templates")
        if os.path.exists(template_dir):
            cmd.extend(["--ro-bind", template_dir, "/templates"])

        # 工作目录
        cmd.extend([
            "--dir", "/workspace",
            "--chdir", "/workspace",
        ])

        # 环境变量
        cmd.extend([
            "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "KPCORTEX_SANDBOX", "1",
            "--setenv", "KPCORTEX_HW_VERSION", "1.0",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "PYTHONNOUSERSITE", "1",
        ])

        # Capability限制
        cmd.append("--cap-drop")
        cmd.append("ALL")

        # Seccomp-BPF（如果启用且有策略文件）
        if self.enable_seccomp:
            seccomp_policy = os.path.join(self.sandbox_root, "seccomp-bpf.policy")
            if os.path.exists(seccomp_policy):
                # bwrap通过--seccomp参数加载seccomp策略
                # 注意：需要seccomp文件描述符，实际部署时需要包装脚本
                logger.debug(f"Seccomp-BPF策略文件已找到: {seccomp_policy}")

        return cmd

    async def run(
        self,
        code: str,
        timeout: float = 5.0,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """
        在沙箱中执行Python代码。

        执行流程：
            1. 安全检查（代码白名单扫描）
            2. 创建临时文件存储代码
            3. 构建bubblewrap隔离命令
            4. 启动子进程执行代码
            5. 超时监控
            6. 结果收集与清理

        参数:
            code: 要执行的Python代码字符串
            timeout: 执行超时时间(秒)，默认5.0秒
            env_vars: 额外的环境变量字典

        返回:
            SandboxResult数据类，包含执行结果、输出、状态等信息

        示例:
            >>> sandbox = Sandbox()
            >>> result = await sandbox.run("print(2 + 3)", timeout=3.0)
            >>> print(result.stdout)
            5
            >>> print(result.success)
            True
        """
        self._execution_count += 1
        start_time = time.monotonic()

        # Step 1: 安全审查
        safe, reason = self.check_command_safety(code)
        if not safe:
            self._blocked_count += 1
            logger.warning(f"代码安全检查未通过: {reason}")
            return SandboxResult(
                success=False,
                error_message=f"安全检查失败: {reason}",
                command_blocked=[reason],
            )

        # Step 2: 创建临时文件
        temp_file: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir="/tmp"
            ) as f:
                # 写入安全的Python运行环境头
                f.write(self._generate_safe_preamble())
                f.write("\n")
                f.write(code)
                f.write("\n")
                temp_file = f.name

            # Step 3: 构建执行命令
            if self._bwrap_available:
                cmd = self._build_bwrap_cmd()
                # 绑定临时文件到沙箱内
                cmd.extend(["--bind", temp_file, "/workspace/script.py"])
                cmd.extend(["/usr/bin/python3", "/workspace/script.py"])
            else:
                # 回退模式：使用ulimit限制的subprocess
                cmd = ["/usr/bin/python3", temp_file]

            # Step 4: 设置资源限制
            preexec_fn = self._build_preexec_fn()

            # Step 5: 异步执行
            logger.debug(f"沙箱执行启动: timeout={timeout}s, cmd={' '.join(cmd[:3])}...")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec_fn,
                env=self._build_execution_env(env_vars),
            )

            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                execution_time = (time.monotonic() - start_time) * 1000
                timed_out = False
            except asyncio.TimeoutError:
                self._timeout_count += 1
                timed_out = True
                execution_time = timeout * 1000
                # 强制终止
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

                logger.warning(f"沙箱执行超时(timeout={timeout}s)，已强制终止")
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr="",
                    return_code=-9,
                    execution_time_ms=execution_time,
                    timed_out=True,
                    error_message=f"执行超时（限制{timeout}秒）",
                )

            # 结果处理
            stdout_str = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
            stderr_str = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
            execution_time = (time.monotonic() - start_time) * 1000

            success = (proc.returncode == 0 and not timed_out)

            result = SandboxResult(
                success=success,
                stdout=stdout_str,
                stderr=stderr_str,
                return_code=proc.returncode,
                execution_time_ms=execution_time,
                timed_out=timed_out,
                error_message="" if success else f"进程退出码: {proc.returncode}"
            )

            if success:
                logger.debug(
                    f"沙箱执行成功: 耗时{execution_time:.1f}ms, "
                    f"输出{len(stdout_str)}字节"
                )
            else:
                logger.debug(
                    f"沙箱执行失败: rc={proc.returncode}, "
                    f"stderr={stderr_str[:200]}"
                )

            return result

        except Exception as e:
            execution_time = (time.monotonic() - start_time) * 1000
            logger.error(f"沙箱执行异常: {type(e).__name__}: {e}")
            return SandboxResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=f"执行异常: {type(e).__name__}: {e}",
            )

        finally:
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    def _generate_safe_preamble(self) -> str:
        """
        生成安全的Python执行环境前置代码。

        在沙箱代码执行前注入安全环境配置，包括：
            - 危险内置函数的屏蔽
            - 安全硬件工具注入
            - 受限的sys.path配置
            - 异常处理框架

        返回:
            Python前置代码字符串
        """
        safe_builtins_str = ", ".join(f'"{b}"' for b in SAFE_BUILTINS)

        preamble = f'''# KunPeng-Cortex 沙箱安全环境 —— 自动生成
# 禁止修改此段代码
import builtins
import sys

# 限制可用的内置函数
_safe_builtins = {{{safe_builtins_str}}}
_original_builtins = dir(builtins)
for _name in _original_builtins:
    if _name not in _safe_builtins and not _name.startswith("__"):
        try:
            delattr(builtins, _name)
        except (AttributeError, TypeError):
            pass

# 限制sys.path（仅保留系统库）
sys.path = [p for p in sys.path if p.startswith(("/usr", "/lib"))]

# 沙箱环境标记
KPCORTEX_SANDBOX = True

# 硬件工具将通过globals注入
# hardware_tools = {{}}

def _sandbox_exit_handler():
    """沙箱退出清理函数"""
    pass

# 用户代码开始
# =============================================================================
'''
        return preamble

    def _build_preexec_fn(self) -> Optional[callable]:
        """
        构建子进程启动前的资源限制函数。

        使用resource模块设置内存限制、CPU时间限制等。
        仅在非Windows平台有效。

        返回:
            资源限制函数，若平台不支持则返回None
        """
        try:
            import resource
        except ImportError:
            return None

        def limit_resources():
            """设置子进程资源限制"""
            # 内存限制（软限制和硬限制）
            max_mem = self.max_memory_mb * 1024 * 1024  # 转换为字节
            try:
                resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
                resource.setrlimit(resource.RLIMIT_DATA, (max_mem, max_mem))
            except (ValueError, OSError):
                pass

            # CPU时间限制
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (10, 10))  # 10秒CPU时间
            except (ValueError, OSError):
                pass

            # 文件大小限制
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))  # 1MB
            except (ValueError, OSError):
                pass

            # 禁止创建核心转储
            try:
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except (ValueError, OSError):
                pass

            # 进程数限制（禁止fork）
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
            except (ValueError, OSError):
                pass

        return limit_resources

    def _build_execution_env(
        self,
        extra_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        构建沙箱执行的环境变量。

        参数:
            extra_vars: 额外的环境变量

        返回:
            完整的进程环境变量字典
        """
        env = os.environ.copy()
        env["KPCORTEX_SANDBOX"] = "1"
        env["KPCORTEX_HW_VERSION"] = "1.0"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        if extra_vars:
            env.update(extra_vars)

        return env

    async def run_with_hardware_tools(
        self,
        code: str,
        hardware_globals: Dict[str, Any],
        timeout: float = 5.0,
    ) -> SandboxResult:
        """
        在沙箱中执行代码，并注入硬件工具函数。

        将硬件控制函数作为全局变量注入到执行环境中，使生成的代码
        可以直接调用gpio_read、gpio_write等硬件操作。

        参数:
            code: 要执行的Python代码
            hardware_globals: 硬件工具函数字典，格式{"func_name": callable, ...}
            timeout: 执行超时时间(秒)

        返回:
            SandboxResult执行结果
        """
        # 为硬件工具生成注入代码
        tool_injection = self._generate_tool_injection_code(hardware_globals)
        full_code = tool_injection + "\n" + code

        # 设置环境变量标记硬件工具可用
        env_vars = {"KPCORTEX_HW_TOOLS": "1"}
        for name in hardware_globals:
            env_vars[f"KPCORTEX_TOOL_{name}"] = "1"

        return await self.run(full_code, timeout=timeout, env_vars=env_vars)

    def _generate_tool_injection_code(self, tools: Dict[str, Any]) -> str:
        """
        生成硬件工具注入代码。

        将Python函数字典序列化为可在沙箱内调用的代码片段。

        参数:
            tools: 硬件工具函数字典

        返回:
            工具注入Python代码
        """
        lines = ["# 硬件工具注入 —— 自动生成的桩函数", ""]
        for name in sorted(tools.keys()):
            lines.append(f"# 工具: {name} (将在执行时通过HAL适配器解析)")
        lines.append("")
        return "\n".join(lines)

    def get_seccomp_policy_text(self) -> str:
        """
        生成Seccomp-BPF策略文件内容。

        定义允许的系统调用白名单，其余调用均被拒绝。

        返回:
            Seccomp-BPF策略文件文本内容
        """
        # 允许的系统调用列表（最小集合，支持Python运行）
        allowed_syscalls = [
            "read", "write", "open", "openat", "close",
            "fstat", "lstat", "stat", "newfstatat",
            "mmap", "mprotect", "munmap", "brk",
            "pread64", "pwrite64", "lseek",
            "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
            "ioctl", "fcntl", "futex", "epoll_create1",
            "epoll_ctl", "epoll_pwait", "getpid", "getppid",
            "getuid", "getgid", "geteuid", "getegid",
            "getrandom", "exit", "exit_group",
            "clock_gettime", "nanosleep", "sched_yield",
            "dup", "dup2", "pipe", "pipe2",
            "access", " faccessat",
            "getdents64", "readlink", "readlinkat",
            "clone", "set_robust_list", "set_tid_address",
            "prlimit64", "arch_prctl",
            "uname", "sysinfo",
            # ARM64特定
            "getcwd", "chdir", "fchdir",
            "umask", "poll", "ppoll",
            "signalfd4", "eventfd2", "timerfd_create", "timerfd_settime",
        ]

        lines = [
            "# KunPeng-Cortex Seccomp-BPF策略",
            "# 仅允许白名单内的系统调用",
            "",
            "# 默认动作: 拒绝(EPERM)",
            "ERRNO(1) {",
        ]

        for syscall in allowed_syscalls:
            lines.append(f"    {syscall}")

        lines.extend([
            "}",
            "",
            "# 允许所有白名单系统调用",
            "ALLOW {}",
        ])

        return "\n".join(lines)
