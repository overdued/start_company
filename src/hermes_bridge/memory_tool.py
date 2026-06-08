#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KunpengMemoryStore — KunPeng-Cortex 记忆存储

同时管理：
- MEMORY.md: 环境事实、硬件知识（2,200 字符上限）
- USER.md: 用户画像、偏好（1,375 字符上限）

遵循 Frozen Snapshot 模式：
- 启动时加载 snapshot → 注入 system prompt
- 会话中写入 → 立即落盘 → 不更新 snapshot
- 下次启动 → 新 snapshot
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


class KunpengMemoryStore:
    """KunPeng-Cortex 记忆存储 —— 适配 Hermes Memory 模式"""

    MEMORY_MAX_CHARS = 2200
    USER_MAX_CHARS = 1375
    SECTION_DELIMITER = "§"

    def __init__(
        self,
        memory_dir: str = "data/memories",
        memory_char_limit: int = MEMORY_MAX_CHARS,
        user_char_limit: int = USER_MAX_CHARS,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.user_file = self.memory_dir / "USER.md"
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit

        # Frozen snapshot —— 启动时捕获，会话中不变
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

        # 内存中缓存的条目
        self.memory_entries: List[Dict[str, Any]] = []
        self.user_entries: List[Dict[str, Any]] = []

        self.load_from_disk()

    def load_from_disk(self) -> None:
        """从磁盘加载 MEMORY.md 和 USER.md，捕获 frozen snapshot"""
        self.memory_entries = self._parse_file(self.memory_file)
        self.user_entries = self._parse_file(self.user_file)

        # 捕获 snapshot
        self._system_prompt_snapshot = {
            "memory": self._render_entries(self.memory_entries),
            "user": self._render_entries(self.user_entries),
        }

    def _parse_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """解析 .md 文件为结构化条目列表"""
        entries = []
        if not filepath.exists():
            return entries

        content = filepath.read_text(encoding="utf-8")
        # 按 § 分隔符分割
        sections = content.split(self.SECTION_DELIMITER)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            lines = section.strip().split("\n")
            title = lines[0].strip() if lines else "未分类"
            body = "\n".join(lines[1:]).strip()
            if body:
                entries.append({
                    "title": title,
                    "content": body,
                    "timestamp": time.time(),
                })
        return entries

    def _render_entries(self, entries: List[Dict[str, Any]]) -> str:
        """将条目列表渲染为 Markdown 字符串"""
        parts = []
        for entry in entries:
            parts.append(f"{self.SECTION_DELIMITER} {entry['title']}")
            parts.append(entry["content"])
        return "\n".join(parts)

    def _write_file(self, filepath: Path, entries: List[Dict[str, Any]]) -> None:
        """将条目列表写入磁盘"""
        content = self._render_entries(entries)
        filepath.write_text(content, encoding="utf-8")

    def add_entry(
        self,
        target: Literal["memory", "user"],
        content: str,
        category: str = "general",
    ) -> Dict[str, Any]:
        """添加记忆条目

        Args:
            target: "memory" 或 "user"
            content: 条目内容
            category: 分类标签

        Returns:
            {"success": bool, "message": str}
        """
        if target == "memory":
            entries = self.memory_entries
            filepath = self.memory_file
            limit = self.memory_char_limit
        elif target == "user":
            entries = self.user_entries
            filepath = self.user_file
            limit = self.user_char_limit
        else:
            return {"success": False, "message": f"无效目标: {target}"}

        # 检查容量
        current_text = self._render_entries(entries)
        if len(current_text) + len(content) > limit:
            # 触发自动压缩
            self.compact(target)
            current_text = self._render_entries(entries)
            if len(current_text) + len(content) > limit:
                return {
                    "success": False,
                    "message": f"{target} 记忆已满（上限 {limit} 字符），无法添加",
                }

        entries.append({
            "title": category,
            "content": content,
            "timestamp": time.time(),
        })
        self._write_file(filepath, entries)
        return {"success": True, "message": f"已添加到 {target}"}

    def remove_entry(self, target: Literal["memory", "user"], substring: str) -> Dict[str, Any]:
        """通过子串匹配删除记忆条目"""
        if target == "memory":
            entries = self.memory_entries
            filepath = self.memory_file
        elif target == "user":
            entries = self.user_entries
            filepath = self.user_file
        else:
            return {"success": False, "message": f"无效目标: {target}"}

        original_count = len(entries)
        entries[:] = [e for e in entries if substring not in e["content"] and substring not in e["title"]]
        removed = original_count - len(entries)

        self._write_file(filepath, entries)
        return {"success": True, "message": f"已删除 {removed} 条匹配条目"}

    def replace_entry(
        self,
        target: Literal["memory", "user"],
        old_substring: str,
        new_content: str,
    ) -> Dict[str, Any]:
        """替换包含指定子串的条目"""
        if target == "memory":
            entries = self.memory_entries
            filepath = self.memory_file
        elif target == "user":
            entries = self.user_entries
            filepath = self.user_file
        else:
            return {"success": False, "message": f"无效目标: {target}"}

        replaced = 0
        for entry in entries:
            if old_substring in entry["content"] or old_substring in entry["title"]:
                entry["content"] = new_content
                entry["timestamp"] = time.time()
                replaced += 1

        if replaced > 0:
            self._write_file(filepath, entries)
        return {"success": replaced > 0, "message": f"已替换 {replaced} 条匹配条目"}

    def read_entries(self, target: Literal["memory", "user", "all"] = "all") -> Dict[str, Any]:
        """读取记忆条目

        Returns:
            {"memory": [...], "user": [...]}
        """
        result = {}
        if target in ("memory", "all"):
            result["memory"] = [
                {"title": e["title"], "content": e["content"], "timestamp": e["timestamp"]}
                for e in self.memory_entries
            ]
        if target in ("user", "all"):
            result["user"] = [
                {"title": e["title"], "content": e["content"], "timestamp": e["timestamp"]}
                for e in self.user_entries
            ]
        return result

    def get_snapshot_for_prompt(self) -> Dict[str, str]:
        """获取 frozen snapshot 用于注入 system prompt

        Returns:
            {"memory": str, "user": str}
        """
        return self._system_prompt_snapshot.copy()

    def compact(self, target: Literal["memory", "user"]) -> Dict[str, Any]:
        """压缩记忆（接近上限时触发）

        策略：
        1. 去重：合并相同 title 的条目
        2. 去旧：删除最早的条目直到容量低于 80%
        """
        if target == "memory":
            entries = self.memory_entries
            filepath = self.memory_file
            limit = self.memory_char_limit
        elif target == "user":
            entries = self.user_entries
            filepath = self.user_file
            limit = self.user_char_limit
        else:
            return {"success": False, "message": f"无效目标: {target}"}

        original_count = len(entries)

        # 去重：合并相同 title
        merged: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            title = entry["title"]
            if title in merged:
                merged[title]["content"] += f"\n{entry['content']}"
                merged[title]["timestamp"] = max(merged[title]["timestamp"], entry["timestamp"])
            else:
                merged[title] = entry.copy()

        entries[:] = list(merged.values())

        # 去旧：删除最早条目直到低于 80% 容量
        while self._render_entries(entries) and len(self._render_entries(entries)) > int(limit * 0.8):
            if entries:
                entries.pop(0)  # 删除最旧的
            else:
                break

        self._write_file(filepath, entries)
        removed = original_count - len(entries)
        return {
            "success": True,
            "message": f"已压缩 {target}: 去重合并 {original_count - len(merged)} 条, 删除旧条目 {removed} 条",
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        memory_text = self._render_entries(self.memory_entries)
        user_text = self._render_entries(self.user_entries)
        return {
            "memory": {
                "entries": len(self.memory_entries),
                "chars": len(memory_text),
                "limit": self.memory_char_limit,
                "usage_percent": round(len(memory_text) / self.memory_char_limit * 100, 1),
            },
            "user": {
                "entries": len(self.user_entries),
                "chars": len(user_text),
                "limit": self.user_char_limit,
                "usage_percent": round(len(user_text) / self.user_char_limit * 100, 1),
            },
        }
