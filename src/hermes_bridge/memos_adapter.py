#!/usr/bin/env python3
"""MemosAdapter — Memos 笔记系统适配器

离线优先的 Memos 兼容存储后端：
- 本地 SQLite 存储（完全离线可用）
- Memos REST API 兼容的数据格式
- 有网时可同步到真实 Memos 服务器
- 支持智能体自动记录用户关键信息、客户需求追踪

Memos API v1 兼容：POST/GET /api/v1/memos, /api/v1/memos/search
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


class MemosStore:
    """KunPeng-Hermes Memos 本地存储

    使用场景：
    - 用户说"记住xxx" → 自动创建 memo
    - 用户询问"上周我说过什么" → 全文搜索历史 memo
    - 客户需求追踪 → 标签分类（#需求, #偏好, #用药, #紧急）
    - 公司迭代 → 按标签统计客户反馈
    """

    MEMO_TABLE = """
        CREATE TABLE IF NOT EXISTS memos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            visibility TEXT DEFAULT 'PRIVATE',
            tags TEXT DEFAULT '[]',
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT '',
            pinned INTEGER DEFAULT 0,
            created_ts REAL NOT NULL,
            updated_ts REAL NOT NULL
        )
    """
    TAG_TABLE = """
        CREATE TABLE IF NOT EXISTS tags (
            name TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0
        )
    """
    FTS_TABLE = """
        CREATE VIRTUAL TABLE IF NOT EXISTS memos_fts USING fts5(
            content, category, source, tags
        )
    """

    # 智能分类标签前缀
    TAG_USER_PREF = "用户偏好"
    TAG_MEDICINE = "用药"
    TAG_EMERGENCY = "紧急"
    TAG_DAILY = "日常"
    TAG_REQUIREMENT = "客户需求"
    TAG_BUG = "问题反馈"
    TAG_FEATURE = "功能建议"
    TAG_HARDWARE = "硬件状态"

    def __init__(self, db_path: str = "data/memos.db", server_url: str = "") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.server_url = server_url  # 远程 Memos 服务器（可选）
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute(self.MEMO_TABLE)
            c.execute(self.TAG_TABLE)
            try:
                c.execute(self.FTS_TABLE)
            except sqlite3.OperationalError:
                pass  # FTS5 may not be available
            # Triggers to keep FTS in sync
            c.execute("""
                INSERT OR IGNORE INTO tags (name, count)
                VALUES ('用户偏好',0),('用药',0),('紧急',0),('日常',0),
                       ('客户需求',0),('问题反馈',0),('功能建议',0),('硬件状态',0)
            """)
            c.commit()

    # ── 核心 API ──

    def create(
        self,
        content: str,
        visibility: str = "PRIVATE",
        tags: List[str] = [],
        category: str = "general",
        source: str = "",
    ) -> Dict[str, Any]:
        """创建一条 memo。用户说"记住xxx"时调用"""
        now = time.time()
        tags_json = json.dumps(tags, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as c:
            cur = c.execute(
                "INSERT INTO memos (content, visibility, tags, category, source, created_ts, updated_ts) VALUES (?,?,?,?,?,?,?)",
                (content, visibility, tags_json, category, source, now, now),
            )
            memo_id = cur.lastrowid
            # 更新 FTS
            try:
                c.execute(
                    "INSERT INTO memos_fts (rowid, content, category, source, tags) VALUES (?,?,?,?,?)",
                    (memo_id, content, category, source, tags_json),
                )
            except sqlite3.OperationalError:
                pass
            # 更新标签计数
            for tag in tags:
                c.execute("UPDATE tags SET count = count + 1 WHERE name = ?", (tag,))
            c.commit()
        return self.get(memo_id)

    def get(self, memo_id: int) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM memos WHERE id = ?", (memo_id,)).fetchone()
            if not row:
                return {}
            return self._row_to_dict(row)

    def list(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        visibility: str = "PRIVATE",
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """列出 memo"""
        sql = "SELECT * FROM memos WHERE 1=1"
        params: list = []
        if category:
            sql += " AND category = ?"; params.append(category)
        if tag:
            sql += " AND tags LIKE ?"; params.append(f'%"{tag}"%')
        if visibility:
            sql += " AND visibility = ?"; params.append(visibility)
        sql += " ORDER BY pinned DESC, updated_ts DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """全文搜索 memo。用户问"上周我说过什么"时调用"""
        # 尝试 FTS5
        try:
            with sqlite3.connect(self.db_path) as c:
                c.row_factory = sqlite3.Row
                rows = c.execute(
                    "SELECT m.* FROM memos m JOIN memos_fts f ON m.id = f.rowid WHERE memos_fts MATCH ? ORDER BY m.updated_ts DESC LIMIT ?",
                    (query, limit),
                ).fetchall()
                if rows:
                    return [self._row_to_dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass
        # 回退 LIKE
        pattern = f"%{query}%"
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM memos WHERE content LIKE ? OR tags LIKE ? ORDER BY updated_ts DESC LIMIT ?",
                (pattern, pattern, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update(self, memo_id: int, content: str = "", tags: List[str] = [], pinned: int = -1) -> Dict[str, Any]:
        """更新 memo"""
        now = time.time()
        fields, params = [], []
        if content:
            fields.append("content = ?"); params.append(content)
        if tags:
            fields.append("tags = ?"); params.append(json.dumps(tags, ensure_ascii=False))
        if pinned >= 0:
            fields.append("pinned = ?"); params.append(pinned)
        fields.append("updated_ts = ?"); params.append(now)
        params.append(memo_id)
        with sqlite3.connect(self.db_path) as c:
            c.execute(f"UPDATE memos SET {', '.join(fields)} WHERE id = ?", params)
            c.commit()
        return self.get(memo_id)

    def archive(self, memo_id: int) -> bool:
        """归档 memo（软删除）"""
        return bool(self.update(memo_id, pinned=-999).get("id"))

    def pin(self, memo_id: int) -> Dict[str, Any]:
        """置顶"""
        return self.update(memo_id, pinned=1)

    # ── 智能体专用方法 ──

    def auto_log(
        self,
        content: str,
        category: Optional[str] = None,
        tags_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """智能体自动记录——根据内容自动分类和打标签

        Args:
            content: 要记录的内容
            category: 手动指定分类（None 则自动推断）

        Returns:
            创建的 memo
        """
        if category is None:
            category = self._infer_category(content)
        tags = self._infer_tags(content, category)
        source = "agent_auto"
        return self.create(content=content, tags=tags, category=category, source=source)

    def _infer_category(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ("药", "吃药", "服药", "血压", "血糖", "病历", "医生")):
            return "health"
        if any(k in t for k in ("喜欢", "不喜欢", "偏好", "习惯", "想要", "经常")):
            return self.TAG_USER_PREF
        if any(k in t for k in ("救命", "摔倒", "紧急", "危险", "着火", "事故")):
            return self.TAG_EMERGENCY
        if any(k in t for k in ("需要", "要求", "建议", "希望改进", "功能", "能不能")):
            return self.TAG_REQUIREMENT
        if any(k in t for k in ("坏了", "问题", "出错", "不行", "没用")):
            return self.TAG_BUG
        if any(k in t for k in ("温度", "湿度", "电机", "GPIO", "传感器", "摄像头")):
            return self.TAG_HARDWARE
        return self.TAG_DAILY

    def _infer_tags(self, text: str, cat: str) -> List[str]:
        tags = [cat]
        # 额外关键词标签
        if "药" in text: tags.append(self.TAG_MEDICINE)
        if "紧急" in text or "救命" in text: tags.append(self.TAG_EMERGENCY)
        if "客户" in text or "用户" in text: tags.append(self.TAG_REQUIREMENT)
        return tags

    def get_user_profile(self) -> Dict[str, Any]:
        """汇总用户画像——从所有 #用户偏好 标签的 memo 中提取"""
        prefs = self.list(tag=self.TAG_USER_PREF, limit=50)
        health = self.list(category="health", limit=50)
        requirements = self.list(tag=self.TAG_REQUIREMENT, limit=50)
        return {
            "preferences": [{"content": m["content"], "time": m["created_ts"]} for m in prefs],
            "health": [{"content": m["content"], "time": m["created_ts"]} for m in health],
            "requirements": [{"content": m["content"], "time": m["created_ts"]} for m in requirements],
            "total_memos": self.count(),
        }

    def get_recent_context(self, limit: int = 10) -> str:
        """获取最近的 memo 作为对话上下文注入"""
        memos = self.list(limit=limit)
        if not memos:
            return ""
        lines = ["\n【用户历史记录（Memos）】"]
        for m in memos:
            tags_str = ", ".join(m.get("tags", [])) if m.get("tags") else ""
            time_str = datetime.fromtimestamp(m["created_ts"]).strftime("%m/%d %H:%M")
            lines.append(f"- [{time_str}] {m['content'][:120]}{' #'+tags_str if tags_str else ''}")
        return "\n".join(lines)

    # ── 统计 ──

    def count(self, category: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) FROM memos"
        params = []
        if category:
            sql += " WHERE category = ?"; params.append(category)
        with sqlite3.connect(self.db_path) as c:
            return c.execute(sql, params).fetchone()[0]

    def tag_stats(self) -> Dict[str, int]:
        """标签统计——用于客户需求分析"""
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute("SELECT name, count FROM tags WHERE count > 0 ORDER BY count DESC").fetchall()
            return {r[0]: r[1] for r in rows}

    def category_stats(self) -> Dict[str, int]:
        """分类统计"""
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute(
                "SELECT category, COUNT(*) as cnt FROM memos GROUP BY category ORDER BY cnt DESC"
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    # ── 工具 ──

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["tags"] = json.loads(d.get("tags", "[]"))
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
        return d

    def export_recent(self, days: int = 30, fmt: str = "markdown") -> str:
        """导出最近 N 天的 memo（用于给客户/团队查看）"""
        cutoff = time.time() - days * 86400
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM memos WHERE updated_ts > ? ORDER BY updated_ts DESC", (cutoff,)
            ).fetchall()
        if not rows:
            return "暂无记录"
        lines = [f"# Memos 导出 ({days}天)\n"]
        prev_cat = ""
        for r in rows:
            d = self._row_to_dict(r)
            cat = d.get("category", "general")
            if cat != prev_cat:
                lines.append(f"\n## {cat}")
                prev_cat = cat
            ts = datetime.fromtimestamp(d["updated_ts"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- **{ts}**: {d['content'][:200]}")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """完整统计"""
        return {
            "total": self.count(),
            "tags": self.tag_stats(),
            "categories": self.category_stats(),
            "db_path": str(self.db_path),
            "server_url": self.server_url or "(本地离线模式)",
        }
