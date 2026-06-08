#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SessionSearchDB — KunPeng-Cortex 会话搜索

将所有用户对话和 Agent 回复索引到 SQLite，支持全文搜索和 LLM 聚焦式摘要。

支持 FTS5（如果可用）或回退到 LIKE 查询。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SessionSearchDB:
    """KunPeng-Cortex 会话搜索 —— 基于 SQLite FTS5"""

    def __init__(self, db_path: str = "data/sessions_fts.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts5_available = self._check_fts5()
        self._init_schema()

    def _check_fts5(self) -> bool:
        """检查 SQLite 是否支持 FTS5"""
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE VIRTUAL TABLE test USING fts5(content)")
            conn.close()
            return True
        except sqlite3.OperationalError:
            return False

    def _init_schema(self) -> None:
        """创建 sessions 表 + messages 表 + FTS5 虚拟表 + triggers"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            # Sessions 表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    summary TEXT
                )
            """)

            # Messages 表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_name TEXT,
                    emotion TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # FTS5 虚拟表（全文索引）
            if self._fts5_available:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                        content,
                        content='messages',
                        content_rowid='id'
                    )
                """)
                # 触发器：保持 FTS5 索引同步
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(rowid, content)
                        VALUES (new.id, new.content);
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                        INSERT INTO messages_fts(rowid, content)
                        VALUES (new.id, new.content);
                    END
                """)

            # 普通索引（FTS5 不可用时使用，或作为辅助）
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp)
            """)

            conn.commit()

    def ensure_session(self, session_id: str) -> None:
        """确保 session 存在于 sessions 表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
                (session_id, time.time()),
            )
            conn.commit()

    def append_interaction(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> int:
        """索引一条会话消息

        Args:
            session_id: 会话唯一标识
            role: "user" | "agent" | "system" | "sensor"
            content: 消息内容
            tool_name: 工具名（可选）
            emotion: 情感状态（可选）

        Returns:
            消息 ID
        """
        self.ensure_session(session_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, tool_name, emotion, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, tool_name, emotion, time.time()),
            )
            conn.commit()
            return cursor.lastrowid

    def search(
        self,
        query: str,
        time_range_days: Optional[int] = None,
        limit: int = 5,
        window: int = 3,
    ) -> List[Dict[str, Any]]:
        """搜索历史会话

        Args:
            query: 搜索关键词
            time_range_days: 时间范围（最近 N 天），None 表示全部
            limit: 返回的最大匹配数
            window: 每条匹配结果前后各包含多少条上下文消息

        Returns:
            匹配结果列表，每条包含：
            - matched_message: 匹配的消息
            - context: 前后 window 条消息
            - session_id: 会话 ID
            - score: 匹配分数
        """
        results = []
        cutoff = None
        if time_range_days:
            cutoff = time.time() - time_range_days * 86400

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if self._fts5_available:
                # 使用 FTS5 全文搜索
                if cutoff:
                    rows = conn.execute(
                        """
                        SELECT m.id, m.session_id, m.role, m.content, m.tool_name,
                               m.emotion, m.timestamp, rank
                        FROM messages m
                        JOIN messages_fts ON m.id = messages_fts.rowid
                        WHERE messages_fts MATCH ? AND m.timestamp > ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (query, cutoff, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT m.id, m.session_id, m.role, m.content, m.tool_name,
                               m.emotion, m.timestamp, rank
                        FROM messages m
                        JOIN messages_fts ON m.id = messages_fts.rowid
                        WHERE messages_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (query, limit),
                    ).fetchall()
            else:
                # 回退到 LIKE 查询
                pattern = f"%{query}%"
                if cutoff:
                    rows = conn.execute(
                        """
                        SELECT id, session_id, role, content, tool_name,
                               emotion, timestamp, 0 as rank
                        FROM messages
                        WHERE content LIKE ? AND timestamp > ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (pattern, cutoff, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT id, session_id, role, content, tool_name,
                               emotion, timestamp, 0 as rank
                        FROM messages
                        WHERE content LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (pattern, limit),
                    ).fetchall()

            for row in rows:
                msg_id = row["id"]
                session_id = row["session_id"]

                # 获取前后上下文
                context_rows = conn.execute(
                    """
                    SELECT role, content, tool_name, emotion, timestamp
                    FROM messages
                    WHERE session_id = ?
                    AND id BETWEEN ? AND ?
                    ORDER BY id
                    """,
                    (session_id, msg_id - window, msg_id + window),
                ).fetchall()

                context = [
                    {
                        "role": c["role"],
                        "content": c["content"],
                        "tool_name": c["tool_name"],
                        "emotion": c["emotion"],
                        "timestamp": c["timestamp"],
                    }
                    for c in context_rows
                ]

                results.append({
                    "matched_message": {
                        "role": row["role"],
                        "content": row["content"],
                        "tool_name": row["tool_name"],
                        "emotion": row["emotion"],
                        "timestamp": row["timestamp"],
                    },
                    "context": context,
                    "session_id": session_id,
                    "score": row["rank"] if row["rank"] else 0,
                })

        return results

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的会话列表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT session_id, created_at, summary
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "summary": r["summary"],
                }
                for r in rows
            ]

    def get_session_messages(
        self, session_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取指定会话的完整消息列表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT role, content, tool_name, emotion, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY id
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "tool_name": r["tool_name"],
                    "emotion": r["emotion"],
                    "timestamp": r["timestamp"],
                }
                for r in rows
            ]

    def update_session_summary(self, session_id: str, summary: str) -> None:
        """更新会话摘要"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET summary = ? WHERE session_id = ?",
                (summary, session_id),
            )
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计"""
        with sqlite3.connect(self.db_path) as conn:
            session_count = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
            message_count = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            return {
                "sessions": session_count,
                "messages": message_count,
                "fts5_available": self._fts5_available,
                "db_path": str(self.db_path),
            }

    def close(self) -> None:
        """关闭数据库连接（上下文管理器使用）"""
        pass  # 使用 with 语句自动管理连接
