#!/usr/bin/env python3
"""voiceprint.py — 声纹识别 (CAM++ 192维嵌入 + 余弦相似度)

核心功能:
- 提取 192 维声纹向量 (L2 归一化)
- 身份验证: 相似度 ≥ 0.35 → 通过
- 待定区间: 0.25 < 相似度 < 0.35 → 待定池
- 身份识别: 与注册库逐一比对取最高
- NumPy 向量索引（Faiss 接口预留，aarch64 无 wheel）

模型: damo/speech_campplus_sv_zh-cn_16k-common (7.2M 参数, 中文 20 万说话人)
"""

from __future__ import annotations

import os
import sqlite3
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EMBEDDING_DIM = 192
VERIFY_THRESHOLD = 0.35    # 验证通过阈值
PENDING_THRESHOLD = 0.25   # 待定区间下界
TARGET_RATE = 16000


class CAMPlusExtractor:
    """CAM++ 声纹向量提取器"""

    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu"):
        from funasr import AutoModel
        kwargs = {
            "model": "damo/speech_campplus_sv_zh-cn_16k-common",
            "device": device,
            "disable_update": True,
        }
        if model_dir:
            kwargs["model"] = model_dir
        self._model = AutoModel(**kwargs)

    def extract(self, wav: np.ndarray) -> np.ndarray:
        """提取 192 维 L2 归一化声纹向量"""
        if len(wav) < TARGET_RATE:  # 至少 1 秒
            pad = TARGET_RATE - len(wav)
            wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
        res = self._model.generate(input=wav, cache={})
        emb = res[0]["spk_embedding"]
        emb = np.asarray(emb, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


class VoiceprintDB:
    """声纹向量数据库 (SQLite3 元信息 + NumPy 向量)

    分层存储:
    - users.db: SQLite3 用户元信息
    - embeddings.npy: NumPy 向量数组 (N × 192)
    """

    def __init__(self, data_dir: str = "data/voice"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "users.db"
        self.emb_path = self.data_dir / "embeddings.npy"
        self.pending_path = self.data_dir / "pending.npy"
        self._init_db()
        self._embeddings: np.ndarray = self._load_npy(self.emb_path)
        self._pending: List[Dict[str, Any]] = []

    def _init_db(self):
        with sqlite3.connect(self.db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_ts REAL NOT NULL,
                    last_seen_ts REAL NOT NULL,
                    sample_count INTEGER DEFAULT 1
                )
            """)
            c.commit()

    def _load_npy(self, path: Path) -> np.ndarray:
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                pass
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    def _save_npy(self):
        np.save(self.emb_path, self._embeddings)

    def enroll(self, name: str, embedding: np.ndarray, role: str = "user") -> int:
        """注册新用户声纹，返回用户 ID"""
        now = time.time()
        with sqlite3.connect(self.db_path) as c:
            cur = c.execute(
                "INSERT INTO users (name, role, created_ts, last_seen_ts, sample_count) VALUES (?,?,?,?,1)",
                (name, role, now, now),
            )
            uid = cur.lastrowid
            c.commit()
        self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])
        self._save_npy()
        return uid

    def update_user(self, user_id: int, embedding: np.ndarray) -> bool:
        """更新用户声纹（平均向量融合）"""
        with sqlite3.connect(self.db_path) as c:
            row = c.execute("SELECT sample_count FROM users WHERE id=?", (user_id,)).fetchone()
            if not row:
                return False
            n = row[0]
            old_emb = self._embeddings[user_id - 1]
            new_emb = (old_emb * n + embedding) / (n + 1)
            new_emb = new_emb / np.linalg.norm(new_emb)
            self._embeddings[user_id - 1] = new_emb
            self._save_npy()
            c.execute("UPDATE users SET sample_count=?, last_seen_ts=? WHERE id=?",
                      (n + 1, time.time(), user_id))
            c.commit()
        return True

    def verify(self, embedding: np.ndarray, user_id: int) -> Tuple[bool, float]:
        """验证是否为指定用户（相似度 ≥ 0.35 通过）"""
        if user_id - 1 >= len(self._embeddings) or user_id < 1:
            return False, 0.0
        target = self._embeddings[user_id - 1]
        sim = float(np.dot(embedding, target))
        return sim >= VERIFY_THRESHOLD, sim

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float, str]:
        """身份识别：返回 (用户信息, 相似度, 状态)

        状态: "matched" | "pending" | "new"
        """
        if len(self._embeddings) == 0:
            return None, 0.0, "new"
        sims = self._embeddings @ embedding  # L2 归一化后内积=余弦相似度
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= VERIFY_THRESHOLD:
            with sqlite3.connect(self.db_path) as c:
                c.row_factory = sqlite3.Row
                row = c.execute("SELECT * FROM users WHERE id=?", (best_idx + 1,)).fetchone()
                if row:
                    c.execute("UPDATE users SET last_seen_ts=? WHERE id=?", (time.time(), best_idx + 1))
                    c.commit()
                    return dict(row), best_sim, "matched"
        if best_sim > PENDING_THRESHOLD:
            return None, best_sim, "pending"
        return None, best_sim, "new"

    def add_pending(self, embedding: np.ndarray, sim: float):
        """加入待定池"""
        self._pending.append({"emb": embedding.copy(), "sim": sim, "ts": time.time()})
        np.save(self.pending_path, np.array([p["emb"] for p in self._pending]))

    def resolve_pending(self, embedding: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float, str]:
        """用新音频解决待定池：多数投票"""
        if not self._pending:
            return None, 0.0, "new"
        user, sim, status = self.identify(embedding)
        if status == "matched":
            self._pending.clear()
            np.save(self.pending_path, np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
            return user, sim, "matched"
        # 平均待定向量再比对
        avg = np.mean([p["emb"] for p in self._pending], axis=0)
        avg = avg / np.linalg.norm(avg)
        user2, sim2, status2 = self.identify(avg)
        if status2 == "matched":
            self._pending.clear()
            np.save(self.pending_path, np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
            return user2, sim2, "matched"
        return None, sim, "pending"

    def list_users(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM users ORDER BY last_seen_ts DESC").fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        return len(self._embeddings)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "users": self.count(),
            "pending": len(self._pending),
            "db_path": str(self.db_path),
            "threshold": VERIFY_THRESHOLD,
        }


def create_extractor(device: str = "cpu") -> Optional[CAMPlusExtractor]:
    """创建 CAM++ 提取器，模型缺失返回 None"""
    try:
        return CAMPlusExtractor(device=device)
    except Exception as e:
        print(f"[WARN] CAM++ 初始化失败: {e}")
        return None
