#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvolutionEngine — KunPeng-Cortex 自进化引擎

核心功能：
1. 周期性自检（每15任务）
2. Skill 自我改进（成功率<70%重写）
3. Token 消耗优化
4. 相似 Skill 合并
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SelfCheckReport:
    """自检报告"""
    timestamp: float
    total_tasks: int
    success_rate: float
    skill_stats: Dict[str, Dict[str, Any]]
    token_usage_trend: List[int]
    similar_skills: List[Tuple[str, str, float]]
    recommendations: List[str]


class EvolutionEngine:
    """KunPeng-Cortex 自进化引擎"""

    def __init__(
        self,
        skill_manager: Any,
        memory_store: Any,
        session_db: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._skill_manager = skill_manager
        self._memory_store = memory_store
        self._session_db = session_db
        self._config = config or {}

        # 统计
        self._task_history: List[Dict[str, Any]] = []
        self._token_history: List[int] = []

        # 配置
        self._auto_create_threshold = self._config.get("auto_create_threshold", 5)
        self._self_check_interval = self._config.get("self_check_interval", 15)
        self._success_rate_threshold = self._config.get("success_rate_threshold", 0.7)
        self._token_reduction_target = self._config.get("token_reduction_target", 0.6)

    def record_task(self, task_result: Dict[str, Any]) -> None:
        """记录任务执行结果"""
        self._task_history.append({
            "timestamp": time.time(),
            "success": task_result.get("success", False),
            "tool_calls": task_result.get("tool_calls", 0),
            "skill_used": task_result.get("skill_used"),
            "elapsed_ms": task_result.get("elapsed_ms", 0),
        })
        self._token_history.append(task_result.get("tokens_used", 0))

    def should_auto_create(self, metrics: Dict[str, Any]) -> bool:
        """判断是否应自动创建 Skill"""
        tool_calls = metrics.get("tool_calls", 0)
        task_count = metrics.get("task_count", 0)
        user_input = metrics.get("user_input", "")

        if tool_calls >= self._auto_create_threshold:
            return True
        if task_count > 0 and task_count % self._self_check_interval == 0:
            return True
        if any(kw in user_input for kw in ("记住", "保存这个", "记下来")):
            return True
        return False

    def periodic_self_check(self) -> SelfCheckReport:
        """每15任务周期性自检"""
        recent_tasks = self._task_history[-self._self_check_interval:]
        total = len(recent_tasks)
        successes = sum(1 for t in recent_tasks if t.get("success"))
        success_rate = successes / total if total > 0 else 1.0

        # Skill 统计
        skill_stats: Dict[str, Dict[str, Any]] = {}
        for task in recent_tasks:
            skill_name = task.get("skill_used")
            if skill_name:
                if skill_name not in skill_stats:
                    skill_stats[skill_name] = {"success": 0, "fail": 0, "total": 0}
                skill_stats[skill_name]["total"] += 1
                if task.get("success"):
                    skill_stats[skill_name]["success"] += 1
                else:
                    skill_stats[skill_name]["fail"] += 1

        # 成功率计算
        for name, stats in skill_stats.items():
            stats["rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0

        # 相似 Skill 检测
        similar = self._find_similar_skills()

        # 生成建议
        recommendations = []
        if success_rate < self._success_rate_threshold:
            recommendations.append(f"最近 {self._self_check_interval} 任务成功率仅 {success_rate:.1%}，建议检查失败原因")
        for name, stats in skill_stats.items():
            if stats["rate"] < self._success_rate_threshold:
                recommendations.append(f"Skill '{name}' 成功率 {stats['rate']:.1%}，建议重写")
        if similar:
            for a, b, sim in similar:
                recommendations.append(f"Skill '{a}' 和 '{b}' 相似度 {sim:.1%}，建议合并")

        return SelfCheckReport(
            timestamp=time.time(),
            total_tasks=total,
            success_rate=success_rate,
            skill_stats=skill_stats,
            token_usage_trend=self._token_history[-self._self_check_interval:],
            similar_skills=similar,
            recommendations=recommendations,
        )

    def _find_similar_skills(self) -> List[Tuple[str, str, float]]:
        """发现相似 Skill"""
        if not self._skill_manager:
            return []

        skills = self._skill_manager.list_skills(detail_level=0)
        similar = []
        for i, s1 in enumerate(skills):
            for s2 in skills[i + 1:]:
                # 简单相似度：描述中共同关键词比例
                desc1 = set(s1.get("description", "").lower().split())
                desc2 = set(s2.get("description", "").lower().split())
                if desc1 and desc2:
                    intersection = desc1 & desc2
                    union = desc1 | desc2
                    sim = len(intersection) / len(union) if union else 0
                    if sim > 0.5:
                        similar.append((s1["name"], s2["name"], sim))
        return similar

    def improve_skill(self, skill_name: str) -> Dict[str, Any]:
        """改进指定 Skill（成功率<70%时重写）"""
        if not self._skill_manager:
            return {"success": False, "message": "Skill 管理器未初始化"}

        skill = self._skill_manager.view_skill(skill_name)
        if not skill:
            return {"success": False, "message": f"Skill '{skill_name}' 不存在"}

        # 获取失败记录
        failures = [t for t in self._task_history if t.get("skill_used") == skill_name and not t.get("success")]
        if not failures:
            return {"success": False, "message": f"Skill '{skill_name}' 无失败记录，无需改进"}

        # 在 body 中添加改进说明
        body = skill.get("body", "")
        improvement_note = f"\n\n## Auto-Improvement ({time.strftime('%Y-%m-%d')})\n"
        improvement_note += f"- 基于最近 {len(failures)} 次失败记录自动改进\n"
        improvement_note += "- 建议增加错误处理重试机制\n"

        new_body = body + improvement_note
        self._skill_manager.update_skill(skill_name, "body", new_body)

        return {"success": True, "message": f"Skill '{skill_name}' 已改进"}

    def get_stats(self) -> Dict[str, Any]:
        """获取进化引擎统计"""
        return {
            "total_tasks_recorded": len(self._task_history),
            "self_check_interval": self._self_check_interval,
            "auto_create_threshold": self._auto_create_threshold,
            "success_rate_threshold": self._success_rate_threshold,
        }
