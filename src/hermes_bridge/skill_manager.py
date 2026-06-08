#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkillManager — KunPeng-Cortex Skill 管理器

支持：
- 手动创建 Skill（agentskills.io 标准格式）
- 自动创建 Skill（复杂任务后触发）
- Progressive Disclosure 加载
- Skill 自我改进
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


class SkillManager:
    """KunPeng-Cortex Skill 管理器"""

    SKILL_DIR = "data/skills"
    MAX_NAME_LENGTH = 64
    MAX_DESCRIPTION_LENGTH = 1024

    def __init__(self, skill_dir: str = "data/skills") -> None:
        self.skill_dir = Path(skill_dir)
        self.skill_dir.mkdir(parents=True, exist_ok=True)
        self._skills_cache: Dict[str, Dict[str, Any]] = {}
        self._load_all_skills()

    def _load_all_skills(self) -> None:
        """从磁盘加载所有 Skill"""
        self._skills_cache.clear()
        for category_dir in self.skill_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for skill_dir in category_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    skill = self._parse_skill_file(skill_file)
                    if skill:
                        self._skills_cache[skill["name"]] = skill

    def _parse_skill_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """解析 SKILL.md 文件"""
        content = filepath.read_text(encoding="utf-8")

        # 解析 YAML frontmatter
        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter = parts[1].strip()
        body = parts[2].strip()

        # 解析 YAML frontmatter（支持多行字符串和列表）
        import yaml
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except Exception:
            # 回退到简单解析
            metadata = {}
            current_key = None
            for line in frontmatter.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line and not line.startswith("-"):
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    metadata[key] = value
                    current_key = key
                elif current_key and line.startswith("-"):
                    item = line.lstrip("-").strip().strip('"').strip("'")
                    if current_key not in metadata:
                        metadata[current_key] = []
                    if isinstance(metadata[current_key], list):
                        metadata[current_key].append(item)

        name = metadata.get("name", filepath.parent.name)
        description = metadata.get("description", "")

        # 从 description 中提取触发关键词（Trigger on: 后的内容）
        trigger_conditions = metadata.get("trigger_conditions", [])
        if not trigger_conditions and description:
            import re
            # 匹配 Trigger on: "keyword1", "keyword2" 格式
            trigger_match = re.search(r'Trigger on:\s*(.+?)(?:\n|$)', description, re.IGNORECASE)
            if trigger_match:
                trigger_text = trigger_match.group(1)
                # 提取引号内的关键词
                trigger_conditions = re.findall(r'["\']([^"\']+)["\']', trigger_text)
                # 如果没有引号，按逗号分割
                if not trigger_conditions:
                    trigger_conditions = [t.strip() for t in trigger_text.split(",")]

        # 从 hermes metadata 中提取 category
        hermes_meta = metadata.get("metadata", {}).get("hermes", {}) if isinstance(metadata.get("metadata"), dict) else {}
        category = hermes_meta.get("category", metadata.get("category", "general"))

        return {
            "name": name,
            "description": description,
            "category": category,
            "trigger_conditions": trigger_conditions,
            "safety_level": metadata.get("safety_level", "low"),
            "auto_created": str(metadata.get("auto_created", "false")).lower() in ("true", "yes", "1"),
            "version": metadata.get("version", "1.0.0"),
            "body": body,
            "filepath": str(filepath),
            "metadata": metadata,
        }

    def _write_skill_file(self, skill: Dict[str, Any]) -> None:
        """将 Skill 写入磁盘"""
        category = skill.get("category", "general")
        name = skill["name"]
        dir_path = self.skill_dir / category / name
        dir_path.mkdir(parents=True, exist_ok=True)

        # 构建 YAML frontmatter
        lines = ["---"]
        lines.append(f'name: {name}')
        lines.append(f'description: >')
        lines.append(f'  {skill.get("description", "")}')
        lines.append(f'category: {category}')
        lines.append(f'version: {skill.get("version", "1.0.0")}')

        triggers = skill.get("trigger_conditions", [])
        if triggers:
            lines.append("trigger_conditions:")
            for t in triggers:
                lines.append(f'  - "{t}"')

        lines.append(f'safety_level: {skill.get("safety_level", "low")}')
        lines.append(f'auto_created: {str(skill.get("auto_created", False)).lower()}')
        lines.append("---")
        lines.append("")
        lines.append(skill.get("body", "# Skill 内容\n"))

        filepath = dir_path / "SKILL.md"
        filepath.write_text("\n".join(lines), encoding="utf-8")

    # ── CRUD ──

    def create_skill(
        self,
        name: str,
        description: str,
        trigger_conditions: List[str],
        procedure: str = "",
        pitfalls: str = "",
        hardware_required: List[str] = None,
        category: str = "general",
        auto_created: bool = False,
    ) -> Dict[str, Any]:
        """创建新 Skill"""
        if len(name) > self.MAX_NAME_LENGTH:
            return {"success": False, "message": f"名称过长（最大 {self.MAX_NAME_LENGTH} 字符）"}

        if name in self._skills_cache:
            return {"success": False, "message": f"Skill '{name}' 已存在"}

        body = f"# {name}\n\n## When to Use\n{description}\n\n## Procedure\n{procedure}\n"
        if pitfalls:
            body += f"\n## Pitfalls\n{pitfalls}\n"
        if hardware_required:
            body += f"\n## Hardware Required\n" + "\n".join(f"- {h}" for h in hardware_required) + "\n"

        skill = {
            "name": name,
            "description": description,
            "category": category,
            "trigger_conditions": trigger_conditions,
            "safety_level": "low",
            "auto_created": auto_created,
            "version": "1.0.0",
            "body": body,
            "metadata": {},
        }

        self._write_skill_file(skill)
        self._skills_cache[name] = skill

        return {"success": True, "message": f"Skill '{name}' 创建成功"}

    def list_skills(
        self,
        category: Optional[str] = None,
        detail_level: int = 0,  # 0=摘要, 1=完整
    ) -> List[Dict[str, Any]]:
        """列出 Skill（Progressive Disclosure）

        Level 0: 只返回 name + description + category
        Level 1: 返回完整内容
        """
        results = []
        for name, skill in self._skills_cache.items():
            if category and skill.get("category") != category:
                continue

            if detail_level == 0:
                results.append({
                    "name": name,
                    "description": skill.get("description", "")[:100],
                    "category": skill.get("category", "general"),
                    "auto_created": skill.get("auto_created", False),
                })
            else:
                results.append({
                    "name": name,
                    "description": skill.get("description", ""),
                    "category": skill.get("category", "general"),
                    "trigger_conditions": skill.get("trigger_conditions", []),
                    "safety_level": skill.get("safety_level", "low"),
                    "auto_created": skill.get("auto_created", False),
                    "body": skill.get("body", ""),
                })
        return results

    def view_skill(self, name: str) -> Optional[Dict[str, Any]]:
        """查看 Skill 详情"""
        return self._skills_cache.get(name)

    def update_skill(self, name: str, field: str, content: str) -> Dict[str, Any]:
        """更新 Skill"""
        if name not in self._skills_cache:
            return {"success": False, "message": f"Skill '{name}' 不存在"}

        skill = self._skills_cache[name]
        if field in ("description", "category", "safety_level", "body"):
            skill[field] = content
        elif field == "trigger_conditions":
            skill[field] = content.split(",") if isinstance(content, str) else content
        else:
            return {"success": False, "message": f"不支持更新的字段: {field}"}

        self._write_skill_file(skill)
        return {"success": True, "message": f"Skill '{name}' 已更新"}

    def delete_skill(self, name: str) -> Dict[str, Any]:
        """删除 Skill"""
        if name not in self._skills_cache:
            return {"success": False, "message": f"Skill '{name}' 不存在"}

        skill = self._skills_cache.pop(name)
        filepath = Path(skill.get("filepath", ""))
        if filepath.exists():
            filepath.unlink()
            # 尝试删除空目录
            try:
                filepath.parent.rmdir()
            except OSError:
                pass

        return {"success": True, "message": f"Skill '{name}' 已删除"}

    # ── 匹配与触发 ──

    def match_skill(self, user_input: str) -> Optional[Dict[str, Any]]:
        """基于触发条件匹配 Skill

        匹配策略（优先级从高到低）：
        1. 精确关键词匹配（trigger_conditions）
        2. 子串匹配
        3. 正则匹配
        """
        user_input_lower = user_input.lower()

        best_match = None
        best_score = 0

        for name, skill in self._skills_cache.items():
            triggers = skill.get("trigger_conditions", [])
            score = 0

            for trigger in triggers:
                trigger_lower = trigger.lower()
                # 精确匹配
                if trigger_lower == user_input_lower:
                    score = 100
                    break
                # 子串匹配
                elif trigger_lower in user_input_lower:
                    score = max(score, 50)
                # 用户输入包含在触发词中
                elif user_input_lower in trigger_lower:
                    score = max(score, 30)

            if score > best_score:
                best_score = score
                best_match = skill

        return best_match if best_score >= 30 else None

    def should_auto_create(self, metrics: Dict[str, Any]) -> bool:
        """判断是否应触发自动创建 Skill

        触发条件（满足任一）：
        1. 单次任务使用了 5+ 个工具/HAL 调用
        2. 每 15 个任务周期性自检
        3. 用户明确说"记住这个操作"
        """
        tool_calls = metrics.get("tool_calls", 0)
        task_count = metrics.get("task_count", 0)
        user_input = metrics.get("user_input", "")

        if tool_calls >= 5:
            return True
        if task_count > 0 and task_count % 15 == 0:
            return True
        if any(kw in user_input for kw in ("记住", "保存这个", "记下来")):
            return True

        return False

    # ── 预设 Skill ──

    def load_preset_skills(self) -> int:
        """加载预设 Skill，返回加载数量"""
        presets_dir = Path(__file__).parent.parent.parent / "hermes_fusion_package" / "skills"
        if not presets_dir.exists():
            return 0

        loaded = 0
        for category_dir in presets_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for skill_dir in category_dir.iterdir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    skill = self._parse_skill_file(skill_file)
                    if skill:
                        # 复制到项目 skills 目录
                        target_dir = self.skill_dir / category_dir.name / skill_dir.name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_file = target_dir / "SKILL.md"
                        if not target_file.exists():
                            target_file.write_text(skill_file.read_text(), encoding="utf-8")
                            self._skills_cache[skill["name"]] = skill
                            loaded += 1

        return loaded

    def get_stats(self) -> Dict[str, Any]:
        """获取 Skill 统计"""
        categories = {}
        for skill in self._skills_cache.values():
            cat = skill.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total": len(self._skills_cache),
            "categories": categories,
            "auto_created": sum(1 for s in self._skills_cache.values() if s.get("auto_created")),
        }
