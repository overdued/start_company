"""
hermes_bridge — KunPeng-Cortex + Hermes Agent 融合桥接层

将 Hermes Agent 的自进化学习能力嫁接到 KunPeng-Cortex 的硬件控制上。
"""

from .memory_tool import KunpengMemoryStore
from .session_search import SessionSearchDB
from .skill_manager import SkillManager
from .evolution_engine import EvolutionEngine

__all__ = ["KunpengMemoryStore", "SessionSearchDB", "SkillManager", "EvolutionEngine"]
