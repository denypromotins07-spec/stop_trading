"""
RL Module Root
Integrates curriculum and HER logic into Ray RLlib training loop.
Bounds replay buffer sizes to respect 3GB RAM limit.
"""

from .curriculum_manager import CurriculumManager, CurriculumStage, TrainingRegime
from .her_buffer import HERBuffer, HindsightExperience

__all__ = [
    "CurriculumManager",
    "CurriculumStage",
    "TrainingRegime",
    "HERBuffer",
    "HindsightExperience",
]
