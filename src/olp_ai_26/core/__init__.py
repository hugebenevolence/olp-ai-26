"""Task-agnostic competition utilities."""

from olp_ai_26.core.colab import ColabPaths
from olp_ai_26.core.config import CompetitionConfig, TrainerConfig
from olp_ai_26.core.metrics import evaluate_metric, get_metric
from olp_ai_26.core.split import SplitResult, make_split

__all__ = [
    "CompetitionConfig",
    "ColabPaths",
    "SplitResult",
    "TrainerConfig",
    "evaluate_metric",
    "get_metric",
    "make_split",
]
