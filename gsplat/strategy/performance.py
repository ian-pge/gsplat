from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy
from .default import DefaultStrategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
@dataclass
class PerformanceStrategy(DefaultStrategy):
    """A strategy optimized for high-quality rendering (e.g. 4K images).

    It inherits from DefaultStrategy but uses settings that prioritize detail and quality
    over training speed or memory usage.

    Key differences:
    - absgrad=True (better structure)
    - Lower pruning thresholds (keep more details)
    - More frequent refinement
    - Revised opacity heuristic
    """

    prune_opa: float = 0.001
    grow_grad2d: float = 0.0008  # Higher threshold for absgrad=True (doc recommends 0.0008)
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 50_000  # Extended for high quality
    reset_every: int = 3000
    refine_every: int = 50       # Refine more often
    pause_refine_after_reset: int = 0
    absgrad: bool = True         # Use absolute gradients
    revised_opacity: bool = True # Use revised opacity
    verbose: bool = True
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"


