"""Match-first GW catalog pipeline.

This package keeps the repository name but makes the match project data and
Siamese retrieval workflow the primary implementation surface.
"""

from .config import MatchRunConfig

__all__ = ["MatchRunConfig"]
