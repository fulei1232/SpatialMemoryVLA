"""Deployment adapters kept separate from training and checkpoint code."""

from .realman_rm65 import RM65Client, RM65SafetyConfig

__all__ = ["RM65Client", "RM65SafetyConfig"]
