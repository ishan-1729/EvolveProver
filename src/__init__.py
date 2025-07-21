"""
EvolveProver - AI Mathematical Reasoning Enhancement System

Arc 1, Episode 1.1: Environment Setup and Model Loading
"""

from .config import CONFIG, setup_logging
from .models import DeepSeekProverGGUF, GenerationResult, ModelDownloader

__version__ = "0.1.0"
__all__ = [
    "CONFIG",
    "setup_logging", 
    "DeepSeekProverGGUF",
    "GenerationResult",
    "ModelDownloader"
]