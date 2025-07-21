"""
Configuration management for EvolveProver project.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for DeepSeek-Prover-V2 model."""
    model_path: str
    model_repo: str
    max_ram_gb: int
    cpu_threads: int
    gpu_enabled: bool
    max_tokens: int
    temperature: float
    top_p: float
    context_length: int

@dataclass
class EvolutionConfig:
    """Configuration for prompt evolution."""
    api_key: str
    api_base: str
    model_name: str

@dataclass
class PathConfig:
    """Configuration for data and output paths."""
    data_dir: Path
    results_dir: Path
    models_dir: Path
    benchmarks_dir: Path
    logs_dir: Path

@dataclass
class EvolveProverConfig:
    """Main configuration class."""
    model: ModelConfig
    evolution: EvolutionConfig
    paths: PathConfig
    log_level: str
    log_file: str

def get_config() -> EvolveProverConfig:
    """Load configuration from environment variables."""
    
    # Create directories if they don't exist
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    results_dir = Path(os.getenv("RESULTS_DIR", "./results"))
    models_dir = Path(os.getenv("MODELS_DIR", "./data/models"))
    benchmarks_dir = Path(os.getenv("BENCHMARKS_DIR", "./data/benchmarks"))
    logs_dir = Path("./logs")
    
    for directory in [data_dir, results_dir, models_dir, benchmarks_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Model configuration
    model_config = ModelConfig(
        model_path=os.getenv("DEEPSEEK_MODEL_PATH", "./data/models/DeepSeek-Prover-V2-7B-Q4_K_M.gguf"),
        model_repo=os.getenv("DEEPSEEK_MODEL_REPO", "unsloth/DeepSeek-Prover-V2-7B-GGUF"),
        max_ram_gb=int(os.getenv("MAX_RAM_GB", "12")),
        cpu_threads=int(os.getenv("CPU_THREADS", "8")),
        gpu_enabled=os.getenv("GPU_ENABLED", "false").lower() == "true",
        max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        context_length=int(os.getenv("CONTEXT_LENGTH", "4096"))
    )
    
    # Evolution configuration
    evolution_config = EvolutionConfig(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        api_base=os.getenv("EVOLUTION_API_BASE", "https://openrouter.ai/api/v1"),
        model_name=os.getenv("EVOLUTION_MODEL", "deepseek/deepseek-r1")
    )
    
    # Path configuration
    path_config = PathConfig(
        data_dir=data_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        benchmarks_dir=benchmarks_dir,
        logs_dir=logs_dir
    )
    
    return EvolveProverConfig(
        model=model_config,
        evolution=evolution_config,
        paths=path_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "./logs/evolveprover.log")
    )

def setup_logging(config: EvolveProverConfig) -> None:
    """Set up logging configuration."""
    import logging
    from pathlib import Path
    
    # Ensure log directory exists
    log_file_path = Path(config.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )

# Global configuration instance
CONFIG = get_config()