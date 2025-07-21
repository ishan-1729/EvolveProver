"""
Utilities for downloading and managing GGUF models.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from ..config import CONFIG

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Handles downloading and managing GGUF models from Hugging Face."""
    
    def __init__(self, repo_id: str = CONFIG.model.model_repo):
        self.repo_id = repo_id
        self.models_dir = CONFIG.paths.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_models(self) -> List[str]:
        """List all available GGUF files in the repository."""
        try:
            files = list_repo_files(self.repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            return gguf_files
        except Exception as e:
            logger.error(f"Failed to list repository files: {e}")
            return []
    
    def download_model(self, filename: str, force_download: bool = False) -> Path:
        """
        Download a specific GGUF model file.
        
        Args:
            filename: Name of the GGUF file to download
            force_download: Whether to re-download if file already exists
            
        Returns:
            Path to the downloaded model file
        """
        local_path = self.models_dir / filename
        
        if local_path.exists() and not force_download:
            logger.info(f"Model {filename} already exists at {local_path}")
            return local_path
        
        try:
            logger.info(f"Downloading {filename} from {self.repo_id}...")
            
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Successfully downloaded {filename} to {downloaded_path}")
            return Path(downloaded_path)
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise
    
    def get_recommended_model(self) -> str:
        """Get the recommended model file based on available RAM."""
        available_models = self.list_available_models()
        
        if not available_models:
            raise ValueError("No GGUF models found in repository")
        
        # Mapping of quantization levels to approximate RAM requirements
        model_priorities = {
            # Format: (model_suffix, min_ram_gb, description)
            "Q4_K_M.gguf": (6, "Medium quality 4-bit quantization (recommended)"),
            "Q4_K_S.gguf": (5, "Small 4-bit quantization (faster)"),
            "Q5_K_M.gguf": (8, "Medium quality 5-bit quantization"),
            "Q6_K.gguf": (10, "6-bit quantization (higher quality)"),
            "Q8_0.gguf": (14, "8-bit quantization (highest quality)"),
            "F16.gguf": (16, "16-bit floating point (original quality)")
        }
        
        max_ram = CONFIG.model.max_ram_gb
        logger.info(f"Selecting model for {max_ram}GB RAM limit")
        
        # Find suitable models
        suitable_models = []
        for model in available_models:
            for suffix, (min_ram, description) in model_priorities.items():
                if model.endswith(suffix) and min_ram <= max_ram:
                    suitable_models.append((model, min_ram, description))
                    break
        
        if not suitable_models:
            logger.warning(f"No models found suitable for {max_ram}GB RAM. Using smallest available.")
            # Return the model with the smallest file (likely most quantized)
            return min(available_models, key=lambda x: len(x))
        
        # Return the highest quality model that fits in RAM
        best_model = max(suitable_models, key=lambda x: x[1])
        logger.info(f"Selected model: {best_model[0]} ({best_model[2]})")
        return best_model[0]
    
    def setup_default_model(self) -> Path:
        """Download and set up the recommended model for this system."""
        recommended_model = self.get_recommended_model()
        return self.download_model(recommended_model)

def verify_model_file(model_path: Path) -> bool:
    """
    Verify that a model file exists and appears valid.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if the model file appears valid
    """
    if not model_path.exists():
        logger.error(f"Model file does not exist: {model_path}")
        return False
    
    # Check file size (GGUF models should be at least 1GB)
    file_size = model_path.stat().st_size
    if file_size < 1_000_000_000:  # 1GB
        logger.error(f"Model file appears too small: {file_size} bytes")
        return False
    
    # Check file extension
    if not model_path.suffix.lower() == '.gguf':
        logger.error(f"Model file does not have .gguf extension: {model_path}")
        return False
    
    logger.info(f"Model file verified: {model_path} ({file_size / 1_000_000_000:.1f}GB)")
    return True

def get_model_info(model_path: Path) -> dict:
    """
    Get basic information about a model file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model information
    """
    if not model_path.exists():
        return {"error": "File does not exist"}
    
    stat = model_path.stat()
    return {
        "name": model_path.name,
        "size_bytes": stat.st_size,
        "size_gb": stat.st_size / 1_000_000_000,
        "modified": stat.st_mtime,
        "path": str(model_path)
    }