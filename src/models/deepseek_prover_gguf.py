"""
DeepSeek-Prover-V2 7B GGUF model wrapper for mathematical reasoning.
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    raise ImportError(
        "llama-cpp-python is required. Install with: pip install llama-cpp-python"
    )

from ..config import CONFIG
from .model_utils import ModelDownloader, verify_model_file, get_model_info

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from model generation."""

    text: str
    tokens_generated: int
    time_taken: float
    tokens_per_second: float
    stop_reason: str


class DeepSeekProverGGUF:
    """
    DeepSeek-Prover-V2 7B model wrapper using GGUF format for CPU inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize the DeepSeek-Prover-V2 model.

        Args:
            model_path: Path to GGUF model file. If None, uses config default.
            n_ctx: Context window size. If None, uses config default.
            n_threads: Number of CPU threads. If None, uses config default.
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU-only).
            verbose: Enable verbose logging from llama.cpp.
        """
        self.model_path = (
            Path(model_path) if model_path else Path(CONFIG.model.model_path)
        )
        self.n_ctx = n_ctx or CONFIG.model.context_length
        self.n_threads = n_threads or CONFIG.model.cpu_threads
        self.n_gpu_layers = n_gpu_layers if CONFIG.model.gpu_enabled else 0
        self.verbose = verbose
        self.model: Optional[Llama] = None

        # Generation parameters
        self.default_params = {
            "max_tokens": CONFIG.model.max_tokens,
            "temperature": CONFIG.model.temperature,
            "top_p": CONFIG.model.top_p,
            "stop": ["<|endoftext|>", "<|end|>", "\n\n---", "```\n\n"],
            "echo": False,
        }

        logger.info(
            f"Initialized DeepSeekProverGGUF with model path: {self.model_path}"
        )

    def download_model_if_needed(self) -> bool:
        """
        Download the model if it doesn't exist locally.

        Returns:
            True if model is available (either existed or was downloaded successfully)
        """
        if self.model_path.exists():
            logger.info(f"Model already exists: {self.model_path}")
            return verify_model_file(self.model_path)

        try:
            logger.info("Model not found locally. Downloading...")
            downloader = ModelDownloader()
            downloaded_path = downloader.setup_default_model()

            # Update model path to the downloaded file
            self.model_path = downloaded_path
            CONFIG.model.model_path = str(downloaded_path)

            return True

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    def load_model(self) -> bool:
        """
        Load the GGUF model into memory.

        Returns:
            True if model loaded successfully
        """
        if self.model is not None:
            logger.info("Model already loaded")
            return True

        if not self.download_model_if_needed():
            logger.error("Cannot load model: download failed")
            return False

        try:
            logger.info("Loading GGUF model...")
            start_time = time.time()

            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                use_mlock=True,  # Keep model in RAM
                use_mmap=True,  # Use memory mapping for efficiency
            )

            load_time = time.time() - start_time
            model_info = get_model_info(self.model_path)

            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Model size: {model_info['size_gb']:.1f}GB")
            logger.info(f"Context window: {self.n_ctx} tokens")
            logger.info(f"CPU threads: {self.n_threads}")
            logger.info(f"GPU layers: {self.n_gpu_layers}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text from the model.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            **kwargs: Additional parameters for generation

        Returns:
            GenerationResult with generated text and metadata
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare generation parameters
        gen_params = self.default_params.copy()
        if max_tokens is not None:
            gen_params["max_tokens"] = max_tokens
        if temperature is not None:
            gen_params["temperature"] = temperature
        if top_p is not None:
            gen_params["top_p"] = top_p
        if stop is not None:
            gen_params["stop"] = stop
        gen_params.update(kwargs)

        logger.debug(f"Generating with params: {gen_params}")

        start_time = time.time()

        try:
            response = self.model(prompt, **gen_params)

            end_time = time.time()
            time_taken = end_time - start_time

            # Extract response data
            generated_text = response["choices"][0]["text"]
            tokens_generated = response["usage"]["completion_tokens"]
            stop_reason = response["choices"][0]["finish_reason"]
            tokens_per_second = tokens_generated / time_taken if time_taken > 0 else 0

            logger.debug(
                f"Generated {tokens_generated} tokens in {time_taken:.2f}s "
                f"({tokens_per_second:.1f} tokens/sec)"
            )

            return GenerationResult(
                text=generated_text,
                tokens_generated=tokens_generated,
                time_taken=time_taken,
                tokens_per_second=tokens_per_second,
                stop_reason=stop_reason,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_stream(
        self, prompt: str, **kwargs
    ) -> Generator[str, None, GenerationResult]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input prompt text
            **kwargs: Generation parameters

        Yields:
            Incremental text chunks

        Returns:
            Final GenerationResult
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        gen_params = self.default_params.copy()
        gen_params.update(kwargs)
        gen_params["stream"] = True

        start_time = time.time()
        full_text = ""
        tokens_generated = 0

        try:
            for chunk in self.model(prompt, **gen_params):
                if chunk["choices"][0]["text"]:
                    text_chunk = chunk["choices"][0]["text"]
                    full_text += text_chunk
                    tokens_generated += 1
                    yield text_chunk

            end_time = time.time()
            time_taken = end_time - start_time
            tokens_per_second = tokens_generated / time_taken if time_taken > 0 else 0

            return GenerationResult(
                text=full_text,
                tokens_generated=tokens_generated,
                time_taken=time_taken,
                tokens_per_second=tokens_per_second,
                stop_reason="length",  # Simplified for streaming
            )

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def encode(self, text: str) -> List[int]:
        """
        Encode text to tokens.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return self.model.tokenize(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return self.model.detokenize(tokens).decode("utf-8")

    def get_context_length(self) -> int:
        """Get the model's context window size."""
        return self.n_ctx

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            logger.info("Unloading model from memory")
            del self.model
            self.model = None

    def __enter__(self):
        """Context manager entry."""
        if not self.load_model():
            raise RuntimeError("Failed to load model")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()

    def __del__(self):
        """Destructor to ensure model is unloaded."""
        if hasattr(self, "model") and self.model is not None:
            self.unload_model()
