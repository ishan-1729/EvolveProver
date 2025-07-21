"""
Test script for Arc 1, Episode 1.1 - Environment Setup and Model Loading
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import CONFIG, setup_logging
from models.model_utils import ModelDownloader, verify_model_file, get_model_info
from models.deepseek_prover_gguf import DeepSeekProverGGUF

def test_configuration():
    """Test that configuration loads correctly."""
    print("=" * 50)
    print("Testing Configuration")
    print("=" * 50)
    
    print(f"Data directory: {CONFIG.paths.data_dir}")
    print(f"Models directory: {CONFIG.paths.models_dir}")
    print(f"Model path: {CONFIG.model.model_path}")
    print(f"Model repo: {CONFIG.model.model_repo}")
    print(f"Max RAM: {CONFIG.model.max_ram_gb}GB")
    print(f"CPU threads: {CONFIG.model.cpu_threads}")
    print(f"GPU enabled: {CONFIG.model.gpu_enabled}")
    print(f"Context length: {CONFIG.model.context_length}")
    print(f"Evolution API: {CONFIG.evolution.api_base}")
    print(f"Evolution model: {CONFIG.evolution.model_name}")
    
    # Check if API key is set
    if CONFIG.evolution.api_key:
        print(f"OpenRouter API key: {'*' * 20}[HIDDEN]")
    else:
        print("⚠️  OpenRouter API key not set")
    
    print("✅ Configuration test passed")
    print()

def test_model_downloader():
    """Test model downloading functionality."""
    print("=" * 50)
    print("Testing Model Downloader")
    print("=" * 50)
    
    downloader = ModelDownloader()
    
    # List available models
    print("Available GGUF models:")
    try:
        models = downloader.list_available_models()
        for model in models[:5]:  # Show first 5
            print(f"  - {model}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return False
    
    # Get recommended model
    try:
        recommended = downloader.get_recommended_model()
        print(f"\nRecommended model for {CONFIG.model.max_ram_gb}GB RAM: {recommended}")
    except Exception as e:
        print(f"❌ Failed to get recommended model: {e}")
        return False
    
    print("✅ Model downloader test passed")
    print()
    return True

def test_model_download():
    """Test downloading the recommended model."""
    print("=" * 50)
    print("Testing Model Download")
    print("=" * 50)
    
    model_path = Path(CONFIG.model.model_path)
    
    if model_path.exists():
        print(f"Model already exists: {model_path}")
        if verify_model_file(model_path):
            info = get_model_info(model_path)
            print(f"Model size: {info['size_gb']:.1f}GB")
            print("✅ Model download test passed (existing file)")
            return True
        else:
            print("❌ Existing model file is invalid")
            return False
    
    # Download model
    print("Model not found. Starting download...")
    print("⚠️  This may take several minutes depending on your internet connection")
    
    try:
        downloader = ModelDownloader()
        downloaded_path = downloader.setup_default_model()
        
        if verify_model_file(downloaded_path):
            info = get_model_info(downloaded_path)
            print(f"✅ Model downloaded successfully: {downloaded_path}")
            print(f"Model size: {info['size_gb']:.1f}GB")
            return True
        else:
            print("❌ Downloaded model file is invalid")
            return False
            
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def test_model_loading():
    """Test loading the GGUF model."""
    print("=" * 50)
    print("Testing Model Loading")
    print("=" * 50)
    
    try:
        # Initialize model wrapper
        print("Initializing DeepSeek-Prover-V2 wrapper...")
        model = DeepSeekProverGGUF(verbose=False)
        
        # Load model
        print("Loading GGUF model (this may take a minute)...")
        start_time = time.time()
        
        if model.load_model():
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.2f}s")
            
            # Test basic properties
            print(f"Context length: {model.get_context_length()} tokens")
            print(f"Model loaded: {model.is_loaded()}")
            
            # Clean up
            model.unload_model()
            print("Model unloaded from memory")
            
            return True
        else:
            print("❌ Failed to load model")
            return False
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_basic_inference():
    """Test basic text generation."""
    print("=" * 50)
    print("Testing Basic Inference")
    print("=" * 50)
    
    test_prompt = """Solve this mathematical problem step by step:

Problem: If x + 3 = 7, what is the value of x?

Solution:"""
    
    try:
        print("Loading model for inference test...")
        with DeepSeekProverGGUF(verbose=False) as model:
            print("Generating response...")
            
            result = model.generate(
                prompt=test_prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            print(f"Generated {result.tokens_generated} tokens in {result.time_taken:.2f}s")
            print(f"Speed: {result.tokens_per_second:.1f} tokens/sec")
            print(f"Stop reason: {result.stop_reason}")
            print("\nGenerated text:")
            print("-" * 30)
            print(result.text)
            print("-" * 30)
            
            # Check if we got a reasonable response
            if result.tokens_generated > 0 and len(result.text.strip()) > 10:
                print("✅ Basic inference test passed")
                return True
            else:
                print("❌ Generated text appears too short or empty")
                return False
                
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def main():
    """Run all tests for Episode 1.1."""
    print("EvolveProver - Arc 1, Episode 1.1 Testing")
    print("Testing Environment Setup and Model Loading")
    print()
    
    # Set up logging
    setup_logging(CONFIG)
    
    # Run tests
    tests = [
        ("Configuration", test_configuration),
        ("Model Downloader", test_model_downloader),
        ("Model Download", test_model_download),
        ("Model Loading", test_model_loading),
        ("Basic Inference", test_basic_inference),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Episode 1.1 setup is complete.")
        print("\nNext steps:")
        print("1. Set up your .env file with OpenRouter API key")
        print("2. Test with different mathematical prompts")
        print("3. Begin Arc 1, Episode 1.2 - Basic Model Interface")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please fix issues before proceeding.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)