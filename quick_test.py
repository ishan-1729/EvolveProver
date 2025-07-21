#!/usr/bin/env python3
"""
Quick test script that directly runs the setup tests without package imports.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    print("EvolveProver - Quick Setup Test")
    print("=" * 50)
    
    # Test 1: Basic imports
    print("Testing basic imports...")
    try:
        import config
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    try:
        from models.model_utils import ModelDownloader
        print("✅ ModelDownloader imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ModelDownloader: {e}")
        return False
    
    try:
        from models.deepseek_prover_gguf import DeepSeekProverGGUF
        print("✅ DeepSeekProverGGUF imported successfully")
    except Exception as e:
        print(f"❌ Failed to import DeepSeekProverGGUF: {e}")
        print("Note: This might be due to missing llama-cpp-python")
        print("Install with: pip install llama-cpp-python")
        return False
    
    # Test 2: Configuration
    print("\nTesting configuration...")
    try:
        cfg = config.CONFIG
        print(f"✅ Configuration loaded")
        print(f"   Models directory: {cfg.paths.models_dir}")
        print(f"   Model repo: {cfg.model.model_repo}")
        print(f"   Max RAM: {cfg.model.max_ram_gb}GB")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 3: Model downloader
    print("\nTesting model downloader...")
    try:
        downloader = ModelDownloader()
        models = downloader.list_available_models()
        print(f"✅ Found {len(models)} available models")
        
        recommended = downloader.get_recommended_model()
        print(f"✅ Recommended model: {recommended}")
    except Exception as e:
        print(f"❌ Model downloader test failed: {e}")
        print("This might be due to network issues or missing Hugging Face access")
        return False
    
    # Test 4: Model initialization (without loading)
    print("\nTesting model initialization...")
    try:
        model = DeepSeekProverGGUF()
        print(f"✅ Model wrapper initialized")
        print(f"   Model path: {model.model_path}")
        print(f"   Context length: {model.n_ctx}")
        print(f"   CPU threads: {model.n_threads}")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Basic setup tests passed!")
    print("\nNow you can try:")
    print("1. python tests/test_setup.py  # Run full test suite")
    print("2. Download and test the actual model")
    print("\nIf you want to download the model immediately:")
    print("python -c \"from src.models.model_utils import ModelDownloader; ModelDownloader().setup_default_model()\"")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Some tests failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n🎉 Ready to proceed!")
        sys.exit(0)