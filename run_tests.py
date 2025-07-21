#!/usr/bin/env python3
"""
Simple CLI script to run EvolveProver tests.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description="Run EvolveProver tests")
    parser.add_argument(
        "--episode", 
        default="1.1",
        help="Episode to test (default: 1.1)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.episode == "1.1":
        # Run Episode 1.1 tests
        try:
            from tests.test_setup import main as test_main
            
            print("Running Arc 1, Episode 1.1 tests...")
            success = test_main()
            
            if success:
                print("\n✅ Episode 1.1 tests completed successfully!")
            else:
                print("\n❌ Episode 1.1 tests failed. Check output above.")
            
            return success
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Make sure you're running from the project root directory.")
            return False
    else:
        print(f"❌ Episode {args.episode} tests not implemented yet.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)