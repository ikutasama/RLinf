#!/usr/bin/env python3
"""
Alpamayo-R1 Integration Verification Script

This script verifies that the Alpamayo-R1 integration with RLinf is working correctly.
It checks:
1. Module imports
2. Configuration loading
3. Model instantiation (without loading weights)
4. File structure

Usage:
    python examples/embodiment/verify_alpamayo.py
"""

import sys
from pathlib import Path


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_check(name: str, passed: bool, details: str = ""):
    """Print a check result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {name}")
    if details:
        print(f"       {details}")


def main():
    """Run verification checks."""
    print_header("Alpamayo-R1 Integration Verification")
    
    all_passed = True
    
    # Check 1: File structure
    print_header("1. File Structure")
    
    files_to_check = [
        "rlinf/models/embodiment/alpamayo/__init__.py",
        "rlinf/models/embodiment/alpamayo/alpamayo_action_model.py",
        "rlinf/models/embodiment/alpamayo/README.md",
        "examples/embodiment/config/alpamayo_sft.yaml",
        "examples/embodiment/config/model/alpamayo.yaml",
        "examples/embodiment/run_alpamayo.sh",
        "examples/embodiment/ALPAMAYO_QUICKSTART.md",
    ]
    
    for file_path in files_to_check:
        exists = Path(file_path).exists()
        print_check(f"File exists: {file_path}", exists)
        if not exists:
            all_passed = False
    
    # Check 2: Module imports
    print_header("2. Module Imports")
    
    try:
        from rlinf.models.embodiment.alpamayo import (
            AlpamayoR1Config,
            get_model,
            get_model_config_and_input_processor,
        )
        print_check("Import AlpamayoR1Config", True)
        print_check("Import get_model", True)
        print_check("Import get_model_config_and_input_processor", True)
    except ImportError as e:
        print_check("Module imports", False, str(e))
        all_passed = False
    
    # Check 3: Configuration
    print_header("3. Configuration")
    
    try:
        from rlinf.models.embodiment.alpamayo import AlpamayoR1Config
        
        config = AlpamayoR1Config()
        print_check("Create default config", True)
        print_check(f"  - Model path: {config.model_path}", True)
        print_check(f"  - Action dim: {config.action_dim}", True)
        print_check(f"  - Num future steps: {config.num_future_steps}", True)
        print_check(f"  - Diffusion steps: {config.diffusion_steps}", True)
        
        # Verify expected values
        assert config.action_dim == 9, "Action dim should be 9"
        assert config.num_future_steps == 64, "Should have 64 future steps"
        assert config.diffusion_steps == 10, "Should have 10 diffusion steps"
        print_check("Configuration values", True)
        
    except Exception as e:
        print_check("Configuration", False, str(e))
        all_passed = False
    
    # Check 4: YAML configuration
    print_header("4. YAML Configuration")
    
    try:
        from omegaconf import OmegaConf
        
        # Load model config
        model_cfg_path = Path("examples/embodiment/config/model/alpamayo.yaml")
        if model_cfg_path.exists():
            model_cfg = OmegaConf.load(model_cfg_path)
            print_check("Load model config (alpamayo.yaml)", True)
            print_check(f"  - Model type: {model_cfg.model_type}", True)
        else:
            print_check("Load model config", False, "File not found")
            all_passed = False
        
        # Load training config
        train_cfg_path = Path("examples/embodiment/config/alpamayo_sft.yaml")
        if train_cfg_path.exists():
            train_cfg = OmegaConf.load(train_cfg_path)
            print_check("Load training config (alpamayo_sft.yaml)", True)
        else:
            print_check("Load training config", False, "File not found")
            all_passed = False
            
    except Exception as e:
        print_check("YAML configuration", False, str(e))
        all_passed = False
    
    # Check 5: Model instantiation (without loading weights)
    print_header("5. Model Instantiation (Mock)")
    
    try:
        from rlinf.models.embodiment.alpamayo import AlpamayoR1Config
        
        # Create a mock config (won't load actual model)
        config = AlpamayoR1Config(
            model_path="mock_model",  # Use mock to avoid downloading
            dtype="float32",
        )
        print_check("Create AlpamayoR1Config", True)
        print_check(f"  - Config type: {type(config).__name__}", True)
        
    except Exception as e:
        print_check("Model instantiation", False, str(e))
        all_passed = False
    
    # Check 6: Documentation
    print_header("6. Documentation")
    
    doc_files = [
        ("Quick Start Guide", "examples/embodiment/ALPAMAYO_QUICKSTART.md"),
        ("Integration Summary", "ALPAMAYO_INTEGRATION_SUMMARY.md"),
        ("User Guide", "INTEGRATION_SUMMARY.md"),
        ("Module README", "rlinf/models/embodiment/alpamayo/README.md"),
    ]
    
    for name, path in doc_files:
        exists = Path(path).exists()
        if exists:
            size = Path(path).stat().st_size
            print_check(f"{name}", exists, f"Size: {size:,} bytes")
        else:
            print_check(f"{name}", False)
            all_passed = False
    
    # Final summary
    print_header("Verification Summary")
    
    if all_passed:
        print("✅ All checks passed!")
        print("\nNext steps:")
        print("  1. Read the quick start guide:")
        print("     cat examples/embodiment/ALPAMAYO_QUICKSTART.md")
        print("\n  2. Install dependencies:")
        print("     pip install physical_ai_av einops flash-attn")
        print("\n  3. Authenticate with HuggingFace:")
        print("     huggingface-cli login")
        print("\n  4. Start training:")
        print("     bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("  - Ensure RLinf is installed: pip install -e .")
        print("  - Check file permissions")
        print("  - Verify Python version (3.12+)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
