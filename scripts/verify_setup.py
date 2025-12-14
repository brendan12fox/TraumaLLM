#!/usr/bin/env python3
"""
Verify Setup for Cloud Execution

Quick check to ensure all files and dependencies are ready for training.
Run this on the cloud instance before training.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent

def check_files():
    """Check that required files exist."""
    print("Checking required files...")
    
    required = {
        'Dataset': PROJECT_ROOT / "outputs" / "instruction_dataset.jsonl",
        'Rules': PROJECT_ROOT / "rules" / "trauma_triage_rules.json",
        'Data copy': PROJECT_ROOT / "data" / "OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx",
    }
    
    all_ok = True
    for name, path in required.items():
        if path.exists():
            size = path.stat().st_size / 1024 / 1024  # MB
            print(f"  ✅ {name}: {path.name} ({size:.1f} MB)")
        else:
            print(f"  ❌ {name}: NOT FOUND at {path}")
            all_ok = False
    
    return all_ok

def check_dependencies():
    """Check that required Python packages are installed."""
    print("\nChecking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'bitsandbytes': 'BitsAndBytes',
        'datasets': 'Datasets',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} not installed. Run: pip install -r requirements.txt")
            all_ok = False
    
    return all_ok

def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ CUDA available")
            print(f"     Device: {device_name}")
            print(f"     Memory: {memory_gb:.1f} GB")
            return True
        else:
            print(f"  ❌ CUDA not available. Training requires GPU.")
            return False
    except ImportError:
        print(f"  ⚠️  PyTorch not installed. Cannot check GPU.")
        return False

def main():
    """Run all checks."""
    print("="*70)
    print("CLOUD SETUP VERIFICATION")
    print("="*70)
    
    files_ok = check_files()
    deps_ok = check_dependencies()
    gpu_ok = check_gpu()
    
    print("\n" + "="*70)
    if files_ok and deps_ok and gpu_ok:
        print("✅ ALL CHECKS PASSED - Ready for training!")
        print("\nRun: python3 scripts/train_lora.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before training")
        if not files_ok:
            print("   → Upload missing files")
        if not deps_ok:
            print("   → Run: pip install -r requirements.txt")
        if not gpu_ok:
            print("   → Ensure GPU instance is running")
        return 1

if __name__ == "__main__":
    sys.exit(main())
