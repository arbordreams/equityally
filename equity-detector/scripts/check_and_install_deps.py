#!/usr/bin/env python3
"""
Dependency Checker and Installer
==================================
Checks for required packages and provides installation guidance.
"""

import sys
import subprocess

REQUIRED_PACKAGES = {
    'transformers': 'transformers',
    'torch': 'torch',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'tqdm': 'tqdm',
}

OPTIONAL_PACKAGES = {
    'umap': 'umap-learn',
}

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("="*80)
    print("DEPENDENCY CHECKER")
    print("="*80)
    print()
    
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()
    
    missing_required = []
    missing_optional = []
    
    print("Checking required packages...")
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if check_package(import_name):
            print(f"  ✓ {import_name}")
        else:
            print(f"  ✗ {import_name} (MISSING)")
            missing_required.append(pip_name)
    
    print()
    print("Checking optional packages...")
    for import_name, pip_name in OPTIONAL_PACKAGES.items():
        if check_package(import_name):
            print(f"  ✓ {import_name}")
        else:
            print(f"  - {import_name} (optional, not found)")
            missing_optional.append(pip_name)
    
    print()
    
    if missing_required:
        print("="*80)
        print("MISSING REQUIRED PACKAGES")
        print("="*80)
        print()
        print("The following packages are required but not installed:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print()
        print("To install missing packages, run:")
        print()
        print(f"  {sys.executable} -m pip install {' '.join(missing_required)}")
        print()
        print("Or use the requirements file:")
        print()
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        print()
        return 1
    
    if missing_optional:
        print("Note: Some optional packages are missing:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print()
        print("The evaluation will run without these optional features.")
        print()
    
    print("="*80)
    print("✓ ALL REQUIRED PACKAGES INSTALLED")
    print("="*80)
    print()
    print("You can proceed with the evaluation:")
    print(f"  {sys.executable} scripts/run_complete_evaluation.py")
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())

