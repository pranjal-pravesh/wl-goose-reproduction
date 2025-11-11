#!/usr/bin/env python3
"""
Verify WL-GOOSE Installation

This script checks that all components are properly installed and configured.
"""

import sys
import os
from pathlib import Path
import subprocess

def print_header(text):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_packages():
    """Check required Python packages."""
    print("\nChecking Python packages...")
    
    required = [
        'numpy',
        'pandas',
        'sklearn',
        'pddlpy',
        'networkx',
        'matplotlib',
        'seaborn',
        'joblib',
        'tqdm'
    ]
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            all_ok = False
    
    return all_ok

def check_fast_downward():
    """Check Fast Downward installation."""
    print("\nChecking Fast Downward...")
    
    fd_path = Path("/Users/pranjal/HomeBase/learned-heuristic-planner/downward/fast-downward.py")
    
    if not fd_path.exists():
        print(f"❌ Fast Downward not found at: {fd_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(fd_path), "--help"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✅ Fast Downward is operational")
            return True
        else:
            print(f"❌ Fast Downward failed to run")
            return False
    except Exception as e:
        print(f"❌ Error running Fast Downward: {e}")
        return False

def check_src_modules():
    """Check that all source modules exist."""
    print("\nChecking source modules...")
    
    project_root = Path("/Users/pranjal/HomeBase/learned-heuristic-planner")
    
    required_modules = [
        'src/ilg_builder.py',
        'src/wl_features.py',
        'src/generate_training_data.py',
        'src/train_models.py',
        'src/gbfs_search.py',
        'src/evaluate.py',
        'src/run_baselines.py',
        'src/analyze_results.py'
    ]
    
    all_ok = True
    for module in required_modules:
        path = project_root / module
        if path.exists():
            print(f"✅ {module}")
        else:
            print(f"❌ {module} (missing)")
            all_ok = False
    
    return all_ok

def check_scripts():
    """Check that all scripts exist."""
    print("\nChecking execution scripts...")
    
    project_root = Path("/Users/pranjal/HomeBase/learned-heuristic-planner")
    
    required_scripts = [
        'scripts/fetch_benchmarks_from_repo.py',
        'scripts/run_training.sh',
        'scripts/run_evaluation.sh'
    ]
    
    all_ok = True
    for script in required_scripts:
        path = project_root / script
        if path.exists():
            # Check if executable
            if os.access(path, os.X_OK):
                print(f"✅ {script} (executable)")
            else:
                print(f"⚠️  {script} (exists but not executable)")
        else:
            print(f"❌ {script} (missing)")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check that required directories exist."""
    print("\nChecking directory structure...")
    
    project_root = Path("/Users/pranjal/HomeBase/learned-heuristic-planner")
    
    required_dirs = [
        'benchmarks',
        'data',
        'models',
        'results',
        'src',
        'scripts'
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        path = project_root / dir_name
        if path.exists() and path.is_dir():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (missing)")
            all_ok = False
    
    return all_ok

def check_benchmarks():
    """Check if benchmarks are downloaded."""
    print("\nChecking benchmarks...")
    
    project_root = Path("/Users/pranjal/HomeBase/learned-heuristic-planner")
    benchmark_dir = project_root / "benchmarks"
    
    if not benchmark_dir.exists():
        print("❌ Benchmarks directory not found")
        return False
    
    # Check a few key domains
    sample_domains = ['blocksworld', 'rovers', 'satellite']
    found_count = 0
    
    for domain in sample_domains:
        domain_path = benchmark_dir / domain
        if domain_path.exists():
            # Check for problem files
            easy_problems = list((domain_path / 'easy').glob('problem*.pddl'))
            hard_problems = list((domain_path / 'hard').glob('problem*.pddl'))
            
            if easy_problems or hard_problems:
                print(f"✅ {domain} ({len(easy_problems)} easy, {len(hard_problems)} hard)")
                found_count += 1
            else:
                print(f"⚠️  {domain} (no problem files found)")
        else:
            print(f"⚠️  {domain} (not downloaded)")
    
    if found_count > 0:
        print(f"\n   Found {found_count}/{len(sample_domains)} sample domains")
        print(f"   Run: python3 scripts/fetch_benchmarks_from_repo.py to download all")
        return True
    else:
        print(f"\n❌ No benchmarks found. Run: python3 scripts/fetch_benchmarks_from_repo.py")
        return False

def main():
    """Run all checks."""
    print_header("WL-GOOSE Installation Verification")
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", check_packages),
        ("Fast Downward", check_fast_downward),
        ("Source Modules", check_src_modules),
        ("Scripts", check_scripts),
        ("Directories", check_directories),
        ("Benchmarks", check_benchmarks)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready to run experiments.")
        print("\nQuick start:")
        print("  1. Download benchmarks: python3 scripts/fetch_benchmarks_from_repo.py")
        print("  2. Run training: ./scripts/run_training.sh")
        print("  3. Run evaluation: ./scripts/run_evaluation.sh")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing Fast Downward: cd downward && ./build.py")
        print("  - Non-executable scripts: chmod +x scripts/*.sh scripts/*.py")
        return 1

if __name__ == "__main__":
    exit(main())


