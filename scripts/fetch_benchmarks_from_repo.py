#!/usr/bin/env python3
"""
Script to fetch benchmark files from the downward-benchmarks repository.
The IPC 2023 Learning Track uses standard benchmark domains.
"""

import os
import subprocess
import shutil
from pathlib import Path

BENCHMARK_DIR = Path("/Users/pranjal/HomeBase/learned-heuristic-planner/benchmarks")
TEMP_DIR = Path("/tmp/downward-benchmarks-temp")

# Domain mappings from repository to our structure
DOMAIN_MAPPINGS = {
    "blocksworld": "blocks",
    "childsnack": "childsnack-opt14-strips",
    "ferry": "ferry",
    "floortile": "floortile-opt11-strips",
    "miconic": "miconic",
    "rovers": "rovers",
    "satellite": "satellite",
    "sokoban": "sokoban-opt08-strips",
    "spanner": "spanner",
    "transport": "transport-opt11-strips"
}

def clone_benchmark_repo():
    """Clone the downward-benchmarks repository."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    print("Cloning downward-benchmarks repository...")
    subprocess.run([
        "git", "clone", 
        "https://github.com/aibasel/downward-benchmarks.git",
        str(TEMP_DIR)
    ], check=True)

def copy_domain_files(repo_domain, our_domain):
    """Copy domain files from repository to our structure."""
    source_dir = TEMP_DIR / repo_domain
    
    if not source_dir.exists():
        print(f"Warning: {repo_domain} not found in repository")
        return
    
    # Find domain file
    domain_files = list(source_dir.glob("domain*.pddl"))
    if not domain_files:
        domain_files = list(source_dir.glob("**/domain*.pddl"))
    
    if domain_files:
        domain_file = domain_files[0]
        # Copy domain file to all difficulty levels
        for level in ["easy", "medium", "hard"]:
            dest = BENCHMARK_DIR / our_domain / level / "domain.pddl"
            shutil.copy(domain_file, dest)
            print(f"  Copied domain file to {our_domain}/{level}/")
    
    # Find problem files
    problem_files = sorted(source_dir.glob("p*.pddl"))
    if not problem_files:
        problem_files = sorted(source_dir.glob("**/p*.pddl"))
    
    if problem_files:
        # Split problems into easy, medium, hard based on size/number
        n = len(problem_files)
        easy_count = min(10, n // 3)
        medium_count = min(10, n // 3)
        
        easy_problems = problem_files[:easy_count]
        medium_problems = problem_files[easy_count:easy_count + medium_count]
        hard_problems = problem_files[easy_count + medium_count:]
        
        # Copy easy problems
        for i, prob in enumerate(easy_problems):
            dest = BENCHMARK_DIR / our_domain / "easy" / f"problem{i+1:02d}.pddl"
            shutil.copy(prob, dest)
        print(f"  Copied {len(easy_problems)} easy problems")
        
        # Copy medium problems
        for i, prob in enumerate(medium_problems):
            dest = BENCHMARK_DIR / our_domain / "medium" / f"problem{i+1:02d}.pddl"
            shutil.copy(prob, dest)
        print(f"  Copied {len(medium_problems)} medium problems")
        
        # Copy hard problems (limit to 20 for practical reasons)
        for i, prob in enumerate(hard_problems[:20]):
            dest = BENCHMARK_DIR / our_domain / "hard" / f"problem{i+1:02d}.pddl"
            shutil.copy(prob, dest)
        print(f"  Copied {len(hard_problems[:20])} hard problems")

def main():
    print("Fetching benchmark files from downward-benchmarks repository...\n")
    
    try:
        clone_benchmark_repo()
        
        for our_domain, repo_domain in DOMAIN_MAPPINGS.items():
            print(f"\nProcessing {our_domain} (from {repo_domain})...")
            copy_domain_files(repo_domain, our_domain)
        
        print("\n✓ Benchmark files downloaded successfully!")
        print(f"Benchmarks are located in: {BENCHMARK_DIR}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nIf automated download fails, manually download benchmarks from:")
        print("  https://github.com/aibasel/downward-benchmarks")
        return 1
    
    finally:
        # Cleanup temp directory
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
    
    return 0

if __name__ == "__main__":
    exit(main())


