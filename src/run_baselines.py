"""
Run Baseline Planners (hFF and LAMA)

Evaluates classical planning baselines on test problems for comparison with WL-GOOSE.

Reference: "Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning" (AAAI 2024)
"""

import os
import sys
import subprocess
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import re


@dataclass
class BaselineResult:
    """Result for a baseline planner on a problem."""
    domain: str
    problem: str
    planner: str
    solved: bool
    plan_cost: int
    time_seconds: float
    nodes_expanded: int = -1  # Not always available


class BaselineEvaluator:
    """Evaluate baseline planners."""
    
    def __init__(self,
                 benchmark_dir: str,
                 fast_downward_path: str,
                 results_dir: str,
                 time_limit: float = 1800.0,
                 memory_limit: str = "4G"):
        """
        Args:
            benchmark_dir: Directory with benchmark problems
            fast_downward_path: Path to fast-downward.py
            results_dir: Directory to save results
            time_limit: Time limit per problem (seconds)
            memory_limit: Memory limit (e.g., "4G")
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.fd_path = fast_downward_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.time_limit = time_limit
        self.memory_limit = memory_limit
    
    def get_test_problems(self, domain: str) -> List[Tuple[str, str]]:
        """Get hard test problems for a domain."""
        domain_dir = self.benchmark_dir / domain
        
        # Find domain file
        domain_file = None
        for level in ['hard', 'medium', 'easy']:
            candidate = domain_dir / level / 'domain.pddl'
            if candidate.exists():
                domain_file = str(candidate)
                break
        
        if not domain_file:
            return []
        
        # Get hard problems
        hard_dir = domain_dir / 'hard'
        if not hard_dir.exists():
            return []
        
        problems = sorted(hard_dir.glob('problem*.pddl'))
        
        return [(domain_file, str(p)) for p in problems]
    
    def run_hff(self, domain_file: str, problem_file: str) -> BaselineResult:
        """
        Run Fast Downward with FF heuristic (GBFS).
        
        Args:
            domain_file: Path to domain PDDL
            problem_file: Path to problem PDDL
            
        Returns:
            BaselineResult
        """
        problem_name = Path(problem_file).stem
        domain_name = Path(domain_file).parent.parent.name
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                start_time = time.time()
                
                # Run Fast Downward with FF heuristic and GBFS
                cmd = [
                    sys.executable,
                    self.fd_path,
                    domain_file,
                    problem_file,
                    "--search", "eager_greedy([ff()])",
                    "--plan-file", os.path.join(tmpdir, "sas_plan")
                ]
                
                result = subprocess.run(
                    cmd,
                    timeout=self.time_limit,
                    capture_output=True,
                    text=True,
                    cwd=tmpdir
                )
                
                elapsed = time.time() - start_time
                
                # Check if plan was found
                plan_file = os.path.join(tmpdir, "sas_plan")
                if os.path.exists(plan_file):
                    with open(plan_file, 'r') as f:
                        plan = [line.strip() for line in f if line.strip() and not line.startswith(';')]
                    
                    # Extract nodes expanded from output
                    nodes_expanded = self._extract_nodes_expanded(result.output)
                    
                    return BaselineResult(
                        domain=domain_name,
                        problem=problem_name,
                        planner='hFF',
                        solved=True,
                        plan_cost=len(plan),  # Unit costs
                        time_seconds=elapsed,
                        nodes_expanded=nodes_expanded
                    )
                else:
                    return BaselineResult(
                        domain=domain_name,
                        problem=problem_name,
                        planner='hFF',
                        solved=False,
                        plan_cost=-1,
                        time_seconds=elapsed
                    )
                    
            except subprocess.TimeoutExpired:
                return BaselineResult(
                    domain=domain_name,
                    problem=problem_name,
                    planner='hFF',
                    solved=False,
                    plan_cost=-1,
                    time_seconds=self.time_limit
                )
            except Exception as e:
                print(f"    Error running hFF on {problem_name}: {e}")
                return BaselineResult(
                    domain=domain_name,
                    problem=problem_name,
                    planner='hFF',
                    solved=False,
                    plan_cost=-1,
                    time_seconds=self.time_limit
                )
    
    def run_lama(self, domain_file: str, problem_file: str) -> BaselineResult:
        """
        Run Fast Downward LAMA planner.
        
        Args:
            domain_file: Path to domain PDDL
            problem_file: Path to problem PDDL
            
        Returns:
            BaselineResult
        """
        problem_name = Path(problem_file).stem
        domain_name = Path(domain_file).parent.parent.name
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                start_time = time.time()
                
                # Run Fast Downward with LAMA
                cmd = [
                    sys.executable,
                    self.fd_path,
                    "--alias", "lama-first",
                    domain_file,
                    problem_file,
                    "--plan-file", os.path.join(tmpdir, "sas_plan")
                ]
                
                result = subprocess.run(
                    cmd,
                    timeout=self.time_limit,
                    capture_output=True,
                    text=True,
                    cwd=tmpdir
                )
                
                elapsed = time.time() - start_time
                
                # Check if plan was found
                plan_files = list(Path(tmpdir).glob("sas_plan*"))
                if plan_files:
                    # Use first plan found
                    with open(plan_files[0], 'r') as f:
                        plan = [line.strip() for line in f if line.strip() and not line.startswith(';')]
                    
                    nodes_expanded = self._extract_nodes_expanded(result.output)
                    
                    return BaselineResult(
                        domain=domain_name,
                        problem=problem_name,
                        planner='LAMA',
                        solved=True,
                        plan_cost=len(plan),
                        time_seconds=elapsed,
                        nodes_expanded=nodes_expanded
                    )
                else:
                    return BaselineResult(
                        domain=domain_name,
                        problem=problem_name,
                        planner='LAMA',
                        solved=False,
                        plan_cost=-1,
                        time_seconds=elapsed
                    )
                    
            except subprocess.TimeoutExpired:
                return BaselineResult(
                    domain=domain_name,
                    problem=problem_name,
                    planner='LAMA',
                    solved=False,
                    plan_cost=-1,
                    time_seconds=self.time_limit
                )
            except Exception as e:
                print(f"    Error running LAMA on {problem_name}: {e}")
                return BaselineResult(
                    domain=domain_name,
                    problem=problem_name,
                    planner='LAMA',
                    solved=False,
                    plan_cost=-1,
                    time_seconds=self.time_limit
                )
    
    @staticmethod
    def _extract_nodes_expanded(output: str) -> int:
        """Extract number of expanded nodes from Fast Downward output."""
        # Look for pattern like "Expanded X state(s)."
        match = re.search(r'Expanded (\d+) state', output)
        if match:
            return int(match.group(1))
        return -1
    
    def evaluate_domain(self, domain: str, planners: List[str] = None) -> List[BaselineResult]:
        """
        Evaluate baselines on a domain.
        
        Args:
            domain: Domain name
            planners: List of planners to run (default: ['hFF', 'LAMA'])
            
        Returns:
            List of BaselineResults
        """
        if planners is None:
            planners = ['hFF', 'LAMA']
        
        print(f"\n{'='*60}")
        print(f"Evaluating baselines for: {domain}")
        print(f"{'='*60}")
        
        # Get test problems
        test_problems = self.get_test_problems(domain)
        
        if not test_problems:
            print(f"No test problems found for {domain}")
            return []
        
        print(f"Found {len(test_problems)} test problems")
        print(f"Planners: {planners}")
        
        all_results = []
        
        # Run each planner
        for planner in planners:
            print(f"\n  Running {planner}...")
            
            for domain_file, problem_file in tqdm(test_problems, desc=f"  {planner}"):
                if planner == 'hFF':
                    result = self.run_hff(domain_file, problem_file)
                elif planner == 'LAMA':
                    result = self.run_lama(domain_file, problem_file)
                else:
                    print(f"Unknown planner: {planner}")
                    continue
                
                all_results.append(result)
            
            # Compute coverage
            solved = sum(1 for r in all_results if r.planner == planner and r.solved)
            print(f"    Coverage: {solved}/{len(test_problems)}")
        
        # Save results
        results_file = self.results_dir / f'{domain}_baselines.json'
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        
        print(f"\n  Results saved to: {results_file}")
        
        return all_results
    
    def evaluate_all_domains(self, domains: List[str] = None, planners: List[str] = None):
        """
        Evaluate baselines on all domains.
        
        Args:
            domains: List of domains (default: all available)
            planners: List of planners (default: ['hFF', 'LAMA'])
        """
        # Find available domains
        if domains is None:
            domains = [d.name for d in self.benchmark_dir.iterdir() 
                      if d.is_dir() and (d / 'hard').exists()]
        
        print(f"Evaluating baselines on {len(domains)} domains")
        
        all_results = {}
        
        for domain in domains:
            try:
                results = self.evaluate_domain(domain, planners)
                all_results[domain] = results
            except Exception as e:
                print(f"Error evaluating {domain}: {e}")
                continue
        
        # Save summary
        summary = self._compute_summary(all_results)
        summary_file = self.results_dir / 'baseline_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Baseline evaluation complete!")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*60}")
    
    def _compute_summary(self, all_results: Dict[str, List[BaselineResult]]) -> Dict:
        """Compute summary statistics."""
        summary = {}
        
        for domain, results in all_results.items():
            if not results:
                continue
            
            domain_summary = {}
            
            # Group by planner
            by_planner = {}
            for r in results:
                if r.planner not in by_planner:
                    by_planner[r.planner] = []
                by_planner[r.planner].append(r)
            
            for planner, planner_results in by_planner.items():
                solved = sum(1 for r in planner_results if r.solved)
                total = len(planner_results)
                coverage = solved / total if total > 0 else 0.0
                
                solved_problems = [r for r in planner_results if r.solved]
                mean_cost = sum(r.plan_cost for r in solved_problems) / len(solved_problems) if solved_problems else 0.0
                mean_time = sum(r.time_seconds for r in solved_problems) / len(solved_problems) if solved_problems else 0.0
                
                domain_summary[planner] = {
                    'coverage': coverage,
                    'solved': solved,
                    'total': total,
                    'mean_cost': mean_cost,
                    'mean_time': mean_time
                }
            
            summary[domain] = domain_summary
        
        return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate baseline planners')
    parser.add_argument('--benchmark-dir', required=True, help='Benchmarks directory')
    parser.add_argument('--fast-downward', required=True, help='Path to fast-downward.py')
    parser.add_argument('--results-dir', required=True, help='Results output directory')
    parser.add_argument('--time-limit', type=float, default=1800.0, help='Time limit per problem (seconds)')
    parser.add_argument('--domains', nargs='+', help='Specific domains to evaluate')
    parser.add_argument('--planners', nargs='+', choices=['hFF', 'LAMA'],
                       help='Planners to run (default: both)')
    
    args = parser.parse_args()
    
    evaluator = BaselineEvaluator(
        benchmark_dir=args.benchmark_dir,
        fast_downward_path=args.fast_downward,
        results_dir=args.results_dir,
        time_limit=args.time_limit
    )
    
    evaluator.evaluate_all_domains(
        domains=args.domains,
        planners=args.planners
    )


if __name__ == "__main__":
    main()


