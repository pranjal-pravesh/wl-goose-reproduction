"""
Evaluate WL-GOOSE Models on Test Set

Runs GBFS with learned heuristics on hard test instances and collects results.

Reference: "Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning" (AAAI 2024)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

from gbfs_search import GreedyBestFirstSearch, LearnedHeuristic, SearchResult


@dataclass
class ProblemResult:
    """Result for a single problem."""
    domain: str
    problem: str
    model_type: str
    seed: int
    solved: bool
    plan_cost: int
    nodes_expanded: int
    nodes_generated: int
    time_seconds: float
    memory_states: int


class WLGOOSEEvaluator:
    """Evaluate trained WL-GOOSE models on test problems."""
    
    def __init__(self,
                 benchmark_dir: str,
                 model_dir: str,
                 data_dir: str,
                 results_dir: str,
                 time_limit: float = 1800.0):
        """
        Args:
            benchmark_dir: Directory with benchmark problems
            model_dir: Directory with trained models
            data_dir: Directory with feature extractors
            results_dir: Directory to save results
            time_limit: Time limit per problem (seconds)
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.time_limit = time_limit
    
    def get_test_problems(self, domain: str) -> List[Tuple[str, str]]:
        """
        Get hard test problems for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of (domain_file, problem_file) tuples
        """
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
    
    def evaluate_problem(self,
                        domain_file: str,
                        problem_file: str,
                        model_path: str,
                        scaler_path: str,
                        feature_extractor_path: str,
                        domain_name: str,
                        model_type: str,
                        seed: int) -> ProblemResult:
        """
        Evaluate a single problem with a specific model.
        
        Args:
            domain_file: Path to domain PDDL
            problem_file: Path to problem PDDL
            model_path: Path to trained model
            scaler_path: Path to scaler
            feature_extractor_path: Path to feature extractor
            domain_name: Domain name
            model_type: Model type identifier
            seed: Random seed used for training
            
        Returns:
            ProblemResult
        """
        problem_name = Path(problem_file).stem
        
        try:
            # Load heuristic
            heuristic = LearnedHeuristic(
                model_path=model_path,
                scaler_path=scaler_path,
                feature_extractor_path=feature_extractor_path
            )
            
            # Run GBFS
            planner = GreedyBestFirstSearch(
                domain_file=domain_file,
                problem_file=problem_file,
                heuristic=heuristic,
                time_limit=self.time_limit
            )
            
            result = planner.search()
            
            return ProblemResult(
                domain=domain_name,
                problem=problem_name,
                model_type=model_type,
                seed=seed,
                solved=result.solved,
                plan_cost=result.plan_cost if result.solved else -1,
                nodes_expanded=result.nodes_expanded,
                nodes_generated=result.nodes_generated,
                time_seconds=result.time_seconds,
                memory_states=result.memory_states
            )
            
        except Exception as e:
            print(f"    Error: {e}")
            return ProblemResult(
                domain=domain_name,
                problem=problem_name,
                model_type=model_type,
                seed=seed,
                solved=False,
                plan_cost=-1,
                nodes_expanded=0,
                nodes_generated=0,
                time_seconds=self.time_limit,
                memory_states=0
            )
    
    def evaluate_domain(self, domain: str, model_types: List[str] = None) -> List[ProblemResult]:
        """
        Evaluate all models for a domain on test problems.
        
        Args:
            domain: Domain name
            model_types: List of model types to evaluate (default: all available)
            
        Returns:
            List of ProblemResults
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {domain}")
        print(f"{'='*60}")
        
        # Get test problems
        test_problems = self.get_test_problems(domain)
        
        if not test_problems:
            print(f"No test problems found for {domain}")
            return []
        
        print(f"Found {len(test_problems)} test problems")
        
        # Find available models
        domain_model_dir = self.model_dir / domain
        if not domain_model_dir.exists():
            print(f"No trained models found for {domain}")
            return []
        
        available_models = [d.name for d in domain_model_dir.iterdir() if d.is_dir()]
        
        if model_types:
            available_models = [m for m in available_models if m in model_types]
        
        if not available_models:
            print(f"No models to evaluate for {domain}")
            return []
        
        print(f"Evaluating model types: {available_models}")
        
        # Feature extractor (shared across all models)
        feature_extractor_path = str(self.data_dir / domain / 'feature_extractor.pkl')
        
        if not Path(feature_extractor_path).exists():
            print(f"Feature extractor not found: {feature_extractor_path}")
            return []
        
        # Evaluate each model type
        all_results = []
        
        for model_type in available_models:
            print(f"\n  Model: {model_type}")
            
            model_type_dir = domain_model_dir / model_type
            
            # Find all seeds
            model_files = sorted(model_type_dir.glob('model_seed*.pkl'))
            
            for model_file in model_files:
                seed = int(model_file.stem.split('seed')[1])
                scaler_file = model_type_dir / f'scaler_seed{seed}.pkl'
                
                # Use dummy scaler path if not exists
                if not scaler_file.exists():
                    scaler_file = model_file  # Will be handled in LearnedHeuristic
                
                print(f"    Seed {seed}: Testing {len(test_problems)} problems...")
                
                # Evaluate each problem
                for domain_file, problem_file in tqdm(test_problems, 
                                                     desc=f"    {model_type} seed{seed}",
                                                     leave=False):
                    result = self.evaluate_problem(
                        domain_file=domain_file,
                        problem_file=problem_file,
                        model_path=str(model_file),
                        scaler_path=str(scaler_file),
                        feature_extractor_path=feature_extractor_path,
                        domain_name=domain,
                        model_type=model_type,
                        seed=seed
                    )
                    
                    all_results.append(result)
                
                # Compute coverage for this seed
                solved = sum(1 for r in all_results if r.seed == seed and r.solved)
                print(f"      Coverage: {solved}/{len(test_problems)}")
        
        # Save results
        domain_results_file = self.results_dir / f'{domain}_results.json'
        with open(domain_results_file, 'w') as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        
        print(f"\n  Results saved to: {domain_results_file}")
        
        return all_results
    
    def evaluate_all_domains(self, domains: List[str] = None, model_types: List[str] = None):
        """
        Evaluate all domains.
        
        Args:
            domains: List of domains (default: all available)
            model_types: List of model types (default: all)
        """
        # Find available domains
        if domains is None:
            domains = [d.name for d in self.model_dir.iterdir() if d.is_dir()]
        
        print(f"Evaluating {len(domains)} domains")
        
        all_results = {}
        
        for domain in domains:
            try:
                results = self.evaluate_domain(domain, model_types)
                all_results[domain] = results
            except Exception as e:
                print(f"Error evaluating {domain}: {e}")
                continue
        
        # Save summary
        summary = self._compute_summary(all_results)
        summary_file = self.results_dir / 'evaluation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Evaluation complete!")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*60}")
    
    def _compute_summary(self, all_results: Dict[str, List[ProblemResult]]) -> Dict:
        """Compute summary statistics."""
        summary = {}
        
        for domain, results in all_results.items():
            if not results:
                continue
            
            # Group by model type and seed
            by_model_seed = {}
            for r in results:
                key = (r.model_type, r.seed)
                if key not in by_model_seed:
                    by_model_seed[key] = []
                by_model_seed[key].append(r)
            
            # Compute coverage and mean cost per model type
            domain_summary = {}
            for (model_type, seed), problems in by_model_seed.items():
                solved = sum(1 for p in problems if p.solved)
                total = len(problems)
                coverage = solved / total if total > 0 else 0.0
                
                solved_problems = [p for p in problems if p.solved]
                mean_cost = np.mean([p.plan_cost for p in solved_problems]) if solved_problems else 0.0
                
                if model_type not in domain_summary:
                    domain_summary[model_type] = []
                
                domain_summary[model_type].append({
                    'seed': seed,
                    'coverage': coverage,
                    'solved': solved,
                    'total': total,
                    'mean_cost': mean_cost
                })
            
            # Aggregate across seeds
            for model_type, seed_results in domain_summary.items():
                coverages = [s['coverage'] for s in seed_results]
                domain_summary[model_type] = {
                    'coverage_mean': np.mean(coverages),
                    'coverage_std': np.std(coverages),
                    'seeds': seed_results
                }
            
            summary[domain] = domain_summary
        
        return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate WL-GOOSE models')
    parser.add_argument('--benchmark-dir', required=True, help='Benchmarks directory')
    parser.add_argument('--model-dir', required=True, help='Models directory')
    parser.add_argument('--data-dir', required=True, help='Training data directory (for feature extractors)')
    parser.add_argument('--results-dir', required=True, help='Results output directory')
    parser.add_argument('--time-limit', type=float, default=1800.0, help='Time limit per problem (seconds)')
    parser.add_argument('--domains', nargs='+', help='Specific domains to evaluate')
    parser.add_argument('--model-types', nargs='+', help='Specific model types to evaluate')
    
    args = parser.parse_args()
    
    evaluator = WLGOOSEEvaluator(
        benchmark_dir=args.benchmark_dir,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        time_limit=args.time_limit
    )
    
    evaluator.evaluate_all_domains(
        domains=args.domains,
        model_types=args.model_types
    )


if __name__ == "__main__":
    main()


