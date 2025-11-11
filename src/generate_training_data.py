"""
Generate Training Data for WL-GOOSE

This module:
1. Runs an optimal planner (Fast Downward with A*) on training instances
2. Extracts states along optimal plan trajectories
3. Computes h*(s) for each state (cost-to-go)
4. Builds ILG representations and extracts WL features
5. Saves training data as (features, h*) pairs

Reference: "Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning" (AAAI 2024)
"""

import os
import sys
import subprocess
import tempfile
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re
import pddlpy
import numpy as np
from tqdm import tqdm

from ilg_builder import ILGBuilder, ILG
from wl_features import WLFeatureExtractor


@dataclass
class TrainingExample:
    """A single training example: state features and h* value."""
    features: np.ndarray
    h_star: float
    domain: str
    problem: str


class OptimalPlannerRunner:
    """Run Fast Downward with A* to compute optimal plans."""
    
    def __init__(self, fast_downward_path: str, timeout: int = 1800):
        """
        Args:
            fast_downward_path: Path to fast-downward.py
            timeout: Timeout in seconds (default 1800 = 30 minutes)
        """
        self.fd_path = fast_downward_path
        self.timeout = timeout
    
    def solve_optimal(self, domain_file: str, problem_file: str) -> Optional[List[str]]:
        """
        Solve a problem optimally using A* with LM-Cut heuristic.
        
        Args:
            domain_file: Path to domain PDDL
            problem_file: Path to problem PDDL
            
        Returns:
            List of action names in the optimal plan, or None if timeout/unsolvable
        """
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Run Fast Downward with optimal configuration
                # Using A* with LM-Cut (admissible and informative)
                # Use absolute paths for input files so they work from any directory
                domain_file_abs = os.path.abspath(domain_file)
                problem_file_abs = os.path.abspath(problem_file)
                
                # Fast Downward writes plans to "sas_plan" in the working directory by default
                plan_file = os.path.join(tmpdir, "sas_plan")
                
                cmd = [
                    sys.executable,  # Use same Python interpreter
                    os.path.abspath(self.fd_path),  # Also use absolute path for FD
                    domain_file_abs,
                    problem_file_abs,
                    "--search", "astar(lmcut())"
                    # Don't use --plan-file, it causes issues with the search component
                ]
                
                result = subprocess.run(
                    cmd,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    cwd=tmpdir
                )
                
                # Check if solution was found
                # Fast Downward returns 0 and prints "Solution found" when successful
                # The message can be in either stdout or stderr
                combined_output = result.stdout + result.stderr
                solution_found = (
                    result.returncode == 0 and 
                    ("Solution found" in result.stdout or "Solution found" in result.stderr or "Solution found!" in combined_output)
                )
                
                if solution_found:
                    # Fast Downward writes to sas_plan in the working directory (tmpdir)
                    if os.path.exists(plan_file):
                        with open(plan_file, 'r') as f:
                            plan = [line.strip() for line in f if line.strip() and not line.startswith(';')]
                        if plan:  # Make sure plan is not empty
                            return plan
                
                # No solution found
                return None
                
            except subprocess.TimeoutExpired:
                return None
            except Exception as e:
                print(f"  Error solving {problem_file}: {e}")
                return None
    
    def extract_states_from_plan(self, domain_file: str, problem_file: str, 
                                  plan: List[str]) -> List[Tuple[set, int]]:
        """
        Execute a plan and extract states with their h* values.
        
        Uses a simpler approach: directly apply action effects based on Blocksworld semantics.
        
        Args:
            domain_file: Path to domain PDDL
            problem_file: Path to problem PDDL
            plan: List of action names
            
        Returns:
            List of (state, h_star) tuples where h_star is cost-to-go
        """
        try:
            # Parse domain and problem
            # Suppress pddlpy parsing warnings (it complains about uppercase :INIT)
            # These warnings are harmless - we parse initial state directly anyway
            import os
            import sys
            
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    domprob = pddlpy.DomainProblem(domain_file, problem_file)
                finally:
                    sys.stderr = old_stderr
            
            # Get initial state - handle empty state by parsing PDDL directly if needed
            initial_state_list = list(domprob.initialstate())
            if not initial_state_list:
                # pddlpy failed to parse (likely due to uppercase :INIT), parse PDDL directly
                initial_state_list = self._parse_initial_state_from_pddl(problem_file)
            
            # Convert to set of frozensets (pddlpy format)
            current_state = set(initial_state_list) if initial_state_list else set()
            
            goal = set(domprob.goals())
            
            # Collect states along trajectory
            states_with_cost = []
            
            # Initial state has h* = length of plan (unit costs)
            states_with_cost.append((current_state.copy(), len(plan)))
            
            # Execute each action using Blocksworld semantics
            for i, action_str in enumerate(plan):
                # Parse action string to get name and parameters
                action_name, params = self._parse_action(action_str)
                
                if not action_name or not params:
                    continue
                
                # Apply action directly using Blocksworld semantics
                new_state = self._apply_blocksworld_action(current_state, action_name, params)
                
                if new_state is not None:
                    current_state = new_state
                    # Record state with remaining cost
                    h_star = len(plan) - (i + 1)
                    states_with_cost.append((current_state.copy(), h_star))
            
            return states_with_cost
            
        except Exception as e:
            print(f"    Error extracting states: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @staticmethod
    def _parse_action(action_str: str) -> Tuple[str, List[str]]:
        """Parse action string like 'stack g i (1)' or '(stack g i)' to ('stack', ['g', 'i'])."""
        action_str = action_str.strip()
        
        # Remove cost suffix like "(1)" if present
        if action_str.endswith(')'):
            # Check if it's a cost suffix
            last_paren = action_str.rfind('(')
            if last_paren > 0:
                # Check if what's after the last ( looks like a number
                cost_part = action_str[last_paren+1:-1].strip()
                if cost_part.isdigit():
                    action_str = action_str[:last_paren].strip()
        
        # Remove outer parentheses if present
        if action_str.startswith('(') and action_str.endswith(')'):
            action_str = action_str[1:-1]
        
        parts = action_str.split()
        action_name = parts[0] if parts else ""
        params = parts[1:] if len(parts) > 1 else []
        
        return action_name, params
    
    @staticmethod
    def _parse_initial_state_from_pddl(problem_file: str) -> List[frozenset]:
        """
        Parse initial state directly from PDDL file.
        Handles both :init and :INIT (uppercase) formats.
        """
        import re
        
        try:
            with open(problem_file, 'r') as f:
                content = f.read()
            
            # Find :init or :INIT section - match until :goal
            init_match = re.search(r':init\s+(.*?)(?=:goal)', content, re.IGNORECASE | re.DOTALL)
            if not init_match:
                return []
            
            init_content = init_match.group(1).strip()
            # Remove trailing closing paren if present
            if init_content.endswith(')'):
                init_content = init_content[:-1].strip()
            
            # Parse facts - they're in format like (CLEAR C) or (ON C E)
            # Use regex to find all parenthesized groups
            facts = []
            # Pattern to match (PREDICATE arg1 arg2 ...)
            fact_pattern = r'\(([^()]+)\)'
            for match in re.finditer(fact_pattern, init_content):
                fact_content = match.group(1).strip()
                parts = fact_content.split()
                if parts:
                    # Convert to frozenset format (lowercase predicate, then args)
                    predicate = parts[0].lower()
                    args = [p.lower() for p in parts[1:]]
                    fact = frozenset([predicate] + args)
                    facts.append(fact)
            
            return facts
            
        except Exception as e:
            return []
    
    @staticmethod
    def _apply_blocksworld_action(state: set, action_name: str, params: List[str]) -> set:
        """
        Apply a Blocksworld action directly to a state.
        
        This is a domain-specific implementation that doesn't rely on pddlpy's
        problematic operator matching.
        """
        action_name = action_name.lower()
        params = [p.lower() for p in params]
        
        # Convert state to a more workable format
        state_facts = {frozenset(f) if isinstance(f, (list, tuple, frozenset)) else frozenset([f]) 
                      for f in state}
        
        new_state_facts = set(state_facts)
        
        try:
            if action_name == 'unstack' and len(params) == 2:
                x, y = params
                # Preconditions: (on x y), (clear x), (handempty)
                # Delete: (on x y), (clear x), (handempty)
                # Add: (holding x), (clear y)
                
                # Remove facts
                to_remove = [
                    frozenset(['on', x, y]),
                    frozenset(['clear', x]),
                    frozenset(['handempty'])
                ]
                for fact in to_remove:
                    new_state_facts.discard(fact)
                
                # Add facts
                new_state_facts.add(frozenset(['holding', x]))
                new_state_facts.add(frozenset(['clear', y]))
                
            elif action_name == 'stack' and len(params) == 2:
                x, y = params
                # Preconditions: (holding x), (clear y)
                # Delete: (holding x), (clear y)
                # Add: (on x y), (clear x), (handempty)
                
                # Remove facts
                to_remove = [
                    frozenset(['holding', x]),
                    frozenset(['clear', y])
                ]
                for fact in to_remove:
                    new_state_facts.discard(fact)
                
                # Add facts
                new_state_facts.add(frozenset(['on', x, y]))
                new_state_facts.add(frozenset(['clear', x]))
                new_state_facts.add(frozenset(['handempty']))
                
            elif action_name == 'pick-up' and len(params) == 1:
                x = params[0]
                # Preconditions: (ontable x) or (on x y), (clear x), (handempty)
                # Delete: (ontable x) or (on x y), (clear x), (handempty)
                # Add: (holding x)
                
                # Remove ontable or on fact
                for fact in list(new_state_facts):
                    fact_list = list(fact)
                    if len(fact_list) >= 2:
                        if fact_list[0] == 'ontable' and fact_list[1] == x:
                            new_state_facts.discard(fact)
                        elif fact_list[0] == 'on' and fact_list[1] == x and len(fact_list) == 3:
                            # Also need to add clear for the block that was under x
                            y = fact_list[2]
                            new_state_facts.discard(fact)
                            new_state_facts.add(frozenset(['clear', y]))
                
                # Remove clear and handempty
                new_state_facts.discard(frozenset(['clear', x]))
                new_state_facts.discard(frozenset(['handempty']))
                
                # Add holding
                new_state_facts.add(frozenset(['holding', x]))
                
            elif action_name == 'put-down' and len(params) == 1:
                x = params[0]
                # Preconditions: (holding x)
                # Delete: (holding x)
                # Add: (ontable x), (clear x), (handempty)
                
                # Remove holding
                new_state_facts.discard(frozenset(['holding', x]))
                
                # Add facts
                new_state_facts.add(frozenset(['ontable', x]))
                new_state_facts.add(frozenset(['clear', x]))
                new_state_facts.add(frozenset(['handempty']))
            
            else:
                # Unknown action
                return None
            
            return new_state_facts
            
        except Exception as e:
            # If application fails, return None
            return None


class TrainingDataGenerator:
    """Generate training data for WL-GOOSE."""
    
    def __init__(self, 
                 benchmark_dir: str,
                 fast_downward_path: str,
                 output_dir: str,
                 timeout: int = 1800):
        """
        Args:
            benchmark_dir: Root directory containing benchmark domains
            fast_downward_path: Path to fast-downward.py
            output_dir: Directory to save training data
            timeout: Timeout per problem in seconds
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.planner = OptimalPlannerRunner(fast_downward_path, timeout)
        self.ilg_builder = ILGBuilder()
    
    def generate_for_domain(self, domain_name: str) -> Dict:
        """
        Generate training data for a single domain.
        
        Args:
            domain_name: Name of the domain (e.g., 'blocksworld')
            
        Returns:
            Dictionary with training statistics
        """
        print(f"\n{'='*60}")
        print(f"Generating training data for: {domain_name}")
        print(f"{'='*60}")
        
        domain_dir = self.benchmark_dir / domain_name
        
        # Find domain file
        domain_file = None
        for level in ['easy', 'medium', 'hard']:
            candidate = domain_dir / level / 'domain.pddl'
            if candidate.exists():
                domain_file = str(candidate)
                break
        
        if not domain_file:
            print(f"Error: No domain file found for {domain_name}")
            return {'domain': domain_name, 'success': False}
        
        print(f"Domain file: {domain_file}")
        
        # Collect training problems (easy + medium)
        training_problems = []
        for level in ['easy', 'medium']:
            level_dir = domain_dir / level
            if level_dir.exists():
                problems = sorted(level_dir.glob('problem*.pddl'))
                training_problems.extend([(level, str(p)) for p in problems])
        
        print(f"Found {len(training_problems)} training problems")
        
        # Generate training examples
        all_ilgs = []
        all_h_stars = []
        solved_count = 0
        total_states = 0
        
        for level, problem_file in tqdm(training_problems, desc="Processing problems"):
            print(f"\n  Solving: {Path(problem_file).name} ({level})")
            
            # Solve optimally
            plan = self.planner.solve_optimal(domain_file, problem_file)
            
            if plan is None:
                print(f"    Failed to solve optimally (timeout or unsolvable)")
                continue
            
            print(f"    Found optimal plan with {len(plan)} actions")
            solved_count += 1
            
            # Extract states and h* values
            states_with_cost = self.planner.extract_states_from_plan(
                domain_file, problem_file, plan
            )
            
            if not states_with_cost:
                print(f"    Failed to extract states")
                continue
            
            print(f"    Extracted {len(states_with_cost)} states")
            total_states += len(states_with_cost)
            
            # Build ILGs for each state
            try:
                # Suppress pddlpy warnings
                import os
                import sys
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    try:
                        domprob = pddlpy.DomainProblem(domain_file, problem_file)
                    finally:
                        sys.stderr = old_stderr
                goal = set(domprob.goals())
                
                for state, h_star in states_with_cost:
                    ilg = self.ilg_builder.build_ilg_from_state(domprob, state, goal)
                    all_ilgs.append(ilg)
                    all_h_stars.append(h_star)
                    
            except Exception as e:
                print(f"    Error building ILGs: {e}")
                continue
        
        print(f"\n  Summary:")
        print(f"    Solved: {solved_count}/{len(training_problems)}")
        print(f"    Total training states: {total_states}")
        
        if len(all_ilgs) == 0:
            print(f"  No training data generated for {domain_name}")
            return {'domain': domain_name, 'success': False, 'states': 0}
        
        # Extract WL features
        print(f"\n  Extracting WL features...")
        feature_extractor = WLFeatureExtractor(num_iterations=4)
        features = feature_extractor.fit_transform(all_ilgs)
        h_stars = np.array(all_h_stars)
        
        print(f"    Feature matrix shape: {features.shape}")
        print(f"    h* range: [{h_stars.min()}, {h_stars.max()}]")
        
        # Save training data
        domain_output_dir = self.output_dir / domain_name
        domain_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features and labels
        np.save(domain_output_dir / 'features.npy', features)
        np.save(domain_output_dir / 'h_star.npy', h_stars)
        
        # Save feature extractor
        feature_extractor.save(str(domain_output_dir / 'feature_extractor.pkl'))
        
        # Save metadata
        metadata = {
            'domain': domain_name,
            'num_training_problems': len(training_problems),
            'num_solved': solved_count,
            'num_states': total_states,
            'feature_dim': features.shape[1],
            'h_star_min': float(h_stars.min()),
            'h_star_max': float(h_stars.max()),
            'h_star_mean': float(h_stars.mean()),
        }
        
        with open(domain_output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  ✓ Training data saved to: {domain_output_dir}")
        
        return metadata
    
    def generate_all_domains(self, domains: List[str]) -> Dict[str, Dict]:
        """
        Generate training data for all specified domains.
        
        Args:
            domains: List of domain names
            
        Returns:
            Dictionary mapping domain names to their metadata
        """
        results = {}
        
        for domain in domains:
            try:
                metadata = self.generate_for_domain(domain)
                results[domain] = metadata
            except Exception as e:
                print(f"\nError processing {domain}: {e}")
                results[domain] = {'domain': domain, 'success': False, 'error': str(e)}
        
        # Save overall summary
        summary_file = self.output_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training data generation complete!")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*60}")
        
        return results


def main():
    """Main entry point for training data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training data for WL-GOOSE')
    parser.add_argument('--benchmark-dir', required=True, help='Path to benchmarks directory')
    parser.add_argument('--fast-downward', required=True, help='Path to fast-downward.py')
    parser.add_argument('--output-dir', required=True, help='Output directory for training data')
    parser.add_argument('--timeout', type=int, default=1800, help='Timeout per problem (seconds)')
    parser.add_argument('--domains', nargs='+', help='Specific domains to process (default: all)')
    
    args = parser.parse_args()
    
    # Default domains from paper
    all_domains = [
        'blocksworld', 'childsnack', 'ferry', 'floortile', 'miconic',
        'rovers', 'satellite', 'sokoban', 'spanner', 'transport'
    ]
    
    domains = args.domains if args.domains else all_domains
    
    generator = TrainingDataGenerator(
        benchmark_dir=args.benchmark_dir,
        fast_downward_path=args.fast_downward,
        output_dir=args.output_dir,
        timeout=args.timeout
    )
    
    generator.generate_all_domains(domains)


if __name__ == "__main__":
    main()


