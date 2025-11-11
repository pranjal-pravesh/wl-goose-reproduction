"""
Greedy Best-First Search with Learned Heuristics

Implements GBFS using WL-GOOSE learned heuristics for automated planning.

Reference: "Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning" (AAAI 2024)
"""

import heapq
import time
import joblib
import pddlpy
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Set, FrozenSet, Dict
from dataclasses import dataclass, field
from collections import deque

from ilg_builder import ILGBuilder
from wl_features import WLFeatureExtractor


@dataclass
class SearchNode:
    """Node in the search tree."""
    state: Set[FrozenSet]
    g_cost: int
    h_value: float
    parent: Optional['SearchNode'] = None
    action: Optional[FrozenSet] = None
    
    def __lt__(self, other):
        """Compare by h-value only (GBFS)."""
        return self.h_value < other.h_value


@dataclass
class SearchResult:
    """Result of a search."""
    solved: bool
    plan: List[FrozenSet] = field(default_factory=list)
    plan_cost: int = 0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    time_seconds: float = 0.0
    memory_states: int = 0


class LearnedHeuristic:
    """Wrapper for learned heuristic model."""
    
    def __init__(self, model_path: str, scaler_path: str, feature_extractor_path: str):
        """
        Load a trained model and its components.
        
        Args:
            model_path: Path to trained model (.pkl)
            scaler_path: Path to scaler (.pkl)
            feature_extractor_path: Path to feature extractor (.pkl)
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if Path(scaler_path).exists() else None
        self.feature_extractor = WLFeatureExtractor.load(feature_extractor_path)
        self.ilg_builder = ILGBuilder()
    
    def compute_heuristic(self, domprob, state: Set[FrozenSet], goal: Set[FrozenSet]) -> float:
        """
        Compute heuristic value for a state.
        
        Args:
            domprob: Parsed domain/problem
            state: Current state
            goal: Goal condition
            
        Returns:
            Heuristic value (estimated cost-to-go)
        """
        try:
            # Build ILG for this state
            ilg = self.ilg_builder.build_ilg_from_state(domprob, state, goal)
            
            # Extract features
            features = self.feature_extractor.transform(ilg)
            
            # Scale if needed
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            # Predict h-value
            h_value = self.model.predict(features)[0]
            
            # Ensure non-negative
            return max(0.0, h_value)
            
        except Exception as e:
            # If feature extraction fails, return high value
            print(f"Warning: Heuristic computation failed: {e}")
            return float('inf')


class GreedyBestFirstSearch:
    """Greedy Best-First Search planner."""
    
    def __init__(self, 
                 domain_file: str, 
                 problem_file: str,
                 heuristic: LearnedHeuristic,
                 time_limit: float = 1800.0,
                 memory_limit: int = 4000000):
        """
        Args:
            domain_file: Path to PDDL domain
            problem_file: Path to PDDL problem
            heuristic: Learned heuristic function
            time_limit: Time limit in seconds
            memory_limit: Maximum number of states to store
        """
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.heuristic = heuristic
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        
        # Parse problem
        # Suppress pddlpy parsing warnings (harmless - just case sensitivity issues)
        import os
        import sys
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                self.domprob = pddlpy.DomainProblem(domain_file, problem_file)
            finally:
                sys.stderr = old_stderr
        self.initial_state = set(self.domprob.initialstate())
        self.goal = set(self.domprob.goals())
    
    def is_goal(self, state: Set[FrozenSet]) -> bool:
        """Check if state satisfies goal."""
        return self.goal.issubset(state)
    
    def get_applicable_actions(self, state: Set[FrozenSet]) -> List[FrozenSet]:
        """Get all applicable actions in a state."""
        applicable = []
        
        for action in self.domprob.ground_actions():
            preconditions = self.domprob.preconditions(action)
            if preconditions.issubset(state):
                applicable.append(action)
        
        return applicable
    
    def apply_action(self, state: Set[FrozenSet], action: FrozenSet) -> Set[FrozenSet]:
        """Apply an action to a state."""
        # Remove delete effects
        new_state = state - self.domprob.delete_effects(action)
        # Add add effects
        new_state = new_state | self.domprob.add_effects(action)
        return new_state
    
    def extract_plan(self, goal_node: SearchNode) -> List[FrozenSet]:
        """Extract plan from goal node."""
        plan = []
        node = goal_node
        
        while node.parent is not None:
            plan.append(node.action)
            node = node.parent
        
        plan.reverse()
        return plan
    
    def search(self) -> SearchResult:
        """
        Run GBFS to find a plan.
        
        Returns:
            SearchResult with plan and statistics
        """
        start_time = time.time()
        
        # Initialize search
        initial_h = self.heuristic.compute_heuristic(self.domprob, self.initial_state, self.goal)
        initial_node = SearchNode(
            state=self.initial_state,
            g_cost=0,
            h_value=initial_h
        )
        
        # Priority queue (min-heap on h-value)
        open_list = [initial_node]
        heapq.heapify(open_list)
        
        # Closed set (visited states)
        closed = set()
        
        # State to node mapping (for duplicate detection)
        state_to_node = {self._state_hash(self.initial_state): initial_node}
        
        nodes_expanded = 0
        nodes_generated = 1
        
        # Search loop
        while open_list:
            # Check time limit
            if time.time() - start_time > self.time_limit:
                return SearchResult(
                    solved=False,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_seconds=time.time() - start_time,
                    memory_states=len(closed)
                )
            
            # Check memory limit
            if len(closed) > self.memory_limit:
                return SearchResult(
                    solved=False,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_seconds=time.time() - start_time,
                    memory_states=len(closed)
                )
            
            # Get best node
            current = heapq.heappop(open_list)
            current_hash = self._state_hash(current.state)
            
            # Skip if already expanded
            if current_hash in closed:
                continue
            
            # Mark as expanded
            closed.add(current_hash)
            nodes_expanded += 1
            
            # Goal test
            if self.is_goal(current.state):
                plan = self.extract_plan(current)
                return SearchResult(
                    solved=True,
                    plan=plan,
                    plan_cost=current.g_cost,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_seconds=time.time() - start_time,
                    memory_states=len(closed)
                )
            
            # Expand node
            applicable_actions = self.get_applicable_actions(current.state)
            
            for action in applicable_actions:
                # Apply action
                successor_state = self.apply_action(current.state, action)
                successor_hash = self._state_hash(successor_state)
                
                # Skip if already expanded
                if successor_hash in closed:
                    continue
                
                # Compute heuristic for successor
                h_value = self.heuristic.compute_heuristic(
                    self.domprob, successor_state, self.goal
                )
                
                # Get action cost (assumed unit cost)
                action_cost = 1
                
                # Create successor node
                successor = SearchNode(
                    state=successor_state,
                    g_cost=current.g_cost + action_cost,
                    h_value=h_value,
                    parent=current,
                    action=action
                )
                
                # Add to open list
                heapq.heappush(open_list, successor)
                nodes_generated += 1
                state_to_node[successor_hash] = successor
        
        # No solution found
        return SearchResult(
            solved=False,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            time_seconds=time.time() - start_time,
            memory_states=len(closed)
        )
    
    @staticmethod
    def _state_hash(state: Set[FrozenSet]) -> int:
        """Create a hash for a state."""
        return hash(frozenset(state))


def format_action(action: FrozenSet) -> str:
    """Format action for output."""
    action_list = sorted(list(action))
    if len(action_list) > 0:
        return f"({' '.join(action_list)})"
    return str(action)


def main():
    """Test GBFS with learned heuristic."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GBFS with learned heuristic')
    parser.add_argument('--domain', required=True, help='Domain PDDL file')
    parser.add_argument('--problem', required=True, help='Problem PDDL file')
    parser.add_argument('--model', required=True, help='Trained model file')
    parser.add_argument('--scaler', required=True, help='Scaler file')
    parser.add_argument('--feature-extractor', required=True, help='Feature extractor file')
    parser.add_argument('--time-limit', type=float, default=1800.0, help='Time limit (seconds)')
    parser.add_argument('--output', help='Output plan file')
    
    args = parser.parse_args()
    
    # Load heuristic
    print("Loading learned heuristic...")
    heuristic = LearnedHeuristic(
        model_path=args.model,
        scaler_path=args.scaler,
        feature_extractor_path=args.feature_extractor
    )
    
    # Create planner
    print(f"Planning for: {args.problem}")
    planner = GreedyBestFirstSearch(
        domain_file=args.domain,
        problem_file=args.problem,
        heuristic=heuristic,
        time_limit=args.time_limit
    )
    
    # Search
    print("Searching...")
    result = planner.search()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Search completed!")
    print(f"{'='*60}")
    print(f"Solved: {result.solved}")
    if result.solved:
        print(f"Plan length: {len(result.plan)}")
        print(f"Plan cost: {result.plan_cost}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Nodes generated: {result.nodes_generated}")
    print(f"Time: {result.time_seconds:.2f} seconds")
    print(f"States in memory: {result.memory_states}")
    
    # Save plan
    if result.solved and args.output:
        with open(args.output, 'w') as f:
            for action in result.plan:
                f.write(format_action(action) + '\n')
        print(f"\nPlan saved to: {args.output}")


if __name__ == "__main__":
    main()


