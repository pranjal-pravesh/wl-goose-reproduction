"""
Instance Learning Graph (ILG) Construction

Implements the ILG representation from the WL-GOOSE paper:
- Builds a graph representation of a lifted planning problem
- Vertices: objects, initial state propositions, goal propositions
- Edges: connections between propositions and their argument objects
- Node colors: distinguish object nodes, achieved propositions, unachieved goals, achieved goals
- Edge labels: argument positions

Reference: "Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning" (AAAI 2024)
"""

import pddlpy
from typing import Dict, List, Tuple, Set, FrozenSet
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class ILG:
    """Instance Learning Graph representation."""
    graph: nx.Graph = field(default_factory=nx.Graph)
    node_colors: Dict[str, Tuple] = field(default_factory=dict)
    edge_labels: Dict[Tuple[str, str], int] = field(default_factory=dict)
    objects: Set[str] = field(default_factory=set)
    propositions: Set[str] = field(default_factory=set)


class ILGBuilder:
    """Build Instance Learning Graphs from PDDL problems."""
    
    def __init__(self):
        self.domain_parser = None
        self.problem_parser = None
    
    def parse_pddl(self, domain_file: str, problem_file: str):
        """Parse PDDL domain and problem files."""
        # Suppress pddlpy parsing warnings (harmless - just case sensitivity issues)
        import os
        import sys
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                domprob = pddlpy.DomainProblem(domain_file, problem_file)
            finally:
                sys.stderr = old_stderr
        return domprob
    
    def build_ilg(self, domain_file: str, problem_file: str) -> ILG:
        """
        Build an Instance Learning Graph from a PDDL problem.
        
        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            
        Returns:
            ILG object containing the graph representation
        """
        # Parse the PDDL files
        domprob = self.parse_pddl(domain_file, problem_file)
        
        ilg = ILG()
        
        # Extract objects, initial state, and goal
        objects = set(domprob.worldobjects())
        initial_state = set(self._frozenset_to_string(fact) for fact in domprob.initialstate())
        goal = set(self._frozenset_to_string(fact) for fact in domprob.goals())
        
        ilg.objects = objects
        
        # Build vertices: V = O ∪ s_0 ∪ G
        # Add object nodes
        for obj in objects:
            ilg.graph.add_node(f"obj:{obj}")
            ilg.node_colors[f"obj:{obj}"] = ("ob",)  # object node
        
        # Add proposition nodes from initial state and goal
        all_propositions = initial_state.union(goal)
        
        for prop in all_propositions:
            prop_node = f"prop:{prop}"
            ilg.graph.add_node(prop_node)
            ilg.propositions.add(prop)
            
            # Determine node color based on whether prop is in s_0, G, or both
            in_initial = prop in initial_state
            in_goal = prop in goal
            
            if in_initial and in_goal:
                # Achieved goal
                ilg.node_colors[prop_node] = ("ag", self._get_predicate_name(prop))
            elif in_initial and not in_goal:
                # Achieved proposition (not a goal)
                ilg.node_colors[prop_node] = ("ap", self._get_predicate_name(prop))
            elif not in_initial and in_goal:
                # Unachieved goal
                ilg.node_colors[prop_node] = ("ug", self._get_predicate_name(prop))
        
        # Build edges: for each proposition p = P(o_1, ..., o_n), add edges (p, o_i)
        for prop in all_propositions:
            prop_node = f"prop:{prop}"
            args = self._get_arguments(prop)
            
            for i, arg in enumerate(args, start=1):
                obj_node = f"obj:{arg}"
                
                if obj_node in ilg.graph:
                    # Add edge with position label
                    ilg.graph.add_edge(prop_node, obj_node)
                    ilg.edge_labels[(prop_node, obj_node)] = i
                    ilg.edge_labels[(obj_node, prop_node)] = i  # undirected
        
        return ilg
    
    def build_ilg_from_state(self, domprob, state: Set[FrozenSet], goal: Set[FrozenSet]) -> ILG:
        """
        Build an ILG from a specific state (used during planning).
        
        Args:
            domprob: Parsed domain/problem object
            state: Current state as set of ground atoms
            goal: Goal condition as set of ground atoms
            
        Returns:
            ILG object
        """
        ilg = ILG()
        
        # Extract objects
        objects = set(domprob.worldobjects())
        ilg.objects = objects
        
        # Convert state and goal to string representations
        state_props = set(self._frozenset_to_string(fact) for fact in state)
        goal_props = set(self._frozenset_to_string(fact) for fact in goal)
        
        # Add object nodes
        for obj in objects:
            ilg.graph.add_node(f"obj:{obj}")
            ilg.node_colors[f"obj:{obj}"] = ("ob",)
        
        # Add proposition nodes
        all_propositions = state_props.union(goal_props)
        
        for prop in all_propositions:
            prop_node = f"prop:{prop}"
            ilg.graph.add_node(prop_node)
            ilg.propositions.add(prop)
            
            in_state = prop in state_props
            in_goal = prop in goal_props
            
            if in_state and in_goal:
                ilg.node_colors[prop_node] = ("ag", self._get_predicate_name(prop))
            elif in_state and not in_goal:
                ilg.node_colors[prop_node] = ("ap", self._get_predicate_name(prop))
            elif not in_state and in_goal:
                ilg.node_colors[prop_node] = ("ug", self._get_predicate_name(prop))
        
        # Build edges
        for prop in all_propositions:
            prop_node = f"prop:{prop}"
            args = self._get_arguments(prop)
            
            for i, arg in enumerate(args, start=1):
                obj_node = f"obj:{arg}"
                
                if obj_node in ilg.graph:
                    ilg.graph.add_edge(prop_node, obj_node)
                    ilg.edge_labels[(prop_node, obj_node)] = i
                    ilg.edge_labels[(obj_node, prop_node)] = i
        
        return ilg
    
    @staticmethod
    def _frozenset_to_string(fact: FrozenSet) -> str:
        """Convert a frozenset fact to a string representation."""
        if isinstance(fact, frozenset):
            fact_list = sorted(list(fact))
            if len(fact_list) > 0:
                # First element is predicate, rest are arguments
                return f"{fact_list[0]}({','.join(fact_list[1:])})"
            return str(fact)
        return str(fact)
    
    @staticmethod
    def _get_predicate_name(prop: str) -> str:
        """Extract predicate name from a proposition string."""
        if "(" in prop:
            return prop.split("(")[0]
        return prop
    
    @staticmethod
    def _get_arguments(prop: str) -> List[str]:
        """Extract arguments from a proposition string."""
        if "(" in prop and ")" in prop:
            args_str = prop.split("(")[1].split(")")[0]
            if args_str:
                return [arg.strip() for arg in args_str.split(",")]
        return []
    
    def visualize_ilg(self, ilg: ILG, output_file: str = None):
        """
        Visualize the ILG using matplotlib (for debugging).
        
        Args:
            ilg: ILG object to visualize
            output_file: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        pos = nx.spring_layout(ilg.graph)
        
        # Separate nodes by type
        object_nodes = [n for n in ilg.graph.nodes() if n.startswith("obj:")]
        prop_nodes = [n for n in ilg.graph.nodes() if n.startswith("prop:")]
        
        # Draw nodes
        nx.draw_networkx_nodes(ilg.graph, pos, nodelist=object_nodes, 
                              node_color='lightblue', label='Objects', node_size=500)
        nx.draw_networkx_nodes(ilg.graph, pos, nodelist=prop_nodes,
                              node_color='lightcoral', label='Propositions', node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(ilg.graph, pos, alpha=0.5)
        
        # Draw labels
        labels = {n: n.split(":")[1][:20] for n in ilg.graph.nodes()}
        nx.draw_networkx_labels(ilg.graph, pos, labels, font_size=8)
        
        plt.legend()
        plt.title("Instance Learning Graph (ILG)")
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def main():
    """Test ILG construction."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python ilg_builder.py <domain.pddl> <problem.pddl>")
        sys.exit(1)
    
    domain_file = sys.argv[1]
    problem_file = sys.argv[2]
    
    builder = ILGBuilder()
    ilg = builder.build_ilg(domain_file, problem_file)
    
    print(f"ILG Statistics:")
    print(f"  Nodes: {ilg.graph.number_of_nodes()}")
    print(f"  Edges: {ilg.graph.number_of_edges()}")
    print(f"  Objects: {len(ilg.objects)}")
    print(f"  Propositions: {len(ilg.propositions)}")
    print(f"\nNode colors:")
    for node, color in list(ilg.node_colors.items())[:10]:
        print(f"  {node}: {color}")
    
    # Optionally visualize
    # builder.visualize_ilg(ilg, "ilg_visualization.png")


if __name__ == "__main__":
    main()


