"""
Weisfeiler-Lehman (WL) Feature Extraction for ILGs

Implements WL color refinement algorithm with edge labels to generate
structural features from Instance Learning Graphs.

The features are histogram-based: count of each WL color across iterations.

Reference: "Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning" (AAAI 2024)
"""

import hashlib
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from dataclasses import dataclass, field

from ilg_builder import ILG


@dataclass
class WLFeatureExtractor:
    """
    Extract WL-based features from Instance Learning Graphs.
    
    Attributes:
        num_iterations: Number of WL refinement iterations (L in paper, default 4)
        color_vocabulary: Set of all colors observed during training
        color_to_index: Mapping from colors to feature indices
    """
    num_iterations: int = 4
    color_vocabulary: Set[str] = field(default_factory=set)
    color_to_index: Dict[str, int] = field(default_factory=dict)
    fitted: bool = False
    
    def fit(self, ilgs: List[ILG]):
        """
        Fit the feature extractor on a collection of ILGs.
        This builds the color vocabulary from the training set.
        
        Args:
            ilgs: List of ILG objects from training data
        """
        all_colors = set()
        
        for ilg in ilgs:
            colors = self._run_wl(ilg, collect_colors=True)
            all_colors.update(colors)
        
        self.color_vocabulary = all_colors
        self.color_to_index = {color: idx for idx, color in enumerate(sorted(all_colors))}
        self.fitted = True
        
        print(f"WL Feature Extractor fitted with {len(self.color_vocabulary)} colors")
    
    def transform(self, ilg: ILG) -> np.ndarray:
        """
        Transform an ILG into a feature vector.
        
        Args:
            ilg: Instance Learning Graph
            
        Returns:
            Feature vector as numpy array of shape (|color_vocabulary|,)
        """
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Run WL and collect color counts
        color_counts = self._run_wl(ilg, collect_colors=False)
        
        # Build feature vector
        feature_vector = np.zeros(len(self.color_vocabulary))
        
        for color, count in color_counts.items():
            if color in self.color_to_index:
                feature_vector[self.color_to_index[color]] = count
        
        return feature_vector
    
    def fit_transform(self, ilgs: List[ILG]) -> np.ndarray:
        """
        Fit the extractor and transform ILGs in one step.
        
        Args:
            ilgs: List of ILG objects
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        self.fit(ilgs)
        return np.array([self.transform(ilg) for ilg in ilgs])
    
    def _run_wl(self, ilg: ILG, collect_colors: bool = False):
        """
        Run Weisfeiler-Lehman color refinement on an ILG.
        
        Args:
            ilg: Instance Learning Graph
            collect_colors: If True, return set of all colors; if False, return color counts
            
        Returns:
            Set of colors (if collect_colors=True) or Counter of colors (if False)
        """
        # Initialize colors from node_colors
        current_colors = {node: self._hash_color(ilg.node_colors[node]) 
                         for node in ilg.graph.nodes()}
        
        all_colors = set(current_colors.values()) if collect_colors else Counter(current_colors.values())
        
        # Run L iterations of WL refinement
        for iteration in range(self.num_iterations):
            next_colors = {}
            
            for node in ilg.graph.nodes():
                # Get current color
                node_color = current_colors[node]
                
                # Collect neighbor colors with edge labels
                neighbor_signature = []
                
                for neighbor in ilg.graph.neighbors(node):
                    neighbor_color = current_colors[neighbor]
                    
                    # Get edge label (argument position)
                    edge_label = ilg.edge_labels.get((node, neighbor), 0)
                    
                    neighbor_signature.append((neighbor_color, edge_label))
                
                # Sort for canonical ordering (important for determinism)
                neighbor_signature.sort()
                
                # Hash: combine own color with neighbor signature
                new_color = self._hash_color((node_color, tuple(neighbor_signature)))
                next_colors[node] = new_color
                
                # Track colors
                if collect_colors:
                    all_colors.add(new_color)
                else:
                    all_colors[new_color] += 1
            
            current_colors = next_colors
        
        return all_colors
    
    @staticmethod
    def _hash_color(color_data) -> str:
        """
        Create a deterministic hash of color data.
        
        Args:
            color_data: Tuple or other hashable representing color information
            
        Returns:
            Hash string
        """
        # Convert to string and hash
        color_str = str(color_data)
        return hashlib.md5(color_str.encode()).hexdigest()[:16]
    
    def save(self, filepath: str):
        """Save the fitted feature extractor."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted extractor")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_iterations': self.num_iterations,
                'color_vocabulary': self.color_vocabulary,
                'color_to_index': self.color_to_index,
                'fitted': self.fitted
            }, f)
        
        print(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a fitted feature extractor."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls(num_iterations=data['num_iterations'])
        extractor.color_vocabulary = data['color_vocabulary']
        extractor.color_to_index = data['color_to_index']
        extractor.fitted = data['fitted']
        
        print(f"Feature extractor loaded from {filepath}")
        return extractor


@dataclass
class WL2LocalFeatureExtractor:
    """
    2-Local WL (2-LWL) feature extractor.
    
    This is an approximation of 2-WL that colors pairs of vertices.
    More expressive than 1-WL but more expensive.
    
    Note: This is a simplified implementation. Full 2-WL would track
    all vertex pairs, which is computationally expensive.
    """
    num_iterations: int = 4
    color_vocabulary: Set[str] = field(default_factory=set)
    color_to_index: Dict[str, int] = field(default_factory=dict)
    fitted: bool = False
    
    def fit(self, ilgs: List[ILG]):
        """Fit on training ILGs."""
        all_colors = set()
        
        for ilg in ilgs:
            colors = self._run_2lwl(ilg, collect_colors=True)
            all_colors.update(colors)
        
        self.color_vocabulary = all_colors
        self.color_to_index = {color: idx for idx, color in enumerate(sorted(all_colors))}
        self.fitted = True
        
        print(f"2-LWL Feature Extractor fitted with {len(self.color_vocabulary)} colors")
    
    def transform(self, ilg: ILG) -> np.ndarray:
        """Transform ILG to feature vector."""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        color_counts = self._run_2lwl(ilg, collect_colors=False)
        
        feature_vector = np.zeros(len(self.color_vocabulary))
        for color, count in color_counts.items():
            if color in self.color_to_index:
                feature_vector[self.color_to_index[color]] = count
        
        return feature_vector
    
    def _run_2lwl(self, ilg: ILG, collect_colors: bool = False):
        """
        Run 2-Local WL refinement.
        
        This simplified version considers:
        - Individual node colors (1-WL)
        - Colors of adjacent node pairs
        """
        # Start with standard 1-WL
        current_colors = {node: self._hash_color(ilg.node_colors[node]) 
                         for node in ilg.graph.nodes()}
        
        all_colors = set(current_colors.values()) if collect_colors else Counter(current_colors.values())
        
        # Add pair colors for adjacent nodes
        pair_colors = {}
        for node in ilg.graph.nodes():
            for neighbor in ilg.graph.neighbors(node):
                # Create canonical pair representation
                pair = tuple(sorted([node, neighbor]))
                if pair not in pair_colors:
                    pair_color = self._hash_color((
                        current_colors[node],
                        current_colors[neighbor],
                        ilg.edge_labels.get((node, neighbor), 0)
                    ))
                    pair_colors[pair] = pair_color
                    
                    if collect_colors:
                        all_colors.add(pair_color)
                    else:
                        all_colors[pair_color] += 1
        
        # Run iterations
        for iteration in range(self.num_iterations):
            next_colors = {}
            
            for node in ilg.graph.nodes():
                node_color = current_colors[node]
                
                # Collect neighbor info including pair colors
                neighbor_signature = []
                for neighbor in ilg.graph.neighbors(node):
                    neighbor_color = current_colors[neighbor]
                    edge_label = ilg.edge_labels.get((node, neighbor), 0)
                    pair = tuple(sorted([node, neighbor]))
                    pair_color = pair_colors.get(pair, "")
                    
                    neighbor_signature.append((neighbor_color, edge_label, pair_color))
                
                neighbor_signature.sort()
                
                new_color = self._hash_color((node_color, tuple(neighbor_signature)))
                next_colors[node] = new_color
                
                if collect_colors:
                    all_colors.add(new_color)
                else:
                    all_colors[new_color] += 1
            
            current_colors = next_colors
            
            # Update pair colors
            new_pair_colors = {}
            for pair in pair_colors:
                n1, n2 = pair
                new_pair_color = self._hash_color((
                    current_colors[n1],
                    current_colors[n2],
                    pair_colors[pair]
                ))
                new_pair_colors[pair] = new_pair_color
                
                if collect_colors:
                    all_colors.add(new_pair_color)
                else:
                    all_colors[new_pair_color] += 1
            
            pair_colors = new_pair_colors
        
        return all_colors
    
    @staticmethod
    def _hash_color(color_data) -> str:
        """Create deterministic hash."""
        color_str = str(color_data)
        return hashlib.md5(color_str.encode()).hexdigest()[:16]
    
    def save(self, filepath: str):
        """Save the fitted feature extractor."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted extractor")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_iterations': self.num_iterations,
                'color_vocabulary': self.color_vocabulary,
                'color_to_index': self.color_to_index,
                'fitted': self.fitted
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a fitted feature extractor."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls(num_iterations=data['num_iterations'])
        extractor.color_vocabulary = data['color_vocabulary']
        extractor.color_to_index = data['color_to_index']
        extractor.fitted = data['fitted']
        
        return extractor


def main():
    """Test WL feature extraction."""
    from ilg_builder import ILGBuilder
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python wl_features.py <domain.pddl> <problem.pddl> [problem2.pddl ...]")
        sys.exit(1)
    
    domain_file = sys.argv[1]
    problem_files = sys.argv[2:]
    
    # Build ILGs
    builder = ILGBuilder()
    ilgs = []
    
    for problem_file in problem_files:
        print(f"Building ILG for {problem_file}...")
        ilg = builder.build_ilg(domain_file, problem_file)
        ilgs.append(ilg)
    
    # Extract features
    print(f"\nExtracting WL features from {len(ilgs)} ILGs...")
    extractor = WLFeatureExtractor(num_iterations=4)
    features = extractor.fit_transform(ilgs)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Number of unique colors: {len(extractor.color_vocabulary)}")
    print(f"\nFirst problem features (first 20 dimensions):")
    print(features[0, :20])


if __name__ == "__main__":
    main()


