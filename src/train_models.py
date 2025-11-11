"""
Train Regression Models for WL-GOOSE

Trains classical machine learning models on WL features:
- Support Vector Regression (SVR) with linear kernel
- SVR with RBF kernel (SVR∞)
- Gaussian Process Regression (GPR)
- SVR with 2-LWL features

Reference: "Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning" (AAAI 2024)
"""

import os
import sys
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import joblib

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


@dataclass
class ModelConfig:
    """Configuration for a model type."""
    name: str
    model_class: type
    param_grid: Dict
    num_seeds: int = 5
    scale_features: bool = True


class WLGOOSETrainer:
    """Train WL-GOOSE models on extracted features."""
    
    # Model configurations from the paper
    MODEL_CONFIGS = {
        'svr_linear': ModelConfig(
            name='SVR (Linear)',
            model_class=SVR,
            param_grid={
                'kernel': ['linear'],
                'C': [0.1, 1.0, 10.0, 100.0],
                'epsilon': [0.01, 0.1, 0.5]
            },
            num_seeds=5,
            scale_features=True
        ),
        'svr_rbf': ModelConfig(
            name='SVR∞ (RBF)',
            model_class=SVR,
            param_grid={
                'kernel': ['rbf'],
                'C': [1.0, 10.0, 100.0, 1000.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.5]
            },
            num_seeds=5,
            scale_features=True
        ),
        'gpr': ModelConfig(
            name='GPR (Dot Product)',
            model_class=GaussianProcessRegressor,
            param_grid={
                'kernel': [DotProduct() + WhiteKernel()],
                'alpha': [1e-10, 1e-8, 1e-6],
                'normalize_y': [True]
            },
            num_seeds=1,  # GPR is deterministic
            scale_features=True
        )
    }
    
    def __init__(self, data_dir: str, model_dir: str):
        """
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self, domain: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Tuple of (features, h_star_values)
        """
        domain_dir = self.data_dir / domain
        
        features = np.load(domain_dir / 'features.npy')
        h_star = np.load(domain_dir / 'h_star.npy')
        
        print(f"  Loaded {len(features)} training examples")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"  h* range: [{h_star.min():.2f}, {h_star.max():.2f}]")
        
        return features, h_star
    
    def train_model(self, 
                    features: np.ndarray, 
                    labels: np.ndarray, 
                    config: ModelConfig,
                    seed: int = 42) -> Tuple[object, object, Dict]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            features: Feature matrix
            labels: Target values (h*)
            config: Model configuration
            seed: Random seed
            
        Returns:
            Tuple of (trained_model, scaler, metrics)
        """
        # Scale features if needed
        scaler = None
        X = features
        
        if config.scale_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(features)
        
        # Create base model
        if config.model_class == SVR:
            base_model = SVR(max_iter=10000)
        elif config.model_class == GaussianProcessRegressor:
            base_model = GaussianProcessRegressor(
                n_restarts_optimizer=5,
                random_state=seed
            )
        else:
            base_model = config.model_class()
        
        # Grid search for hyperparameters
        print(f"    Running grid search with {len(config.param_grid)} parameters...")
        
        grid_search = GridSearchCV(
            base_model,
            config.param_grid,
            cv=min(5, len(features) // 10),  # 5-fold CV or less if not enough data
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, labels)
        
        best_model = grid_search.best_estimator_
        
        # Compute metrics
        train_score = best_model.score(X, labels)
        predictions = best_model.predict(X)
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        metrics = {
            'train_r2': train_score,
            'train_mse': mse,
            'train_mae': mae,
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_  # Convert back to positive MSE
        }
        
        print(f"    Best params: {grid_search.best_params_}")
        print(f"    Train R²: {train_score:.4f}")
        print(f"    Train MAE: {mae:.4f}")
        
        return best_model, scaler, metrics
    
    def train_domain(self, domain: str, model_types: List[str] = None):
        """
        Train all models for a single domain.
        
        Args:
            domain: Domain name
            model_types: List of model type keys (default: all)
        """
        print(f"\n{'='*60}")
        print(f"Training models for: {domain}")
        print(f"{'='*60}")
        
        # Load training data
        try:
            features, h_star = self.load_training_data(domain)
        except Exception as e:
            print(f"Error loading data for {domain}: {e}")
            return
        
        if len(features) == 0:
            print(f"No training data for {domain}, skipping")
            return
        
        # Determine which models to train
        if model_types is None:
            model_types = list(self.MODEL_CONFIGS.keys())
        
        # Train each model type
        for model_type in model_types:
            if model_type not in self.MODEL_CONFIGS:
                print(f"Unknown model type: {model_type}")
                continue
            
            config = self.MODEL_CONFIGS[model_type]
            print(f"\n  Training {config.name}...")
            
            # Create output directory
            model_output_dir = self.model_dir / domain / model_type
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Train with multiple seeds (or once for GPR)
            all_metrics = []
            
            for seed in range(config.num_seeds):
                if config.num_seeds > 1:
                    print(f"    Seed {seed + 1}/{config.num_seeds}...")
                
                try:
                    model, scaler, metrics = self.train_model(
                        features, h_star, config, seed=seed
                    )
                    
                    all_metrics.append(metrics)
                    
                    # Save model
                    model_file = model_output_dir / f'model_seed{seed}.pkl'
                    joblib.dump(model, model_file)
                    
                    # Save scaler if used
                    if scaler is not None:
                        scaler_file = model_output_dir / f'scaler_seed{seed}.pkl'
                        joblib.dump(scaler, scaler_file)
                    
                except Exception as e:
                    print(f"    Error training seed {seed}: {e}")
                    continue
            
            # Compute and save aggregate metrics
            if all_metrics:
                aggregate_metrics = {
                    'domain': domain,
                    'model_type': model_type,
                    'num_seeds': len(all_metrics),
                    'train_r2_mean': np.mean([m['train_r2'] for m in all_metrics]),
                    'train_r2_std': np.std([m['train_r2'] for m in all_metrics]),
                    'train_mae_mean': np.mean([m['train_mae'] for m in all_metrics]),
                    'train_mae_std': np.std([m['train_mae'] for m in all_metrics]),
                    'all_runs': all_metrics
                }
                
                metrics_file = model_output_dir / 'metrics.json'
                with open(metrics_file, 'w') as f:
                    json.dump(aggregate_metrics, f, indent=2)
                
                print(f"\n    Summary:")
                print(f"      R² = {aggregate_metrics['train_r2_mean']:.4f} ± {aggregate_metrics['train_r2_std']:.4f}")
                print(f"      MAE = {aggregate_metrics['train_mae_mean']:.4f} ± {aggregate_metrics['train_mae_std']:.4f}")
                print(f"    Saved to: {model_output_dir}")
    
    def train_all_domains(self, domains: List[str] = None, model_types: List[str] = None):
        """
        Train models for all domains.
        
        Args:
            domains: List of domain names (default: all available)
            model_types: List of model types (default: all)
        """
        # Find all domains if not specified
        if domains is None:
            domains = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        
        print(f"Training models for {len(domains)} domains")
        print(f"Model types: {model_types if model_types else 'all'}")
        
        for domain in domains:
            try:
                self.train_domain(domain, model_types)
            except Exception as e:
                print(f"Error training {domain}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Model training complete!")
        print(f"Models saved to: {self.model_dir}")
        print(f"{'='*60}")


def main():
    """Main entry point for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train WL-GOOSE models')
    parser.add_argument('--data-dir', required=True, help='Directory with training data')
    parser.add_argument('--model-dir', required=True, help='Output directory for models')
    parser.add_argument('--domains', nargs='+', help='Specific domains to train (default: all)')
    parser.add_argument('--model-types', nargs='+', 
                       choices=['svr_linear', 'svr_rbf', 'gpr'],
                       help='Model types to train (default: all)')
    
    args = parser.parse_args()
    
    trainer = WLGOOSETrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir
    )
    
    trainer.train_all_domains(
        domains=args.domains,
        model_types=args.model_types
    )


if __name__ == "__main__":
    main()


