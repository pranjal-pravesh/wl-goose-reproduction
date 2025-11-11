"""
Analyze and Visualize WL-GOOSE Results

Compares WL-GOOSE models with baseline planners, generates tables and plots.

Reference: "Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning" (AAAI 2024)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsAnalyzer:
    """Analyze and visualize experimental results."""
    
    def __init__(self, results_dir: str, output_dir: str = None):
        """
        Args:
            results_dir: Directory with result files
            output_dir: Directory for plots and tables (default: results_dir/analysis)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / 'analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_wlgoose_results(self) -> pd.DataFrame:
        """Load WL-GOOSE evaluation results into a DataFrame."""
        all_results = []
        
        # Find all domain result files
        result_files = list(self.results_dir.glob('*_results.json'))
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            for entry in data:
                all_results.append(entry)
        
        if not all_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        return df
    
    def load_baseline_results(self) -> pd.DataFrame:
        """Load baseline planner results into a DataFrame."""
        all_results = []
        
        # Find all baseline result files
        result_files = list(self.results_dir.glob('*_baselines.json'))
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            for entry in data:
                all_results.append(entry)
        
        if not all_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        return df
    
    def compute_coverage_by_domain(self, df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """
        Compute coverage (fraction of problems solved) by domain.
        
        Args:
            df: DataFrame with results
            group_by: Column to group by ('model_type' or 'planner')
            
        Returns:
            DataFrame with coverage statistics
        """
        if df.empty:
            return pd.DataFrame()
        
        # Group by domain and model/planner
        grouped = df.groupby(['domain', group_by])
        
        coverage_data = []
        
        for (domain, model), group in grouped:
            solved = group['solved'].sum()
            total = len(group)
            coverage = solved / total if total > 0 else 0.0
            
            coverage_data.append({
                'domain': domain,
                group_by: model,
                'coverage': coverage,
                'solved': solved,
                'total': total
            })
        
        return pd.DataFrame(coverage_data)
    
    def aggregate_across_seeds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate WL-GOOSE results across multiple seeds."""
        if df.empty or 'seed' not in df.columns:
            return df
        
        # Group by domain, model_type, and problem
        grouped = df.groupby(['domain', 'model_type', 'problem'])
        
        aggregated = []
        
        for (domain, model_type, problem), group in grouped:
            solved_count = group['solved'].sum()
            num_seeds = len(group)
            
            # Average over seeds
            aggregated.append({
                'domain': domain,
                'model_type': model_type,
                'problem': problem,
                'solved': solved_count > 0,  # Solved by at least one seed
                'solved_fraction': solved_count / num_seeds,
                'mean_plan_cost': group[group['solved']]['plan_cost'].mean() if solved_count > 0 else -1,
                'mean_time': group[group['solved']]['time_seconds'].mean() if solved_count > 0 else -1,
                'mean_nodes_expanded': group[group['solved']]['nodes_expanded'].mean() if solved_count > 0 else -1
            })
        
        return pd.DataFrame(aggregated)
    
    def create_coverage_table(self):
        """Create a coverage comparison table."""
        print("\nGenerating coverage comparison table...")
        
        # Load results
        wlgoose_df = self.load_wlgoose_results()
        baseline_df = self.load_baseline_results()
        
        if wlgoose_df.empty and baseline_df.empty:
            print("No results found!")
            return
        
        # Aggregate WL-GOOSE across seeds
        if not wlgoose_df.empty:
            wlgoose_agg = self.aggregate_across_seeds(wlgoose_df)
            wlgoose_coverage = self.compute_coverage_by_domain(wlgoose_agg, 'model_type')
        else:
            wlgoose_coverage = pd.DataFrame()
        
        # Compute baseline coverage
        if not baseline_df.empty:
            baseline_coverage = self.compute_coverage_by_domain(baseline_df, 'planner')
        else:
            baseline_coverage = pd.DataFrame()
        
        # Combine into one table
        all_coverage = []
        
        # Get all domains
        domains = set()
        if not wlgoose_coverage.empty:
            domains.update(wlgoose_coverage['domain'].unique())
        if not baseline_coverage.empty:
            domains.update(baseline_coverage['domain'].unique())
        
        # Build table
        for domain in sorted(domains):
            row = {'Domain': domain}
            
            # Add baseline results
            if not baseline_coverage.empty:
                for planner in ['hFF', 'LAMA']:
                    planner_data = baseline_coverage[
                        (baseline_coverage['domain'] == domain) & 
                        (baseline_coverage['planner'] == planner)
                    ]
                    if not planner_data.empty:
                        coverage = planner_data.iloc[0]['coverage']
                        solved = planner_data.iloc[0]['solved']
                        total = planner_data.iloc[0]['total']
                        row[planner] = f"{solved}/{total} ({coverage:.2%})"
                    else:
                        row[planner] = "N/A"
            
            # Add WL-GOOSE results
            if not wlgoose_coverage.empty:
                for model_type in ['svr_linear', 'svr_rbf', 'gpr']:
                    model_data = wlgoose_coverage[
                        (wlgoose_coverage['domain'] == domain) & 
                        (wlgoose_coverage['model_type'] == model_type)
                    ]
                    if not model_data.empty:
                        coverage = model_data.iloc[0]['coverage']
                        solved = model_data.iloc[0]['solved']
                        total = model_data.iloc[0]['total']
                        row[model_type] = f"{solved}/{total} ({coverage:.2%})"
                    else:
                        row[model_type] = "N/A"
            
            all_coverage.append(row)
        
        # Create DataFrame
        coverage_table = pd.DataFrame(all_coverage)
        
        # Save as CSV
        csv_file = self.output_dir / 'coverage_table.csv'
        coverage_table.to_csv(csv_file, index=False)
        print(f"Coverage table saved to: {csv_file}")
        
        # Print to console
        print("\nCoverage Comparison:")
        print(coverage_table.to_string(index=False))
        
        return coverage_table
    
    def plot_coverage_comparison(self):
        """Create coverage comparison bar plots."""
        print("\nGenerating coverage comparison plots...")
        
        # Load results
        wlgoose_df = self.load_wlgoose_results()
        baseline_df = self.load_baseline_results()
        
        if wlgoose_df.empty and baseline_df.empty:
            print("No results found!")
            return
        
        # Aggregate and compute coverage
        coverage_data = []
        
        if not wlgoose_df.empty:
            wlgoose_agg = self.aggregate_across_seeds(wlgoose_df)
            wlgoose_coverage = self.compute_coverage_by_domain(wlgoose_agg, 'model_type')
            for _, row in wlgoose_coverage.iterrows():
                coverage_data.append({
                    'Domain': row['domain'],
                    'Method': row['model_type'].replace('svr_linear', 'SVR (Linear)')
                                              .replace('svr_rbf', 'SVR∞ (RBF)')
                                              .replace('gpr', 'GPR'),
                    'Coverage': row['coverage']
                })
        
        if not baseline_df.empty:
            baseline_coverage = self.compute_coverage_by_domain(baseline_df, 'planner')
            for _, row in baseline_coverage.iterrows():
                coverage_data.append({
                    'Domain': row['domain'],
                    'Method': row['planner'],
                    'Coverage': row['coverage']
                })
        
        if not coverage_data:
            print("No coverage data to plot!")
            return
        
        df = pd.DataFrame(coverage_data)
        
        # Create plot
        plt.figure(figsize=(14, 8))
        
        # Get unique domains
        domains = sorted(df['Domain'].unique())
        
        # Pivot for grouped bar plot
        pivot_df = df.pivot(index='Domain', columns='Method', values='Coverage')
        
        # Plot
        ax = pivot_df.plot(kind='bar', figsize=(14, 8), width=0.8)
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel('Coverage (Fraction of Problems Solved)', fontsize=12)
        ax.set_title('Coverage Comparison: WL-GOOSE vs Baselines', fontsize=14, fontweight='bold')
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        plot_file = self.output_dir / 'coverage_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Coverage plot saved to: {plot_file}")
        plt.close()
    
    def plot_plan_quality_comparison(self):
        """Compare plan quality (cost) when multiple methods solve the same problem."""
        print("\nGenerating plan quality comparison...")
        
        # Load results
        wlgoose_df = self.load_wlgoose_results()
        baseline_df = self.load_baseline_results()
        
        if wlgoose_df.empty or baseline_df.empty:
            print("Need both WL-GOOSE and baseline results for plan quality comparison!")
            return
        
        # Aggregate WL-GOOSE
        wlgoose_agg = self.aggregate_across_seeds(wlgoose_df)
        
        # Find commonly solved problems
        wlgoose_solved = set(zip(wlgoose_agg[wlgoose_agg['solved']]['domain'],
                                wlgoose_agg[wlgoose_agg['solved']]['problem']))
        baseline_solved = set(zip(baseline_df[baseline_df['solved']]['domain'],
                                 baseline_df[baseline_df['solved']]['problem']))
        
        common_problems = wlgoose_solved & baseline_solved
        
        if not common_problems:
            print("No commonly solved problems found!")
            return
        
        print(f"Found {len(common_problems)} commonly solved problems")
        
        # Compare costs
        quality_data = []
        
        for domain, problem in common_problems:
            # WL-GOOSE cost (best model)
            wl_costs = wlgoose_agg[
                (wlgoose_agg['domain'] == domain) & 
                (wlgoose_agg['problem'] == problem) &
                (wlgoose_agg['solved'])
            ]['mean_plan_cost']
            
            # Baseline costs
            baseline_costs = baseline_df[
                (baseline_df['domain'] == domain) & 
                (baseline_df['problem'] == problem) &
                (baseline_df['solved'])
            ]
            
            if not wl_costs.empty and not baseline_costs.empty:
                wl_cost = wl_costs.min()  # Best WL-GOOSE model
                
                for _, baseline_row in baseline_costs.iterrows():
                    quality_data.append({
                        'domain': domain,
                        'problem': problem,
                        'WL-GOOSE': wl_cost,
                        'Baseline': baseline_row['plan_cost'],
                        'Baseline_Method': baseline_row['planner']
                    })
        
        if not quality_data:
            print("No quality data to analyze!")
            return
        
        df = pd.DataFrame(quality_data)
        
        # Save to CSV
        csv_file = self.output_dir / 'plan_quality_comparison.csv'
        df.to_csv(csv_file, index=False)
        print(f"Plan quality data saved to: {csv_file}")
        
        # Compute statistics
        for baseline_method in df['Baseline_Method'].unique():
            subset = df[df['Baseline_Method'] == baseline_method]
            wl_better = (subset['WL-GOOSE'] < subset['Baseline']).sum()
            baseline_better = (subset['WL-GOOSE'] > subset['Baseline']).sum()
            tied = (subset['WL-GOOSE'] == subset['Baseline']).sum()
            
            print(f"\nVs {baseline_method}:")
            print(f"  WL-GOOSE better: {wl_better}")
            print(f"  {baseline_method} better: {baseline_better}")
            print(f"  Tied: {tied}")
    
    def generate_full_report(self):
        """Generate complete analysis report."""
        print("\n" + "="*60)
        print("WL-GOOSE Experimental Results Analysis")
        print("="*60)
        
        self.create_coverage_table()
        self.plot_coverage_comparison()
        self.plot_plan_quality_comparison()
        
        print("\n" + "="*60)
        print(f"Analysis complete! Results saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze WL-GOOSE results')
    parser.add_argument('--results-dir', required=True, help='Directory with result files')
    parser.add_argument('--output-dir', help='Output directory for analysis (default: results-dir/analysis)')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()


