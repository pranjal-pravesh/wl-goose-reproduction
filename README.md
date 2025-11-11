# WL-GOOSE: Learning Reliable Heuristics with Classical Machine Learning

This repository contains a complete implementation of the WL-GOOSE approach from the AAAI 2024 paper:

**"Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning"**  
by Dillon Z. Chen, Felipe Trevizan, and Sylvie Thiébaux

## Overview

WL-GOOSE uses:
- **Instance Learning Graphs (ILG)**: Graph representations of planning problems
- **Weisfeiler-Lehman (WL) algorithm**: Feature extraction from graphs
- **Classical Machine Learning**: SVR and GPR for learning heuristic functions
- **Greedy Best-First Search**: Planning with learned heuristics

The approach achieves state-of-the-art performance among learning-based planners, outperforming the FF heuristic and matching LAMA on multiple domains.

## Project Structure

```
learned-heuristic-planner/
├── benchmarks/          # IPC 2023 benchmark problems
│   ├── blocksworld/
│   ├── rovers/
│   └── ...
├── data/                # Generated training data
│   └── {domain}/
│       ├── features.npy
│       ├── h_star.npy
│       └── feature_extractor.pkl
├── models/              # Trained regression models
│   └── {domain}/
│       ├── svr_linear/
│       ├── svr_rbf/
│       └── gpr/
├── results/             # Evaluation results
│   ├── {domain}_results.json
│   ├── {domain}_baselines.json
│   └── analysis/
│       ├── coverage_table.csv
│       └── coverage_comparison.png
├── src/                 # Core implementation
│   ├── ilg_builder.py           # ILG construction
│   ├── wl_features.py           # WL feature extraction
│   ├── generate_training_data.py # Training data generation
│   ├── train_models.py          # Model training
│   ├── gbfs_search.py           # GBFS with learned heuristics
│   ├── evaluate.py              # Model evaluation
│   ├── run_baselines.py         # Baseline evaluation
│   └── analyze_results.py       # Results analysis
├── scripts/             # Execution scripts
│   ├── download_benchmarks.sh
│   ├── fetch_benchmarks_from_repo.py
│   ├── run_training.sh
│   └── run_evaluation.sh
├── downward/            # Fast Downward planner
└── requirements.txt     # Python dependencies
```

## Installation

### Prerequisites

1. **Python 3.8+** with virtual environment
2. **Fast Downward planner** (already cloned and built)
3. **Git** for downloading benchmarks

### Setup

```bash
# Navigate to project directory
cd /Users/pranjal/HomeBase/learned-heuristic-planner

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify Fast Downward is built
./downward/fast-downward.py --help
```

## Usage

### 1. Download Benchmarks

```bash
# Download IPC benchmarks
python3 scripts/fetch_benchmarks_from_repo.py
```

This creates a directory structure with 10 domains (Blocksworld, Rovers, Satellite, etc.), each split into Easy, Medium, and Hard problems.

### 2. Generate Training Data

Generate training data by solving Easy and Medium problems optimally and extracting state features:

```bash
# Full training pipeline (may take hours)
./scripts/run_training.sh

# Or manually:
python3 src/generate_training_data.py \
    --benchmark-dir benchmarks \
    --fast-downward downward/fast-downward.py \
    --output-dir data \
    --timeout 1800 \
    --domains blocksworld rovers satellite
```

This will:
- Solve training problems with A* + LM-Cut (30 min timeout per problem)
- Extract states along optimal trajectories
- Build ILG representations
- Extract WL features (L=4 iterations)
- Save training pairs (φ(s), h*(s))

### 3. Train Models

Train SVR and GPR models on the extracted features:

```bash
python3 src/train_models.py \
    --data-dir data \
    --model-dir models \
    --domains blocksworld rovers satellite
```

This trains:
- **SVR (Linear kernel)**: 5 runs with different seeds
- **SVR∞ (RBF kernel)**: 5 runs with hyperparameter tuning
- **GPR (Dot product kernel)**: 1 run (deterministic)

### 4. Evaluate on Test Problems

Evaluate trained models on Hard test problems:

```bash
# Full evaluation pipeline
./scripts/run_evaluation.sh

# Or manually:
python3 src/evaluate.py \
    --benchmark-dir benchmarks \
    --model-dir models \
    --data-dir data \
    --results-dir results \
    --time-limit 1800 \
    --domains blocksworld rovers satellite
```

### 5. Run Baseline Comparisons

Evaluate hFF and LAMA baselines:

```bash
python3 src/run_baselines.py \
    --benchmark-dir benchmarks \
    --fast-downward downward/fast-downward.py \
    --results-dir results \
    --time-limit 1800 \
    --domains blocksworld rovers satellite \
    --planners hFF LAMA
```

### 6. Analyze Results

Generate comparison tables and plots:

```bash
python3 src/analyze_results.py \
    --results-dir results
```

This creates:
- Coverage comparison table (CSV)
- Coverage comparison plot (PNG)
- Plan quality analysis

## Key Implementation Details

### Instance Learning Graphs (ILG)

ILGs encode a planning problem as a graph:
- **Vertices**: Objects (O), initial state propositions (s₀), goal propositions (G)
- **Edges**: Connections between propositions and their argument objects
- **Node colors**: 
  - `ob`: object
  - `ap`: achieved proposition (in s₀, not in G)
  - `ug`: unachieved goal (in G, not in s₀)
  - `ag`: achieved goal (in both s₀ and G)
- **Edge labels**: Argument positions (1, 2, ...)

### WL Feature Extraction

Weisfeiler-Lehman color refinement:
1. Initialize colors from node colors
2. For L=4 iterations:
   - Update each node's color based on:
     - Its current color
     - Multiset of (neighbor_color, edge_label) pairs
   - Hash to create new color
3. Generate feature vector: histogram of all colors across iterations

### Model Training

- **Target**: h*(s) = optimal cost-to-go for state s
- **Features**: WL color histogram (dimension ~1000-5000 depending on domain)
- **Models**:
  - SVR with linear kernel: Fast, interpretable
  - SVR with RBF kernel: More expressive, better coverage
  - GPR: Bayesian, provides uncertainty estimates

### Search

Greedy Best-First Search (GBFS):
- Priority queue ordered by h(s) only (no g-cost)
- For each state: build ILG → extract features → predict h(s)
- No tie-breaking, no secondary heuristics

## Expected Results

Based on the paper, WL-GOOSE should:
- **Outperform hFF** on most domains
- **Match or beat LAMA** on domains like Blocksworld, Childsnack, Floortile, Satellite
- **Plan quality**: Near-optimal (trained on h*)
- **Training time**: CPU-only, minutes to hours per domain
- **Inference time**: Fast (sparse SVR support vectors)

## Troubleshooting

### Import Errors

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Fast Downward Not Found

```bash
# Rebuild Fast Downward
cd downward
./build.py
```

### pddlpy Parsing Errors

Some PDDL files may have syntax that pddlpy doesn't handle well. Check:
- PDDL version compatibility
- Domain and problem file formatting

### Out of Memory

For large domains, reduce:
- Number of training problems
- WL iterations (default L=4)
- Feature dimensionality

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{chen2024return,
  title={Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning},
  author={Chen, Dillon Z. and Trevizan, Felipe and Thi{\'e}baux, Sylvie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## License

This implementation is for research and educational purposes. The Fast Downward planner has its own license (GPL v3).

## Contact

For questions about this implementation, please open an issue on the repository.

---

**Note**: This is a reproduction of the WL-GOOSE approach. Performance may vary depending on:
- Benchmark versions
- Planner configurations  
- Hardware specifications
- Random seed variations

