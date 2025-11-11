# Reproduce WL-GOOSE Experiments End-to-End

## 1. Benchmarks and Environment Setup

**Obtain IPC 2023 Learning Track Benchmarks**

- Download all 10 domains: Blocksworld, Childsnack, Ferry, Floortile, Miconic, Rovers, Satellite, Sokoban, Spanner, Transport
- Each domain has Easy, Medium, and Hard instances
- Training set: Easy + Medium; Test set: Hard
- Create directory structure: `benchmarks/{domain}/{easy,medium,hard}/`

**Install Scorpion Planner**

- Clone/build Scorpion optimal planner for generating training labels
- Scorpion is used to compute h*(s) with 30-minute timeout per instance
- Alternative: Use Fast Downward with optimal A* if Scorpion unavailable

## 2. Core Implementation Components

**ILG (Instance Learning Graph) Construction** (`src/ilg_builder.py`)

- Parse PDDL problems to extract: predicates P, objects O, initial state s_0, goal G
- Build graph vertices: V = O ∪ s_0 ∪ G
- Build edges: for each atom p = P(o_1,...,o_n) in s_0 ∪ G, add edges (p, o_i)
- Node colors: `ob` for objects, `(ap, P)` for achieved props, `(ug, P)` for unachieved goals, `(ag, P)` for achieved goals
- Edge labels: argument position i

**WL Feature Extraction** (`src/wl_features.py`)

- Implement Weisfeiler-Lehman color refinement with edge labels
- Run L=4 iterations (fixed hyperparameter from paper)
- Color update: hash(own_color, multiset of (neighbor_color, edge_label))
- Generate histogram feature vector over all observed colors
- Store color vocabulary from training set

**2-LWL Features** (optional, `src/wl_features.py`)

- Implement 2-Local WL approximation for 2-WL expressivity
- More expensive but more expressive than standard WL

## 3. Training Data Generation

**Extract Optimal State-Cost Pairs** (`src/generate_training_data.py`)

- For each training instance (Easy + Medium):
  - Run Scorpion with 30-minute timeout to find optimal plan
  - If solved: extract all states along trajectory
  - For each state s: compute h*(s) = remaining cost to goal
  - Build ILG(s), extract WL features φ(s)
  - Store training pair: (φ(s), h*(s))
- Save training data per domain: `data/{domain}/training_features.pkl`

## 4. Model Training

**Train Regression Models** (`src/train_models.py`)

- Use scikit-learn for all models
- **SVR with linear kernel**: `sklearn.svm.SVR(kernel='linear')`
- **SVR∞ with RBF kernel**: `sklearn.svm.SVR(kernel='rbf')`, tune C and gamma
- **GPR with dot-product kernel**: `sklearn.gaussian_process.GaussianProcessRegressor`
- Train 5 runs with different seeds for SVR (report mean performance)
- Train 1 run for GPR (deterministic)
- Save trained models: `models/{domain}/{model_type}_seed{i}.pkl`

**Hyperparameter Tuning**

- Grid search for SVR regularization C and RBF width γ
- Use cross-validation on training set
- Record best hyperparameters per domain

## 5. Evaluation with GBFS

**Implement GBFS with Learned Heuristic** (`src/gbfs_search.py`)

- Greedy Best-First Search: priority queue by h(s) only
- No tie-breaking, no g-cost consideration
- For test state s:
  - Build ILG(s)
  - Extract WL features φ(s) using training vocabulary
  - Predict h(s) from trained model
  - Use as priority in GBFS
- Integrate with Fast Downward translator for PDDL parsing

**Run Evaluation on Hard Instances** (`src/evaluate.py`)

- For each domain and each trained model:
  - Run GBFS on all Hard test instances
  - Track: coverage (problems solved), plan cost, nodes expanded, search time
  - Use reasonable time/memory limits (match paper's setup)
- Save results: `results/{domain}/{model_type}_results.json`

## 6. Baseline Comparisons

**Run hFF Baseline**

- Use Fast Downward with FF heuristic and GBFS
- Command: `./fast-downward.py domain.pddl problem.pddl --search "eager_greedy([ff()])"`
- Run on same Hard instances

**Run LAMA Baseline** (optional)

- Use Fast Downward LAMA configuration
- Command: `./fast-downward.py --alias lama-first domain.pddl problem.pddl`

## 7. Results Analysis

**Aggregate and Visualize Results** (`src/analyze_results.py`)

- Compute coverage per domain (# solved / # total)
- Compare plan quality (cost) when multiple methods solve same problem
- Generate plots: coverage comparison, plan quality comparison
- Create summary tables matching paper's Table 1/2 format
- Statistical significance tests if needed

## 8. Key Files to Create

```
/Users/pranjal/HomeBase/learned-heuristic-planner/
├── benchmarks/          # IPC 2023 benchmarks
├── data/                # Generated training data
├── models/              # Trained regression models
├── results/             # Evaluation results
├── src/
│   ├── ilg_builder.py   # ILG graph construction
│   ├── wl_features.py   # WL feature extraction
│   ├── generate_training_data.py  # Training data pipeline
│   ├── train_models.py  # Model training
│   ├── gbfs_search.py   # GBFS with learned heuristic
│   ├── evaluate.py      # Full evaluation script
│   └── analyze_results.py  # Results analysis
├── scripts/
│   ├── download_benchmarks.sh
│   ├── run_training.sh
│   └── run_evaluation.sh
└── requirements.txt     # Already exists
```

## 9. Critical Implementation Details

- **WL iterations**: L=4 (fixed, from paper)
- **Training timeout**: 30 minutes per instance for Scorpion
- **All domains have unit costs** (simplifies cost computation)
- **No GPU required** (CPU-only classical ML)
- **Feature sparsity**: SVR gives sparse support vectors for fast inference
- **Reproducibility**: Fix random seeds for SVR (5 runs), report mean ± std

## 10. Validation Checkpoints

After each major step, verify:

1. ILG construction matches paper's definition (Section 6)
2. WL features are histograms with correct dimensionality
3. Training data contains h*(s) values from optimal plans
4. Models converge and produce reasonable heuristic values
5. GBFS finds plans on test problems
6. Coverage results trend toward paper's findings (WL-GOOSE > hFF)