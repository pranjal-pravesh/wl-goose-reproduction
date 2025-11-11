# WL-GOOSE Quick Start Guide

## Complete End-to-End Reproduction

This guide walks you through reproducing the WL-GOOSE experiments from start to finish.

## Prerequisites Checklist

- [x] Fast Downward installed at `downward/`
- [x] Python virtual environment activated
- [x] Dependencies installed from `requirements.txt`

## Step-by-Step Execution

### 1. Verify Environment (5 minutes)

```bash
# Activate virtual environment
source venv/bin/activate

# Check Fast Downward
./downward/fast-downward.py --help

# Check Python packages
python3 -c "import pddlpy, sklearn, networkx; print('All imports OK')"
```

### 2. Download Benchmarks (5-10 minutes)

```bash
# Download IPC benchmark problems
python3 scripts/fetch_benchmarks_from_repo.py

# Verify benchmarks were downloaded
ls benchmarks/blocksworld/easy/
ls benchmarks/blocksworld/medium/
ls benchmarks/blocksworld/hard/
```

Expected output: Domain files and problem files in each directory.

### 3. Quick Test - Single Domain (30-60 minutes)

Test the pipeline on a single easy domain (Blocksworld) before running full experiments:

```bash
# Generate training data for Blocksworld only
python3 src/generate_training_data.py \
    --benchmark-dir benchmarks \
    --fast-downward downward/fast-downward.py \
    --output-dir data \
    --timeout 300 \
    --domains blocksworld

# This will:
# - Solve easy and medium Blocksworld problems optimally
# - Extract states along optimal trajectories
# - Build ILG graphs and extract WL features
# - Save training data to data/blocksworld/
```

Expected output:
```
Generating training data for: blocksworld
Found X training problems
Solving: problem01.pddl (easy)
  Found optimal plan with Y actions
  Extracted Z states
...
✓ Training data saved to: data/blocksworld
```

```bash
# Train models on Blocksworld
python3 src/train_models.py \
    --data-dir data \
    --model-dir models \
    --domains blocksworld

# This trains:
# - SVR with linear kernel (5 seeds)
# - SVR with RBF kernel (5 seeds)
# - GPR with dot product kernel (1 seed)
```

Expected output:
```
Training models for: blocksworld
Loaded X training examples
Feature dimension: Y

Training SVR (Linear)...
  Seed 1/5...
    Best params: {'C': 10.0, ...}
    Train R²: 0.XX
...
Summary:
  R² = 0.XX ± 0.XX
  MAE = Y.YY ± Y.YY
```

```bash
# Evaluate on Blocksworld hard problems
python3 src/evaluate.py \
    --benchmark-dir benchmarks \
    --model-dir models \
    --data-dir data \
    --results-dir results \
    --time-limit 300 \
    --domains blocksworld

# Run baselines
python3 src/run_baselines.py \
    --benchmark-dir benchmarks \
    --fast-downward downward/fast-downward.py \
    --results-dir results \
    --time-limit 300 \
    --domains blocksworld \
    --planners hFF LAMA

# Analyze results
python3 src/analyze_results.py --results-dir results
```

Expected output:
```
Evaluating: blocksworld
Found X test problems
Model: svr_rbf
  Seed 0: Testing X problems...
    Coverage: Y/X

Coverage Comparison:
Domain       hFF          LAMA         svr_linear   svr_rbf      gpr
blocksworld  A/X (P%)     B/X (Q%)     C/X (R%)     D/X (S%)     E/X (T%)
```

### 4. Full Experiment Run (Multiple Hours to Days)

Once the test succeeds, run on all domains:

```bash
# Full training pipeline
./scripts/run_training.sh

# This will process all 10 domains:
# blocksworld, childsnack, ferry, floortile, miconic,
# rovers, satellite, sokoban, spanner, transport
#
# Time estimate:
# - Small domains (Blocksworld, Ferry): 30-60 min each
# - Medium domains (Rovers, Satellite): 1-3 hours each
# - Large domains (Sokoban): 3-6 hours each
#
# Total: 10-30 hours depending on hardware
```

```bash
# Full evaluation pipeline
./scripts/run_evaluation.sh

# Time estimate: 5-15 hours for all domains
```

### 5. Inspect Results

```bash
# View coverage table
cat results/analysis/coverage_table.csv

# View coverage plot
open results/analysis/coverage_comparison.png  # macOS
# or
xdg-open results/analysis/coverage_comparison.png  # Linux

# Detailed results per domain
cat results/blocksworld_results.json
cat results/blocksworld_baselines.json
```

## Interpreting Results

### Coverage Metrics

Coverage = (Problems Solved) / (Total Hard Problems)

- **Good performance**: Coverage > 0.5 (50%)
- **Competitive with hFF**: Coverage within 10% of hFF
- **State-of-the-art**: Coverage > hFF

### Expected Patterns (from paper)

Based on the WL-GOOSE paper, you should see:

1. **Blocksworld**: WL-GOOSE ≥ LAMA ≥ hFF
2. **Rovers**: WL-GOOSE ≈ hFF < LAMA
3. **Satellite**: WL-GOOSE ≥ LAMA ≥ hFF
4. **Sokoban**: LAMA > WL-GOOSE ≈ hFF

### Plan Quality

For problems solved by multiple methods:
- WL-GOOSE plans should be near-optimal (trained on h*)
- hFF plans may be longer
- LAMA often finds high-quality plans

## Troubleshooting

### "No training data generated"

**Problem**: Optimal planner timed out on all training problems.

**Solution**:
- Increase `--timeout` (e.g., 3600 for 1 hour)
- Use fewer/easier training problems
- Check that Fast Downward is working: `./downward/fast-downward.py --help`

### "Feature extractor not found"

**Problem**: Training data generation didn't complete successfully.

**Solution**:
- Re-run training data generation for that domain
- Check `data/{domain}/` exists and contains `feature_extractor.pkl`

### "Out of memory during training"

**Problem**: Training dataset is too large for available RAM.

**Solution**:
- Reduce number of training problems (edit `run_training.sh`)
- Use fewer WL iterations (edit `wl_features.py`, change L=4 to L=3)
- Train on smaller domains first

### Evaluation is very slow

**Problem**: GBFS with learned heuristic expands many nodes.

**Solution**:
- This is expected for some problems
- Reduce `--time-limit` to skip hard problems faster
- Check that feature extraction is working correctly

## Validation Checklist

After running experiments, verify:

- [ ] Training data generated for each domain (`data/{domain}/features.npy` exists)
- [ ] Models trained for each domain (`models/{domain}/svr_rbf/model_seed0.pkl` exists)
- [ ] Evaluation results saved (`results/{domain}_results.json` exists)
- [ ] Baseline results saved (`results/{domain}_baselines.json` exists)
- [ ] Analysis completed (`results/analysis/coverage_table.csv` exists)
- [ ] Coverage plot generated (`results/analysis/coverage_comparison.png` exists)

## Next Steps

### Experiment with Variations

Try modifying key parameters:

1. **WL iterations**: Change L in `wl_features.py`
   - L=3: Faster, less expressive
   - L=5: Slower, more expressive

2. **Training timeout**: Adjust in `run_training.sh`
   - Higher timeout → more training data → better models
   - Lower timeout → faster pipeline

3. **Model types**: Train only specific models
   ```bash
   python3 src/train_models.py \
       --data-dir data \
       --model-dir models \
       --model-types svr_rbf  # Only train RBF kernel
   ```

4. **Search time limit**: Adjust in `run_evaluation.sh`
   - Higher limit → more problems solved
   - Lower limit → faster evaluation

### Compare with Paper Results

The original paper reports results on IPC 2023 Learning Track benchmarks. Compare your coverage numbers with:

- Table 1 in the paper (coverage by domain)
- Figure 4 in the paper (coverage comparison plot)

Minor differences are expected due to:
- Different hardware
- Random seed variation
- Planner version differences
- Benchmark file versions

## Advanced Usage

### Custom Domains

To add a new domain:

1. Create directory: `benchmarks/mydomain/`
2. Add subdirectories: `easy/`, `medium/`, `hard/`
3. Place PDDL files in each subdirectory
4. Run training and evaluation with `--domains mydomain`

### Debugging Individual Components

Test each module separately:

```bash
# Test ILG construction
python3 src/ilg_builder.py \
    benchmarks/blocksworld/easy/domain.pddl \
    benchmarks/blocksworld/easy/problem01.pddl

# Test WL feature extraction
python3 src/wl_features.py \
    benchmarks/blocksworld/easy/domain.pddl \
    benchmarks/blocksworld/easy/problem01.pddl \
    benchmarks/blocksworld/easy/problem02.pddl

# Test GBFS with a specific model
python3 src/gbfs_search.py \
    --domain benchmarks/blocksworld/hard/domain.pddl \
    --problem benchmarks/blocksworld/hard/problem01.pddl \
    --model models/blocksworld/svr_rbf/model_seed0.pkl \
    --scaler models/blocksworld/svr_rbf/scaler_seed0.pkl \
    --feature-extractor data/blocksworld/feature_extractor.pkl \
    --time-limit 300
```

## Getting Help

If you encounter issues:

1. Check the detailed README.md
2. Verify all prerequisites are installed
3. Run the quick test on Blocksworld first
4. Check log files for error messages
5. Ensure sufficient disk space (10+ GB)
6. Ensure sufficient RAM (8+ GB recommended)

---

**Ready to start?** Run the quick test now:

```bash
source venv/bin/activate
python3 scripts/fetch_benchmarks_from_repo.py
python3 src/generate_training_data.py --benchmark-dir benchmarks --fast-downward downward/fast-downward.py --output-dir data --timeout 300 --domains blocksworld
```


