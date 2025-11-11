#!/bin/bash

# Run the complete evaluation pipeline for WL-GOOSE
# This script evaluates trained models and baselines, then analyzes results

set -e

# Configuration
PROJECT_ROOT="/Users/pranjal/HomeBase/learned-heuristic-planner"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmarks"
DATA_DIR="${PROJECT_ROOT}/data"
MODEL_DIR="${PROJECT_ROOT}/models"
RESULTS_DIR="${PROJECT_ROOT}/results"
FAST_DOWNWARD="${PROJECT_ROOT}/downward/fast-downward.py"

# Domains to evaluate
DOMAINS=(
    "blocksworld"
    "rovers"
    "satellite"
    "sokoban"
    "transport"
)

echo "========================================"
echo "WL-GOOSE Evaluation Pipeline"
echo "========================================"
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Benchmarks: ${BENCHMARK_DIR}"
echo "Models: ${MODEL_DIR}"
echo "Results: ${RESULTS_DIR}"
echo ""

cd "${PROJECT_ROOT}"

# Step 1: Evaluate WL-GOOSE models
echo "Step 1: Evaluating WL-GOOSE models..."
echo "========================================"

python3 src/evaluate.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --data-dir "${DATA_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --time-limit 300 \
    --domains "${DOMAINS[@]}"

if [ $? -ne 0 ]; then
    echo "Warning: Some WL-GOOSE evaluations failed"
fi

# Step 2: Evaluate baseline planners
echo ""
echo "Step 2: Evaluating baseline planners..."
echo "========================================"

python3 src/run_baselines.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --fast-downward "${FAST_DOWNWARD}" \
    --results-dir "${RESULTS_DIR}" \
    --time-limit 300 \
    --domains "${DOMAINS[@]}" \
    --planners hFF LAMA

if [ $? -ne 0 ]; then
    echo "Warning: Some baseline evaluations failed"
fi

# Step 3: Analyze results
echo ""
echo "Step 3: Analyzing results..."
echo "========================================"

python3 src/analyze_results.py \
    --results-dir "${RESULTS_DIR}"

if [ $? -ne 0 ]; then
    echo "Warning: Results analysis encountered errors"
fi

echo ""
echo "========================================"
echo "Evaluation pipeline complete!"
echo "========================================"
echo "Results: ${RESULTS_DIR}"
echo "Analysis: ${RESULTS_DIR}/analysis"
echo ""


