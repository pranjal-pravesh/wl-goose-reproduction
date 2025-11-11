#!/bin/bash

# Run the complete training pipeline for WL-GOOSE
# This script generates training data and trains all models

set -e

# Configuration
PROJECT_ROOT="/Users/pranjal/HomeBase/learned-heuristic-planner"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmarks"
DATA_DIR="${PROJECT_ROOT}/data"
MODEL_DIR="${PROJECT_ROOT}/models"
FAST_DOWNWARD="${PROJECT_ROOT}/downward/fast-downward.py"

# Domains to train on
DOMAINS=(
    "blocksworld"
    "rovers"
    "satellite"
    "sokoban"
    "transport"
)

echo "========================================"
echo "WL-GOOSE Training Pipeline"
echo "========================================"
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Benchmarks: ${BENCHMARK_DIR}"
echo "Output data: ${DATA_DIR}"
echo "Output models: ${MODEL_DIR}"
echo ""

# Step 1: Generate training data
echo "Step 1: Generating training data..."
echo "========================================"

cd "${PROJECT_ROOT}"

python3 src/generate_training_data.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --fast-downward "${FAST_DOWNWARD}" \
    --output-dir "${DATA_DIR}" \
    --timeout 1800 \
    --domains "${DOMAINS[@]}"

if [ $? -ne 0 ]; then
    echo "Error generating training data!"
    exit 1
fi

# Step 2: Train models
echo ""
echo "Step 2: Training models..."
echo "========================================"

python3 src/train_models.py \
    --data-dir "${DATA_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --domains "${DOMAINS[@]}"

if [ $? -ne 0 ]; then
    echo "Error training models!"
    exit 1
fi

echo ""
echo "========================================"
echo "Training pipeline complete!"
echo "========================================"
echo "Training data: ${DATA_DIR}"
echo "Trained models: ${MODEL_DIR}"
echo ""


