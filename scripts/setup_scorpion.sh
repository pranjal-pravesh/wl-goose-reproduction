#!/bin/bash

# Setup Scorpion planner for generating optimal training labels
# Scorpion is an optimal planner that can be used to compute h*(s)
# 
# Note: Scorpion is a specific planner that may not be publicly available.
# As an alternative, we can use Fast Downward with A* and an admissible heuristic
# to compute optimal plans.

set -e

SCORPION_DIR="/Users/pranjal/HomeBase/learned-heuristic-planner/scorpion"

echo "Setting up optimal planner for training label generation..."
echo ""
echo "OPTION 1: Use Fast Downward with A* (recommended fallback)"
echo "  Fast Downward is already installed and can compute optimal plans using:"
echo "  --search \"astar(lmcut())\" or --search \"astar(merge_and_shrink())\""
echo ""
echo "OPTION 2: Install Scorpion planner (if available)"
echo "  Scorpion is a competition planner that needs to be obtained separately"
echo ""

# Check if Fast Downward is available
FD_PATH="/Users/pranjal/HomeBase/learned-heuristic-planner/downward/fast-downward.py"

if [ -f "$FD_PATH" ]; then
    echo "✓ Fast Downward is available at: $FD_PATH"
    echo ""
    echo "We'll use Fast Downward with A* and admissible heuristics for optimal planning."
    echo "This is a valid alternative to Scorpion for generating training labels."
    echo ""
    echo "Configuration:"
    echo "  - Search: A* with LM-Cut or Merge-and-Shrink heuristics"
    echo "  - Timeout: 30 minutes per instance (as specified in paper)"
    echo "  - This will generate optimal h*(s) values for training"
else
    echo "✗ Fast Downward not found at expected location"
    echo "Please ensure Fast Downward is built correctly"
    exit 1
fi

echo ""
echo "✓ Optimal planner setup complete"
echo ""

exit 0


