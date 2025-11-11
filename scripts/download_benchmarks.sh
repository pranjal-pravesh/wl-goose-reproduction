#!/bin/bash

# Download IPC 2023 Learning Track Benchmarks
# The benchmarks are typically available from the IPC repository or conference site

set -e

BENCHMARK_DIR="/Users/pranjal/HomeBase/learned-heuristic-planner/benchmarks"

echo "Downloading IPC 2023 Learning Track Benchmarks..."

# The IPC 2023 benchmarks are typically hosted on GitHub
# For now, we'll clone the standard IPC benchmark repository

# Clone the IPC 2023 benchmarks repository
cd "$BENCHMARK_DIR"

# IPC benchmarks are often available from planning.domains or GitHub
# We'll use a standard approach to get common planning benchmarks

DOMAINS=(
    "blocksworld"
    "childsnack"
    "ferry"
    "floortile"
    "miconic"
    "rovers"
    "satellite"
    "sokoban"
    "spanner"
    "transport"
)

echo "Creating directory structure for 10 domains..."

for domain in "${DOMAINS[@]}"; do
    mkdir -p "${BENCHMARK_DIR}/${domain}/easy"
    mkdir -p "${BENCHMARK_DIR}/${domain}/medium"
    mkdir -p "${BENCHMARK_DIR}/${domain}/hard"
    echo "Created directories for ${domain}"
done

echo ""
echo "Benchmark directory structure created at: ${BENCHMARK_DIR}"
echo ""
echo "NOTE: You need to obtain the actual IPC 2023 Learning Track benchmark files."
echo "These are typically available from:"
echo "  - https://github.com/aibasel/downward-benchmarks"
echo "  - https://github.com/AI-Planning/pddl-generators"
echo "  - The IPC 2023 competition website"
echo ""
echo "Place the PDDL files in the appropriate directories:"
echo "  - Easy problems in: benchmarks/{domain}/easy/"
echo "  - Medium problems in: benchmarks/{domain}/medium/"
echo "  - Hard problems in: benchmarks/{domain}/hard/"
echo ""
echo "Each domain should have a domain.pddl file and problem files (problem01.pddl, etc.)"

exit 0


