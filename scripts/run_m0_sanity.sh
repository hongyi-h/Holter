#!/bin/bash
# M0: Data sanity checks (R001-R004)
# Run on any machine with Python + numpy (no GPU needed)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== M0: Data Sanity Checks ==="
echo "Data dir: ${DATA_DIR:-data/DMS}"
echo ""

python scripts/sanity_check.py \
    --data_dir "${DATA_DIR:-data/DMS}" \
    --output "sanity_results.json"

echo ""
echo "Results saved to sanity_results.json"
echo "Review before proceeding to M1."
