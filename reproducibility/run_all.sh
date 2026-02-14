#!/usr/bin/env bash
set -euo pipefail

python3 database/build_db.py
python3 reproducibility/hash_inputs.py
bash scripts/ci_validate_database.sh

echo "Reproducibility pipeline completed."
