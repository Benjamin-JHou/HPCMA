#!/usr/bin/env bash
set -euo pipefail

python3 database/build_db.py
python3 reproducibility/hash_inputs.py

python3 - <<'PY'
import sqlite3

con = sqlite3.connect('release/hpcma_atlas.sqlite')
try:
    cur = con.execute('select * from gene_profile_view limit 1')
    cols = [d[0] for d in cur.description]
finally:
    con.close()

expected = ['gene', 'disease', 'celltype', 'evidence_type', 'source_id']
if cols != expected:
    raise SystemExit(f'Unexpected gene_profile_view columns: {cols}')
print('Database query contract validated')
PY
