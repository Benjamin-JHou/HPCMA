# Manuscript Claim Map

| Claim ID | Claim Statement | Evidence Table | Generating Script | Notes |
|---|---|---|---|---|
| C1 | Atlas links genes, diseases, and cell types in a queryable resource. | `atlas_edges`, `gene_profile_view` | `database/build_db.py` | Derived from `atlas_resource/hypertension_atlas_master_table.csv` |
| C2 | Resource outputs are reproducible from versioned inputs with checksums. | `reproducibility/manifest.json` | `reproducibility/hash_inputs.py` | Includes SHA-256 and run parameters |
| C3 | Query contracts are stable for reviewer replication. | `gene_profile_view` contract | `scripts/ci_validate_database.sh` | Enforced in CI and regression tests |
