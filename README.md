# Hypertension Pan-Comorbidity Multi-Modal Atlas (HPCMA)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HPCMA is an open, reproducible research resource for hypertension-related multi-comorbidity biology. The repository focuses on auditable atlas tables, schema-backed query contracts, and reproducibility checks.

## What This Repository Provides

- Canonical atlas resource tables in `atlas_resource/`
- Relational schema and SQLite build pipeline in `database/`
- Reproducibility workflow in `reproducibility/`
- Core analysis scripts in `scripts/`
- Regression and contract tests in `tests/`

## Scope Boundary

- Primary purpose: reproducible research resource release
- API endpoints in `src/inference/api_server.py` are supplementary demonstration only
- No clinical deployment readiness claim is made

See: `docs/research_scope_and_claim_boundaries.md`

## Quick Start

```bash
python3 database/build_db.py
python3 reproducibility/hash_inputs.py
bash reproducibility/run_all.sh
python -m pytest tests -q
```

## Core Resource Tables

- `atlas_resource/hypertension_atlas_master_table.csv`
- `atlas_resource/prioritized_causal_genes.csv`
- `atlas_resource/gene_disease_celltype_annotation.csv`
- `atlas_resource/clinical_translation_table.csv`
- `atlas_resource/mechanism_axis_clusters.csv`
- `atlas_resource/ldsc_genetic_correlation_matrix.csv`
- `atlas_resource/coloc_results.csv`
- `atlas_resource/multilayer_network_edges.csv`

All canonical table fields are standardized to `lower_snake_case`.

## Query Contract

Database query interfaces are stabilized through:

- `database/queries/query_disease_network.sql`
- `database/queries/query_gene_profile.sql`
- `database/queries/query_celltype_links.sql`
- `tests/test_query_regression.py`

## Documentation Index

- Resource usage: `ATLAS_USAGE_GUIDE.md`
- Data provenance map: `docs/data_provenance_table.md`
- Claim traceability: `docs/manuscript_claim_map.md`
- Query expected outputs: `docs/query_examples_expected_outputs.md`
- Reviewer checklist: `docs/reviewer_checklist.md`
- Supplementary API notice: `docs/supplementary_api_notice.md`

## Citation

If you use HPCMA resources, please cite the related manuscript and link this repository.

## License

MIT. See `LICENSE`.
