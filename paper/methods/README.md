# HPCMA Methods Supplement (Resource-Paper Version)

## Scope and Intent

This supplement documents the reproducibility and database-verifiable components used for the HPCMA resource manuscript. The objective is to provide an auditable computational resource (data snapshot, schema, query contracts, and reproducibility workflow), not to claim full rerun of all downstream experimental pipelines in this revision.

## M1. Data Snapshot and Provenance

The resource is anchored by a fixed input snapshot and file-level provenance metadata.

- Primary provenance table: `docs/data_provenance_table.md`
- Reproducibility manifest: `reproducibility/manifest.json`
- Input checksum generator: `reproducibility/hash_inputs.py`

The manifest records timestamp, pipeline version, input paths, and content hashes. These artifacts are used as the canonical traceability layer for reviewer inspection.

## M2. Atlas Database Build and Quality Controls

A relational packaging step materializes resource tables into SQLite for deterministic query and review.

- Schema definition: `database/schema.sql`
- Build script: `database/build_db.py`
- QC checks: `database/qc_checks.sql`
- Query templates: `database/queries/*.sql`

Build command:

```bash
python database/build_db.py
```

Post-build QC is executed against the generated SQLite artifact to ensure schema presence and table-level consistency required by manuscript tables/figures.

## M3. Query Contract Validation

SQL outputs that support manuscript claims are validated through regression-style tests and fixed query templates.

- Query regression test: `tests/test_query_regression.py`
- Expected output mapping: `docs/query_examples_expected_outputs.md`
- Query templates:
  - `database/queries/query_disease_network.sql`
  - `database/queries/query_gene_profile.sql`
  - `database/queries/query_celltype_links.sql`

This contract-first pattern reduces drift between manuscript narrative and resource outputs.

## M4. Reproducibility Manifest Workflow

An end-to-end reproducibility gate verifies artifact integrity and claim traceability.

- Driver: `reproducibility/run_all.sh`
- Coverage tests:
  - `tests/test_data_provenance.py`
  - `tests/test_schema_integrity.py`
  - `tests/test_claim_traceability.py`
  - `tests/test_docs_claim_boundaries.py`

The workflow regenerates/validates manifest-bound artifacts and ensures cited resource outputs remain reproducible.

## M5. Supplementary API Boundary

The inference API included in this repository is a supplementary demonstration channel only.

- API implementation: `src/inference/api_server.py`
- Boundary checks: `tests/test_api_disclaimer.py`

The API is explicitly labeled as non-clinical and non-deployment; it is not part of the core evidentiary chain for the resource paper.

## M6. Out-of-Scope Elements in This Revision

The following are intentionally not claimed as fully rerun evidence within this quick submission package:

- Full empirical rerun of all step2-7 research pipelines on newly curated real-data cohorts.
- Operational readiness for care settings of the supplementary API channel.

These boundaries are enforced by repository documentation and test guards to avoid over-claiming.

## M7. Verification Commands

```bash
python -m pytest tests -q
bash reproducibility/run_all.sh
```

Successful execution of these commands constitutes the minimal reproducibility evidence set for this manuscript version.
