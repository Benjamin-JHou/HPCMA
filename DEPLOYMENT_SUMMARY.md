# HPCMA Reproducibility Summary

## Purpose

This file tracks the reproducibility release status of the public HPCMA resource repository.

## Public Release Intent

The repository is curated for research reuse and reviewer auditability:

- canonical atlas tables (`atlas_resource/`)
- relational schema and query contracts (`database/`)
- reproducibility manifest workflow (`reproducibility/`)
- tests for schema, provenance, and claims (`tests/`)

Non-core authoring assets (for example manuscript drafts and private working notebooks) are intentionally excluded from public tracking.

## Validation Commands

```bash
python -m pytest tests -q
bash reproducibility/run_all.sh
```

## Expected Validation Outcomes

- test suite passes
- SQLite atlas database is rebuilt deterministically
- provenance manifest is regenerated
- query contract check passes

## Scope Policy

- This repository is a reproducible research resource
- Supplementary API components are demonstration-only artifacts
- Clinical deployment claims are out of scope

Reference: `docs/research_scope_and_claim_boundaries.md`
