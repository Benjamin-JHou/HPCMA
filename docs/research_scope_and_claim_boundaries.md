# Research Scope and Claim Boundaries

## Purpose

This project is framed as a reproducible atlas resource for hypertension pan-comorbidity research using public and harmonized datasets.

## Allowed Primary Claims

- Resource construction workflow is transparent and reproducible.
- Data harmonization, quality checks, and atlas tables are auditable.
- Released artifacts can be queried and reused for research.

## Disallowed Primary Claims

- Clinical deployment readiness.
- Patient-level decision support validity in routine care.
- Regulatory-grade model performance claims without external clinical validation.

## Supplementary Components

- The API server and risk-scoring endpoints are supplementary demonstration artifacts.
- Supplementary artifacts must be clearly labeled "demonstration only".

## Reviewer-Facing Policy

- Every manuscript claim must map to a specific table, script, and reproducibility record.
- Simulated outputs must be isolated from real-data outputs and labeled explicitly.

## Public Repository Curation Policy

- Include only materials required for reproducibility and research reuse.
- Exclude private manuscript drafting assets, teaching notebooks, and internal planning notes from Git tracking.
- Keep canonical data interfaces in `atlas_resource/` with standardized `lower_snake_case` fields.
