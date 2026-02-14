# HPCMA SCI Resource Paper Design (Bioinformatics Track)

**Date:** 2026-02-14  
**Project:** Hypertension Pan-Comorbidity Multi-Modal Atlas (HPCMA)

## 1. Publication Positioning

- **Target venue style:** Bioinformatics / Briefings in Bioinformatics.
- **Paper type:** Public resource + reproducible pipeline.
- **Primary claim:** A reproducible multi-source hypertension comorbidity atlas resource built from public datasets.
- **Non-claims:**
  - No clinical deployment readiness claim.
  - No individual-level clinical decision validity claim.
  - Prediction API is supplementary demonstration only.

## 2. Scientific Narrative

- **Main contribution axis:** data standardization, resource integration, queryable atlas, reproducibility.
- **Secondary axis:** transparent QC and robustness checks.
- **Supplementary axis:** API demo and exploratory risk scoring workflow.

## 3. Evidence Framework (Reviewer-Facing)

### 3.1 Core Evidence in Main Text
- Data source coverage matrix (trait, source, release version, access date, sample size).
- Harmonization metrics (mapping success, SNP retention, missingness and exclusion rates).
- Cross-dataset consistency checks (effect direction consistency, duplicate handling, sample size sanity checks).
- Atlas stability analysis (bootstrap/re-sampling stability).

### 3.2 Resource/Database Evidence
- Publish a real research database artifact (DuckDB/SQLite), not CSV-only deliverables.
- Provide schema with primary keys, unique constraints, and foreign keys.
- Include provenance fields (`source_id`, `source_version`, `download_date`, `checksum`).
- Provide fixed SQL query examples with expected outputs for reproducibility checks.

### 3.3 Statistical Reporting Requirements
- Every major result includes effect size, confidence interval, and multiple-testing control.
- Sensitivity analyses are mandatory for key conclusions.

## 4. Engineering and Documentation Repositioning

- Remove or downgrade over-claims like "production-ready" and "clinical deployment ready".
- Explicitly label simulated or demonstration-only components.
- Separate outputs by evidence level:
  - `real_data_results/`
  - `synthetic_demo_results/`
- Keep API and model-serving materials in supplementary documentation only.

## 5. Risk Control Priorities

1. Stop claim drift between docs and code.
2. Replace pseudo-health checks with real dependency checks.
3. Enforce CI as a gate (remove fail-open patterns).
4. Add schema and query regression tests for resource integrity.
5. Add manuscript claim-to-evidence traceability mapping.

## 6. Deliverables for Submission Package

- Database artifact (`.duckdb` or `.sqlite`) and schema SQL.
- Reproducibility manifest with input hashes and run parameters.
- End-to-end rebuild script.
- Reviewer checklist and claim map linking conclusions to files/tables.
- Main text figures/tables generated only from real-data results.

## 7. Acceptance Criteria

- A third-party reviewer can rebuild core resource outputs from documented inputs and scripts.
- Main text claims can be traced to auditable tables and scripts.
- No unsupported clinical deployment language remains in main paper artifacts.
- Supplementary API demo clearly marked as non-clinical and non-primary evidence.
