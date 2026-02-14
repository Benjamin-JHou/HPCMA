# HPCMA SCI Resource Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reposition HPCMA into a reviewer-auditable Bioinformatics-style resource paper pipeline with real-data evidence boundaries, database-backed deliverables, and fail-closed reproducibility checks.

**Architecture:** Keep the existing analysis pipeline, but split outputs into real-data vs synthetic-demo channels, build a relational research database artifact (DuckDB/SQLite) from real outputs, and enforce integrity through schema constraints plus regression tests. Align manuscript-facing docs to evidence-backed claims only.

**Tech Stack:** Python 3.9+, pandas, duckdb/sqlite3, pytest, GitHub Actions, SQL schema/check scripts.

---

**Required skills during execution:** `@test-driven-development`, `@systematic-debugging`, `@verification-before-completion`.

### Task 1: Create Evidence Boundary and Artifact Policy

**Files:**
- Create: `docs/research_scope_and_claim_boundaries.md`
- Modify: `README.md`
- Modify: `DEPLOYMENT_SUMMARY.md`
- Modify: `atlas_resource/README.md`
- Test: `tests/test_docs_claim_boundaries.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_no_production_ready_claims_in_main_docs():
    targets = [
        Path("README.md"),
        Path("DEPLOYMENT_SUMMARY.md"),
        Path("atlas_resource/README.md"),
    ]
    banned = ["production-ready", "clinical deployment ready", "deployment-ready"]
    for p in targets:
        text = p.read_text(encoding="utf-8").lower()
        assert not any(term in text for term in banned)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_claim_boundaries.py::test_no_production_ready_claims_in_main_docs -v`  
Expected: FAIL because current docs still contain claim-drift language.

**Step 3: Write minimal implementation**

- Add explicit scope boundary document.
- Replace over-claim wording in listed docs.
- Add sentence: prediction API is supplementary demonstration only.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_docs_claim_boundaries.py::test_no_production_ready_claims_in_main_docs -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/research_scope_and_claim_boundaries.md README.md DEPLOYMENT_SUMMARY.md atlas_resource/README.md tests/test_docs_claim_boundaries.py
git commit -m "docs: enforce SCI claim boundaries and remove deployment overstatements"
```

### Task 2: Separate Real vs Synthetic Result Channels

**Files:**
- Create: `results/real_data/.gitkeep`
- Create: `results/synthetic_demo/.gitkeep`
- Modify: `scripts/step2_genetic_architecture.py`
- Modify: `scripts/step3_causal_gene_prioritization.py`
- Modify: `scripts/step5_multimodal_prediction.py`
- Modify: `scripts/step7_validation.py`
- Test: `tests/test_output_channeling.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_step_scripts_expose_output_mode_flag():
    scripts = [
        "scripts/step2_genetic_architecture.py",
        "scripts/step3_causal_gene_prioritization.py",
        "scripts/step5_multimodal_prediction.py",
        "scripts/step7_validation.py",
    ]
    for p in scripts:
        text = Path(p).read_text(encoding="utf-8")
        assert "--output-mode" in text or "OUTPUT_MODE" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_channeling.py::test_step_scripts_expose_output_mode_flag -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Add output mode arg/env: `real_data` vs `synthetic_demo`.
- Route outputs to `results/real_data/` or `results/synthetic_demo/`.
- Add visible log label for evidence mode.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_output_channeling.py::test_step_scripts_expose_output_mode_flag -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/step2_genetic_architecture.py scripts/step3_causal_gene_prioritization.py scripts/step5_multimodal_prediction.py scripts/step7_validation.py results/real_data/.gitkeep results/synthetic_demo/.gitkeep tests/test_output_channeling.py
git commit -m "refactor: split real-data and synthetic-demo output channels"
```

### Task 3: Build Database Schema and Deterministic Builder

**Files:**
- Create: `database/schema.sql`
- Create: `database/build_db.py`
- Create: `database/qc_checks.sql`
- Create: `release/.gitkeep`
- Test: `tests/test_schema_integrity.py`

**Step 1: Write the failing test**

```python
import sqlite3
from pathlib import Path


def test_schema_contains_core_constraints():
    schema = Path("database/schema.sql").read_text(encoding="utf-8").lower()
    assert "primary key" in schema
    assert "foreign key" in schema
    assert "unique" in schema
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema_integrity.py::test_schema_contains_core_constraints -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Define core tables: traits, genes, diseases, celltypes, atlas_edges, provenance.
- Add PK/UK/FK constraints.
- Build script writes `release/hpcma_atlas.sqlite` (or `.duckdb`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema_integrity.py::test_schema_contains_core_constraints -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add database/schema.sql database/build_db.py database/qc_checks.sql release/.gitkeep tests/test_schema_integrity.py
git commit -m "feat: add auditable atlas database schema and builder"
```

### Task 4: Add Provenance and Checksum Tracking

**Files:**
- Modify: `database/build_db.py`
- Create: `reproducibility/manifest.json`
- Create: `reproducibility/hash_inputs.py`
- Test: `tests/test_data_provenance.py`

**Step 1: Write the failing test**

```python
import json
from pathlib import Path


def test_manifest_has_required_provenance_fields():
    data = json.loads(Path("reproducibility/manifest.json").read_text(encoding="utf-8"))
    required = ["inputs", "checksums", "pipeline_version", "run_parameters"]
    for key in required:
        assert key in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_provenance.py::test_manifest_has_required_provenance_fields -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Generate file checksums for real-data inputs.
- Store source id/version/date/checksum into manifest and database provenance table.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_provenance.py::test_manifest_has_required_provenance_fields -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add database/build_db.py reproducibility/manifest.json reproducibility/hash_inputs.py tests/test_data_provenance.py
git commit -m "feat: add provenance manifest and checksum tracking"
```

### Task 5: Add Query Regression and Release SQL Examples

**Files:**
- Create: `database/queries/query_gene_profile.sql`
- Create: `database/queries/query_disease_network.sql`
- Create: `database/queries/query_celltype_links.sql`
- Create: `tests/test_query_regression.py`
- Create: `docs/query_examples_expected_outputs.md`

**Step 1: Write the failing test**

```python
import sqlite3


def test_gene_query_returns_stable_columns():
    con = sqlite3.connect("release/hpcma_atlas.sqlite")
    cur = con.execute("select * from gene_profile_view limit 1")
    cols = [d[0] for d in cur.description]
    assert cols == ["gene", "disease", "celltype", "evidence_type", "source_id"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_query_regression.py::test_gene_query_returns_stable_columns -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Add stable views and SQL templates.
- Document expected columns and one fixed-output snapshot.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_query_regression.py::test_gene_query_returns_stable_columns -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add database/queries/query_gene_profile.sql database/queries/query_disease_network.sql database/queries/query_celltype_links.sql tests/test_query_regression.py docs/query_examples_expected_outputs.md
git commit -m "test: add query regression suite and stable SQL examples"
```

### Task 6: Make CI Fail-Closed and Add DB Validation Job

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `scripts/ci_validate_database.sh`
- Test: `tests/test_ci_contract.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_ci_has_no_fail_open_or_true():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "|| true" not in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ci_contract.py::test_ci_has_no_fail_open_or_true -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Remove fail-open `|| true` from test/lint/typecheck.
- Add DB build + QC + query regression steps.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ci_contract.py::test_ci_has_no_fail_open_or_true -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add .github/workflows/ci.yml scripts/ci_validate_database.sh tests/test_ci_contract.py
git commit -m "ci: enforce fail-closed checks and database validation"
```

### Task 7: Reframe API as Supplementary Demonstration

**Files:**
- Modify: `src/inference/api_server.py`
- Modify: `tests/test_api.py`
- Create: `docs/supplementary_api_notice.md`
- Test: `tests/test_api_disclaimer.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_api_has_demo_only_disclaimer():
    text = Path("src/inference/api_server.py").read_text(encoding="utf-8").lower()
    assert "demonstration only" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_disclaimer.py::test_api_has_demo_only_disclaimer -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Add explicit demo-only disclaimer in API module and docs.
- Remove production clinical wording from API endpoint docs.
- Adjust API tests to verify disclaimer and schema consistency only.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_disclaimer.py::test_api_has_demo_only_disclaimer -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/inference/api_server.py tests/test_api.py docs/supplementary_api_notice.md tests/test_api_disclaimer.py
git commit -m "docs(api): mark inference API as supplementary demonstration"
```

### Task 8: Manuscript Traceability Package

**Files:**
- Create: `docs/manuscript_claim_map.md`
- Create: `docs/reviewer_checklist.md`
- Create: `docs/data_provenance_table.md`
- Create: `reproducibility/run_all.sh`
- Test: `tests/test_claim_traceability.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_claim_map_has_table_and_script_references():
    text = Path("docs/manuscript_claim_map.md").read_text(encoding="utf-8")
    assert "Claim ID" in text
    assert "Evidence Table" in text
    assert "Generating Script" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_claim_traceability.py::test_claim_map_has_table_and_script_references -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Build claim-to-evidence mapping template.
- Add reviewer reproducibility checklist.
- Add one-command rebuild script and usage notes.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_claim_traceability.py::test_claim_map_has_table_and_script_references -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/manuscript_claim_map.md docs/reviewer_checklist.md docs/data_provenance_table.md reproducibility/run_all.sh tests/test_claim_traceability.py
git commit -m "docs: add manuscript claim traceability and reviewer reproducibility pack"
```

### Task 9: Final Verification Gate

**Files:**
- Modify: `docs/plans/2026-02-14-hpcma-sci-resource-implementation.md` (mark completed checks)
- Test: full suite in `tests/`

**Step 1: Run format and lint checks**

Run: `black --check src tests scripts database reproducibility`  
Expected: PASS.

**Step 2: Run unit and regression tests**

Run: `pytest tests -v`  
Expected: PASS with no fail-open bypass.

**Step 3: Run database build and QC checks**

Run: `python database/build_db.py && python reproducibility/hash_inputs.py`  
Expected: DB artifact and manifest generated without integrity violations.

**Step 4: Run reproducibility script**

Run: `bash reproducibility/run_all.sh`  
Expected: deterministic rebuild of release artifacts and summary tables.

**Step 5: Commit final verification metadata**

```bash
git add docs/plans/2026-02-14-hpcma-sci-resource-implementation.md reproducibility/manifest.json release/
git commit -m "chore: finalize SCI resource readiness verification"
```

---

## Execution Status (2026-02-14)

- Completed: Task 1, Task 2, Task 3, Task 4, Task 5, Task 6, Task 7, Task 8.
- Partial: Task 9 (verification executed with environment constraints).

### Verification Evidence Captured

- Reproducibility pipeline: `bash reproducibility/run_all.sh` passes.
- Database contract: `bash scripts/ci_validate_database.sh` passes.
- Added test contracts execute via Python import runner for:
  - `tests/test_docs_claim_boundaries.py`
  - `tests/test_output_channeling.py`
  - `tests/test_schema_integrity.py`
  - `tests/test_data_provenance.py`
  - `tests/test_query_regression.py`
  - `tests/test_ci_contract.py`
  - `tests/test_api_disclaimer.py`
  - `tests/test_claim_traceability.py`

### Known Environment Constraint

- `python3 -m pytest tests -q` fails in this environment because `pytest` is not installed and network access prevents dependency installation.
