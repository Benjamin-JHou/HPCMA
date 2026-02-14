# Project Cleanup And Schema Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean redundant artifacts from repository workspace and unify canonical atlas table fields globally to a consistent snake_case contract.

**Architecture:** Keep `atlas_resource/` as the single source of truth for public atlas tables. Remove untracked duplicate outputs and cache artifacts that add noise. Normalize headers for canonical atlas CSVs, then make ingestion code and key docs/scripts consume the new schema.

**Tech Stack:** Git, Bash, Python stdlib csv/pathlib, SQLite build script, pytest.

---

### Task 1: Workspace Cleanup (safe + requested)

**Files/Paths:**
- Delete: `.pytest_cache/`, `**/__pycache__/`, `*.DS_Store`
- Delete untracked duplicate outputs in `results/` and root `figures/` (except tracked `figures/resource_reproducibility_flow.png`)
- Keep: `data/` raw data directory

**Step 1:** Collect untracked files from `git status --short`.

**Step 2:** Remove only confirmed redundant categories.

**Step 3:** Re-check `git status --short` and ensure no tracked files were removed unintentionally.

### Task 2: Canonical Atlas Field Normalization

**Files:**
- Modify: `atlas_resource/*.csv` (8 canonical atlas tables)
- Create: `scripts/normalize_atlas_tables.py`

**Step 1:** Convert headers to `lower_snake_case` for all canonical atlas tables.

**Step 2:** Preserve row data and file order.

**Step 3:** Run normalizer script and inspect transformed headers.

### Task 3: Code And Docs Contract Alignment

**Files:**
- Modify: `database/build_db.py`
- Modify: `atlas_resource/README.md`
- Modify key references in docs/scripts from `results/<core_table>.csv` to `atlas_resource/<core_table>.csv` for canonical tables.

**Step 1:** Update `build_db.py` to read normalized headers.

**Step 2:** Keep backward compatibility via alias fallback where practical.

**Step 3:** Update repository-facing documentation and key script references.

### Task 4: Verification

**Commands:**
- `python -m pytest tests -q`
- `bash reproducibility/run_all.sh`
- `git status --short --branch`

**Expected:** tests pass, reproducibility pipeline passes, workspace cleaner with reduced redundant artifacts.
