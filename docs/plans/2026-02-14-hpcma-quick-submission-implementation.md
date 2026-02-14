# HPCMA Quick Submission Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a fast, reviewer-defensible resource-paper package by aligning code/tests/manuscript figures with auditable evidence.

**Architecture:** Execute in the existing isolated worktree branch, harden test execution environment, merge into main, then refactor manuscript methods and figure mapping to match database-backed reproducibility artifacts.

**Tech Stack:** Git, Python (local env with pytest), SQLite, Markdown, matplotlib/graph rendering.

---

### Task 1: Ensure Pytest Is Runnable and Passing

**Files:**
- Modify: `.github/workflows/ci.yml` (if env assumptions need correction)
- Modify: `tests/*` (only if test collection/runtime errors appear)

**Step 1: Discover usable python/pytest environment**
Run: `which -a python3 python pytest && conda env list || true`
Expected: Identify at least one interpreter with pytest.

**Step 2: Run full test suite**
Run: `python -m pytest tests -q` (using discovered interpreter)
Expected: PASS.

**Step 3: Fix failures minimally**
If failing, patch failing tests/code and rerun.

**Step 4: Verify pass status**
Run: same pytest command again.
Expected: PASS.

### Task 2: Merge Feature Branch Into Main

**Files:**
- No new files expected.

**Step 1: Verify branches are clean**
Run: `git status --short` on both worktree and main repo.
Expected: clean.

**Step 2: Merge**
Run on main repo: `git merge --no-ff feature/sci-resource-readiness`
Expected: merge commit created or fast-forward completed.

**Step 3: Verify merge content**
Run: `git log --oneline -n 20`
Expected: feature commits present in main history.

### Task 3: Rewrite Methods for Resource Evidence Consistency

**Files:**
- Modify: `paper/methods/README.md`

**Step 1: Add resource-paper method sections**
Include:
- data snapshot/provenance references
- schema/database build workflow
- query contract validation
- reproducibility manifest generation
- supplementary API boundary

**Step 2: Remove unsupported full-empirical claims**
Strip language implying full rerun of all toolchains if not actually executed.

**Step 3: Verify terminology boundaries**
Run: `rg -n "production-ready|clinical deployment|deployment-ready" paper/methods/README.md -i`
Expected: no risky deployment claims.

### Task 4: Update Paper Figure Mapping and Narrative

**Files:**
- Modify: `paper/README.md`

**Step 1: Reclassify figures**
Primary: atlas/network/resource/reproducibility figures.
Supplementary: prediction-model-centric figures.

**Step 2: Ensure text reflects resource narrative**
Adjust figure captions/index language accordingly.

### Task 5: Generate Reproducibility Flow Figure

**Files:**
- Create/Modify: `figures/resource_reproducibility_flow.png`
- Create/Modify: `paper/figures/resource_reproducibility_flow.png`
- Optional source: `scripts/generate_reproducibility_flow_figure.py`

**Step 1: Generate figure from script**
Run script and save PNG.

**Step 2: Copy/sync to paper figures**
Ensure both locations contain same updated figure.

**Step 3: Verify references**
Update `paper/README.md` figure table.

### Task 6: Final Verification and Cleanup

**Files:**
- Modify: `docs/plans/2026-02-14-hpcma-quick-submission-implementation.md` (status section)

**Step 1: Re-run tests**
Run: `python -m pytest tests -q`
Expected: PASS.

**Step 2: Re-run reproducibility checks**
Run: `bash reproducibility/run_all.sh`
Expected: PASS.

**Step 3: Manuscript consistency grep**
Run: `rg -n "production-ready|clinical deployment|deployment-ready" README.md DEPLOYMENT_SUMMARY.md paper/README.md paper/methods/README.md -i`
Expected: no unsupported over-claim lines in target narrative files.

**Step 4: Commit in logical chunks**
Use separate commits for tests/env fixes, merge, methods rewrite, figure updates.
