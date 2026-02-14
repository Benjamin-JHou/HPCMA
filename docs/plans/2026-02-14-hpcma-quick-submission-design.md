# HPCMA Quick Submission Design (Resource-Paper Track)

**Date:** 2026-02-14
**Scope:** Fast submission-hardening path aligned with Bioinformatics-style resource manuscript expectations.

## Objective

Deliver a reviewer-defensible package by prioritizing evidence-consistent manuscript language, reproducibility artifacts, and stable software contracts, without full end-to-end re-execution of simulated analysis steps.

## Confirmed Boundary

- In scope:
  - Make `pytest` runnable and pass repository tests in an available local environment.
  - Merge `feature/sci-resource-readiness` into `main`.
  - Rewrite methods text to match auditable resource outputs and constraints.
  - Update/synchronize manuscript figure set and add reproducibility-flow figure.
- Out of scope:
  - Full replacement and rerun of all simulated steps 2-7 with new raw-data pipelines.

## Manuscript Positioning

- Primary claim: reproducible resource and database-backed atlas release.
- Supplementary claim: API exists for demonstration only.
- Explicitly avoid unsupported deployment/clinical-operational claims.

## Methods Rewrite Strategy

`paper/methods/README.md` will be rewritten to contain:
- Data sources snapshot and table-level provenance framing.
- Resource assembly and schema constraints.
- Database build process and query contract checks.
- Reproducibility manifest and checksum workflow.
- Clear scope statement that predictive API and model-serving outputs are supplementary.

## Figure Strategy

- Keep resource-core figures as primary.
- Move predictive/clinical deployment-oriented figures to supplementary framing.
- Add `resource_reproducibility_flow.png` illustrating input -> schema -> DB -> query -> manifest chain.
- Keep `paper/figures` synchronized with manuscript figure index and root `figures` artifacts where needed.

## Success Criteria

1. `pytest` runs and passes in a local available environment.
2. `main` contains the feature branch changes with clean merge history.
3. Methods and figure narrative contain no evidence drift against implemented artifacts.
4. Reproducibility flow figure is generated and referenced in paper docs.
5. Final grep checks confirm removal of over-claim terms in manuscript-facing docs.
