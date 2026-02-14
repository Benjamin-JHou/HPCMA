# Reviewer Reproducibility Checklist

1. Build the database artifact: `python3 database/build_db.py`
2. Generate provenance manifest: `python3 reproducibility/hash_inputs.py`
3. Validate SQL query contract: `bash scripts/ci_validate_database.sh`
4. Run traceability checks: review `docs/manuscript_claim_map.md`
5. Confirm API is supplementary only: `docs/supplementary_api_notice.md`
