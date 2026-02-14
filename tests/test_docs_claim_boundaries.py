from pathlib import Path


def test_no_production_ready_claims_in_main_docs():
    targets = [
        Path("README.md"),
        Path("DEPLOYMENT_SUMMARY.md"),
        Path("atlas_resource/README.md"),
    ]
    banned = ["production-ready", "clinical deployment ready", "deployment-ready"]
    for path in targets:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(term in text for term in banned), f"banned claim found in {path}"
