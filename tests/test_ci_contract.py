from pathlib import Path


def test_ci_has_no_fail_open_or_true():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "|| true" not in text
