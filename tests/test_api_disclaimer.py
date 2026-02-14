from pathlib import Path


def test_api_has_demo_only_disclaimer():
    text = Path("src/inference/api_server.py").read_text(encoding="utf-8").lower()
    assert "demonstration only" in text
