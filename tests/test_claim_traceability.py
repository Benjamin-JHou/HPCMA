from pathlib import Path


def test_claim_map_has_table_and_script_references():
    text = Path("docs/manuscript_claim_map.md").read_text(encoding="utf-8")
    assert "Claim ID" in text
    assert "Evidence Table" in text
    assert "Generating Script" in text
