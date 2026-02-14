from pathlib import Path


def test_schema_contains_core_constraints():
    schema = Path("database/schema.sql").read_text(encoding="utf-8").lower()
    assert "primary key" in schema
    assert "foreign key" in schema
    assert "unique" in schema
