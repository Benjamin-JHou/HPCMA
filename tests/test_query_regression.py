import sqlite3
from pathlib import Path


def test_gene_query_returns_stable_columns():
    assert Path("database/queries/query_gene_profile.sql").exists()
    con = sqlite3.connect("release/hpcma_atlas.sqlite")
    try:
        cur = con.execute("select * from gene_profile_view limit 1")
        cols = [d[0] for d in cur.description]
    finally:
        con.close()
    assert cols == ["gene", "disease", "celltype", "evidence_type", "source_id"]
