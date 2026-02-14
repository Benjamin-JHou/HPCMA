#!/usr/bin/env python3
import csv
import hashlib
import sqlite3
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "database" / "schema.sql"
INPUT_MASTER = ROOT / "atlas_resource" / "hypertension_atlas_master_table.csv"
OUTPUT_DB = ROOT / "release" / "hpcma_atlas.sqlite"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> None:
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    conn = sqlite3.connect(OUTPUT_DB)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))

    source_id = "atlas_master_v1"
    checksum = sha256sum(INPUT_MASTER)
    conn.execute(
        """
        INSERT OR REPLACE INTO provenance_sources
        (source_id, source_file, source_version, download_date, checksum_sha256)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            source_id,
            str(INPUT_MASTER.relative_to(ROOT)),
            "v1.0.0",
            str(date.today()),
            checksum,
        ),
    )

    gene_map = {}
    disease_map = {}
    celltype_map = {}

    with INPUT_MASTER.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def read_value(row: dict, *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    for row in rows:
        gene = read_value(row, "gene", "Gene")
        disease = read_value(row, "disease", "Disease")
        celltype = read_value(row, "cell_type", "CellType")

        if gene and gene not in gene_map:
            cur = conn.execute("INSERT INTO genes(symbol) VALUES (?)", (gene,))
            gene_map[gene] = cur.lastrowid

        if disease and disease not in disease_map:
            cur = conn.execute("INSERT INTO diseases(disease_name) VALUES (?)", (disease,))
            disease_map[disease] = cur.lastrowid

        if celltype and celltype not in celltype_map:
            cur = conn.execute("INSERT INTO cell_types(cell_type_name) VALUES (?)", (celltype,))
            celltype_map[celltype] = cur.lastrowid

    for row in rows:
        gene = read_value(row, "gene", "Gene")
        disease = read_value(row, "disease", "Disease")
        celltype = read_value(row, "cell_type", "CellType")

        if not gene or not disease:
            continue

        gene_id = gene_map[gene]
        disease_id = disease_map[disease]
        cell_type_id = celltype_map.get(celltype)

        def to_float(value):
            try:
                return float(value) if value not in (None, "") else None
            except ValueError:
                return None

        conn.execute(
            """
            INSERT OR IGNORE INTO atlas_edges(
                gene_id, disease_id, cell_type_id, mr_beta, pph4,
                mechanism_axis, clinical_intervention, priority_score,
                total_influence, evidence_type, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gene_id,
                disease_id,
                cell_type_id,
                to_float(read_value(row, "mr_beta", "MR_Beta")),
                to_float(read_value(row, "pph4", "PPH4")),
                read_value(row, "mechanism_axis", "Mechanism_Axis") or None,
                read_value(row, "clinical_intervention", "Clinical_Intervention") or None,
                to_float(read_value(row, "priority_score", "Priority_Score")),
                to_float(read_value(row, "total_influence", "Total_Influence")),
                "MR+coloc+atlas",
                source_id,
            ),
        )

    conn.commit()

    counts = {
        "genes": conn.execute("SELECT COUNT(*) FROM genes").fetchone()[0],
        "diseases": conn.execute("SELECT COUNT(*) FROM diseases").fetchone()[0],
        "cell_types": conn.execute("SELECT COUNT(*) FROM cell_types").fetchone()[0],
        "atlas_edges": conn.execute("SELECT COUNT(*) FROM atlas_edges").fetchone()[0],
    }
    conn.close()

    print(f"Built: {OUTPUT_DB}")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    build()
