#!/usr/bin/env python3
"""Normalize canonical atlas table headers to lower_snake_case."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATLAS_DIR = ROOT / "atlas_resource"

TABLES = [
    "hypertension_atlas_master_table.csv",
    "prioritized_causal_genes.csv",
    "gene_disease_celltype_annotation.csv",
    "clinical_translation_table.csv",
    "mechanism_axis_clusters.csv",
    "ldsc_genetic_correlation_matrix.csv",
    "coloc_results.csv",
    "multilayer_network_edges.csv",
]


def to_snake_case(name: str) -> str:
    s = name.strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    return s


def normalize_table(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"skip(empty): {path}")
            return
        old_fields = list(reader.fieldnames)
        new_fields = [to_snake_case(col) for col in old_fields]

        if len(set(new_fields)) != len(new_fields):
            raise ValueError(f"Header collision in {path}: {new_fields}")

        rows = []
        for row in reader:
            rows.append({new: row.get(old, "") for old, new in zip(old_fields, new_fields)})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"normalized: {path.name}")


def main() -> None:
    for table in TABLES:
        normalize_table(ATLAS_DIR / table)


if __name__ == "__main__":
    main()
