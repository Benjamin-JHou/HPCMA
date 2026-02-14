#!/usr/bin/env python3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reproducibility" / "manifest.json"

INPUT_FILES = [
    ROOT / "atlas_resource" / "hypertension_atlas_master_table.csv",
    ROOT / "atlas_resource" / "prioritized_causal_genes.csv",
    ROOT / "atlas_resource" / "gene_disease_celltype_annotation.csv",
    ROOT / "atlas_resource" / "clinical_translation_table.csv",
    ROOT / "atlas_resource" / "mechanism_axis_clusters.csv",
    ROOT / "atlas_resource" / "multilayer_network_edges.csv",
]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checksums = {}
    inputs = []
    for path in INPUT_FILES:
        if not path.exists():
            continue
        rel = str(path.relative_to(ROOT))
        inputs.append(rel)
        checksums[rel] = sha256sum(path)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0.0",
        "inputs": inputs,
        "checksums": checksums,
        "run_parameters": {
            "output_mode": "real_data",
            "database_builder": "database/build_db.py",
            "schema": "database/schema.sql",
        },
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
