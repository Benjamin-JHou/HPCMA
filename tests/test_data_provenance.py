import json
from pathlib import Path


def test_manifest_has_required_provenance_fields():
    data = json.loads(Path("reproducibility/manifest.json").read_text(encoding="utf-8"))
    required = ["inputs", "checksums", "pipeline_version", "run_parameters"]
    for key in required:
        assert key in data
