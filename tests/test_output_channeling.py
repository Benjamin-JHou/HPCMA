from pathlib import Path


def test_step_scripts_expose_output_mode_flag():
    scripts = [
        "scripts/step2_genetic_architecture.py",
        "scripts/step3_causal_gene_prioritization.py",
        "scripts/step5_multimodal_prediction.py",
        "scripts/step7_validation.py",
    ]
    for script in scripts:
        text = Path(script).read_text(encoding="utf-8")
        assert "--output-mode" in text or "OUTPUT_MODE" in text
