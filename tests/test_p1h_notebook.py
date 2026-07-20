"""P1-H release test: publication notebook is valid JSON with compilable code cells."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "phase1_hardening_platform_demo.ipynb"


def test_P1H_13_publication_notebook_is_valid_and_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 8
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 4
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell_{index}", "exec")


def test_P1H_14_notebook_exposes_pipeline_reproducibility_and_claim_boundary():
    text = NOTEBOOK.read_text(encoding="utf-8")
    for required in (
        "phase1_hardening.yaml",
        "Phase1Pipeline",
        "receiver-position mismatch",
        "Born",
        "DBIM",
        "CSI",
        "clinical",
    ):
        assert required in text
