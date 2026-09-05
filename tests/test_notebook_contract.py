import json
import re
from pathlib import Path

NOTEBOOK = Path("notebooks/bleve-pressure-modeling.ipynb")


def test_notebook_is_valid_json_with_professional_metadata() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["metadata"]["kernelspec"]["name"] == "python3"
    assert len(payload["cells"]) == 29

    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_notebook_contains_no_local_identity_or_machine_specific_paths() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in payload["cells"]).lower()

    assert "/users/" not in text
    assert re.search(r"\b\d{8}\b", text) is None
    assert re.search(r"\b[a-z]{4}[_ -]?\d{4}\b", text) is None
