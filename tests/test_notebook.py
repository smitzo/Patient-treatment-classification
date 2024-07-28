import json
from pathlib import Path


def test_companion_notebook_cells_execute_in_order(tmp_path, monkeypatch):
    notebook = json.loads(Path("classification_lab.ipynb").read_text())
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            exec(compile("".join(cell["source"]), "classification_lab.ipynb", "exec"), namespace)
    assert (tmp_path / "reports/notebook-iris.json").exists()
