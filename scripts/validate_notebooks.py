from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_NOTEBOOKS = [
    "content/notebooks/forensic_classification_challenge.ipynb",
    "content/notebooks/package_diagnostics.ipynb",
]
REQUIRED_PACKAGE_IMPORTS = [
    "import dateutil",
    "import pandas as pd",
    "import sklearn",
]
REQUIRED_PYODIDE_RUNTIME_PACKAGES = [
    "python-dateutil",
    "pandas",
    "scikit-learn",
]
REQUIRED_BOOTSTRAP_SUCCESS_SNIPPETS = [
    "Loaded Pyodide packages:",
    "python-dateutil",
    "pandas",
    "scikit-learn",
]
REQUIRED_VERSION_SNIPPETS = [
    "dateutil.__version__",
    "pd.__version__",
    "sklearn.__version__",
]


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_notebook(path: Path) -> None:
    notebook = load_notebook(path)
    assert notebook["nbformat"] >= 4, f"Unexpected nbformat for {path}"
    cells = notebook.get("cells", [])
    assert cells, f"Notebook has no cells: {path}"
    assert any(cell.get("cell_type") == "markdown" for cell in cells), f"Notebook needs markdown cells: {path}"
    assert any(cell.get("cell_type") == "code" for cell in cells), f"Notebook needs code cells: {path}"
    source_text = "\n".join("".join(cell.get("source", [])) for cell in cells)
    assert "piplite.install" not in source_text, f"Notebook should rely on preloaded Pyodide packages instead of piplite.install: {path}"
    assert "pyodide_js.loadPackage(" not in source_text, (
        f"Notebook must use the Python-side Pyodide loader, not pyodide_js.loadPackage, in {path}"
    )

    bootstrap_cell_index = None
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        cell_source = "".join(cell.get("source", []))
        if "import pyodide" in cell_source and "await pyodide.load_package(" in cell_source:
            bootstrap_cell_index = index
            for package_name in REQUIRED_PYODIDE_RUNTIME_PACKAGES:
                assert package_name in cell_source, f"Pyodide bootstrap is missing package '{package_name}' in {path}"
            for expected_snippet in REQUIRED_BOOTSTRAP_SUCCESS_SNIPPETS:
                assert expected_snippet in cell_source, (
                    f"Pyodide bootstrap must print a visible success message after loading in {path}"
                )
            break
    assert bootstrap_cell_index is not None, f"Notebook must include a Pyodide package bootstrap cell before imports: {path}"

    first_required_import_cell_index = None
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        cell_source = "".join(cell.get("source", []))
        if any(expected_import in cell_source for expected_import in REQUIRED_PACKAGE_IMPORTS):
            first_required_import_cell_index = index
            break
    assert first_required_import_cell_index is not None, f"Notebook is missing required import cells: {path}"
    assert bootstrap_cell_index < first_required_import_cell_index, (
        f"Pyodide package bootstrap cell must run before dateutil/pandas/sklearn imports in {path}"
    )

    for expected_snippet in [*REQUIRED_PACKAGE_IMPORTS, *REQUIRED_VERSION_SNIPPETS]:
        assert expected_snippet in source_text, f"Missing '{expected_snippet}' in {path}"
    if path.name == "forensic_classification_challenge.ipynb":
        for expected_snippet in ["MODEL_CANDIDATES", "class_label", "mystery_samples.csv", "training_samples.csv"]:
            assert expected_snippet in source_text, f"Missing '{expected_snippet}' in {path}"
    if path.name == "package_diagnostics.ipynb":
        for expected_snippet in ["platform", "diagnostics", "training_samples.csv"]:
            assert expected_snippet in source_text, f"Missing '{expected_snippet}' in {path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate repository notebooks.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    for relative_path in REQUIRED_NOTEBOOKS:
        validate_notebook(args.root / relative_path)

    print("Notebook validation passed")


if __name__ == "__main__":
    main()
