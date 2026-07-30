from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_NOTEBOOKS = [
    "content/notebooks/forensic_classification_challenge.ipynb",
    "content/notebooks/package_diagnostics.ipynb",
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
    assert "piplite.install" in source_text, f"Notebook should install browser-side packages: {path}"
    if path.name == "forensic_classification_challenge.ipynb":
        for expected_snippet in ["MODEL_CANDIDATES", "class_label", "mystery_samples.csv", "training_samples.csv"]:
            assert expected_snippet in source_text, f"Missing '{expected_snippet}' in {path}"
    if path.name == "package_diagnostics.ipynb":
        for expected_snippet in ["platform", "sklearn", "training_samples.csv"]:
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
