from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import DEFAULT_FEATURE_COLUMNS, write_datasets

REQUIRED_FILES = [
    ".github/workflows/deploy.yml",
    "README.md",
    "LICENSE",
    "requirements.txt",
    "jupyter-lite.json",
    "jupyter_lite_config.json",
    "pages/index.html",
    "content/notebooks/forensic_classification_challenge.ipynb",
    "content/notebooks/package_diagnostics.ipynb",
    "content/data/training_samples.csv",
    "content/data/mystery_samples.csv",
    "scripts/generate_synthetic_data.py",
    "scripts/validate_repository.py",
    "scripts/validate_notebooks.py",
    "tests/test_repository.py",
]
PYODIDE_KERNEL_PLUGIN = "@jupyterlite/pyodide-kernel-extension:kernel"
PYODIDE_RUNTIME_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.1/full/pyodide.mjs"
REQUIRED_PYODIDE_PACKAGES = ["python-dateutil", "pandas", "scikit-learn"]


def assert_exists(root: Path) -> None:
    missing = [relative_path for relative_path in REQUIRED_FILES if not (root / relative_path).exists()]
    assert not missing, f"Missing required files: {missing}"


def validate_csv(path: Path, expect_label: bool) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        expected = ["sample_id", *DEFAULT_FEATURE_COLUMNS]
        if expect_label:
            expected = [*expected, "class_label"]
        assert fieldnames == expected, f"Unexpected columns in {path}: {fieldnames}"
        rows = list(reader)
        assert rows, f"CSV is empty: {path}"
        assert all(row["sample_id"] for row in rows), f"Missing sample_id values in {path}"
        if expect_label:
            assert all(row["class_label"] for row in rows), f"Missing class_label values in {path}"


def validate_generated_data(root: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        generated_training = temporary_root / "training.csv"
        generated_mystery = temporary_root / "mystery.csv"
        write_datasets(generated_training, generated_mystery, force=True)
        committed_training = root / "content/data/training_samples.csv"
        committed_mystery = root / "content/data/mystery_samples.csv"
        assert committed_training.read_text(encoding="utf-8") == generated_training.read_text(encoding="utf-8"), "Committed training CSV does not match generator output"
        assert committed_mystery.read_text(encoding="utf-8") == generated_mystery.read_text(encoding="utf-8"), "Committed mystery CSV does not match generator output"


def validate_workflow(root: Path) -> None:
    workflow_text = (root / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    for expected_snippet in ["actions/deploy-pages@v4", "actions/upload-pages-artifact@v3", "jupyter lite build", "pytest -q"]:
        assert expected_snippet in workflow_text, f"Workflow is missing '{expected_snippet}'"


def validate_jupyterlite_config(path: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    jupyter_config = config.get("jupyter-config-data", {})
    plugin_settings = jupyter_config.get("litePluginSettings", {}).get(PYODIDE_KERNEL_PLUGIN, {})

    assert plugin_settings, f"Missing litePluginSettings for {PYODIDE_KERNEL_PLUGIN} in {path}"
    assert plugin_settings.get("pyodideUrl") == PYODIDE_RUNTIME_URL, f"Unexpected Pyodide runtime URL in {path}"
    assert plugin_settings.get("disablePyPIFallback") is True, f"PyPI fallback should be disabled in {path}"

    packages = plugin_settings.get("loadPyodideOptions", {}).get("packages", [])
    assert packages == REQUIRED_PYODIDE_PACKAGES, f"Unexpected preloaded Pyodide packages in {path}: {packages}"


def validate_pages(root: Path) -> None:
    landing_page = (root / "pages/index.html").read_text(encoding="utf-8")
    for expected_snippet in ["Launch participant notebook", "lite/lab/index.html?path=notebooks/forensic_classification_challenge.ipynb", "package_diagnostics.ipynb"]:
        assert expected_snippet in landing_page, f"Landing page is missing '{expected_snippet}'"


def validate_built_site(site_dir: Path) -> None:
    assert (site_dir / ".nojekyll").exists(), f"Missing .nojekyll in built site: {site_dir / '.nojekyll'}"
    assert (site_dir / "index.html").exists(), f"Missing landing page in built site: {site_dir / 'index.html'}"
    assert (site_dir / "lite" / "lab" / "index.html").exists(), f"Missing JupyterLite lab app in built site: {site_dir / 'lite' / 'lab' / 'index.html'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate repository structure and generated assets.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--site-dir", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    assert_exists(root)
    validate_csv(root / "content/data/training_samples.csv", expect_label=True)
    validate_csv(root / "content/data/mystery_samples.csv", expect_label=False)
    validate_generated_data(root)
    validate_workflow(root)
    validate_jupyterlite_config(root / "jupyter-lite.json")
    validate_pages(root)
    if args.site_dir:
        validate_built_site((root / args.site_dir).resolve() if not args.site_dir.is_absolute() else args.site_dir)

    print("Repository validation passed")


if __name__ == "__main__":
    main()
