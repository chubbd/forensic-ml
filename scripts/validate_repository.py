from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import DEFAULT_FEATURE_COLUMNS, write_datasets

try:
    from jupyterlite_pyodide_kernel.constants import PYODIDE_VERSION as KERNEL_PYODIDE_VERSION
except ImportError:  # pragma: no cover - exercised in CI/runtime with installed deps
    KERNEL_PYODIDE_VERSION = "314.0.1"

REQUIRED_FILES = [
    ".github/workflows/deploy.yml",
    "README.md",
    "LICENSE",
    "requirements.txt",
    "jupyter-lite.json",
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
REQUIRED_PYODIDE_PACKAGES = ["python-dateutil", "pandas", "scikit-learn"]
COMM_WHEEL_URL = "https://files.pythonhosted.org/packages/60/97/891a0971e1e4a8c5d2b20bbe0e524dc04548d2307fee33cdeba148fd4fc7/comm-0.2.3-py3-none-any.whl"
COMM_WHEEL_FILENAME = "comm-0.2.3-py3-none-any.whl"
COMM_PACKAGE_NAME = "comm"
COMM_VERSION = "0.2.3"
PYODIDE_DISTRIBUTION_URL = f"https://github.com/pyodide/pyodide/releases/download/{KERNEL_PYODIDE_VERSION}/pyodide-{KERNEL_PYODIDE_VERSION}.tar.bz2"
EXPECTED_PYODIDE_URL = "./static/pyodide/pyodide.mjs"
EXPECTED_PIPLITE_INDEX_URL = "./pypi/all.json"


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def url_path(url: str) -> str:
    return urlsplit(url).path


def pyodide_kernel_settings(config: dict) -> dict:
    return config.get("jupyter-config-data", {}).get("litePluginSettings", {}).get(PYODIDE_KERNEL_PLUGIN, {})


def package_names_from_lock(path: Path) -> set[str]:
    lock = load_json(path)
    packages = lock.get("packages", {})
    names = {
        normalize_package_name(package.get("name", package_name))
        for package_name, package in packages.items()
    }
    return {name for name in names if name}


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
    for expected_snippet in [
        "actions/checkout@v7",
        "actions/setup-python@v7",
        "actions/configure-pages@v6",
        "actions/upload-pages-artifact@v5",
        "actions/deploy-pages@v5",
        "jupyter lite build",
        "pytest -q",
    ]:
        assert expected_snippet in workflow_text, f"Workflow is missing '{expected_snippet}'"


def validate_jupyterlite_config(path: Path) -> None:
    config = load_json(path)
    plugin_settings = pyodide_kernel_settings(config)

    assert plugin_settings, f"Missing litePluginSettings for {PYODIDE_KERNEL_PLUGIN} in {path}"
    assert plugin_settings.get("pyodideUrl") == EXPECTED_PYODIDE_URL, f"Source config should point at the bundled Pyodide runtime in {path}"
    assert plugin_settings.get("disablePyPIFallback") is True, f"PyPI fallback should be disabled in {path}"

    packages = plugin_settings.get("loadPyodideOptions", {}).get("packages", [])
    assert packages == REQUIRED_PYODIDE_PACKAGES, f"Unexpected preloaded Pyodide packages in {path}: {packages}"

    pyodide_addon = config.get("PyodideAddon", {})
    assert pyodide_addon.get("pyodide_url") == PYODIDE_DISTRIBUTION_URL, f"Unexpected Pyodide distribution URL in {path}"

    piplite_addon = config.get("PipliteAddon", {})
    assert piplite_addon.get("piplite_urls") == [COMM_WHEEL_URL], f"Unexpected PipliteAddon wheel URLs in {path}"


def validate_pages(root: Path) -> None:
    landing_page = (root / "pages/index.html").read_text(encoding="utf-8")
    for expected_snippet in ["Launch participant notebook", "lite/lab/index.html?path=notebooks/forensic_classification_challenge.ipynb", "package_diagnostics.ipynb"]:
        assert expected_snippet in landing_page, f"Landing page is missing '{expected_snippet}'"


def validate_built_site(site_dir: Path) -> None:
    assert (site_dir / ".nojekyll").exists(), f"Missing .nojekyll in built site: {site_dir / '.nojekyll'}"
    assert (site_dir / "index.html").exists(), f"Missing landing page in built site: {site_dir / 'index.html'}"
    assert (site_dir / "lite" / "lab" / "index.html").exists(), f"Missing JupyterLite lab app in built site: {site_dir / 'lite' / 'lab' / 'index.html'}"

    lite_dir = site_dir / "lite"
    built_config_path = lite_dir / "jupyter-lite.json"
    built_config = load_json(built_config_path)
    plugin_settings = pyodide_kernel_settings(built_config)

    assert plugin_settings, f"Missing litePluginSettings for {PYODIDE_KERNEL_PLUGIN} in {built_config_path}"
    built_pyodide_url = plugin_settings.get("pyodideUrl")
    if built_pyodide_url is not None:
        assert url_path(built_pyodide_url) == EXPECTED_PYODIDE_URL, f"Built site should serve a bundled Pyodide runtime from {built_config_path}"
    piplite_urls = plugin_settings.get("pipliteUrls", [])
    assert len(piplite_urls) == 1, f"Built site should expose exactly one local piplite wheel index from {built_config_path}"
    assert url_path(piplite_urls[0]) == EXPECTED_PIPLITE_INDEX_URL, f"Built site should expose the local piplite wheel index from {built_config_path}"

    bundled_pyodide_dir = lite_dir / "static" / "pyodide"
    bundled_runtime_lock = bundled_pyodide_dir / "pyodide-lock.json"
    assert (bundled_pyodide_dir / "pyodide.mjs").exists(), f"Missing bundled Pyodide runtime in {bundled_pyodide_dir}"
    assert bundled_runtime_lock.exists(), f"Missing bundled Pyodide metadata lockfile in {bundled_runtime_lock}"
    bundled_runtime_packages = package_names_from_lock(bundled_runtime_lock)
    missing_runtime_packages = {normalize_package_name(package) for package in REQUIRED_PYODIDE_PACKAGES} - bundled_runtime_packages
    assert not missing_runtime_packages, f"Missing required Pyodide packages in {bundled_runtime_lock}: {sorted(missing_runtime_packages)}"
    bundled_runtime_lock_data = load_json(bundled_runtime_lock)
    for package_name in REQUIRED_PYODIDE_PACKAGES:
        file_name = bundled_runtime_lock_data["packages"][package_name]["file_name"]
        assert not str(file_name).startswith("http"), f"Required Pyodide package should be served locally in {bundled_runtime_lock}: {package_name} -> {file_name}"

    piplite_dir = lite_dir / "pypi"
    piplite_index = piplite_dir / "all.json"
    assert piplite_index.exists(), f"Missing local piplite index in {piplite_index}"
    piplite_index_data = load_json(piplite_index)
    comm_releases = piplite_index_data.get(COMM_PACKAGE_NAME, {}).get("releases", {}).get(COMM_VERSION, [])
    assert any(
        release.get("filename") == COMM_WHEEL_FILENAME and url_path(release.get("url", "")) == f"./{COMM_WHEEL_FILENAME}"
        for release in comm_releases
    ), f"Missing bundled comm wheel metadata from {piplite_index}"
    assert (piplite_dir / COMM_WHEEL_FILENAME).exists(), f"Missing bundled comm wheel in {piplite_dir}"


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
