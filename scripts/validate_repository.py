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


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def url_path(url: str) -> str:
    return urlsplit(url).path


def is_http_url(value: str) -> bool:
    parts = urlsplit(value)
    return parts.scheme in {"http", "https"} or value.startswith("//")


def resolve_lite_asset_path(lite_dir: Path, asset_url: str) -> Path:
    asset_path = Path(url_path(asset_url))
    if asset_path.is_absolute():
        return lite_dir / asset_path.as_posix().lstrip("/")
    return (lite_dir / asset_path).resolve()


def lock_package_entry(lock_data: dict, package_name: str) -> dict:
    normalized_target = normalize_package_name(package_name)
    packages = lock_data.get("packages", {})
    for package_key, package_data in packages.items():
        candidate = package_data if isinstance(package_data, dict) else {}
        candidate_name = normalize_package_name(candidate.get("name", package_key))
        if candidate_name == normalized_target:
            return candidate
    raise AssertionError(f"Missing required Pyodide package metadata for {package_name}")


def discover_local_piplite_index_manifests(lite_dir: Path, piplite_urls: list[object]) -> list[Path]:
    candidate_index_paths: list[Path] = []
    for piplite_url in piplite_urls:
        raw_value = str(piplite_url)
        if not raw_value or is_http_url(raw_value):
            continue
        resolved_path = resolve_lite_asset_path(lite_dir, raw_value)
        if resolved_path.suffix.lower() == ".json":
            candidate_index_paths.append(resolved_path)
    candidate_index_paths.extend(sorted((lite_dir / "pypi").glob("*.json")))
    return list(dict.fromkeys(candidate_index_paths))


def load_local_piplite_index_manifest(path: Path) -> dict:
    index_data = load_json(path)
    assert isinstance(index_data, dict), f"Invalid local piplite index manifest structure in {path}: expected JSON object"
    return index_data


def comm_releases_from_index(index_data: dict) -> list[dict]:
    package_data = index_data.get(COMM_PACKAGE_NAME, {})
    if not isinstance(package_data, dict):
        return []
    releases = package_data.get("releases", {})
    if not isinstance(releases, dict):
        return []
    version_releases = releases.get(COMM_VERSION, [])
    if not isinstance(version_releases, list):
        return []
    return [release for release in version_releases if isinstance(release, dict)]


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
    for piplite_url in piplite_urls:
        assert not is_http_url(str(piplite_url)), f"Built site should expose only local piplite index URLs in {built_config_path}: {piplite_url}"

    bundled_pyodide_dir = lite_dir / "static" / "pyodide"
    bundled_runtime_lock = bundled_pyodide_dir / "pyodide-lock.json"
    assert (bundled_pyodide_dir / "pyodide.mjs").exists(), f"Missing bundled Pyodide runtime in {bundled_pyodide_dir}"
    assert bundled_runtime_lock.exists(), f"Missing bundled Pyodide metadata lockfile in {bundled_runtime_lock}"
    bundled_runtime_packages = package_names_from_lock(bundled_runtime_lock)
    missing_runtime_packages = {normalize_package_name(package) for package in REQUIRED_PYODIDE_PACKAGES} - bundled_runtime_packages
    assert not missing_runtime_packages, f"Missing required Pyodide packages in {bundled_runtime_lock}: {sorted(missing_runtime_packages)}"
    bundled_runtime_lock_data = load_json(bundled_runtime_lock)
    for package_name in REQUIRED_PYODIDE_PACKAGES:
        package_entry = lock_package_entry(bundled_runtime_lock_data, package_name)
        file_name = str(package_entry.get("file_name", ""))
        assert file_name, f"Missing file_name for required Pyodide package in {bundled_runtime_lock}: {package_name}"
        assert not is_http_url(file_name), f"Required Pyodide package should be served locally in {bundled_runtime_lock}: {package_name} -> {file_name}"

    wheel_paths = sorted(lite_dir.rglob(COMM_WHEEL_FILENAME))
    assert wheel_paths, f"Missing bundled comm wheel in built JupyterLite output: {COMM_WHEEL_FILENAME}"

    candidate_index_paths = discover_local_piplite_index_manifests(lite_dir, piplite_urls)
    assert candidate_index_paths, f"Missing local piplite index manifest in built JupyterLite output under {lite_dir / 'pypi'}"

    for index_path in candidate_index_paths:
        assert index_path.exists(), f"Built site references missing local piplite index manifest: {index_path}"

    comm_reference_found = False
    for index_path in candidate_index_paths:
        index_data = load_local_piplite_index_manifest(index_path)
        comm_releases = comm_releases_from_index(index_data)
        for release in comm_releases:
            if release.get("filename") != COMM_WHEEL_FILENAME:
                continue
            release_url = str(release.get("url", ""))
            assert release_url, f"Missing URL for bundled comm wheel metadata in {index_path}"
            assert not is_http_url(release_url), f"Bundled comm wheel metadata should use local URL in {index_path}: {release_url}"
            release_path = Path(url_path(release_url))
            release_candidates = {
                (index_path.parent / release_path).resolve(),
                resolve_lite_asset_path(lite_dir, release_url),
            }
            assert any(candidate.exists() for candidate in release_candidates), (
                f"Bundled comm wheel metadata references missing local file from {index_path}: {release_url}"
            )
            comm_reference_found = True
            break
        if comm_reference_found:
            break

    assert comm_reference_found, (
        f"Missing bundled comm wheel metadata in local piplite index manifest(s): "
        f"{[path.as_posix() for path in candidate_index_paths]}"
    )


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
