from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_repository import validate_built_site, validate_jupyterlite_config
from scripts.validate_notebooks import validate_notebook


def test_data_generator_creates_expected_schema(tmp_path) -> None:
    training_path = tmp_path / "training.csv"
    mystery_path = tmp_path / "mystery.csv"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_synthetic_data.py"),
            "--training-out",
            str(training_path),
            "--mystery-out",
            str(mystery_path),
            "--rows-per-class",
            "3",
            "--mystery-count",
            "2",
            "--seed",
            "99",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote" in completed.stdout
    with training_path.open("r", encoding="utf-8", newline="") as handle:
        training_rows = list(csv.DictReader(handle))
    with mystery_path.open("r", encoding="utf-8", newline="") as handle:
        mystery_rows = list(csv.DictReader(handle))

    assert len(training_rows) == 12
    assert len(mystery_rows) == 2
    assert "class_label" in training_rows[0]
    assert "class_label" not in mystery_rows[0]


def test_repository_validation_script_passes() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_repository.py"), "--root", str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Repository validation passed" in completed.stdout


def test_notebook_validation_script_passes() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_notebooks.py"), "--root", str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Notebook validation passed" in completed.stdout


def test_notebook_validation_requires_version_diagnostics(tmp_path) -> None:
    notebook_path = tmp_path / "missing_diagnostics.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# Example\n"]},
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": [
                            "import dateutil\n",
                            "import pandas as pd\n",
                            "import sklearn\n",
                        ],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_notebook(notebook_path)
    except AssertionError as exc:
        assert "dateutil.__version__" in str(exc)
    else:
        raise AssertionError("validate_notebook should require package version diagnostics")


def test_notebook_validation_accepts_preloaded_package_diagnostics(tmp_path) -> None:
    notebook_path = tmp_path / "with_diagnostics.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# Example\n"]},
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": [
                            "import dateutil\n",
                            "import pandas as pd\n",
                            "import sklearn\n",
                            'print(f\"python-dateutil {dateutil.__version__}\")\n',
                            'print(f\"pandas {pd.__version__}\")\n',
                            'print(f\"scikit-learn {sklearn.__version__}\")\n',
                        ],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    validate_notebook(notebook_path)


def test_jupyterlite_config_requires_preloaded_pyodide_packages(tmp_path) -> None:
    config_path = tmp_path / "jupyter-lite.json"
    config_path.write_text(
        json.dumps(
            {
                "PyodideAddon": {
                    "pyodide_url": "https://github.com/pyodide/pyodide/releases/download/314.0.1/pyodide-314.0.1.tar.bz2"
                },
                "PipliteAddon": {
                    "piplite_urls": [
                        "https://files.pythonhosted.org/packages/60/97/891a0971e1e4a8c5d2b20bbe0e524dc04548d2307fee33cdeba148fd4fc7/comm-0.2.3-py3-none-any.whl"
                    ],
                },
                "jupyter-config-data": {
                    "litePluginSettings": {
                        "@jupyterlite/pyodide-kernel-extension:kernel": {
                            "pyodideUrl": "./static/pyodide/pyodide.mjs",
                            "disablePyPIFallback": True,
                            "loadPyodideOptions": {
                                "packages": ["pandas", "scikit-learn"],
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_jupyterlite_config(config_path)
    except AssertionError as exc:
        assert "Unexpected preloaded Pyodide packages" in str(exc)
    else:
        raise AssertionError("validate_jupyterlite_config should require the preloaded Pyodide package set")


def test_jupyterlite_config_rejects_runtime_cdn_override(tmp_path) -> None:
    config_path = tmp_path / "jupyter-lite.json"
    config_path.write_text(
        json.dumps(
            {
                "PyodideAddon": {
                    "pyodide_url": "https://github.com/pyodide/pyodide/releases/download/314.0.1/pyodide-314.0.1.tar.bz2"
                },
                "PipliteAddon": {
                    "piplite_urls": [
                        "https://files.pythonhosted.org/packages/60/97/891a0971e1e4a8c5d2b20bbe0e524dc04548d2307fee33cdeba148fd4fc7/comm-0.2.3-py3-none-any.whl"
                    ],
                },
                "jupyter-config-data": {
                    "litePluginSettings": {
                        "@jupyterlite/pyodide-kernel-extension:kernel": {
                            "pyodideUrl": "https://cdn.jsdelivr.net/pyodide/v314.0.1/full/pyodide.mjs",
                            "disablePyPIFallback": True,
                            "loadPyodideOptions": {
                                "packages": ["python-dateutil", "pandas", "scikit-learn"],
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_jupyterlite_config(config_path)
    except AssertionError as exc:
        assert "should point at the bundled Pyodide runtime" in str(exc)
    else:
        raise AssertionError("validate_jupyterlite_config should reject remote Pyodide runtime overrides")


def test_built_site_validation_requires_local_pyodide_runtime_and_lock(tmp_path) -> None:
    site_dir = tmp_path / "site"
    lite_dir = site_dir / "lite"
    (lite_dir / "lab").mkdir(parents=True)
    (lite_dir / "static" / "pyodide").mkdir(parents=True)
    (lite_dir / "pypi").mkdir(parents=True)
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    (site_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (lite_dir / "lab" / "index.html").write_text("<html></html>", encoding="utf-8")
    (lite_dir / "static" / "pyodide" / "pyodide.mjs").write_text("// runtime", encoding="utf-8")

    lock_packages = {
        package_name: {"name": package_name, "version": "1.0.0", "file_name": f"{package_name}.whl", "sha256": package_name}
        for package_name in [
            "python-dateutil",
            "pandas",
            "scikit-learn",
            "comm",
        ]
    }
    lock_text = json.dumps({"packages": lock_packages})
    (lite_dir / "static" / "pyodide" / "pyodide-lock.json").write_text(lock_text, encoding="utf-8")
    (lite_dir / "pypi" / "all.json").write_text('{"comm": [{"filename": "comm-0.2.3-py3-none-any.whl"}]}', encoding="utf-8")
    (lite_dir / "pypi" / "comm-0.2.3-py3-none-any.whl").write_text("", encoding="utf-8")
    (lite_dir / "jupyter-lite.json").write_text(
        json.dumps(
            {
                "jupyter-config-data": {
                    "litePluginSettings": {
                        "@jupyterlite/pyodide-kernel-extension:kernel": {
                            "pyodideUrl": "./static/pyodide/pyodide.mjs",
                            "disablePyPIFallback": True,
                            "pipliteUrls": ["./pypi/all.json"],
                            "loadPyodideOptions": {
                                "packages": [
                                    "python-dateutil",
                                    "pandas",
                                    "scikit-learn",
                                ],
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    validate_built_site(site_dir)
