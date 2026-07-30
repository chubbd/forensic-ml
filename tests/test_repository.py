from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def test_notebook_validation_requires_python_dateutil(tmp_path) -> None:
    notebook_path = tmp_path / "missing_dateutil.ipynb"
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
                            "import piplite\n",
                            'await piplite.install(["pandas", "scikit-learn"])\n',
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
        assert "python-dateutil" in str(exc)
    else:
        raise AssertionError("validate_notebook should require python-dateutil")


def test_notebook_validation_accepts_python_dateutil(tmp_path) -> None:
    notebook_path = tmp_path / "with_dateutil.ipynb"
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
                            "import piplite\n",
                            'await piplite.install(["python-dateutil", "pandas", "scikit-learn"])\n',
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
