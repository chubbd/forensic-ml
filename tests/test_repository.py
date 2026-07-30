from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
