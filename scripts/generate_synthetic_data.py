from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable

DEFAULT_CLASSES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
DEFAULT_FEATURE_COLUMNS = [f"measurement_{index:02d}" for index in range(1, 7)]
DEFAULT_CLASS_CENTERS = {
    "ALPHA": [1.2, 2.8, 1.7, 3.4, 2.5, 1.1],
    "BRAVO": [4.4, 3.9, 4.8, 3.5, 4.1, 4.9],
    "CHARLIE": [7.5, 6.2, 7.1, 6.7, 7.6, 6.4],
    "DELTA": [9.1, 8.7, 8.5, 9.3, 8.8, 9.0],
}


def _sample_measurements(rng: random.Random, class_name: str) -> list[float]:
    center = DEFAULT_CLASS_CENTERS[class_name]
    return [round(max(0.0, value + rng.gauss(0, 0.22)), 4) for value in center]


def build_training_rows(classes: Iterable[str], rows_per_class: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for class_name in classes:
        for sample_index in range(1, rows_per_class + 1):
            row = {
                "sample_id": f"train-{class_name.lower()}-{sample_index:03d}",
                "class_label": class_name,
            }
            row.update(zip(DEFAULT_FEATURE_COLUMNS, _sample_measurements(rng, class_name)))
            rows.append(row)
    return rows


def build_mystery_rows(classes: list[str], row_count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed + 1_000)
    class_cycle = [classes[(index + 1) % len(classes)] for index in range(row_count)]
    rows: list[dict[str, object]] = []
    for sample_index, class_name in enumerate(class_cycle, start=1):
        row = {"sample_id": f"mystery-{sample_index:03d}"}
        row.update(zip(DEFAULT_FEATURE_COLUMNS, _sample_measurements(rng, class_name)))
        rows.append(row)
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]], force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_datasets(training_out: Path, mystery_out: Path, rows_per_class: int = 24, mystery_count: int = 4, seed: int = 42, force: bool = False) -> None:
    classes = list(DEFAULT_CLASSES)
    training_rows = build_training_rows(classes, rows_per_class, seed)
    mystery_rows = build_mystery_rows(classes, mystery_count, seed)
    write_csv(training_out, ["sample_id", *DEFAULT_FEATURE_COLUMNS, "class_label"], training_rows, force=force)
    write_csv(mystery_out, ["sample_id", *DEFAULT_FEATURE_COLUMNS], mystery_rows, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate generic synthetic forensic classification CSV files.")
    parser.add_argument("--training-out", type=Path, default=Path("content/data/training_samples.csv"))
    parser.add_argument("--mystery-out", type=Path, default=Path("content/data/mystery_samples.csv"))
    parser.add_argument("--rows-per-class", type=int, default=24)
    parser.add_argument("--mystery-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    write_datasets(
        training_out=args.training_out,
        mystery_out=args.mystery_out,
        rows_per_class=args.rows_per_class,
        mystery_count=args.mystery_count,
        seed=args.seed,
        force=args.force,
    )
    print(f"Wrote {args.training_out} and {args.mystery_out}")


if __name__ == "__main__":
    main()
