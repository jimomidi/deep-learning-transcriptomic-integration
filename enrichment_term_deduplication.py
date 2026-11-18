"""
Identify duplicate annotation terms within functional enrichment result
tables (e.g. GO Biological Process, KEGG, Reactome exports).

Enrichment tools occasionally emit the same annotation term more than once
across different result blocks. This script flags any row whose ``Term``
value is not unique within a given result file and writes the duplicated
rows out for manual review, for each configured enrichment result file.

Usage
-----
    python enrichment_term_deduplication.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from utils import print_header, save_dataframe


def find_duplicate_terms(path: Path | str, term_column: str = "Term") -> pd.DataFrame:
    """Return all rows of ``path`` whose ``term_column`` value is duplicated."""
    df = pd.read_csv(path)
    if term_column not in df.columns:
        raise ValueError(f"{path} has no '{term_column}' column.")
    return df[df.duplicated(subset=term_column, keep=False)]


def main() -> None:
    config.ensure_output_dirs()

    for label, path in config.ENRICHMENT_FILES.items():
        print_header(f"CHECKING DUPLICATE TERMS: {label}")
        if not Path(path).exists():
            print(f"Skipping {label}: file not found at {path}")
            continue
        duplicates = find_duplicate_terms(path)
        print(f"{len(duplicates)} duplicated row(s) found.")
        output_path = config.RESULTS_DIR / f"{label.lower()}_duplicate_terms.csv"
        save_dataframe(duplicates, output_path, index=False)


if __name__ == "__main__":
    main()
