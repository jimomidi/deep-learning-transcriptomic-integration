"""
Identify genes with shared differential expression across two conditions.

This script takes two independently derived differential-expression (DE)
result tables (one per dataset/condition) and:

1. Splits each table into up- and down-regulated gene sets using its
   log2 fold-change sign.
2. Computes the overall overlap between the two DE gene sets, irrespective
   of direction.
3. Computes direction-specific overlap categories:
   - concordant up-regulation (up in both conditions)
   - concordant down-regulation (down in both conditions)
   - discordant regulation (up in condition A / down in condition B, and
     the reverse)

Each DE result table is expected to be a CSV file with at minimum the
columns ``gene`` and ``logFC`` (additional columns such as ``P.Value`` are
preserved but not required for this step, since significance filtering is
assumed to already be reflected in the input table).

Usage
-----
    python differential_expression_overlap.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from utils import print_header, save_dataframe


def load_de_table(path: Path | str, gene_col: str = "gene", logfc_col: str = "logFC") -> pd.DataFrame:
    """Load a differential expression result table and validate required columns."""
    df = pd.read_csv(path)
    missing = {gene_col, logfc_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    return df


def split_up_down(
    de_df: pd.DataFrame, gene_col: str = "gene", logfc_col: str = "logFC"
) -> tuple[pd.Series, pd.Series]:
    """Split a DE table into up-regulated and down-regulated gene name sets."""
    up_genes = de_df.loc[de_df[logfc_col] > 0, gene_col]
    down_genes = de_df.loc[de_df[logfc_col] < 0, gene_col]
    return up_genes, down_genes


def compute_direction_specific_overlap(
    up_a: pd.Series,
    down_a: pd.Series,
    up_b: pd.Series,
    down_b: pd.Series,
) -> dict[str, list[str]]:
    """
    Compute all overlap categories between two DE gene sets.

    Returns a dictionary with keys:
        'shared_any_direction', 'concordant_up', 'concordant_down',
        'discordant_a_up_b_down', 'discordant_a_down_b_up'
    """
    set_up_a, set_down_a = set(up_a), set(down_a)
    set_up_b, set_down_b = set(up_b), set(down_b)
    all_a = set_up_a | set_down_a
    all_b = set_up_b | set_down_b

    return {
        "shared_any_direction": sorted(all_a & all_b),
        "concordant_up": sorted(set_up_a & set_up_b),
        "concordant_down": sorted(set_down_a & set_down_b),
        "discordant_a_up_b_down": sorted(set_up_a & set_down_b),
        "discordant_a_down_b_up": sorted(set_down_a & set_up_b),
    }


def summarize_overlap(overlap: dict[str, list[str]]) -> pd.DataFrame:
    """Build a tidy long-format summary table (category, gene) from the overlap dict."""
    rows = [
        {"category": category, "gene": gene}
        for category, genes in overlap.items()
        for gene in genes
    ]
    return pd.DataFrame(rows, columns=["category", "gene"])


def main() -> None:
    config.ensure_output_dirs()

    print_header("LOADING DIFFERENTIAL EXPRESSION RESULTS")
    de_a = load_de_table(config.DATASET_A_DEG_PATH)
    de_b = load_de_table(config.DATASET_B_DEG_PATH)
    print(f"Dataset A DE table: {de_a.shape[0]} genes")
    print(f"Dataset B DE table: {de_b.shape[0]} genes")

    up_a, down_a = split_up_down(de_a)
    up_b, down_b = split_up_down(de_b)

    print_header("COMPUTING SHARED DYSREGULATED GENES")
    overlap = compute_direction_specific_overlap(up_a, down_a, up_b, down_b)
    for category, genes in overlap.items():
        print(f"{category}: {len(genes)} gene(s)")

    summary_df = summarize_overlap(overlap)
    save_dataframe(
        summary_df,
        config.RESULTS_DIR / "shared_differential_expression_overlap.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
