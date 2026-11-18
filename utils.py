"""
Shared helper functions used across the analysis scripts in this repository.

Keeping these routines in one module avoids duplicating I/O, gene-panel
handling, and plotting boilerplate across the individual pipeline stages.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed the standard library, NumPy, and (if available) TensorFlow RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def configure_plot_style(overrides: dict | None = None) -> None:
    """Apply the shared matplotlib style defined in config.PLOT_STYLE."""
    import matplotlib.pyplot as plt

    style = dict(config.PLOT_STYLE)
    if overrides:
        style.update(overrides)
    plt.rcParams.update(style)


def print_header(title: str) -> None:
    """Print a visually distinct section header to stdout."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_dataframe(df: pd.DataFrame, path: Path | str, index: bool = True) -> None:
    """Save a DataFrame to CSV, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Saved: {path}")


def load_gene_panel(path: Path | str = config.GENE_PANEL_PATH) -> list[str]:
    """
    Load the candidate gene panel used throughout the pipeline.

    Expected format: a plain text file with one gene identifier per line.
    Blank lines and lines starting with '#' are ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gene panel file not found: {path}\n"
            "Create a plain text file with one gene identifier per line."
        )
    genes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith("#"):
                genes.append(gene)
    return genes


def load_probe_gene_mapping(
    path: Path | str = config.PROBE_GENE_MAPPING_PATH,
) -> dict[str, list[str]]:
    """
    Load a probe-to-gene mapping table for microarray-based datasets.

    Expected format: a two-column CSV with headers ``probe_id, gene_symbol``.
    A gene may map to more than one probe; all probes are collected into a
    list keyed by gene symbol.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Probe-to-gene mapping file not found: {path}\n"
            "Create a two-column CSV with headers 'probe_id,gene_symbol'."
        )
    mapping_df = pd.read_csv(path)
    mapping: dict[str, list[str]] = {}
    for probe_id, gene_symbol in zip(mapping_df["probe_id"], mapping_df["gene_symbol"]):
        mapping.setdefault(str(gene_symbol).strip(), []).append(str(probe_id).strip())
    return mapping


def align_feature_order(df: pd.DataFrame, feature_list: Iterable[str]) -> pd.DataFrame:
    """Return ``df`` restricted and reordered to ``feature_list`` (columns).

    Missing features are added as all-NaN columns so that downstream code can
    rely on a fixed, consistent column order.
    """
    out = df.copy()
    for feature in feature_list:
        if feature not in out.columns:
            out[feature] = np.nan
    return out[list(feature_list)]


def low_expression_prevalence(
    expression_df: pd.DataFrame,
    threshold: float = config.EXPRESSION_THRESHOLD,
) -> pd.Series:
    """
    Fraction of samples in which each feature (column) falls below ``threshold``.

    ``expression_df`` is expected in samples-by-features orientation.
    """
    return (expression_df < threshold).sum(axis=0) / expression_df.shape[0]


def genes_low_in_both_groups(
    group_a_expr: pd.DataFrame,
    group_b_expr: pd.DataFrame,
    expression_threshold: float = config.EXPRESSION_THRESHOLD,
    prevalence_threshold: float = config.PREVALENCE_THRESHOLD,
) -> pd.Series:
    """
    Boolean mask (indexed by feature) flagging genes that are low-expressed
    in more than ``prevalence_threshold`` of samples in *both* groups.

    A gene is retained downstream unless it is low in both groups, so that a
    gene expressed in only one of the two conditions is not discarded.
    """
    frac_low_a = low_expression_prevalence(group_a_expr, expression_threshold)
    frac_low_b = low_expression_prevalence(group_b_expr, expression_threshold)
    common_index = frac_low_a.index.union(frac_low_b.index)
    frac_low_a = frac_low_a.reindex(common_index, fill_value=1.0)
    frac_low_b = frac_low_b.reindex(common_index, fill_value=1.0)
    return (frac_low_a >= prevalence_threshold) & (frac_low_b >= prevalence_threshold)
