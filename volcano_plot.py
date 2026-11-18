"""
Generate volcano plots highlighting a candidate gene panel on top of a
genome-wide (or panel-wide) differential expression background.

Each input DE table is expected to contain at least the columns:
    - a feature identifier column (gene symbol or probe ID)
    - 'logFC'
    - 'P.Value'

When the DE table is indexed by probe ID rather than gene symbol (as is
common for microarray platforms), supply a probe-to-gene mapping so that a
gene with multiple probes can be resolved to a single representative point:
the probe with the largest absolute log2 fold change is used.

Usage
-----
    python volcano_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

import config
from utils import configure_plot_style, load_gene_panel, load_probe_gene_mapping, print_header

# A palette large enough to give every highlighted gene a distinct color.
_DEFAULT_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#FFFF33", "#A65628", "#F781BF", "#999999", "#1B9E77",
    "#39FF14", "#7570B3", "#FFD92F", "#66A61E", "#E6AB02",
    "#A6761D", "#666666", "#E7298A", "#1F78B4", "#B15928",
    "#6A3D9A", "#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072",
    "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9",
    "#BC80BD", "#CCEBC5", "#FFED6F", "#17BECF", "#AEC7E8",
    "#FF9896", "#98DF8A", "#C5B0D5", "#C49C94", "#F7B6D2",
]


def resolve_representative_probe(
    df: pd.DataFrame,
    gene_to_probes: dict[str, list[str]],
    feature_col: str,
    logfc_col: str = "logFC",
) -> dict[str, str]:
    """
    For each gene mapped to one or more probes, pick a single representative
    probe/row to plot: the one with the largest absolute log2 fold change.

    Genes mapped to exactly one probe present in ``df`` are passed through
    unchanged.
    """
    representative: dict[str, str] = {}
    for gene, probes in gene_to_probes.items():
        candidates = [p for p in probes if p in set(df[feature_col])]
        if not candidates:
            continue
        if len(candidates) == 1:
            representative[gene] = candidates[0]
            continue
        best_probe, best_value = None, -np.inf
        for probe in candidates:
            value = df.loc[df[feature_col] == probe, logfc_col].abs().max()
            if pd.notna(value) and value > best_value:
                best_value, best_probe = value, probe
        if best_probe is not None:
            representative[gene] = best_probe
    return representative


def plot_volcano(
    df: pd.DataFrame,
    highlight_genes: list[str],
    output_path: Path | str,
    feature_col: str = "gene",
    logfc_col: str = "logFC",
    pvalue_col: str = "P.Value",
    p_cut: float = 0.05,
    fc_cut: float = 1.0,
    x_limit: float | None = None,
    gene_to_representative_id: dict[str, str] | None = None,
    show_legend: bool = True,
    palette: list[str] | None = None,
) -> None:
    """
    Draw and save a volcano plot with a highlighted candidate gene panel.

    Parameters
    ----------
    df:
        Differential expression table containing ``feature_col``, ``logfc_col``,
        and ``pvalue_col``.
    highlight_genes:
        Genes to highlight with distinct colors.
    gene_to_representative_id:
        Optional mapping from gene symbol to the identifier used as the row
        key in ``df`` (e.g. a representative probe ID). If omitted,
        ``highlight_genes`` values are looked up directly in ``feature_col``.
    """
    palette = palette or _DEFAULT_PALETTE
    if len(highlight_genes) > len(palette):
        raise ValueError("Palette is smaller than the number of genes to highlight.")

    gene_to_id = gene_to_representative_id or {g: g for g in highlight_genes}
    highlighted_ids = {gene_to_id[g] for g in highlight_genes if g in gene_to_id}

    y_cut = -np.log10(p_cut)
    x_limit = x_limit or (df[logfc_col].abs().max() * 1.08)

    fig, ax = plt.subplots(figsize=(5.5, 6))

    background = ~df[feature_col].astype(str).isin(highlighted_ids)
    ax.scatter(
        df.loc[background, logfc_col],
        -np.log10(df.loc[background, pvalue_col]),
        color="black", alpha=0.4, s=22, edgecolor="none",
    )

    ax.axvline(x=-fc_cut, color="black", linestyle="--", lw=1)
    ax.axvline(x=fc_cut, color="black", linestyle="--", lw=1)
    ax.axhline(y=y_cut, color="black", linestyle="--", lw=1)

    ax.set_xlim(-x_limit, x_limit)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)

    ax.add_patch(Polygon(
        [[fc_cut, y_cut], [x_limit, y_cut], [x_limit, ylim[1]], [fc_cut, ylim[1]]],
        facecolor="#FFB3B3", alpha=0.35, edgecolor="none", zorder=0,
    ))
    ax.add_patch(Polygon(
        [[-x_limit, y_cut], [-fc_cut, y_cut], [-fc_cut, ylim[1]], [-x_limit, ylim[1]]],
        facecolor="#B3E0FF", alpha=0.35, edgecolor="none", zorder=0,
    ))

    legend_handles = []
    for color, gene in zip(palette, highlight_genes):
        feature_id = gene_to_id.get(gene)
        if feature_id is None:
            continue
        subset = df[df[feature_col].astype(str) == str(feature_id)]
        if subset.empty:
            continue
        ax.scatter(
            subset[logfc_col], -np.log10(subset[pvalue_col]),
            color=color, s=110, edgecolor="none", zorder=10,
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor=color, markeredgecolor="none", label=gene)
        )

    ax.set_xlabel("log2 Fold Change", fontsize=16, fontweight="bold", labelpad=8)
    ax.set_ylabel("-log10 P-value", fontsize=16, fontweight="bold", labelpad=8)

    if show_legend and legend_handles:
        ax.legend(
            handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, ncol=max(1, len(legend_handles) // 8), fontsize=9,
            columnspacing=1.2, handletextpad=0.5, markerscale=1.5,
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_dataset_a_volcano(gene_panel: list[str]) -> None:
    """Volcano plot for the microarray-based dataset (probe-level DE table)."""
    df = pd.read_csv(config.DATASET_A_VOLCANO_PATH)
    gene_to_probes = load_probe_gene_mapping()
    representative = resolve_representative_probe(df, gene_to_probes, feature_col="gene")
    plot_volcano(
        df,
        highlight_genes=[g for g in gene_panel if g in representative],
        output_path=config.RESULTS_DIR / "volcano_dataset_a.png",
        p_cut=config.DATASET_A_PVALUE_CUTOFF,
        fc_cut=config.DATASET_A_LOGFC_CUTOFF,
        gene_to_representative_id=representative,
    )


def run_dataset_b_volcano(gene_panel: list[str]) -> None:
    """Volcano plot for the RNA-seq-based dataset (gene-level DE table)."""
    df = pd.read_csv(config.DATASET_B_VOLCANO_PATH)
    plot_volcano(
        df,
        highlight_genes=gene_panel,
        output_path=config.RESULTS_DIR / "volcano_dataset_b.png",
        p_cut=config.DATASET_B_PVALUE_CUTOFF,
        fc_cut=config.DATASET_B_LOGFC_CUTOFF,
    )


def main() -> None:
    config.ensure_output_dirs()
    configure_plot_style()

    gene_panel = load_gene_panel()

    print_header("VOLCANO PLOT: DATASET A")
    run_dataset_a_volcano(gene_panel)

    print_header("VOLCANO PLOT: DATASET B")
    run_dataset_b_volcano(gene_panel)


if __name__ == "__main__":
    main()
