"""
Compare pairwise co-expression structure among a candidate gene panel between
a case group and a control group, within each dataset.

For each dataset, this script:
1. Builds a gene-by-sample expression matrix restricted to the candidate
   gene panel.
2. Computes the Pearson correlation matrix among genes separately in the
   case group and the control group.
3. Computes the differential correlation matrix (case minus control).
4. Renders a clustered heatmap of the differential correlation matrix.

Usage
-----
    python differential_coexpression_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage

import config
from utils import (
    configure_plot_style,
    genes_low_in_both_groups,
    load_gene_panel,
    load_probe_gene_mapping,
    print_header,
)


def compute_differential_coexpression(
    expr_case: pd.DataFrame, expr_control: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the case-minus-control differential gene-gene correlation matrix.

    Both inputs are expected in samples-by-genes orientation (as returned by
    ``DataFrame.corr()``, which correlates columns pairwise) and restricted
    to the same set of genes.
    """
    corr_case = expr_case.corr()
    corr_control = expr_control.corr()
    return corr_case - corr_control


def plot_differential_coexpression_heatmap(
    corr_diff: pd.DataFrame,
    output_path: Path | str,
    title: str | None = None,
) -> None:
    """Render a clustered heatmap of a differential co-expression matrix."""
    linkage_matrix = linkage(corr_diff.T, method="average")

    grid = sns.clustermap(
        corr_diff,
        col_linkage=linkage_matrix,
        cmap="PiYG_r",
        center=0,
        figsize=(6, 6.5),
        row_cluster=False,
        col_cluster=True,
        linewidths=0.3,
        linecolor="black",
        vmin=-1,
        vmax=1,
        dendrogram_ratio=(0.0, 0.15),
        cbar_pos=None,
    )

    heatmap_pos = grid.ax_heatmap.get_position()
    cbar_ax = grid.fig.add_axes(
        [heatmap_pos.x1 + 0.01, heatmap_pos.y0, 0.03, heatmap_pos.height]
    )
    plt.colorbar(grid.ax_heatmap.collections[0], cax=cbar_ax)

    grid.ax_heatmap.set_xticklabels(corr_diff.columns, rotation=90, fontsize=8)
    grid.ax_heatmap.set_yticklabels(corr_diff.index, rotation=0, fontsize=8)
    grid.ax_heatmap.yaxis.set_ticks_position("left")
    grid.ax_heatmap.yaxis.set_label_position("left")
    grid.ax_heatmap.tick_params(axis="y", labelleft=True, labelright=False)
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    if title:
        grid.ax_heatmap.set_title(title)

    for spine in grid.ax_heatmap.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"Saved: {output_path}")


def run_dataset_a_coexpression(gene_panel: list[str]) -> None:
    """Case-vs-control differential co-expression for the microarray dataset."""
    expr_df = pd.read_csv(config.DATASET_A_EXPRESSION_PATH, index_col=0)
    metadata = pd.read_csv(config.DATASET_A_METADATA_PATH)

    metadata["group"] = metadata[config.GROUP_ANNOTATION_COLUMN].str.extract(
        config.GROUP_ANNOTATION_PATTERN
    )
    metadata = metadata[metadata[config.SAMPLE_ID_COLUMN].isin(expr_df.columns)]

    control_samples = metadata.loc[
        metadata["group"] == config.CONTROL_GROUP_LABEL, config.SAMPLE_ID_COLUMN
    ]
    case_samples = metadata.loc[
        metadata["group"] != config.CONTROL_GROUP_LABEL, config.SAMPLE_ID_COLUMN
    ]

    gene_to_probes = load_probe_gene_mapping()
    used_probes: set[str] = set()
    gene_expression = {}
    for gene in gene_panel:
        probes = [p for p in gene_to_probes.get(gene, []) if p in expr_df.index]
        available = [p for p in probes if p not in used_probes] or probes[:1]
        if not available:
            continue
        used_probes.update(available)
        gene_expression[gene] = expr_df.loc[available].mean(axis=0)

    expr_gene = pd.DataFrame(gene_expression).T  # genes x samples
    corr_diff = compute_differential_coexpression(
        expr_gene[case_samples].T, expr_gene[control_samples].T
    )
    plot_differential_coexpression_heatmap(
        corr_diff,
        config.RESULTS_DIR / "differential_coexpression_dataset_a.png",
        title="Dataset A: Case - Control",
    )


def run_dataset_b_coexpression(gene_panel: list[str]) -> None:
    """Case-vs-control differential co-expression for the RNA-seq dataset."""
    control_df = pd.read_csv(config.DATASET_B_CONTROL_EXPRESSION_PATH)
    control_df = control_df[control_df[config.CONTROL_GENE_ID_COLUMN].isin(gene_panel)]
    control_df = control_df.set_index(config.CONTROL_GENE_ID_COLUMN)
    expr_control = control_df.drop(columns=[config.CONTROL_SAMPLE_LABEL_COLUMN], errors="ignore").T
    expr_control = expr_control.apply(pd.to_numeric, errors="coerce")

    case_df = pd.read_csv(config.DATASET_B_CASE_EXPRESSION_PATH)
    case_df[config.CASE_GENE_ID_COLUMN] = (
        case_df[config.CASE_GENE_ID_COLUMN].astype(str).str.replace(r"\..*", "", regex=True)
    )
    case_df = case_df[case_df[config.CASE_GENE_NAME_COLUMN].isin(gene_panel)]
    expr_case = case_df.pivot_table(
        index=config.CASE_SAMPLE_ID_COLUMN,
        columns=config.CASE_GENE_NAME_COLUMN,
        values=config.CASE_EXPRESSION_VALUE_COLUMN,
        aggfunc="mean",
    )

    all_genes = sorted(set(expr_control.columns) | set(expr_case.columns))
    expr_control = expr_control.reindex(columns=all_genes)
    expr_case = expr_case.reindex(columns=all_genes)

    both_low = genes_low_in_both_groups(expr_control, expr_case)

    expr_control = np.log2(expr_control + 0.01)
    expr_case = np.log2(expr_case + 0.01)

    corr_diff = compute_differential_coexpression(expr_case, expr_control).fillna(0)
    for gene in both_low[both_low].index:
        corr_diff.loc[gene, :] = 0
        corr_diff.loc[:, gene] = 0

    plot_differential_coexpression_heatmap(
        corr_diff,
        config.RESULTS_DIR / "differential_coexpression_dataset_b.png",
        title="Dataset B: Case - Control",
    )


def main() -> None:
    config.ensure_output_dirs()
    configure_plot_style()
    gene_panel = load_gene_panel()

    print_header("DIFFERENTIAL CO-EXPRESSION: DATASET A")
    run_dataset_a_coexpression(gene_panel)

    print_header("DIFFERENTIAL CO-EXPRESSION: DATASET B")
    run_dataset_b_coexpression(gene_panel)


if __name__ == "__main__":
    main()
